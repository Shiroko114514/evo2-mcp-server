# Evo2-7B Bioinformatics MCP Server

一个把 NVIDIA 托管的 **Evo2-7B Forward API** 封装成 MCP (Model Context Protocol) Tools 的服务器，让 Claude Code、Cursor、Codex 等 Agent 可以用自然语言驱动 Evo2：

```
Agent
  ↓
MCP Tool
  ↓
Evo2 MCP Server（本项目）
  ↓
NVIDIA Evo2-7B Forward API
  ↓
forward outputs → likelihood / variant scores
  ↓
Agent
```

```text
POST https://health.api.nvidia.com/v1/biology/arc/evo2-7b/forward
Authorization: Bearer $NVIDIA_API_KEY
```

---

## 1. 项目介绍

提供 5 个 MCP Tools：

| Tool | 作用 |
|---|---|
| `evo2_forward` | 对 DNA sequence 执行 Evo2-7B forward inference，返回指定 layer 的输出统计（或保存原始 tensor） |
| `evo2_score` | 计算 Evo2 模型对 DNA sequence 的 model-based likelihood（total / mean / per-position） |
| `evo2_variant_score` | 比较单个 nucleotide variant 对 Evo2 sequence likelihood 的影响（Δ log-likelihood） |
| `evo2_batch_score` | 批量比较多个 nucleotide variants（复用一次 WT forward，并发受限，自动去重） |
| `evo2_score_fasta` | 对 FASTA 中每个 record 评分（本地路径受 `EVO2_MCP_ALLOWED_DIRS` 沙箱限制） |

本项目是 **Bioinformatics MCP Tool Server**，不是简单的 HTTP API wrapper：

- 自动校验 / 规范化 DNA sequence（大写、去空白、非法字符明确报错）
- 基于官方语义计算 likelihood（byte-level tokenizer + causal shift，见 §17）
- 输出分层设计（summary / raw / save），防止 MCP context 爆炸
- 完整错误分类（400/401/403/404/408/413/422/429/5xx/timeout）+ retry/backoff
- API Key 只从环境变量读取，绝不写死、绝不进日志
- 配套批量脚本：`scripts/score_fasta.py`（批量 FASTA 评分 + **embedding 提取**，见 §18）与 `scripts/analyze_run.py`（聚类/分类/回归下游分析，见 §19）

## 2. Evo2 API 介绍

[Evo2](https://github.com/ArcInstitute/evo2)（Arc Institute / NVIDIA）是 DNA 基础模型（StripedHyena2 架构），7B 版 32 层、Apache-2.0、训练上下文达 1M bp。NVIDIA 提供托管 NIM 服务：

- **Forward 端点**：`POST https://health.api.nvidia.com/v1/biology/arc/evo2-7b/forward`
- **请求体**（官方 OpenAPI `ForwardInputs`）：
  ```json
  { "sequence": "ACGTACGT...", "output_layers": ["output_layer"] }
  ```
  `output_layers` 支持 1–100 个 layer 名（如 `output_layer`、`decoder.layers.24.mlp.linear_fc2`、`decoder.layers.3.self_attention`、`embedding`、`decoder.final_norm`）。
- **响应体**（官方 OpenAPI `ForwardOutputs`）：`{"data": "<base64 编码的 NPZ>", "elapsed_ms": <int>}`；超大响应可能以 `Content-Type: application/zip`（原始 NPZ 字节）返回。
- **`output_layer` = 最终 logits**，shape `[seq_len, batch_size, 512]`（512 是 byte-level tokenizer 的 padded vocabulary size）。

> ⚠️ **Deprecation 提醒（2026-08-24 核实）**：build.nvidia.com 上托管的 `arc/evo2-7b` 端点已标记 **Deprecated**（页面提示 "This NIM Endpoint has been deprecated"）。官方 NIM 文档（`docs.nvidia.com/nim/bionemo/evo2/latest/`）描述的 API 与托管端点完全一致；如托管端点不可用，可改为自托管 NIM 容器，把 `EVO2_MCP_BASE_URL` 指向 `http://localhost:8000/biology/arc/evo2`。

### 已核实的官方事实（实现依据，2026-08-24）

| 项目 | 结论 | 来源 |
|---|---|---|
| 响应格式 | JSON `{"data": base64-NPZ, "elapsed_ms"}`；或 `application/zip` 原始 NPZ | NVIDIA NIM endpoints 文档 + 托管 OpenAPI schema（`ForwardOutputs`） |
| `output_layer` shape | `[seq_len, batch_size, 512]`，float，**就是 logits** | 同上（"Final output/logits"） |
| vocabulary 大小 | 512（padded）；byte-level tokenizer，每 bp = 1 个 token | NVIDIA 文档 + Arc/vortex `CharLevelTokenizer(512)` |
| A/C/G/T → logits index | A=65, C=67, T=84, G=71（ASCII 字节值） | NVIDIA 文档原文 + `np.frombuffer(text.encode(), np.uint8)` |
| BOS/EOS/offset | 默认无 BOS（Arc `score_sequences` `prepend_bos=False`）；eod_id=0, pad_id=1 | Arc `evo2/models.py` + `evo2/scoring.py` |
| likelihood 计算 | `log_softmax(logits, -1)` 后 causal shift：`logits[:, :-1]` vs `input_ids[:, 1:]`；**position 0 不参与计分**，长度 N 的序列得 N-1 个分数 | Arc `evo2/scoring.py` `logits_to_logprobs` |
| 特殊 token | 输出中只有 A/C/G/T 4 个 token 有意义（NIM 文档原文） | NVIDIA 文档 |

### 实测与文档的差异（2026-08-24 live 验证，任务要求记录实际接口）

用真实 key 探测托管 `health.api.nvidia.com` 端点后发现**文档中的 layer 名在托管端不适用**：

| 请求的 layer 名 | 托管 API 实际行为 |
|---|---|
| `output_layer`（文档名） | ❌ `422 {"error":"StripedHyena has no attribute 'output_layer'"}` |
| `decoder.layers.N.*` / `embedding` / `final_norm` | ❌ 422 `has no attribute` |
| **`unembed`**（模型属性名） | ✅ **最终 logits**：NPZ key `unembed.output`，shape `(1, seq_len, 512)`，dtype **float64** |
| `embedding_layer` | ✅ `embedding_layer.output`，`(1, seq, 4096)` |
| `norm` | ✅ `norm.output`，`(1, seq, 4096)` |
| `blocks.N.mlp` / `blocks.N` | ✅ `blocks.N.mlp.output`，`(1, seq, 4096)` |

应对措施（已实现并 live 通过）：

- 新增 `EVO2_MCP_LOGITS_LAYER` 配置（默认 `auto`）：评分工具先尝试文档名 `output_layer`，若收到 `422 has no attribute`（托管端）自动切换到 `unembed` 并缓存，之后不再重复探测；自托管 NIM 2.x 容器则一次成功，零额外请求。
- NPZ 解析器同时支持裸 key（`output_layer`）和 `<name>.output`（`unembed.output`）两种 key 格式，并按"末维 = 512"启发式兜底。
- `422` 错误信息现在会提示托管端可用的属性名。

> 如果返回的 `seq_len` 与输入序列长度不一致（例如服务端加了 padding/BOS），本服务器**拒绝**计算 likelihood 并返回 raw 统计 + 明确说明，绝不猜测对齐。

## 3. NVIDIA API Key 获取方法

1. 打开 <https://build.nvidia.com/>，右上角 **Get API Key**（需要登录 NVIDIA 账号）。
2. 创建一个 key（形如 `nvapi-xxxxxxxx...`）。
3. 把它设置到环境变量，**不要写进代码 / 配置 / Git**：
   ```bash
   export NVIDIA_API_KEY="nvapi-xxxxxxxx"
   ```
   或者复制 `.env.example` 为 `.env` 并填入（`.env` 已被 `.gitignore` 忽略）。

## 4. 安装

```bash
# 推荐：pip / uv
pip install -e ".[dev]"
# 或
uv sync --extra dev

# 推荐（本项目自带）：pixi 项目本地环境
pixi install
pixi run test
```

要求 Python >= 3.10（推荐 3.11+）。核心依赖：`mcp>=2.0`、`httpx>=0.27`、`numpy>=1.26`、`pydantic>=2.6`、`python-dotenv>=1.0`。分析脚本（`scripts/analyze_run.py`）额外需要 dev 依赖：`scikit-learn`、`pandas`、`matplotlib`。

## 5. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `NVIDIA_API_KEY` | 无（必填） | API Key，只从这里读取 |
| `EVO2_MCP_BASE_URL` | `https://health.api.nvidia.com/v1/biology/arc/evo2-7b` | 服务地址（自托管 NIM 时修改） |
| `EVO2_MCP_TIMEOUT` | `120` | HTTP 读超时（秒） |
| `EVO2_MCP_MAX_RETRIES` | `4` | 408/429/5xx 最大重试次数 |
| `EVO2_MCP_MAX_CONCURRENCY` | `2` | batch/FASTA 并发上限 |
| `EVO2_MCP_ALLOWED_DIRS` | 空 | FASTA 允许读取的目录（`:` 分隔） |
| `EVO2_MCP_OUTPUT_DIR` | `./output` | `mode="save"` 输出目录（也是 save_path 唯一允许的位置） |
| `EVO2_MCP_ALLOW_AMBIGUOUS` | `0` | 设为 `1` 允许 N 碱基透传（见 §15） |
| `EVO2_MCP_MAX_SEQUENCE_LENGTH` | `1000000` | 序列长度硬上限 |
| `EVO2_MCP_RAW_INLINE_MAX` | `4096` | `mode="raw"` 允许内联的 tensor 元素总数上限 |
| `EVO2_MCP_MAX_PER_POSITION` | `5000` | per-position 列表返回上限（超出取头尾） |
| `EVO2_MCP_LOGITS_LAYER` | `auto` | 评分用的 logits layer 名：`auto` 自动探测（文档名 `output_layer` 失败时切到托管端 `unembed`）；也可显式指定 |

## 6. CLI 启动

```bash
# 三种方式等价
python -m evo2_mcp
evo2-mcp
uv run evo2-mcp      # 用 uv 管理的项目环境
# pixi 环境：
pixi run evo2-mcp
```

服务器通过 **stdio** 与 MCP 客户端通信，正常启动后不会有输出（等待 MCP 握手）。

## 7. MCP 配置

### Claude Code（`.mcp.json`）

```json
{
  "mcpServers": {
    "evo2": {
      "command": "uv",
      "args": ["run", "evo2-mcp"],
      "env": {
        "NVIDIA_API_KEY": "${NVIDIA_API_KEY}"
      }
    }
  }
}
```

> 注意：`${NVIDIA_API_KEY}` 是否被客户端展开取决于客户端实现。**最稳妥**的方式是直接填入真实 key：

```json
{
  "mcpServers": {
    "evo2": {
      "command": "uv",
      "args": ["run", "evo2-mcp"],
      "env": {
        "NVIDIA_API_KEY": "YOUR_API_KEY"
      }
    }
  }
}
```

但 **千万不要把含真实 key 的 `.mcp.json` 提交到 Git**（把该文件加入 `.gitignore`，或用环境变量/密钥管理工具注入）。也可以不传 `env`，让服务器进程自己从环境或 `.env` 读取 `NVIDIA_API_KEY`：

```json
{
  "mcpServers": {
    "evo2": {
      "command": "uv",
      "args": ["run", "evo2-mcp"]
    }
  }
}
```

### Cursor（`~/.cursor/mcp.json` 或项目 `.cursor/mcp.json`）

```json
{
  "mcpServers": {
    "evo2": {
      "command": "uv",
      "args": ["run", "evo2-mcp"],
      "env": { "NVIDIA_API_KEY": "YOUR_API_KEY" }
    }
  }
}
```

### Codex（`~/.codex/config.toml`）

```toml
[mcp_servers.evo2]
command = "uv"
args = ["run", "evo2-mcp"]
env = { "NVIDIA_API_KEY" = "YOUR_API_KEY" }
```

### 自托管 NIM

```json
{
  "mcpServers": {
    "evo2": {
      "command": "uv",
      "args": ["run", "evo2-mcp"],
      "env": {
        "EVO2_MCP_BASE_URL": "http://localhost:8000/biology/arc/evo2"
      }
    }
  }
}
```

## 8. Tool 列表

### `evo2_forward(sequence, output_layers=["output_layer"], mode="summary", save_path=None)`

对 DNA sequence 执行 Evo2-7B forward inference。`mode`：

- `"summary"`（默认）：每层返回 `shape / dtype / min / max / mean / std`，context 安全；
- `"save"`：把原始 tensor 存成 `.npz`（`output/evo2_forward_<时间戳>.npz`），返回路径；
- `"raw"`：内联完整 tensor（仅当元素总数 ≤ `EVO2_MCP_RAW_INLINE_MAX`，默认 4096，防止 context 爆炸）。

> **layer 名注意**：托管 `health.api.nvidia.com` 端点接受的是模型属性名（logits 用 **`unembed`**，另有 `embedding_layer`、`norm`、`blocks.N.mlp`），文档名 `output_layer`/`decoder.layers.N.*` 只适用于自托管 NIM 2.x 容器。`evo2_score`/`evo2_variant_score`/`evo2_batch_score`/`evo2_score_fasta` 会自动探测，无需手动指定；只有直接调用 `evo2_forward` 时需要按实际端点选名。

### `evo2_score(sequence, include_per_position=False)`

计算 Evo2 对序列的 likelihood：

```json
{
  "sequence_length": 123,
  "total_log_likelihood": -123.45,
  "mean_log_likelihood": -1.2345,
  "scored_positions": 122,
  "per_position_log_likelihood": null,
  "method_notes": "...",
  "disclaimer": "..."
}
```

语义（与 Arc 官方实现一致）：`logits[i]` 预测位置 `i+1` 的碱基，在完整 512 vocab 上做 log-softmax 后取目标碱基字节索引；**position 0 不参与计分**，因此 `scored_positions = length - 1`，`mean` 是对这 N-1 个值取平均。`per_position_log_likelihood[k]` 对应序列 0-based 位置 `k+1`（即 1-based 位置 `k+2`）。若 API 返回的 `seq_len` 无法与序列对齐，**不伪造结果**，返回 raw 统计并明确说明：

```text
Likelihood calculation is not supported until the API output format is verified.
```

### `evo2_variant_score(sequence, position, ref, alt, coordinate="1-based", include_per_position=False)`

```json
{
  "position": 100,
  "ref": "A",
  "alt": "G",
  "wildtype_log_likelihood": -500.1,
  "mutant_log_likelihood": -500.5,
  "delta_log_likelihood": -0.4,
  "interpretation": "The mutant sequence is less likely than the wildtype under Evo2-7B ... (NOT a clinical pathogenicity call)"
}
```

校验链：position 范围 → coordinate 换算 → ref 必须与序列该位一致 → ref≠alt → 1-based 位置 1（0-based 0）不可评分（causal LM 无法给第一个 token 赋概率）→ 明确报错。

### `evo2_batch_score(sequence, variants, coordinate="1-based")`

```json
{
  "sequence_length": 300,
  "wildtype_log_likelihood": -1200.0,
  "variants": [
    { "position": 100, "ref": "A", "alt": "G", "delta_log_likelihood": -0.42 },
    { "position": 200, "ref": "C", "alt": "T", "delta_log_likelihood": 0.13 }
  ]
}
```

- **WT forward 只计算一次**并复用于所有 variant；
- 相同 `(position, alt)` 的 mutant 只 forward 一次（memoize）；
- 并发受 `EVO2_MCP_MAX_CONCURRENCY` 限制（默认 2，尊重 NVIDIA rate limit）；
- 单个 variant 失败不影响整个 batch（逐条返回 `error`）。

### `evo2_score_fasta(fasta_path=None, fasta_text=None)`

```text
>sequence_1
ACGTACGT...
>sequence_2
TTGGCCAA...
```

- `fasta_text`：内联 FASTA（默认可用，有大小/记录数上限）；
- `fasta_path`：仅当文件位于 `EVO2_MCP_ALLOWED_DIRS` 内才允许读取，否则明确拒绝；
- 逐条返回 `total_log_likelihood / mean_log_likelihood`，单条错误不影响其余。

## 9. 使用示例

```json
{
  "sequence": "acgtACGT acgt",           // 小写 + 空白自动处理
  "output_layers": ["output_layer"],
  "mode": "summary"
}
```

返回：

```json
{
  "sequence_length": 12,
  "requested_output_layers": ["output_layer"],
  "returned_layers": ["output_layer"],
  "layer_stats": [
    { "name": "output_layer", "shape": [12, 1, 512], "dtype": "float32",
      "size": 6144, "min": -3.21, "max": 4.02, "mean": 0.01, "std": 0.98 }
  ],
  "api": { "elapsed_ms": 87 }
}
```

Agent 想要完整 logits 时：

```json
{ "sequence": "ACGT...", "mode": "save" }
```

```json
{
  "saved": true,
  "path": "/abs/path/output/evo2_forward_20260824_153000.npz",
  "bytes_on_disk": 24576,
  "layer_stats": [...]
}
```

## 10. FASTA 示例

```json
{
  "fasta_text": ">geneA\nACGTACGTACGT\n>geneB\nTTGGCCAATTGG"
}
```

（或 `"fasta_path": "/data/genomes/genes.fa"`，需配置 `EVO2_MCP_ALLOWED_DIRS=/data/genomes`）

大批量 FASTA（如整目录的 enhancer/promoter）评分 + 提取 embedding，用 `scripts/score_fasta.py`（见 §18）—— 每次运行输出独立 run 文件夹（`scores.csv` + `embeddings.npz`）。

## 11. Variant scoring 示例

```json
{
  "sequence": "ACGTACGTACGTACGTACGT",
  "position": 10,
  "ref": "A",
  "alt": "G"
}
```

## 12. Batch scoring 示例

```json
{
  "sequence": "ACGTACGTACGTACGTACGT",
  "variants": [
    { "position": 10, "ref": "A", "alt": "G" },
    { "position": 12, "ref": "T", "alt": "C" },
    { "position": 14, "ref": "A", "alt": "T" }
  ]
}
```

典型 Agent 工作流（对应"分析序列上所有 SNP，找 Evo2 score 变化最大的前 20 个"）：

```text
读取输入 → 解析 DNA / VCF → 生成 WT / mutant → evo2_batch_score
→ 按 |delta_log_likelihood| 排序 → 取前 20 → 保存 CSV → 解释结果
```

## 13. 错误处理

| HTTP | 含义 | 本服务器行为 |
|---|---|---|
| 400 | Bad Request（含非法序列等） | 直接报错，附响应摘要 |
| 401 | **API Key 无效** | 直接报错，提示检查 `NVIDIA_API_KEY` |
| 403 | 无权限（托管端点已弃用等） | 直接报错，提示可能原因 |
| 404 | 路径不存在 | 直接报错，提示检查 `EVO2_MCP_BASE_URL` |
| 408 | 服务端超时 | **有限重试**（backoff）后报错 |
| 413 | Payload 过大 | 直接报错，提示减小序列或 layer 数量 |
| 422 | 参数校验失败 | 直接报错，附详情 |
| 429 | **Rate limit** | **重试 + exponential backoff**（尊重 `Retry-After`，上限 60s），上限 `EVO2_MCP_MAX_RETRIES` |
| 5xx | NVIDIA 服务端错误 | **有限重试**后报错 |
| timeout | 请求超时（`EVO2_MCP_TIMEOUT` 秒） | 明确报错：`NVIDIA Evo2 API request timed out.`，不抛裸 traceback |

所有错误通过 MCP 返回结构化 JSON：`{"error": "Evo2APIError", "message": "..."}`。`evo2_batch_score` 中单条失败返回 `{"error": ..., "status": ...}`，不中断整批。

## 14. Rate limit

NVIDIA 托管 NIM 有 rate limit。设计对策：

- `EVO2_MCP_MAX_CONCURRENCY`（默认 2）限制并发；
- 429 → exponential backoff（1s, 2s, 4s, 8s, 16s…，封顶 30s + 抖动；若有 `Retry-After` 优先遵循但封顶 60s）；
- 重试上限 `EVO2_MCP_MAX_RETRIES`（默认 4），**不会无限重试**；
- batch 内 WT 只算一次、相同 mutant 去重，减少请求数。

## 15. 安全说明

- **API Key**：只从 `NVIDIA_API_KEY` 环境变量（或 `.env`）读取；代码中没有任何硬编码 key；日志只记录 URL、序列长度、layer 名，**不记录序列内容和 key**；错误信息只包含响应前 500 字符摘要。
- **序列隐私**：所有日志/错误只含 `preview`（如 `ACGT...GCTA (len=12345)`）。
- **路径沙箱**：
  - FASTA 读取仅限 `EVO2_MCP_ALLOWED_DIRS`；未配置时拒绝一切本地路径；
  - `mode="save"` 的 `save_path` 必须位于 `EVO2_MCP_OUTPUT_DIR` 内；
- **`.gitignore`** 已包含 `.env`、`*.env`、`output/`、`*.npz`。
- **N 碱基**：默认拒绝并明确报错（Evo2 模型未在含糊碱基上评测过，文档只保证 A/C/G/T 有意义）。如确需透传 N，用 `EVO2_MCP_ALLOW_AMBIGUOUS=1` 启动 —— 这是显式选择，不是静默丢弃。
- **Don't execute**：本服务器不做任何 shell 执行；Agent 通过 FASTA/序列输入只能触发受限的 HTTP 请求。

## 16. 生物学解释限制

- Evo2 score 是 **model-based sequence likelihood change**，不是实验证据，更不是临床致病性诊断。
- `delta_log_likelihood < 0` 只能解读为"突变体序列在模型下**更不可能**"，不能解读为"致病"。
- 需要 downstream 验证（实验、群体频率、ClinVar 注释、蛋白结构影响等）才能谈 pathogenicity。
- 每个 Tool 的 description 都带有以下声明（MCP 客户端可见）：

```text
This is a DNA foundation model inference tool. It does not provide clinical
diagnosis. Model scores should not be interpreted as pathogenicity labels
without additional validation.
```

## 17. 实现依据与验证来源（2026-08-24）

- NVIDIA NIM for Evo 2 — Endpoints：<https://docs.nvidia.com/nim/bionemo/evo2/latest/endpoints.html>
- NVIDIA NIM for Evo 2 — Quickstart：<https://docs.nvidia.com/nim/bionemo/evo2/latest/quickstart-guide.html>
- NVIDIA 托管 API 参考（arc/evo2-7b-forward OpenAPI schema）：<https://docs.api.nvidia.com/nim/reference/arc-evo2-7b-infer>
- **托管端点实测（2026-08-24，真实 key）**：`output_layer` 返回 422 `StripedHyena has no attribute 'output_layer'`；`unembed` 返回 logits（NPZ key `unembed.output`，shape `(1, seq, 512)`，float64）—— 因此评分工具默认 `EVO2_MCP_LOGITS_LAYER=auto` 自动探测
- **批量实测（2026-08-25，真实 key，3800+ 条 K562 enhancer/promoter）**：
  - 序列 **> ~100 kb 时托管端返回 422**（PyTorch `canUse32BitIndexMath` 限制）—— 批量跑用 `--skip-longer-than 100000`；
  - 并发下 layer 名自动探测的**竞态已修复**（`forward_logits` 用局部变量记录尝试名）并加了回归测试；
  - embedding 提取：`norm`/`embedding_layer`/`blocks.N` 均可用，shape `(1, seq, 4096)` float64（mean-pool 后 4096 维）。
- Arc Institute Evo2 仓库（`scoring.py`、`models.py`）：<https://github.com/ArcInstitute/evo2>
- vortex `CharLevelTokenizer`（Evo2 官方 tokenizer 实现）：PyPI `vtx` 1.1.0 源码 `vortex/model/tokenizer.py`
- Evo2 模型卡：<https://huggingface.co/ArcInstitute/evo2_7b>

若 NVIDIA 调整 API，请以官方最新文档为准；`EVO2_MCP_BASE_URL` 可随时切换。

## 18. 批量评分与 Embedding 提取（scripts/score_fasta.py）

MCP Tools 适合 Agent 交互式调用；**大批量 FASTA 评分**用配套脚本 `scripts/score_fasta.py`（真实 API 已用 3800+ 条 K562 enhancer/promoter 验证）。

**每次运行自动创建一个带时间戳的独立文件夹**：

```text
output/run_20260825_104403/
├── scores.csv            # 每序列一行：id, header, length, total/mean LL, ...
│                         #   + embedding_key（与 embeddings.npz 的 record_ids 对齐）
└── embeddings.npz        # embeddings: (n, 4096) float32 mean-pooled 矩阵
                          # record_ids: 与矩阵行一一对应的键（来源__序列id）
```

```bash
# 小样本（指定 id）
.pixi/envs/dev/bin/python scripts/score_fasta.py \
  --fasta /path/cis/enhancers.fa /path/cis/promoters.fa \
  --ids K562_TE_629,K562_MPT_6842 --allow-ambiguous

# 全量（跳过 >100kb —— 托管端对该长度返回 422；保留原始 embedding）
.pixi/envs/dev/bin/python scripts/score_fasta.py \
  --fasta /path/cis/enhancers.fa /path/cis/promoters.fa \
         /path/trans/enhancers.fa /path/trans/promoters.fa \
  --skip-longer-than 100000 --allow-ambiguous \
  --embedding-layer norm --keep-raw-embeddings
```

关键参数：

| 参数 | 说明 |
|---|---|
| `--embedding-layer norm\|blocks.31\|embedding_layer\|none` | 提取哪个 layer 的 embedding（默认 `norm`）；`none` 只评分 |
| `--keep-raw-embeddings` | 额外保存每条**原始逐位** embedding `(1, seq, 4096)` 到 `embeddings_raw/`（**磁盘占用大**：10 kb 序列 ≈ 328 MB；默认不存） |
| `--skip-longer-than 100000` | 跳过超过该长度的序列（托管 API 限制，见 §17） |
| `--allow-ambiguous` | 允许 N 碱基透传（5 条含 N 的序列照常跑并带 caveat 警告） |
| `--max-concurrency 2` | 并发数（默认 2，尊重 rate limit） |
| `--out / --embeddings-out / --embedding-raw-dir` | 覆盖默认 run 文件夹布局 |

**效率设计**：每条序列只发一次请求（`output_layers=["unembed","norm"]`，logits 与 embedding 同取）；logits layer 名整次运行只探测一次；相同并发由信号量限制。

## 19. Embedding 关联与下游分析（scripts/analyze_run.py）

`embeddings.npz` 是 mean-pooled 的序列表示（每序列一个 4096 维向量），适合直接做聚类、分类、回归。加载与关联：

```python
import csv, numpy as np

run = "output/run_20260825_104403"
rows = list(csv.DictReader(open(f"{run}/scores.csv")))
d = np.load(f"{run}/embeddings.npz", allow_pickle=True)
X = d["embeddings"]                        # (n, 4096) float32
ids = [str(x) for x in d["record_ids"]]    # 与 X 行一一对应
key_to_row = {r["embedding_key"]: r for r in rows if r.get("embedding_key")}
scores = [key_to_row[k] for k in ids]      # scores[i] ↔ X[i]
```

一键跑完整分析（KMeans 聚类、enhancer-vs-promoter 分类、embedding→likelihood 回归、PCA 图）：

```bash
.pixi/envs/dev/bin/python scripts/analyze_run.py output/run_20260825_104403 --k 3
```

输出 `analysis_<run>.npz`（合并后的 `X` + `keys`）和 `analysis_<run>.png`。分析要点：

- 4096 维向量做相似度/聚类前先**单位化**（脚本已做）；
- 小样本时分类/回归自动跳过（保护阈值 ≥6 条）；全量 3806 条跑完后这些分析才有统计意义；
- 键 `cis_enhancers__xxx` → 类别 enhancers、区域 cis；`trans_*` 同理（`parse_source` 可改标签维度做 cis-vs-trans 分类）。

## 开发与测试

```bash
pixi install          # 或 pip install -e ".[dev]"
pixi run test         # 运行 pytest（全部 mock，不调用真实 API）
```

离线测试 **91 passed / 4 skipped**（skip = live 门控）。覆盖：序列校验、大小写/空白归一化、非法字符、缺 API key、forward 请求构造、401/408/429/5xx/timeout、layer 名自动探测（含并发竞态回归）、variant 校验、variant/batch 评分数学正确性（对照 Arc 语义独立复算）、NPZ 解码（JSON base64 / zip / 旧版 JSON tensor / `<layer>.output` key）、FASTA 沙箱、MCP session 集成、脚本纯函数（pooling/键名）等。

**真实 API 冒烟测试**（需要真实 key，默认跳过）。key 从 `.env` 自动读取（`Settings.from_env()` 已加载）：

```bash
EVO2_MCP_RUN_LIVE=1 .pixi/envs/dev/bin/python -m pytest tests/test_live_api.py -v -s
```

live 测试会真实请求 NVIDIA 端点，验证：logits layer 自动探测（`unembed`）、真实 NPZ 解析（`(1, seq, 512)` float64）、`evo2_score` 与手工从原始 logits 复算一致、variant score。

## 项目结构

```text
.
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
├── src/evo2_mcp/
│   ├── __main__.py      # python -m evo2_mcp 入口
│   ├── config.py        # 环境变量配置
│   ├── sequence.py      # DNA 校验/归一化
│   ├── api_client.py    # HTTP 客户端（retry/backoff/错误分类/响应解码 + layer 自动探测）
│   ├── forward_output.py# NPZ 解码 + likelihood 计算 + embedding 提取
│   ├── fasta.py         # FASTA 解析 + 读取沙箱
│   ├── tools.py         # 5 个 Tool 的实现
│   └── server.py        # MCP server（stdio）
├── scripts/
│   ├── score_fasta.py   # 批量 FASTA 评分 + embedding 提取（每次运行独立 run 文件夹）
│   └── analyze_run.py   # 下游分析：加载/关联 → 聚类/分类/回归 + PCA 图
├── tests/               # pytest（全 mock，91 用例）+ 可选 live test
└── output/              # mode="save" 的 .npz 输出 + run_*/ 运行结果（git 忽略）
```
