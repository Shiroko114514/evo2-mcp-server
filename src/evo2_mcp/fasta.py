"""FASTA reading with an allowed-directories sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evo2_mcp.config import Settings


@dataclass
class FastaRecord:
    header: str  # everything after ">" on the header line
    id: str      # first whitespace-delimited token of the header
    sequence: str  # concatenated raw sequence (validation happens later)


class FastaError(RuntimeError):
    pass


def parse_fasta(text: str) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    buf: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append(_finalise(header, buf))
            header = line[1:].strip()
            buf = []
        else:
            if header is None:
                raise FastaError("FASTA data does not start with a '>' header line")
            buf.append(line.strip())
    if header is not None:
        records.append(_finalise(header, buf))
    if not records:
        raise FastaError("No FASTA records found")
    return records


def _finalise(header: str, buf: list[str]) -> FastaRecord:
    seq = "".join(buf)
    rec_id = header.split()[0] if header else ""
    return FastaRecord(header=header, id=rec_id, sequence=seq)


def read_fasta_path(path: str, settings: Settings) -> str:
    """Resolve `path`, enforce the allowed-directories sandbox, return file text."""
    if not settings.allowed_dirs:
        raise FastaError(
            "Reading local FASTA files is disabled: no allowed directories are configured. "
            "Set EVO2_MCP_ALLOWED_DIRS to a colon-separated list of directories the server "
            "may read from, or pass `fasta_text=...` instead of `fasta_path=...`."
        )
    resolved = Path(path).expanduser().resolve()
    if not any(_is_within(resolved, allowed) for allowed in settings.allowed_dirs):
        raise FastaError(
            f"Refusing to read {resolved}: not inside any allowed directory "
            f"({', '.join(str(d) for d in settings.allowed_dirs)}). "
            "Update EVO2_MCP_ALLOWED_DIRS if you want to allow this path."
        )
    if not resolved.is_file():
        raise FastaError(f"FASTA path is not a regular file: {resolved}")
    return resolved.read_text(encoding="utf-8", errors="replace")


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False
