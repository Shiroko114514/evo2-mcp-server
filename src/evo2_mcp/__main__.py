"""Package entrypoint.

* `python -m evo2_mcp`   → runs __main__ directly.
* Console script `evo2-mcp` (see pyproject.toml) → calls `main`.
"""

from evo2_mcp.server import main

if __name__ == "__main__":
    main()
