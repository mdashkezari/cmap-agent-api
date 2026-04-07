from __future__ import annotations

"""Backwards-compatible shim.

cmap-agent-sync now runs the richer metadata sync (CMAP API catalog_sql + SQL stats + tblDataset_References).
"""

from cmap_agent.sync.metadata_sync import main

if __name__ == "__main__":
    main()
