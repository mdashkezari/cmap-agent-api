from __future__ import annotations

from cmap_agent.tools.registry import Tool, ToolRegistry
from cmap_agent.tools.metadata_query_tool import (
    QueryMetadataArgs,
    query_metadata,
    SCHEMA_SUMMARY,
)
from cmap_agent.tools.catalog_tools import (
    CatalogSearchArgs, CatalogSearchVariablesArgs, CatalogSearchROIArgs, CatalogSearchKBFArgs, DatasetMetadataArgs, ListVariablesArgs,
    CountDatasetsArgs, DatasetSummaryArgs,
    catalog_search, catalog_search_variables, catalog_search_roi, catalog_search_kb_first, dataset_metadata, list_variables,
    count_datasets, dataset_summary
)
from cmap_agent.tools.cmap_tools import (
    SpaceTimeArgs, TimeSeriesArgs, DepthProfileArgs, ClimatologyArgs,
    PlotTimeseriesArgs, PlotMapArgs,
    cmap_space_time, cmap_time_series, cmap_depth_profile, cmap_climatology,
    plot_timeseries, plot_map
)
from cmap_agent.tools.colocalize_tool import ColocalizeArgs, cmap_colocalize
from cmap_agent.tools.kb_tools import KBSearchArgs, kb_search
from cmap_agent.tools.web_tools import WebSearchArgs, web_search

def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()

    # KB / retrieval tools
    reg.register(Tool(
        name="kb.search",
        description="Semantic search over the CMAP knowledge base (datasets, variables, references).",
        args_model=KBSearchArgs,
        fn=kb_search,
    ))

    # Catalog cache tools (lexical / structured)
    reg.register(Tool(
        name="catalog.search",
        description="Search CMAP datasets by text over cached metadata (table/name/description/keywords).",
        args_model=CatalogSearchArgs,
        fn=catalog_search,
    ))
    reg.register(Tool(
        name="catalog.search_roi",
        description=(
            "Find ALL datasets that spatially overlap a Region Of Interest (ROI) bounding box. "
            "Use this as the FIRST tool when the user asks 'what datasets cover region X' or 'what data is available at lat/lon bounds'. "
            "Returns datasets regardless of variable content — follow up with catalog.search_kb_first using specific variable names (e.g. 'nitrate phosphate') to filter by scientific relevance. "
            "Optionally filtered by make (Observation/Model/Assimilation) and sensor (in-Situ/Satellite)."
        ),
        args_model=CatalogSearchROIArgs,
        fn=catalog_search_roi,
    ))
    reg.register(Tool(
        name="catalog.search_kb_first",
        description=(
            "KB-first semantic dataset discovery: uses the Chroma knowledge base to find relevant datasets/variables, then applies optional constraints like ROI overlap, make, and sensor."
        ),
        args_model=CatalogSearchKBFArgs,
        fn=catalog_search_kb_first,
    ))
    reg.register(Tool(
        name="catalog.search_variables",
        description="Search CMAP variables (across datasets) by semantic text over cached variable metadata (short name, long name, keywords, units, dataset name).",
        args_model=CatalogSearchVariablesArgs,
        fn=catalog_search_variables,
    ))
    reg.register(Tool(
        name="catalog.dataset_metadata",
        description="Get dataset + variable metadata (including references) from cached catalog by table name.",
        args_model=DatasetMetadataArgs,
        fn=dataset_metadata,
    ))
    reg.register(Tool(
        name="catalog.list_variables",
        description="List variables for a dataset table (from cached catalog).",
        args_model=ListVariablesArgs,
        fn=list_variables,
    ))

    reg.register(Tool(
        name="catalog.count_datasets",
        description="Return the total number of datasets in the cached CMAP catalog.",
        args_model=CountDatasetsArgs,
        fn=count_datasets,
    ))

    reg.register(Tool(
        name="catalog.dataset_summary",
        description="Get a compact overview for one dataset by table name or text query (description/keywords).",
        args_model=DatasetSummaryArgs,
        fn=dataset_summary,
    ))

    # CMAP data tools
    reg.register(Tool(
        name="cmap.space_time",
        description="Retrieve raw CMAP space-time subset for a single variable; returns CSV by default (set format='parquet' for parquet).",
        args_model=SpaceTimeArgs,
        fn=cmap_space_time,
    ))
    reg.register(Tool(
        name="cmap.time_series",
        description="Retrieve raw CMAP time series subset at point/box; returns CSV by default (set format='parquet' for parquet).",
        args_model=TimeSeriesArgs,
        fn=cmap_time_series,
    ))
    reg.register(Tool(
        name="cmap.depth_profile",
        description="Retrieve raw CMAP depth profile subset; returns CSV by default (set format='parquet' for parquet).",
        args_model=DepthProfileArgs,
        fn=cmap_depth_profile,
    ))

    reg.register(Tool(
        name="cmap.climatology",
        description=(
            "Compute on-the-fly climatology for qualified gridded datasets using CMAP's uspAggregate logic. "
            "Returns CSV by default (set format='parquet' for parquet). "
            "If a dataset lacks the required helper field (month/week/dayofyear/year), this tool will return a structured error."
        ),
        args_model=ClimatologyArgs,
        fn=cmap_climatology,
    ))

    reg.register(Tool(
        name="cmap.colocalize",
        description=(
            "Colocalize / integrate / join / match a *small* source dataset (CMAP table or inline CSV/parquet) with one-or-more CMAP target datasets "
            "using pycmap.Sample(). The output preserves the source schema and adds the colocalized target variables."
        ),
        args_model=ColocalizeArgs,
        fn=cmap_colocalize,
    ))

    # Visualization tools (custom plots; output PNG artifacts)
    reg.register(Tool(
        name="viz.plot_timeseries",
        description="Create a custom time-series plot PNG from a CMAP subset artifact or inline rows.",
        args_model=PlotTimeseriesArgs,
        fn=plot_timeseries,
    ))
    reg.register(Tool(
        name="viz.plot_map",
        description="Create a custom map plot PNG from a CMAP subset artifact or inline rows. Shows labeled lat/lon ticks; supports Cartopy projections via the 'projection' argument (default PlateCarree).",
        args_model=PlotMapArgs,
        fn=plot_map,
    ))

    # Metadata SQL tool
    reg.register(Tool(
        name="catalog.query_metadata",
        description=(
            "Execute a read-only SELECT query against CMAP metadata tables to answer "
            "structural questions not covered by other catalog tools. Use for: "
            "cruise details (ship, chief scientist, dates, region), scientists on a cruise, "
            "which program a dataset belongs to, what CMAP regions a dataset covers, "
            "dataset references/citations, server locations, or collection listings. "
            "Do NOT use for variable data retrieval — use pycmap tools for that. "
            "The query must use SQL Server syntax, reference only metadata tables, "
            "and include TOP N. "
            "Schema:\n" + SCHEMA_SUMMARY
        ),
        args_model=QueryMetadataArgs,
        fn=query_metadata,
    ))

    # Web search tool (optional; requires TAVILY_API_KEY)
    reg.register(Tool(
        name="web.search",
        description="Search the public web for additional scientific/contextual information (optional).",
        args_model=WebSearchArgs,
        fn=web_search,
    ))

    return reg
