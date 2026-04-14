-- drop_agent_catalog_tables.sql
--
-- Removes the agent.Catalog* tables that were used as an intermediate
-- cache between the live CMAP metadata tables and the agent runtime.
--
-- These tables are no longer needed because:
--   - The agent runtime reads directly from udfCatalog() (in-memory cache)
--   - The KB sync (cmap-agent-sync-kb) reads directly from udfCatalog()
--     and tblDataset_References
--   - References are fetched from dbo.tblDataset_References via Dataset_ID
--
-- Prerequisites: deploy cmap-agent v0.2.55+ before running this script.
-- Running this while an older version of the agent is active will break it.

IF OBJECT_ID('agent.CatalogDatasetReferences', 'U') IS NOT NULL
    DROP TABLE agent.CatalogDatasetReferences;

IF OBJECT_ID('agent.CatalogVariables', 'U') IS NOT NULL
    DROP TABLE agent.CatalogVariables;

IF OBJECT_ID('agent.CatalogDatasets', 'U') IS NOT NULL
    DROP TABLE agent.CatalogDatasets;

-- Optionally drop the agent schema itself if no other agent.* tables remain.
-- Check first: SELECT * FROM sys.tables WHERE SCHEMA_NAME(schema_id) = 'agent'
-- If only agent.ThreadState remains, keep the schema.
-- IF SCHEMA_ID('agent') IS NOT NULL DROP SCHEMA agent;
