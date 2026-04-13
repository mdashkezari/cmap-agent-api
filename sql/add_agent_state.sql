-- Migration: add AgentState column to agent.Threads
--
-- Additive change only: nullable column, no constraints, no default value.
-- Safe to run on a live production database — SQL Server adds a nullable
-- column without locking the table on modern compatibility levels.
--
-- Rollback: the column can be dropped after reverting to v135 if desired,
-- but it is harmless to leave in place (v135 code never reads it).
--
-- Run once before deploying cmap-agent v136.

IF NOT EXISTS (
    SELECT 1 FROM sys.columns
    WHERE object_id = OBJECT_ID('agent.Threads')
      AND name = 'AgentState'
)
BEGIN
    ALTER TABLE agent.Threads ADD AgentState NVARCHAR(MAX) NULL;
    PRINT 'agent.Threads.AgentState column added.';
END
ELSE
BEGIN
    PRINT 'agent.Threads.AgentState already exists — skipped.';
END
GO
