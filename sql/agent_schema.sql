-- agent_schema.sql
--
-- Creates all agent.* tables required by cmap-agent.
-- Safe to run on an existing database — all statements are idempotent.
-- Run this once before the first deployment, then re-run after any schema
-- update; existing tables and data are never dropped or truncated.
--
-- Tables created:
--   agent.Threads          — conversation sessions
--   agent.Messages         — individual messages within a thread
--   agent.ToolRuns         — tool call log (arguments, results, timing)
--   agent.ThreadSummaries  — rolling/final memory summaries
--
-- Note: agent.Catalog* tables (CatalogDatasets, CatalogVariables,
-- CatalogDatasetReferences) were removed in v0.2.55. The agent runtime
-- now reads directly from udfCatalog() and dbo.tblDataset_References.
-- Run sql/drop_agent_catalog_tables.sql to remove those tables if still present.

IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'agent')
BEGIN
    EXEC('CREATE SCHEMA agent');
END
GO

-- Threads
IF OBJECT_ID('agent.Threads', 'U') IS NULL
BEGIN
CREATE TABLE agent.Threads (
    ThreadId           UNIQUEIDENTIFIER NOT NULL DEFAULT NEWID(),
    UserId             INT NOT NULL,
    Title              NVARCHAR(200) NULL,
    CreatedAt          DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    UpdatedAt          DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    IsArchived         BIT NOT NULL DEFAULT 0,
    ClientTag          NVARCHAR(50) NULL,
    AgentState         NVARCHAR(MAX) NULL,
    PRIMARY KEY (ThreadId)
);
CREATE INDEX IX_Threads_User_Updated ON agent.Threads(UserId, UpdatedAt DESC);
END
ELSE
BEGIN
    -- Idempotent migration: add AgentState if not already present
    IF NOT EXISTS (
        SELECT 1 FROM sys.columns
        WHERE object_id = OBJECT_ID('agent.Threads') AND name = 'AgentState'
    )
    BEGIN
        ALTER TABLE agent.Threads ADD AgentState NVARCHAR(MAX) NULL;
        PRINT 'agent.Threads.AgentState column added.';
    END
END
GO

-- Messages
IF OBJECT_ID('agent.Messages', 'U') IS NULL
BEGIN
CREATE TABLE agent.Messages (
    MessageId          BIGINT IDENTITY(1,1) NOT NULL,
    ThreadId           UNIQUEIDENTIFIER NOT NULL,
    UserId             INT NOT NULL,
    Role               NVARCHAR(20) NOT NULL,    -- system | user | assistant | tool
    Content            NVARCHAR(MAX) NULL,
    ContentJson        NVARCHAR(MAX) NULL,       -- optional JSON payload
    ModelProvider      NVARCHAR(30) NULL,
    ModelName          NVARCHAR(80) NULL,
    CreatedAt          DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    TokenCountIn       INT NULL,
    TokenCountOut      INT NULL,
    ParentMessageId    BIGINT NULL,
    PRIMARY KEY (MessageId),
    CONSTRAINT FK_Messages_Thread FOREIGN KEY (ThreadId) REFERENCES agent.Threads(ThreadId)
);
CREATE INDEX IX_Messages_Thread_Created ON agent.Messages(ThreadId, CreatedAt ASC);
CREATE INDEX IX_Messages_User_Created   ON agent.Messages(UserId, CreatedAt DESC);
END
GO

-- Tool Runs
IF OBJECT_ID('agent.ToolRuns', 'U') IS NULL
BEGIN
CREATE TABLE agent.ToolRuns (
    ToolRunId          BIGINT IDENTITY(1,1) NOT NULL,
    ThreadId           UNIQUEIDENTIFIER NOT NULL,
    MessageId          BIGINT NULL,
    UserId             INT NOT NULL,
    ToolName           NVARCHAR(100) NOT NULL,
    ToolArgsJson       NVARCHAR(MAX) NOT NULL,
    ToolResultJson     NVARCHAR(MAX) NULL,
    ToolResultPreview  NVARCHAR(MAX) NULL,
    StartedAt          DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    EndedAt            DATETIME2(3) NULL,
    Status             NVARCHAR(20) NOT NULL DEFAULT 'ok',  -- ok | error | timeout | blocked
    ErrorMessage       NVARCHAR(2000) NULL,
    PRIMARY KEY (ToolRunId),
    CONSTRAINT FK_ToolRuns_Thread FOREIGN KEY (ThreadId) REFERENCES agent.Threads(ThreadId)
);
CREATE INDEX IX_ToolRuns_Thread_Started ON agent.ToolRuns(ThreadId, StartedAt DESC);
END
GO

-- Summaries (memory)
IF OBJECT_ID('agent.ThreadSummaries', 'U') IS NULL
BEGIN
CREATE TABLE agent.ThreadSummaries (
    SummaryId          BIGINT IDENTITY(1,1) NOT NULL,
    ThreadId           UNIQUEIDENTIFIER NOT NULL,
    UserId             INT NOT NULL,
    SummaryType        NVARCHAR(30) NOT NULL,   -- rolling | final | preferences
    SummaryText        NVARCHAR(MAX) NOT NULL,
    SummaryJson        NVARCHAR(MAX) NULL,
    CreatedAt          DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    PRIMARY KEY (SummaryId),
    CONSTRAINT FK_Summaries_Thread FOREIGN KEY (ThreadId) REFERENCES agent.Threads(ThreadId)
);
CREATE INDEX IX_Summaries_Thread_Created ON agent.ThreadSummaries(ThreadId, CreatedAt DESC);
END
GO
