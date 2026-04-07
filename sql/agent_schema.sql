
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
    PRIMARY KEY (ThreadId)
);
CREATE INDEX IX_Threads_User_Updated ON agent.Threads(UserId, UpdatedAt DESC);
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

-- Catalog cache (rich metadata + authoritative stats + references)
IF OBJECT_ID('agent.CatalogDatasets', 'U') IS NULL
BEGIN
CREATE TABLE agent.CatalogDatasets (
    TableName          NVARCHAR(200) NOT NULL,
    DatasetId          INT NULL,
    ShortName          NVARCHAR(450) NULL,
    DatasetName        NVARCHAR(MAX) NULL,
    Description        NVARCHAR(MAX) NULL,
    Keywords           NVARCHAR(MAX) NULL,
    DataSource         NVARCHAR(MAX) NULL,
    Distributor        NVARCHAR(MAX) NULL,
    Acknowledgement    NVARCHAR(MAX) NULL,

    Make               NVARCHAR(MAX) NULL,
    Sensor             NVARCHAR(MAX) NULL,
    ProcessLevel       NVARCHAR(200) NULL,
    StudyDomain         NVARCHAR(200) NULL,
    TemporalResolution NVARCHAR(400) NULL,
    SpatialResolution  NVARCHAR(400) NULL,
    Units              NVARCHAR(1000) NULL,
    Comments           NVARCHAR(MAX) NULL,
    Regions            NVARCHAR(1000) NULL,

    -- Authoritative coverage from tblDataset_Stats
    TimeMin            DATETIME2(3) NULL,
    TimeMax            DATETIME2(3) NULL,
    LatMin             FLOAT NULL,
    LatMax             FLOAT NULL,
    LonMin             FLOAT NULL,
    LonMax             FLOAT NULL,
    DepthMin           FLOAT NULL,
    DepthMax           FLOAT NULL,
    HasDepth           BIT NULL,

    UpdatedAt          DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    PRIMARY KEY (TableName)
);
CREATE INDEX IX_CatalogDatasets_ShortName ON agent.CatalogDatasets(ShortName);
CREATE INDEX IX_CatalogDatasets_DatasetId ON agent.CatalogDatasets(DatasetId);
END
GO

IF OBJECT_ID('agent.CatalogVariables', 'U') IS NULL
BEGIN
CREATE TABLE agent.CatalogVariables (
    TableName          NVARCHAR(200) NOT NULL,
    VarName            NVARCHAR(200) NOT NULL,
    LongName           NVARCHAR(500) NULL,
    Unit               NVARCHAR(100) NULL,
    Keywords           NVARCHAR(MAX) NULL,
    UpdatedAt          DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    PRIMARY KEY (TableName, VarName),
    CONSTRAINT FK_CatalogVariables_Dataset FOREIGN KEY (TableName) REFERENCES agent.CatalogDatasets(TableName)
);
CREATE INDEX IX_CatalogVariables_VarName ON agent.CatalogVariables(VarName);
END
GO

IF OBJECT_ID('agent.CatalogDatasetReferences', 'U') IS NULL
BEGIN
CREATE TABLE agent.CatalogDatasetReferences (
    TableName          NVARCHAR(200) NOT NULL,
    DatasetId          INT NULL,
    ReferenceId        INT NOT NULL,
    Reference          NVARCHAR(MAX) NOT NULL,
    UpdatedAt          DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    PRIMARY KEY (TableName, ReferenceId),
    CONSTRAINT FK_CatalogDatasetReferences_Dataset FOREIGN KEY (TableName) REFERENCES agent.CatalogDatasets(TableName)
);
CREATE INDEX IX_CatalogDatasetReferences_DatasetId ON agent.CatalogDatasetReferences(DatasetId);
END
GO

