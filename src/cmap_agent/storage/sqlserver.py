from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Iterable
import platform

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from cmap_agent.config.settings import settings

def _build_sqlalchemy_url() -> str:
    # Prefer DSN if provided
    if settings.CMAP_SQLSERVER_DSN:
        # Example: "mssql+pyodbc:///?odbc_connect=DSN=CMAP_PROD;TrustServerCertificate=yes"
        trust = "yes" if settings.CMAP_SQLSERVER_TRUST_CERT else "no"
        return f"mssql+pyodbc:///?odbc_connect=DSN={settings.CMAP_SQLSERVER_DSN};TrustServerCertificate={trust}"
    # Else host-based
    if not (settings.CMAP_SQLSERVER_HOST and settings.CMAP_SQLSERVER_DB and settings.CMAP_SQLSERVER_USER and settings.CMAP_SQLSERVER_PASSWORD):
        raise RuntimeError("SQL Server connection not configured. Set CMAP_SQLSERVER_DSN or HOST/DB/USER/PASSWORD.")

    trust = "yes" if settings.CMAP_SQLSERVER_TRUST_CERT else "no"

    # driver = settings.CMAP_SQLSERVER_DRIVER.replace(" ", "+")
    if platform.system().lower().find("windows") != -1:
        driver = "{SQL Server}"
    elif platform.system().lower().find("darwin") != -1:
        driver = "/usr/local/lib/libtdsodbc.so"
    elif platform.system().lower().find("linux") != -1:
        driver = "/usr/lib/x86_64-linux-gnu/odbc/libtdsodbc.so"

    driver = driver.replace(" ", "+")    
    user = settings.CMAP_SQLSERVER_USER
    pwd = settings.CMAP_SQLSERVER_PASSWORD
    host = settings.CMAP_SQLSERVER_HOST
    port = settings.CMAP_SQLSERVER_PORT
    db = settings.CMAP_SQLSERVER_DB
    return f"mssql+pyodbc://{user}:{pwd}@{host}:{port}/{db}?driver={driver}&TrustServerCertificate={trust}"

@dataclass
class SQLServerStore:
    engine: Engine

    @classmethod
    def from_env(cls) -> "SQLServerStore":
        url = _build_sqlalchemy_url()
        engine = create_engine(url, pool_pre_ping=True, pool_recycle=3600, future=True)
        return cls(engine=engine)

    def create_thread(self, user_id: int, title: str | None = None, client_tag: str | None = None) -> str:
        thread_id = str(uuid.uuid4())
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO agent.Threads (ThreadId, UserId, Title, ClientTag)
                VALUES (:thread_id, :user_id, :title, :client_tag)
            """), dict(thread_id=thread_id, user_id=user_id, title=title, client_tag=client_tag))
        return thread_id

    def touch_thread(self, thread_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE agent.Threads SET UpdatedAt = SYSUTCDATETIME()
                WHERE ThreadId = :thread_id
            """), dict(thread_id=thread_id))

    def add_message(
        self,
        thread_id: str,
        user_id: int,
        role: str,
        content: str | None,
        content_json: str | None = None,
        model_provider: str | None = None,
        model_name: str | None = None,
        token_in: int | None = None,
        token_out: int | None = None,
        parent_message_id: int | None = None,
    ) -> int:
        with self.engine.begin() as conn:
            res = conn.execute(text("""
                INSERT INTO agent.Messages
                    (ThreadId, UserId, Role, Content, ContentJson, ModelProvider, ModelName, TokenCountIn, TokenCountOut, ParentMessageId)
                OUTPUT INSERTED.MessageId
                VALUES
                    (:thread_id, :user_id, :role, :content, :content_json, :model_provider, :model_name, :token_in, :token_out, :parent_message_id)
            """), dict(
                thread_id=thread_id, user_id=user_id, role=role,
                content=content, content_json=content_json,
                model_provider=model_provider, model_name=model_name,
                token_in=token_in, token_out=token_out, parent_message_id=parent_message_id
            ))
            msg_id = int(res.scalar_one())
        self.touch_thread(thread_id)
        return msg_id

    def add_tool_run(
        self,
        thread_id: str,
        user_id: int,
        tool_name: str,
        tool_args_json: str,
        message_id: int | None = None,
        status: str = "ok",
        tool_result_json: str | None = None,
        tool_result_preview: str | None = None,
        error_message: str | None = None,
    ) -> int:
        with self.engine.begin() as conn:
            res = conn.execute(text("""
                INSERT INTO agent.ToolRuns
                    (ThreadId, MessageId, UserId, ToolName, ToolArgsJson, ToolResultJson, ToolResultPreview, Status, ErrorMessage, EndedAt)
                OUTPUT INSERTED.ToolRunId
                VALUES
                    (:thread_id, :message_id, :user_id, :tool_name, :tool_args_json, :tool_result_json, :tool_result_preview, :status, :error_message, SYSUTCDATETIME())
            """), dict(
                thread_id=thread_id, message_id=message_id, user_id=user_id,
                tool_name=tool_name, tool_args_json=tool_args_json,
                tool_result_json=tool_result_json, tool_result_preview=tool_result_preview,
                status=status, error_message=error_message
            ))
            run_id = int(res.scalar_one())
        return run_id

    def get_recent_messages(self, thread_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT TOP (:limit) MessageId, Role, Content, ContentJson, CreatedAt
                FROM agent.Messages
                WHERE ThreadId = :thread_id
                ORDER BY CreatedAt DESC
            """), dict(limit=limit, thread_id=thread_id)).mappings().all()
        # Return chronological
        return list(reversed([dict(r) for r in rows]))

    def get_latest_summary(self, thread_id: str, summary_type: str = "rolling") -> str | None:
        with self.engine.begin() as conn:
            row = conn.execute(text("""
                SELECT TOP 1 SummaryText
                FROM agent.ThreadSummaries
                WHERE ThreadId = :thread_id AND SummaryType = :summary_type
                ORDER BY CreatedAt DESC
            """), dict(thread_id=thread_id, summary_type=summary_type)).scalar()
        return str(row) if row is not None else None

    def add_summary(self, thread_id: str, user_id: int, summary_type: str, summary_text: str, summary_json: str | None = None) -> int:
        with self.engine.begin() as conn:
            res = conn.execute(text("""
                INSERT INTO agent.ThreadSummaries (ThreadId, UserId, SummaryType, SummaryText, SummaryJson)
                OUTPUT INSERTED.SummaryId
                VALUES (:thread_id, :user_id, :summary_type, :summary_text, :summary_json)
            """), dict(
                thread_id=thread_id, user_id=user_id, summary_type=summary_type,
                summary_text=summary_text, summary_json=summary_json
            ))
            sid = int(res.scalar_one())
        return sid

    def get_thread_title(self, thread_id: str, user_id: int) -> str | None:
        with self.engine.begin() as conn:
            row = conn.execute(text("""
                SELECT TOP 1 Title
                FROM agent.Threads
                WHERE ThreadId = :thread_id AND UserId = :user_id
            """), dict(thread_id=thread_id, user_id=user_id)).scalar()
        if row is None:
            return None
        title = str(row).strip()
        return title if title else None

    def set_thread_title(self, thread_id: str, user_id: int, title: str) -> None:
        title = (title or "").strip()
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE agent.Threads
                SET Title = :title, UpdatedAt = SYSUTCDATETIME()
                WHERE ThreadId = :thread_id AND UserId = :user_id
            """), dict(title=title, thread_id=thread_id, user_id=user_id))

    def list_threads(self, user_id: int, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """List threads for a user, newest first.

        Includes a last-message preview and latest summary (if present).
        """
        with self.engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT
                    t.ThreadId,
                    t.Title,
                    t.CreatedAt,
                    t.UpdatedAt,
                    lastmsg.Role  AS LastRole,
                    lastmsg.Content AS LastMessage,
                    firstuser.Content AS FirstUserMessage,
                    lasts.SummaryText AS SummaryText
                FROM agent.Threads t
                OUTER APPLY (
                    SELECT TOP 1 Role, Content
                    FROM agent.Messages m
                    WHERE m.ThreadId = t.ThreadId
                    ORDER BY m.CreatedAt DESC
                ) lastmsg
                OUTER APPLY (
                    SELECT TOP 1 Content
                    FROM agent.Messages m
                    WHERE m.ThreadId = t.ThreadId AND m.Role = 'user'
                    ORDER BY m.CreatedAt ASC
                ) firstuser
                OUTER APPLY (
                    SELECT TOP 1 SummaryText
                    FROM agent.ThreadSummaries s
                    WHERE s.ThreadId = t.ThreadId AND s.UserId = t.UserId
                    ORDER BY s.CreatedAt DESC
                ) lasts
                WHERE t.UserId = :user_id
                ORDER BY t.UpdatedAt DESC
                OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
            """), dict(user_id=user_id, limit=limit, offset=offset)).mappings().all()
        return [dict(r) for r in rows]

    def list_thread_messages(
        self,
        thread_id: str,
        user_id: int,
        limit: int = 200,
        offset: int = 0,
        include_json: bool = False,
    ) -> list[dict[str, Any]]:
        """List messages for a thread (chronological), enforcing ownership via Threads.UserId."""
        # NOTE: agent.Threads and agent.Messages both have CreatedAt. Always qualify message columns
        # to avoid "Ambiguous column name" errors when joining.
        cols: list[str] = [
            "m.MessageId AS MessageId",
            "m.Role AS Role",
            "m.Content AS Content",
            "m.CreatedAt AS CreatedAt",
        ]
        if include_json:
            cols.append("m.ContentJson AS ContentJson")
        select_cols = ", ".join(cols)
        with self.engine.begin() as conn:
            rows = conn.execute(text(f"""
                SELECT {select_cols}
                FROM agent.Messages m
                INNER JOIN agent.Threads t ON t.ThreadId = m.ThreadId
                WHERE m.ThreadId = :thread_id AND t.UserId = :user_id
                ORDER BY m.CreatedAt ASC
                OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
            """), dict(thread_id=thread_id, user_id=user_id, limit=limit, offset=offset)).mappings().all()
        return [dict(r) for r in rows]

    def resolve_user_id_by_api_key(
        self,
        api_key: str,
        table: str = "dbo.tblAPI_keys",
        api_key_column: str = "API_Key",
        user_id_column: str = "User_ID",
    ) -> int | None:
        """Resolve CMAP User_ID for a given API key.

        Notes:
        - table/column identifiers are expected to be validated by caller
          (to avoid SQL injection through identifier interpolation).
        - api_key is passed as a bound parameter.
        """

        if not api_key:
            return None
        q = f"SELECT TOP 1 [{user_id_column}] FROM {table} WHERE [{api_key_column}] = :api_key"
        with self.engine.begin() as conn:
            row = conn.execute(text(q), dict(api_key=api_key)).scalar()
        if row is None:
            return None
        try:
            return int(row)
        except Exception:
            return None

    # Placeholder: adapt to CMAP Users/API key table
    def load_cmap_api_key(self, user_id: int) -> str | None:
        # Prefer explicit fallback if provided.
        if settings.CMAP_API_KEY_FALLBACK:
            return settings.CMAP_API_KEY_FALLBACK

        # Best-effort load from tblAPI_keys for dev / backward compatibility.
        try:
            table = getattr(settings, "CMAP_AGENT_AUTH_APIKEY_TABLE", "dbo.tblAPI_keys")
            key_col = getattr(settings, "CMAP_AGENT_AUTH_APIKEY_COLUMN", "API_Key")
            user_col = getattr(settings, "CMAP_AGENT_AUTH_USERID_COLUMN", "User_ID")
            q = f"SELECT TOP 1 [{key_col}] FROM {table} WHERE [{user_col}] = :user_id"
            with self.engine.begin() as conn:
                row = conn.execute(text(q), dict(user_id=int(user_id))).scalar()
            return str(row) if row is not None else None
        except Exception:
            return None
