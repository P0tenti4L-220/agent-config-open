"""Read-only GitHub MCP client with a narrow, validated adapter.

The model never receives the full GitHub tool set.  Only ``search_issues`` and
``issue_read`` are exposed through this wrapper, the repository is fixed by
configuration, and unknown result shapes are errors rather than empty lists.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import shutil
import threading
from contextlib import AsyncExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

from .contracts import MCPConfig, ResearchLayer
from .errors import ToolFailure
from .github_scope import SearchScope, parse_issue_identity
from .tools import IssueReadOutcome, IssueSearchOutcome

ALLOWED_TOOLS = ("search_issues", "issue_read")


class GitHubMCPBackend:
    """Synchronous facade over an asyncio MCP session running in one thread."""

    def __init__(self, config: MCPConfig) -> None:
        self.config = config
        self.available = False
        self.unavailable_reason: str | None = None if config.enabled else "mcp_not_configured"
        self._session: ClientSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._startup_error: str | None = None
        self._schemas: dict[str, dict[str, Any]] = {}
        self._seen_issues: set[str] = set()
        self._issues_checked = False

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> bool:
        if not self.config.enabled:
            return False
        if self.available:
            return True
        command = self.config.command or ""
        resolved = shutil.which(command) if not os.path.sep in command else command
        if not resolved or not Path(resolved).exists():
            self.unavailable_reason = "mcp_command_not_found"
            return False
        self._thread = threading.Thread(target=self._thread_main, name="mcp-github", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=self.config.startup_timeout_seconds):
            self.unavailable_reason = "mcp_startup_timeout"
            self._shutdown()
            return False
        if self._startup_error or self._session is None:
            self.unavailable_reason = self._startup_error or "mcp_startup_failed"
            return False
        self.available = True
        self.unavailable_reason = None
        return True

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()

    async def _serve(self) -> None:
        try:
            async with AsyncExitStack() as stack:
                child_env = {**os.environ}
                token = os.environ.get(self.config.token_env)
                if token:
                    # The official server reads the conventional variable even
                    # when the user stores the token under a different name.
                    child_env.setdefault("GITHUB_PERSONAL_ACCESS_TOKEN", token)
                params = StdioServerParameters(
                    command=self.config.command or "",
                    args=list(self.config.args),
                    env=child_env,
                )
                read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(
                    ClientSession(
                        read_stream,
                        write_stream,
                        read_timeout_seconds=timedelta(seconds=self.config.call_timeout_seconds),
                    )
                )
                await session.initialize()
                listed = await session.list_tools()
                tools = {tool.name: tool for tool in listed.tools}
                missing = [name for name in ALLOWED_TOOLS if name not in tools]
                if missing:
                    raise RuntimeError(f"required MCP tools are missing: {', '.join(missing)}")
                for name in ALLOWED_TOOLS:
                    schema = tools[name].inputSchema
                    if not isinstance(schema, dict) or not isinstance(schema.get("properties"), dict):
                        raise RuntimeError(f"{name} has no usable input schema")
                self._schemas = {name: dict(tools[name].inputSchema) for name in ALLOWED_TOOLS}
                self._session = session
                self._ready.set()
                assert self._loop is not None
                await self._loop.run_in_executor(None, self._stop.wait)
        except Exception as exc:  # noqa: BLE001 - recorded as a compatibility failure
            self._startup_error = f"mcp_startup_failed:{type(exc).__name__}:{exc}"[:500]
            self._ready.set()

    def close(self) -> None:
        self._shutdown()

    def _shutdown(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._session = None
        self.available = False

    def __enter__(self) -> "GitHubMCPBackend":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- tools --------------------------------------------------------------
    @property
    def issues_checked(self) -> bool:
        return self._issues_checked

    def search_issues(
        self,
        query: str,
        limit: int,
        *,
        scope: SearchScope | None = None,
    ) -> IssueSearchOutcome:
        self._require_session("search_issues")
        if scope is None:
            if not self.config.repo:
                raise ToolFailure("out_of_scope_issue", "no configured repository is available for this search")
            scope = SearchScope(layer=ResearchLayer.CURRENT, repo=self.config.repo)
        # The wrapper owns the scope qualifier; user text cannot widen it.
        clean_query = " ".join(
            token
            for token in query.split()
            if not token.lower().startswith(("repo:", "org:", "user:", "is:"))
        )
        full_query = f"{clean_query} {scope.qualifier} is:issue".strip()
        arguments: dict[str, Any] = {"query": full_query}
        properties = self._schemas["search_issues"].get("properties", {})
        if "page" in properties:
            arguments["page"] = 1
        if "perPage" in properties:
            arguments["perPage"] = limit
        elif "per_page" in properties:
            arguments["per_page"] = limit
        if "fields" in properties:
            # Search locates an issue; reading its body requires issue_read.
            allowed = properties["fields"].get("items", {}).get("enum")
            wanted = ["number", "title", "state", "html_url", "updated_at"]
            arguments["fields"] = [f for f in wanted if not allowed or f in allowed]
        payload = self._call("search_issues", arguments)
        items, total = _extract_items(payload)
        issues: list[dict[str, Any]] = []
        for item in items[:limit]:
            normalized = _normalize_scoped_issue(item, expected_repo=scope.repo)
            issues.append(normalized)
            self._seen_issues.add(normalized["issue_ref"])
        self._issues_checked = True
        complete = _is_complete(len(issues), total, limit)
        return IssueSearchOutcome(
            issues=issues,
            total_count=total,
            complete=complete,
            query_scope=scope.label,
            query=full_query,
            layer=scope.layer.value,
            incomplete=None,
        )

    def read_issue(self, issue_ref: str, *, allow_any_repo: bool = False) -> IssueReadOutcome:
        self._require_session("issue_read")
        owner, repo, number = _parse_issue_ref(issue_ref)
        if not allow_any_repo:
            if f"{owner}/{repo}" != (self.config.repo or ""):
                raise ToolFailure("out_of_scope_issue", "issue is outside the configured repository")
            if not self._issues_checked or issue_ref not in self._seen_issues:
                raise ToolFailure(
                    "out_of_scope_issue",
                    "issue_ref was not returned by github_search_issues in this run",
                )
        arguments: dict[str, Any] = {}
        properties = self._schemas["issue_read"].get("properties", {})
        for key, value in (
            ("owner", owner),
            ("repo", repo),
            ("issue_number", number),
            ("method", "get"),
        ):
            if key in properties:
                arguments[key] = value
        if "issue_number" not in arguments:
            raise ToolFailure("unsupported_tool_schema", "issue_read has no issue_number parameter")
        payload = self._call("issue_read", arguments)
        issue = _extract_issue(payload)
        # The response must confirm the requested repository and number; the
        # configuration repository is never stamped onto a foreign issue.
        normalized = _normalize_scoped_issue(
            issue, expected_repo=f"{owner}/{repo}", expected_number=number
        )
        if allow_any_repo and normalized["issue_ref"] != issue_ref:
            raise ToolFailure("ambiguous_issue_identity", "issue_read returned a different issue than requested")
        url = normalized["html_url"]
        comments: list[dict[str, Any]] = []
        comments_error: str | None = None
        if "method" in properties:
            comment_args = dict(arguments)
            comment_args["method"] = "get_comments"
            try:
                comment_payload = self._call("issue_read", comment_args)
                comments = _extract_comments(comment_payload, limit=10)
            except ToolFailure as exc:
                comments_error = f"{exc.code}:{exc.message}"[:300]
        return IssueReadOutcome(
            issue=normalized,
            comments=comments,
            url=url,
            updated_at=normalized.get("updated_at"),
            comments_error=comments_error,
        )

    # -- protocol -----------------------------------------------------------
    def _require_session(self, tool: str) -> None:
        if not self.available or self._session is None or self._loop is None:
            raise ToolFailure(
                self.unavailable_reason or "mcp_unavailable",
                "GitHub MCP is not available; remote history was not checked",
            )
        if tool not in ALLOWED_TOOLS:
            raise ToolFailure("unknown_tool", "tool is not in the MCP allowlist")

    def _call(self, tool: str, arguments: dict[str, Any]) -> Any:
        assert self._session is not None and self._loop is not None
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(tool, arguments),
            self._loop,
        )
        try:
            result: CallToolResult = future.result(timeout=self.config.call_timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise ToolFailure("mcp_timeout", f"{tool} exceeded the MCP call timeout", retryable=True) from exc
        except Exception as exc:  # noqa: BLE001 - transport/session failure
            raise ToolFailure(
                "mcp_transport_error",
                f"{tool} failed at the MCP transport layer: {type(exc).__name__}",
                retryable=True,
            ) from exc
        if result.isError:
            message = _text_content(result) or "MCP tool reported an error"
            retryable = any(token in message.lower() for token in ("timeout", "rate limit", "502", "503", "temporar"))
            raise ToolFailure("mcp_tool_error", message[:500], retryable=retryable)
        if result.structuredContent is not None:
            return result.structuredContent
        text = _text_content(result)
        if text is None:
            raise ToolFailure("unsupported_result", f"{tool} returned no structured or text payload")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ToolFailure("unsupported_result", f"{tool} returned non-JSON text") from exc


# ---------------------------------------------------------------------------
# Payload adapters
# ---------------------------------------------------------------------------


def _text_content(result: CallToolResult) -> str | None:
    pieces: list[str] = []
    for item in result.content:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            pieces.append(text)
    return "\n".join(pieces) if pieces else None


def _extract_items(payload: Any) -> tuple[list[dict[str, Any]], int | None]:
    if isinstance(payload, list):
        items = payload
        total = None
    elif isinstance(payload, dict):
        items = payload.get("items")
        total = payload.get("total_count")
        if items is None and isinstance(payload.get("issues"), list):
            items = payload["issues"]
        if items is None:
            raise ToolFailure("unsupported_result", "GitHub search payload has no items list")
        if total is not None and (isinstance(total, bool) or not isinstance(total, int) or total < 0):
            raise ToolFailure("unsupported_result", "GitHub total_count is not a non-negative integer")
    else:
        raise ToolFailure("unsupported_result", "GitHub search payload is neither a list nor an object")
    if not isinstance(items, list):
        raise ToolFailure("unsupported_result", "GitHub search items is not a list")
    for item in items:
        if not isinstance(item, dict):
            raise ToolFailure("unsupported_result", "GitHub search item is not an object")
    return items, total


def _normalize_scoped_issue(
    item: dict[str, Any],
    *,
    expected_repo: str | None,
    expected_number: int | None = None,
) -> dict[str, Any]:
    owner, repo, number, html_url = parse_issue_identity(
        item,
        expected_repo=expected_repo,
        allow_expected_fallback=expected_repo is not None,
    )
    if expected_number is not None and number != expected_number:
        raise ToolFailure("ambiguous_issue_identity", "issue_read returned a different issue number")
    title = item.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ToolFailure("unsupported_result", "issue title is missing")
    state = item.get("state") if isinstance(item.get("state"), str) else "unknown"
    updated = item.get("updated_at") if isinstance(item.get("updated_at"), str) else None
    body = item.get("body") if isinstance(item.get("body"), str) else None
    return {
        "issue_ref": f"{owner}/{repo}#{number}",
        "owner": owner,
        "repo": repo,
        "number": number,
        "title": title[:500],
        "state": state,
        "html_url": html_url,
        "updated_at": updated,
        "body": body[:10000] if body else None,
    }


def _extract_issue(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("issue"), dict):
        return payload["issue"]
    if isinstance(payload, dict) and "number" in payload:
        return payload
    raise ToolFailure("unsupported_result", "issue_read payload is not an issue object")


def _extract_comments(payload: Any, limit: int) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        comments = payload.get("comments")
    elif isinstance(payload, list):
        comments = payload
    else:
        raise ToolFailure("unsupported_result", "comments payload is neither a list nor an object")
    if not isinstance(comments, list):
        raise ToolFailure("unsupported_result", "comments field is not a list")
    normalized: list[dict[str, Any]] = []
    for comment in comments[:limit]:
        if not isinstance(comment, dict):
            raise ToolFailure("unsupported_result", "comment is not an object")
        body = comment.get("body")
        normalized.append(
            {
                "author": (comment.get("user") or {}).get("login") if isinstance(comment.get("user"), dict) else None,
                "created_at": comment.get("created_at") if isinstance(comment.get("created_at"), str) else None,
                "body": body[:2000] if isinstance(body, str) else None,
            }
        )
    return normalized


def _is_complete(returned: int, total: int | None, limit: int) -> bool:
    if total is not None:
        return returned >= min(total, limit) and total <= limit
    return returned < limit


def _parse_issue_ref(issue_ref: str) -> tuple[str, str, int]:
    try:
        repository, raw_number = issue_ref.rsplit("#", 1)
        owner, repo = repository.split("/", 1)
        number = int(raw_number)
    except (ValueError, AttributeError) as exc:
        raise ToolFailure("invalid_issue_ref", "issue_ref must look like owner/repo#123") from exc
    if number < 1:
        raise ToolFailure("invalid_issue_ref", "issue number must be positive")
    return owner, repo, number