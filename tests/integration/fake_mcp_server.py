"""A synthetic MCP server used only by integration tests.

It is not a second production service.  Modes let tests exercise pagination,
isError, permission failures and unknown payload shapes over the real stdio
MCP protocol.
"""

from __future__ import annotations

import sys

from mcp.server.fastmcp import FastMCP

MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"

ISSUES = [
    {
        "number": 1,
        "title": "Adapter missing for items/checks formats",
        "state": "open",
        "html_url": "https://example.invalid/repo/issues/1",
        "updated_at": "2026-10-24T20:00:00Z",
        "body": "Synthetic issue about an explicit adapter.",
    },
    {
        "number": 2,
        "title": "Truncated tool output loses evidence",
        "state": "closed",
        "html_url": "https://example.invalid/repo/issues/2",
        "updated_at": "2026-10-23T20:00:00Z",
        "body": "Synthetic issue about truncation.",
    },
    {
        "number": 3,
        "title": "Verification status distinction",
        "state": "open",
        "html_url": "https://example.invalid/repo/issues/3",
        "updated_at": "2026-10-22T20:00:00Z",
        "body": "Synthetic issue about not_run.",
    },
]

mcp = FastMCP("fake-github", log_level="ERROR")


@mcp.tool()
def search_issues(query: str, page: int = 1, perPage: int = 5) -> dict:
    """Search the synthetic issue list."""
    if MODE == "iserror":
        raise RuntimeError("permission denied by synthetic policy")
    if MODE == "unknown":
        return {"unexpected": "shape", "total_count": "not-an-int"}
    if MODE == "pagination":
        return {"total_count": 10, "items": ISSUES[:3]}
    if MODE == "empty":
        return {"total_count": 0, "items": []}
    if MODE == "baditem":
        return {"total_count": 1, "items": [{"number": True, "title": "bad"}]}
    return {"total_count": len(ISSUES), "items": ISSUES[:perPage]}


@mcp.tool()
def issue_read(owner: str, repo: str, issue_number: int, method: str = "get") -> dict:
    """Read one synthetic issue or its comments."""
    if MODE == "iserror":
        raise RuntimeError("permission denied by synthetic policy")
    issue = next((item for item in ISSUES if item["number"] == issue_number), None)
    if issue is None:
        raise ValueError("issue not found")
    if method == "get_comments":
        return {
            "comments": [
                {
                    "user": {"login": "synthetic-user"},
                    "created_at": "2026-10-24T21:00:00Z",
                    "body": "Synthetic comment body.",
                }
            ]
        }
    return issue


@mcp.tool()
def delete_issue(issue_number: int) -> dict:
    """A deliberately dangerous tool that must never be exposed to the model."""
    return {"deleted": issue_number}


if __name__ == "__main__":
    mcp.run(transport="stdio")