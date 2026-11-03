"""Program-owned GitHub search scope and issue identity validation.

The model may suggest keywords, but the repository qualifier, the layer and
the public fallback are decided by configuration and the controller.  This
module compiles a bounded query from structured terms, rejects GitHub search
operators the caller is not allowed to inject, and derives every issue identity
from the trusted response rather than from the configured repository.
"""

from __future__ import annotations

import re
from typing import Any, Iterable
from urllib.parse import urlsplit

from pydantic import field_validator, model_validator

from .contracts import ISSUE_REF_RE, ResearchLayer, StrictModel
from .errors import ToolFailure

GITHUB_HOSTS = {"github.com", "www.github.com", "api.github.com"}

# GitHub search qualifiers that would widen or retarget the program scope.
_FORBIDDEN_PREFIXES = (
    "repo:",
    "org:",
    "user:",
    "is:",
    "in:",
    "label:",
    "author:",
    "assignee:",
    "mentions:",
    "team:",
    "milestone:",
    "project:",
    "type:",
    "state:",
    "comments:",
    "reactions:",
    "created:",
    "updated:",
    "closed:",
    "language:",
    "license:",
    "topic:",
    "linked:",
    "subscribed:",
    "involves:",
    "archived:",
    "visibility:",
    "sort:",
    "order:",
)
_FORBIDDEN_WORDS = {"and", "or", "not"}
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.][A-Za-z0-9_.+#/-]{0,63}$")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{2,63}")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "error",
    "errors",
    "exception",
    "traceback",
    "failed",
    "failure",
    "when",
    "while",
    "after",
    "before",
    "during",
    "into",
    "http",
    "https",
    "true",
    "false",
    "none",
    "null",
    "self",
}


class ScopeViolation(ValueError):
    """The query contains a term the program does not allow to leave."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.code = "invalid_query_scope"


class SearchScope(StrictModel):
    layer: ResearchLayer
    repo: str | None = None
    public: bool = False

    @field_validator("repo")
    @classmethod
    def _repo_shape(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
            raise ValueError("scope repo must look like owner/name")
        return value

    @model_validator(mode="after")
    def _shape(self) -> "SearchScope":
        if self.public:
            if self.repo is not None:
                raise ValueError("public scope must not name a repository")
        elif self.repo is None:
            raise ValueError("current/dependency scope requires a repository")
        return self

    @property
    def qualifier(self) -> str:
        if self.public:
            return "is:public"
        return f"repo:{self.repo}"

    @property
    def label(self) -> str:
        return "public" if self.public else f"repo:{self.repo}"

    @property
    def query_scope(self) -> str:
        return self.label


def scope_for(layer: ResearchLayer, repo: str | None) -> SearchScope:
    if layer == ResearchLayer.PUBLIC:
        return SearchScope(layer=layer, public=True)
    if not repo:
        raise ToolFailure("invalid_query_scope", "a repository is required for this layer")
    return SearchScope(layer=layer, repo=repo)


def compile_query(terms: Iterable[str], scope: SearchScope) -> str:
    """Join validated terms with the program-owned scope qualifier."""

    tokens = _validated_tokens(terms)
    if not tokens:
        raise ScopeViolation("the query has no allowed search terms")
    if len(tokens) > 6:
        raise ScopeViolation("the query has too many terms for a bounded search")
    return f"{' '.join(tokens)} {scope.qualifier} is:issue".strip()


def _validated_tokens(terms: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for raw in terms:
        if not isinstance(raw, str):
            raise ScopeViolation("query terms must be strings")
        for token in raw.split():
            lowered = token.lower()
            if lowered in _FORBIDDEN_WORDS or token.startswith(("-", "@")):
                raise ScopeViolation(f"search operator or negation is not allowed: {token[:40]}")
            if lowered.startswith(_FORBIDDEN_PREFIXES):
                raise ScopeViolation(f"repository or qualifier override is not allowed: {token[:40]}")
            if any(char in token for char in "\"'`()[]{}|=\\"):
                raise ScopeViolation("quotes, grouping and expression syntax are not allowed")
            if not _TOKEN_RE.fullmatch(token):
                raise ScopeViolation(f"unsupported search token: {token[:40]}")
            if token not in tokens:
                tokens.append(token)
    return tokens


def derive_terms(text: str, *, limit: int = 4) -> list[str]:
    """Extract conservative public identifiers from untrusted prose."""

    found: list[str] = []
    for match in _IDENTIFIER_RE.findall(text or ""):
        token = match.strip(".")
        if token.lower() in _STOPWORDS or len(token) < 3:
            continue
        if token not in found:
            found.append(token)
        if len(found) >= limit:
            break
    return found


def normalize_repo(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().rstrip("/")
    scp = re.fullmatch(r"git@github\.com:([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?", value)
    if scp:
        return f"{scp.group(1)}/{scp.group(2)}"
    match = re.fullmatch(r"(?:https?://(?:www\.)?github\.com/)?([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?", value)
    if not match:
        return None
    return f"{match.group(1)}/{match.group(2)}"


def issue_ref_for(repo: str, number: int) -> str:
    return f"{repo}#{number}"


def split_issue_ref(issue_ref: str) -> tuple[str, str, int]:
    if not isinstance(issue_ref, str) or not re.fullmatch(ISSUE_REF_RE, issue_ref):
        raise ToolFailure("invalid_issue_ref", "issue_ref must look like owner/repo#123")
    repository, raw_number = issue_ref.rsplit("#", 1)
    owner, repo = repository.split("/", 1)
    return owner, repo, int(raw_number)


def _url_identity(url: str) -> tuple[str, str, int] | None:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    if host not in GITHUB_HOSTS:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if host == "api.github.com" and len(parts) == 5 and parts[0] == "repos" and parts[3] == "issues":
        owner, repo, raw_number = parts[1], parts[2], parts[4]
    elif host != "api.github.com" and len(parts) == 4 and parts[2] == "issues":
        owner, repo, raw_number = parts[0], parts[1], parts[3]
    else:
        return None
    try:
        number = int(raw_number)
    except ValueError:
        return None
    if number < 1:
        return None
    return owner, repo, number


def parse_issue_identity(
    item: dict[str, Any],
    *,
    expected_repo: str | None = None,
    allow_expected_fallback: bool = True,
) -> tuple[str, str, int, str]:
    """Return ``(owner, repo, number, html_url)`` from a trusted response.

    Repository data, API URLs and HTML URLs are cross-checked when more than
    one is present.  A response that names another repository is rejected at
    the current/dependency layers instead of being relabelled.
    """

    if not isinstance(item, dict):
        raise ToolFailure("unsupported_result", "issue result is not an object")
    number = item.get("number")
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise ToolFailure("unsupported_result", "issue number is missing or not a positive integer")

    candidates: list[tuple[str, str, int]] = []
    repository = item.get("repository")
    if isinstance(repository, dict):
        for key in ("full_name", "name_with_owner", "fullname"):
            value = repository.get(key)
            if isinstance(value, str):
                repository, owner, repo = _split_repo(value)
                if repository:
                    candidates.append((owner, repo, number))
                break
    for key in ("repository_url", "html_url", "url"):
        value = item.get(key)
        if isinstance(value, str):
            parsed = _url_identity(value)
            if parsed is not None:
                candidates.append(parsed)
    raw_ref = item.get("issue_ref")
    if isinstance(raw_ref, str) and re.fullmatch(ISSUE_REF_RE, raw_ref):
        owner, repo, ref_number = split_issue_ref(raw_ref)
        candidates.append((owner, repo, ref_number))

    distinct = {candidate for candidate in candidates}
    if len(distinct) > 1:
        raise ToolFailure("ambiguous_issue_identity", "issue response names more than one repository or number")
    if not distinct:
        if expected_repo and allow_expected_fallback:
            owner, repo = expected_repo.split("/", 1)
            resolved = (owner, repo, number)
        else:
            raise ToolFailure(
                "missing_issue_identity",
                "issue identity is not present in the response and cannot be assumed",
            )
    else:
        resolved = distinct.pop()

    owner, repo, resolved_number = resolved
    if resolved_number != number:
        raise ToolFailure("ambiguous_issue_identity", "issue number disagrees with the response URL")
    repository_name = f"{owner}/{repo}"
    if expected_repo and repository_name.lower() != expected_repo.lower():
        raise ToolFailure(
            "out_of_scope_issue",
            "issue belongs to a repository outside the authorized search scope",
        )

    html_url = item.get("html_url")
    if not isinstance(html_url, str) or not html_url.startswith("https://"):
        html_url = f"https://github.com/{repository_name}/issues/{number}"
    else:
        parsed = _url_identity(html_url)
        if parsed is not None and parsed != resolved:
            raise ToolFailure("ambiguous_issue_identity", "issue URL disagrees with the response repository")
    return owner, repo, number, html_url


def _split_repo(value: str) -> tuple[str | None, str, str]:
    normalized = normalize_repo(value)
    if not normalized:
        return None, "", ""
    owner, repo = normalized.split("/", 1)
    return normalized, owner, repo
