"""Local FastAPI application for public Trace task review."""

from __future__ import annotations

import os
from pathlib import Path
import secrets
import threading
from typing import Any, Sequence
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ..calibration import discover_calibration_summaries
from .assets import AssetIndex, build_asset_index
from .index import (
    ReviewIndex,
    build_review_index,
    is_path_within,
    load_sample_payload,
    sample_view,
)
from .overlay import render_annotation_overlay
from .security import DEFAULT_TOKEN_ENV, validate_review_bind
from .store import (
    ASSET_DECISIONS,
    AUDIT_FIELDS,
    ISSUE_CATEGORIES,
    ISSUE_SEVERITIES,
    ISSUE_STATUSES,
    ReviewStore,
)

_HERE = Path(__file__).resolve().parent
_TEMPLATES = _HERE / "templates"
_STATIC = _HERE / "static"


class ReviewAppState:
    """Mutable local app state with atomic index replacement."""

    def __init__(
        self, *, review_root: Path, repo_root: Path, database_path: Path
    ) -> None:
        self.review_root = review_root.resolve()
        self.repo_root = repo_root.resolve()
        self.store = ReviewStore(database_path)
        self._lock = threading.RLock()
        self._index = build_review_index(self.review_root)
        self._assets: AssetIndex | None = None
        self._calibrations, self._calibration_errors = discover_calibration_summaries(
            self.review_root
        )

    @property
    def index(self) -> ReviewIndex:
        with self._lock:
            return self._index

    @property
    def assets(self) -> AssetIndex:
        with self._lock:
            if self._assets is None:
                self._assets = build_asset_index(self.review_root)
            return self._assets

    @property
    def calibrations(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._calibrations)

    @property
    def calibration_errors(self) -> list[str]:
        with self._lock:
            return list(self._calibration_errors)

    def reload(self) -> ReviewIndex:
        fresh = build_review_index(self.review_root)
        calibrations, calibration_errors = discover_calibration_summaries(
            self.review_root
        )
        with self._lock:
            self._index = fresh
            self._assets = None
            self._calibrations = calibrations
            self._calibration_errors = calibration_errors
            return fresh


def create_review_app(
    *,
    review_root: Path | str,
    database_path: Path | str | None = None,
    repo_root: Path | str | None = None,
    auth_token: str | None = None,
    auth_token_env: str = DEFAULT_TOKEN_ENV,
    trusted_hosts: Sequence[str] | None = None,
) -> FastAPI:
    """Create the local review app.

    A configured token is entered through the login form and stored only in an
    HTTP-only session cookie. Query-string authentication is not supported.
    """

    root = Path(review_root).expanduser().resolve()
    repository = (
        Path(repo_root).expanduser().resolve() if repo_root else _infer_repo_root(root)
    )
    database = (
        Path(database_path).expanduser().resolve()
        if database_path is not None
        else _default_database_path(root)
    )
    configured_token = (
        str(auth_token).strip()
        if auth_token is not None
        else str(os.environ.get(auth_token_env, "")).strip()
    )
    allowed_hosts = _trusted_hosts(trusted_hosts)
    state = ReviewAppState(
        review_root=root, repo_root=repository, database_path=database
    )

    app = FastAPI(title="Trace Review", docs_url=None, redoc_url=None)
    app.state.trace_review = state
    templates = Jinja2Templates(directory=str(_TEMPLATES))

    @app.exception_handler(ValueError)
    async def invalid_review_input(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.middleware("http")
    async def authentication_and_headers(request: Request, call_next: Any) -> Response:
        if request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
            rejection = _unsafe_request_rejection(request)
            if rejection:
                return JSONResponse({"detail": rejection}, status_code=403)
        if configured_token and not _public_auth_path(request.url.path):
            if not bool(request.session.get("authenticated", False)):
                if request.url.path.startswith("/api/"):
                    return JSONResponse(
                        {"detail": "authentication required"}, status_code=401
                    )
                next_path = quote(request.url.path, safe="/")
                return RedirectResponse(f"/login?next={next_path}", status_code=303)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self'; form-action 'self'; frame-ancestors 'none'",
        )
        return response

    # SessionMiddleware must wrap the auth middleware so request.session exists.
    app.add_middleware(
        SessionMiddleware,
        secret_key=secrets.token_urlsafe(48),
        same_site="strict",
        https_only=False,
        session_cookie="trace_review_session",
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    def render(request: Request, template: str, **context: Any) -> HTMLResponse:
        common = {
            "request": request,
            "audit_fields": AUDIT_FIELDS,
            "issue_categories": sorted(ISSUE_CATEGORIES),
            "issue_severities": sorted(ISSUE_SEVERITIES),
            "asset_decisions": sorted(ASSET_DECISIONS),
        }
        common.update(context)
        return templates.TemplateResponse(
            request=request,
            name=template,
            context=common,
        )

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request, next: str = "/") -> HTMLResponse:
        return render(request, "login.html", next_path=_safe_next(next), error="")

    @app.post("/login")
    async def login_submit(request: Request) -> Response:
        form = await request.form()
        supplied = str(form.get("token", ""))
        next_path = _safe_next(str(form.get("next", "/")))
        if configured_token and secrets.compare_digest(supplied, configured_token):
            request.session.clear()
            request.session["authenticated"] = True
            return RedirectResponse(next_path, status_code=303)
        return render(request, "login.html", next_path=next_path, error="Invalid token")

    @app.post("/logout")
    async def logout(request: Request) -> RedirectResponse:
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        index = state.index
        domains = []
        for domain in index.domains.values():
            tasks = [
                task for task in index.tasks.values() if task.domain == domain.domain
            ]
            domains.append(
                {
                    "domain": domain.domain,
                    "scene_count": len(domain.scenes),
                    "task_count": len(tasks),
                    "materialized_count": sum(task.materialized for task in tasks),
                    "sample_count": sum(task.sample_count for task in tasks),
                }
            )
        return render(
            request,
            "index.html",
            index=index,
            domains=domains,
            calibration_count=len(state.calibrations),
            calibration_errors=state.calibration_errors,
        )

    @app.get("/domains/{domain}", response_class=HTMLResponse)
    async def domain_page(request: Request, domain: str) -> HTMLResponse:
        index = state.index
        domain_record = index.domains.get(domain)
        if domain_record is None:
            raise HTTPException(status_code=404, detail="unknown domain")
        scenes = []
        for scene_id in domain_record.scenes:
            scene = index.scenes[f"{domain}/{scene_id}"]
            tasks = [index.tasks[task_id] for task_id in scene.tasks]
            scenes.append(
                {
                    "scene": scene,
                    "task_count": len(tasks),
                    "materialized_count": sum(task.materialized for task in tasks),
                    "sample_count": sum(task.sample_count for task in tasks),
                }
            )
        return render(request, "domain.html", domain=domain_record, scenes=scenes)

    @app.get("/domains/{domain}/scenes/{scene_id}", response_class=HTMLResponse)
    async def scene_page(request: Request, domain: str, scene_id: str) -> HTMLResponse:
        index = state.index
        scene = index.scenes.get(f"{domain}/{scene_id}")
        if scene is None:
            raise HTTPException(status_code=404, detail="unknown scene")
        tasks = [index.tasks[task_id] for task_id in scene.tasks]
        return render(request, "scene.html", scene=scene, tasks=tasks)

    @app.get(
        "/domains/{domain}/scenes/{scene_id}/tasks/{task_id}",
        response_class=HTMLResponse,
    )
    async def task_page(
        request: Request, domain: str, scene_id: str, task_id: str
    ) -> HTMLResponse:
        task = _task_or_404(state.index, domain, scene_id, task_id)
        samples = [state.index.samples[uid] for uid in task.samples]
        audit = (
            state.store.get_task_audit(task_id, task.recipe_digest)
            if task.materialized
            else _empty_audit(task_id)
        )
        issues = (
            state.store.list_issues(task_id=task_id, recipe_digest=task.recipe_digest)
            if task.materialized
            else []
        )
        calibrations = [
            summary for summary in state.calibrations if task_id in summary["task_ids"]
        ]
        return render(
            request,
            "task.html",
            task=task,
            samples=samples,
            audit=audit,
            issues=issues,
            calibrations=calibrations,
        )

    @app.get(
        "/domains/{domain}/scenes/{scene_id}/tasks/{task_id}/queries/{query_id}",
        response_class=HTMLResponse,
    )
    async def query_page(
        request: Request,
        domain: str,
        scene_id: str,
        task_id: str,
        query_id: str,
    ) -> HTMLResponse:
        task = _task_or_404(state.index, domain, scene_id, task_id)
        samples = [
            state.index.samples[uid]
            for uid in task.samples
            if state.index.samples[uid].query_id == query_id
        ]
        if query_id not in task.query_counts:
            raise HTTPException(status_code=404, detail="unknown materialized query")
        return render(
            request,
            "query.html",
            task=task,
            query_id=query_id,
            samples=samples,
        )

    @app.post("/domains/{domain}/scenes/{scene_id}/tasks/{task_id}/audit")
    async def task_audit_submit(
        request: Request, domain: str, scene_id: str, task_id: str
    ) -> RedirectResponse:
        task = _materialized_task_or_409(state.index, domain, scene_id, task_id)
        form = await request.form()
        values = {field: field in form for field in AUDIT_FIELDS}
        state.store.update_task_audit(
            task_id,
            task.recipe_digest,
            values=values,
            notes=str(form.get("notes", "")),
            updated_by=str(form.get("updated_by", "")),
        )
        return RedirectResponse(
            f"/domains/{domain}/scenes/{scene_id}/tasks/{task_id}", status_code=303
        )

    @app.post("/domains/{domain}/scenes/{scene_id}/tasks/{task_id}/issues")
    async def task_issue_submit(
        request: Request, domain: str, scene_id: str, task_id: str
    ) -> RedirectResponse:
        task = _materialized_task_or_409(state.index, domain, scene_id, task_id)
        form = await request.form()
        issue = state.store.create_issue(
            title=str(form.get("title", "")),
            body=str(form.get("body", "")),
            author=str(form.get("author", "")),
            category=str(form.get("category", "other")),
            severity=str(form.get("severity", "issue")),
            domain=domain,
            scene_id=scene_id,
            task_id=task_id,
            recipe_digest=task.recipe_digest,
        )
        return RedirectResponse(f"/issues/{issue['id']}", status_code=303)

    @app.get("/samples/{sample_uid}", response_class=HTMLResponse)
    async def sample_page(request: Request, sample_uid: str) -> HTMLResponse:
        index = state.index
        sample = index.samples.get(sample_uid)
        if sample is None:
            raise HTTPException(status_code=404, detail="unknown sample")
        try:
            payload = load_sample_payload(index, sample)
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=409, detail=f"sample is unavailable: {exc}"
            ) from exc
        return render(
            request,
            "sample.html",
            sample=sample,
            sample_data=sample_view(payload),
            issues=state.store.list_issues(
                sample_uid=sample_uid,
                recipe_digest=sample.recipe_digest,
                sample_semantic_hash=sample.semantic_hash,
            ),
        )

    @app.post("/samples/{sample_uid}/issues")
    async def sample_issue_submit(
        request: Request, sample_uid: str
    ) -> RedirectResponse:
        sample = state.index.samples.get(sample_uid)
        if sample is None:
            raise HTTPException(status_code=404, detail="unknown sample")
        form = await request.form()
        issue = state.store.create_issue(
            title=str(form.get("title", "")),
            body=str(form.get("body", "")),
            author=str(form.get("author", "")),
            category=str(form.get("category", "other")),
            severity=str(form.get("severity", "issue")),
            domain=sample.domain,
            scene_id=sample.scene_id,
            task_id=sample.task_id,
            sample_uid=sample.uid,
            recipe_digest=sample.recipe_digest,
            sample_semantic_hash=sample.semantic_hash,
        )
        return RedirectResponse(f"/issues/{issue['id']}", status_code=303)

    @app.get("/media/{media_id}")
    async def sample_media(media_id: str) -> FileResponse:
        path = state.index.media.get(media_id)
        if (
            path is None
            or not path.is_file()
            or not is_path_within(state.review_root, path)
        ):
            raise HTTPException(status_code=404, detail="unknown media")
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @app.get("/overlays/{sample_uid}.png")
    async def sample_overlay(sample_uid: str) -> Response:
        index = state.index
        sample = index.samples.get(sample_uid)
        if sample is None:
            raise HTTPException(status_code=404, detail="unknown sample")
        image_path = index.media.get(sample.media_id)
        if (
            image_path is None
            or not image_path.is_file()
            or not is_path_within(state.review_root, image_path)
        ):
            raise HTTPException(
                status_code=404, detail="sample image is not materialized"
            )
        try:
            payload = load_sample_payload(index, sample)
            annotation = payload.get("annotation_gt", {})
            png = render_annotation_overlay(
                image_path, annotation if isinstance(annotation, dict) else {}
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=409, detail=f"overlay unavailable: {exc}"
            ) from exc
        return Response(
            png, media_type="image/png", headers={"Cache-Control": "no-store"}
        )

    @app.get("/issues", response_class=HTMLResponse)
    async def issues_page(request: Request, status: str = "") -> HTMLResponse:
        selected_status = status if status in ISSUE_STATUSES else None
        return render(
            request,
            "issues.html",
            issues=state.store.list_issues(status=selected_status),
            selected_status=status,
        )

    @app.get("/issues/{issue_id}", response_class=HTMLResponse)
    async def issue_page(request: Request, issue_id: str) -> HTMLResponse:
        try:
            issue = state.store.get_issue(issue_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown issue") from exc
        return render(
            request,
            "issue.html",
            issue=issue,
            comments=state.store.issue_comments(issue_id),
        )

    @app.post("/issues/{issue_id}/comments")
    async def issue_comment_submit(request: Request, issue_id: str) -> RedirectResponse:
        form = await request.form()
        try:
            state.store.add_comment(
                issue_id,
                body=str(form.get("body", "")),
                author=str(form.get("author", "")),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown issue") from exc
        return RedirectResponse(f"/issues/{issue_id}", status_code=303)

    @app.post("/issues/{issue_id}/status")
    async def issue_status_submit(request: Request, issue_id: str) -> RedirectResponse:
        form = await request.form()
        try:
            state.store.set_issue_status(issue_id, str(form.get("status", "open")))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown issue") from exc
        return RedirectResponse(f"/issues/{issue_id}", status_code=303)

    @app.get("/assets/{kind}", response_class=HTMLResponse)
    async def assets_page(request: Request, kind: str) -> HTMLResponse:
        normalized = "three_d" if kind in {"3d", "three-d", "three_d"} else kind
        records = state.assets.records.get(normalized)
        if records is None:
            raise HTTPException(status_code=404, detail="unknown asset catalog")
        reviews = {
            record["asset_id"]: record
            for record in state.store.asset_reviews()
            if record["kind"] == normalized
        }
        return render(
            request,
            "assets.html",
            kind=normalized,
            assets=records,
            reviews=reviews,
        )

    @app.post("/assets/{kind}/{asset_id}/review")
    async def asset_review_submit(
        request: Request, kind: str, asset_id: str
    ) -> RedirectResponse:
        normalized = "three_d" if kind in {"3d", "three-d", "three_d"} else kind
        records = state.assets.records.get(normalized)
        if records is None or asset_id not in {record.id for record in records}:
            raise HTTPException(status_code=404, detail="unknown asset")
        form = await request.form()
        state.store.set_asset_review(
            asset_id,
            kind=normalized,
            decision=str(form.get("decision", "improve")),
            notes=str(form.get("notes", "")),
            updated_by=str(form.get("updated_by", "")),
        )
        return RedirectResponse(f"/assets/{normalized}", status_code=303)

    @app.get("/assets/media/{media_id}")
    async def asset_media(media_id: str) -> FileResponse:
        path = state.assets.media.get(media_id)
        if (
            path is None
            or not path.is_file()
            or not is_path_within(state.review_root, path)
        ):
            raise HTTPException(status_code=404, detail="unknown asset media")
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @app.get("/api/index")
    async def api_index() -> dict[str, Any]:
        index = state.index
        return {
            "schema": "trace-review-app-index-v1",
            "built_at": index.built_at,
            "domain_count": len(index.domains),
            "scene_count": len(index.scenes),
            "task_count": len(index.tasks),
            "materialized_task_count": index.materialized_task_count,
            "sample_count": index.sample_count,
            "errors": list(index.errors),
        }

    @app.post("/api/reload")
    async def api_reload() -> dict[str, Any]:
        index = state.reload()
        return {
            "status": "reloaded",
            "built_at": index.built_at,
            "sample_count": index.sample_count,
        }

    @app.post("/reload")
    async def reload_page() -> RedirectResponse:
        state.reload()
        return RedirectResponse("/", status_code=303)

    @app.get("/api/tasks/{task_id}/audit")
    async def api_task_audit(task_id: str) -> dict[str, Any]:
        task = state.index.tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="unknown task")
        if not task.materialized:
            raise HTTPException(status_code=409, detail="task is not materialized")
        return state.store.get_task_audit(task_id, task.recipe_digest)

    @app.patch("/api/tasks/{task_id}/audit")
    async def api_task_audit_update(request: Request, task_id: str) -> dict[str, Any]:
        task = state.index.tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="unknown task")
        if not task.materialized:
            raise HTTPException(status_code=409, detail="task is not materialized")
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=422, detail="expected a JSON object")
        values = body.get("values", body)
        if not isinstance(values, dict):
            raise HTTPException(
                status_code=422, detail="audit values must be an object"
            )
        unknown = set(values) - set(AUDIT_FIELDS)
        if "values" not in body:
            unknown -= {"notes", "updated_by"}
            values = {
                key: value for key, value in values.items() if key in AUDIT_FIELDS
            }
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"unknown audit fields: {sorted(unknown)!r}",
            )
        try:
            return state.store.update_task_audit(
                task_id,
                task.recipe_digest,
                values=values,
                notes=str(body.get("notes", "")),
                updated_by=str(body.get("updated_by", "")),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def serve_review_app(
    *,
    review_root: Path | str,
    host: str = "127.0.0.1",
    port: int = 8000,
    database_path: Path | str | None = None,
    repo_root: Path | str | None = None,
    auth_token_env: str = DEFAULT_TOKEN_ENV,
    log_level: str = "info",
    trusted_hosts: Sequence[str] | None = None,
) -> None:
    """Validate the bind and run the local app with Uvicorn."""

    token = validate_review_bind(host, token_env=auth_token_env)
    app = create_review_app(
        review_root=review_root,
        database_path=database_path,
        repo_root=repo_root,
        auth_token=token,
        auth_token_env=auth_token_env,
        trusted_hosts=tuple(trusted_hosts or ()) + (str(host),),
    )
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised without review extra.
        raise RuntimeError(
            "serve requires the Trace review optional dependencies"
        ) from exc
    uvicorn.run(app, host=str(host), port=int(port), log_level=str(log_level))


def _task_or_404(index: ReviewIndex, domain: str, scene_id: str, task_id: str):
    task = index.tasks.get(task_id)
    if task is None or task.domain != domain or task.scene_id != scene_id:
        raise HTTPException(status_code=404, detail="unknown task")
    return task


def _materialized_task_or_409(
    index: ReviewIndex, domain: str, scene_id: str, task_id: str
):
    task = _task_or_404(index, domain, scene_id, task_id)
    if not task.materialized:
        raise HTTPException(status_code=409, detail="task is not materialized")
    return task


def _infer_repo_root(review_root: Path) -> Path:
    if review_root.name == "task-reviews" and review_root.parent.name == "review":
        return review_root.parents[1]
    return Path.cwd().resolve()


def _default_database_path(review_root: Path) -> Path:
    if review_root.name == "task-reviews":
        return review_root.parent / "feedback" / "review_feedback.sqlite"
    return review_root / "feedback" / "review_feedback.sqlite"


def _safe_next(value: str) -> str:
    text = str(value).strip()
    if (
        not text.startswith("/")
        or text.startswith("//")
        or "\\" in text
        or any(ord(character) < 32 for character in text)
    ):
        return "/"
    return text


def _public_auth_path(path: str) -> bool:
    return path == "/healthz" or path == "/login" or path.startswith("/static/")


def _trusted_hosts(extra: Sequence[str] | None) -> list[str]:
    values = {"127.0.0.1", "localhost", "::1", "testserver"}
    for value in extra or ():
        normalized = str(value).strip()
        if not normalized:
            continue
        if "*" in normalized:
            raise ValueError("trusted_hosts must not contain wildcards")
        if any(
            ord(character) <= 32 or character in {"/", "\\"} for character in normalized
        ):
            raise ValueError("trusted_hosts entries must be host names, not URLs")
        values.add(normalized)
    return sorted(values)


def _unsafe_request_rejection(request: Request) -> str:
    if str(request.headers.get("sec-fetch-site", "")).lower() == "cross-site":
        return "cross-site state-changing requests are not accepted"
    expected = _origin_tuple(str(request.url))
    origin = str(request.headers.get("origin", "")).strip()
    if origin:
        if _origin_tuple(origin) != expected:
            return "request Origin does not match this review application"
        return ""
    referer = str(request.headers.get("referer", "")).strip()
    if referer and _origin_tuple(referer) != expected:
        return "request Referer does not match this review application"
    return ""


def _origin_tuple(value: str) -> tuple[str, str, int | None] | None:
    try:
        parsed = urlsplit(str(value))
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        return None
    if scheme not in {"http", "https"} or not host:
        return None
    if port is None:
        port = 80 if scheme == "http" else 443
    return scheme, host, port


def _empty_audit(task_id: str) -> dict[str, Any]:
    return {
        "task_id": str(task_id),
        "recipe_digest": "",
        **{field: False for field in AUDIT_FIELDS},
        "notes": "",
        "updated_by": "",
        "updated_at": "",
    }


__all__ = ["ReviewAppState", "create_review_app", "serve_review_app"]
