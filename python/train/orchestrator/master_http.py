"""CherryPy-powered HTTP control surface for the orchestrator master."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from train.orchestrator.master_runtime import OrchestratorMasterRuntime

try:
    import cherrypy
except ModuleNotFoundError:  # pragma: no cover - exercised via CLI/runtime guard
    class _CherryPyStub:
        request = None
        response = None

        class HTTPError(RuntimeError):
            pass

        @staticmethod
        def expose(function: Any) -> Any:
            return function

    cherrypy = _CherryPyStub()  # type: ignore[assignment]


class MasterHttpServer:
    """Small CherryPy wrapper used by ``ek_master.py`` and HTTP smoke tests."""

    def __init__(
        self,
        *,
        runtime: OrchestratorMasterRuntime,
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        _require_cherrypy()
        self._runtime = runtime
        self._host = host
        self._port = port
        self._root = _MasterWebRoot(runtime)
        self._started = False

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def start(self) -> None:
        if self._started:
            return
        assert cherrypy is not None
        cherrypy.config.update(
            {
                "engine.autoreload.on": False,
                "checker.on": False,
                "log.screen": False,
                "server.socket_host": self._host,
                "server.socket_port": self._port,
            }
        )
        cherrypy.tree.apps.clear()
        cherrypy.tree.mount(self._root, "/")
        cherrypy.engine.start()
        self._port = int(cherrypy.server.httpserver.socket.getsockname()[1])
        self._started = True

    def block(self) -> None:
        assert cherrypy is not None
        cherrypy.engine.block()

    def stop(self) -> None:
        if not self._started:
            return
        assert cherrypy is not None
        cherrypy.engine.exit()
        cherrypy.tree.apps.clear()
        self._started = False


class _MasterWebRoot:
    def __init__(self, runtime: OrchestratorMasterRuntime) -> None:
        self.api = _ApiRoot(runtime)
        self._runtime = runtime

    @property
    def runtime(self) -> OrchestratorMasterRuntime:
        return self._runtime

    @cherrypy.expose  # type: ignore[misc]
    def index(self) -> bytes:
        return _html_response(_render_status_page())

    @cherrypy.expose  # type: ignore[misc]
    def healthz(self) -> bytes:
        return _text_response("ok\n")


class _ApiRoot:
    def __init__(self, runtime: OrchestratorMasterRuntime) -> None:
        self.v1 = _ApiV1Root(runtime)

    @cherrypy.expose  # type: ignore[misc]
    def index(self) -> bytes:
        return _json_response(
            {
                "api_versions": ["v1"],
                "paths": [
                    "/api/v1/bootstrap",
                    "/api/v1/runtime",
                    "/api/v1/summary",
                    "/api/v1/status",
                    "/api/v1/spec",
                ],
            }
        )


class _ApiV1Root:
    def __init__(self, runtime: OrchestratorMasterRuntime) -> None:
        self.runtime = _RuntimeApi(runtime)
        self.spec = _SpecApi(runtime)
        self.submit = _SubmitApi(runtime)
        self._runtime = runtime

    @cherrypy.expose  # type: ignore[misc]
    def index(self) -> bytes:
        return _json_response(
            {
                "endpoints": [
                    "/api/v1/bootstrap",
                    "/api/v1/runtime",
                    "/api/v1/summary",
                    "/api/v1/status",
                    "/api/v1/spec",
                    "/api/v1/submit",
                ]
            }
        )

    @cherrypy.expose  # type: ignore[misc]
    def bootstrap(self, limit: str = "20") -> bytes:
        _require_method("GET")
        return _json_response(self._runtime.bootstrap(limit=_parse_limit(limit)))

    @cherrypy.expose  # type: ignore[misc]
    def summary(self) -> bytes:
        _require_method("GET")
        return _json_response(self._runtime.latest_summary())

    @cherrypy.expose  # type: ignore[misc]
    def status(self, limit: str = "20") -> bytes:
        _require_method("GET")
        return _json_response(self._runtime.status_snapshot(limit=_parse_limit(limit)))


class _RuntimeApi:
    def __init__(self, runtime: OrchestratorMasterRuntime) -> None:
        self._runtime = runtime

    @cherrypy.expose  # type: ignore[misc]
    def index(self) -> bytes:
        _require_method("GET")
        return _json_response(self._runtime.runtime_status())

    @cherrypy.expose  # type: ignore[misc]
    def start(self) -> bytes:
        _require_method("POST")
        return _json_response(self._runtime.start_loop())

    @cherrypy.expose  # type: ignore[misc]
    def stop(self) -> bytes:
        _require_method("POST")
        return _json_response(self._runtime.stop_loop())

    @cherrypy.expose  # type: ignore[misc]
    def pause(self) -> bytes:
        _require_method("POST")
        return _json_response(self._runtime.pause_loop())

    @cherrypy.expose  # type: ignore[misc]
    def resume(self) -> bytes:
        _require_method("POST")
        return _json_response(self._runtime.resume_loop())

    @cherrypy.expose  # type: ignore[misc]
    def reconcile(self) -> bytes:
        _require_method("POST")
        return _json_response(self._runtime.reconcile_once())

    @cherrypy.expose  # type: ignore[misc]
    def requeue_expired(self) -> bytes:
        _require_method("POST")
        return _json_response(self._runtime.requeue_expired_tasks())


class _SpecApi:
    def __init__(self, runtime: OrchestratorMasterRuntime) -> None:
        self._runtime = runtime
        self.lineages = _NamedSpecCollectionApi(runtime, "lineages")
        self.label_jobs = _NamedSpecCollectionApi(runtime, "label_jobs")
        self.idle_phase10_jobs = _NamedSpecCollectionApi(runtime, "idle_phase10_jobs")

    @cherrypy.expose  # type: ignore[misc]
    def index(self) -> bytes:
        _require_method("GET")
        return _json_response(self._runtime.get_spec_dict())

    @cherrypy.expose  # type: ignore[misc]
    def patch(self) -> bytes:
        _require_method("POST")
        payload = _read_json_object()
        return _json_response(self._runtime.patch_spec(payload))

    @cherrypy.expose  # type: ignore[misc]
    def replace(self) -> bytes:
        _require_method("POST")
        payload = _read_json_object()
        return _json_response(self._runtime.replace_spec_from_payload(payload))


class _NamedSpecCollectionApi:
    def __init__(self, runtime: OrchestratorMasterRuntime, collection_name: str) -> None:
        self._runtime = runtime
        self._collection_name = collection_name

    @cherrypy.expose  # type: ignore[misc]
    def index(self) -> bytes:
        _require_method("GET")
        spec = self._runtime.get_spec_dict()
        return _json_response(spec.get(self._collection_name, []))

    @cherrypy.expose  # type: ignore[misc]
    def default(self, entry_name: str) -> bytes:
        if cherrypy.request.method == "GET":
            spec = self._runtime.get_spec_dict()
            for entry in list(spec.get(self._collection_name) or []):
                if str(dict(entry).get("name")) == entry_name:
                    return _json_response(entry)
            raise cherrypy.HTTPError(404, f"{self._collection_name} entry not found: {entry_name}")
        _require_method("POST")
        payload = _read_json_object()
        try:
            updated = self._runtime.update_named_spec_entry(
                collection_name=self._collection_name,
                entry_name=entry_name,
                patch=payload,
            )
        except KeyError as exc:
            raise cherrypy.HTTPError(404, str(exc)) from exc
        return _json_response(updated)


class _SubmitApi:
    def __init__(self, runtime: OrchestratorMasterRuntime) -> None:
        self._runtime = runtime

    @cherrypy.expose  # type: ignore[misc]
    def index(self) -> bytes:
        return _json_response({"paths": ["phase10", "label", "idle_phase10"]})

    @cherrypy.expose  # type: ignore[misc]
    def phase10(self) -> bytes:
        _require_method("POST")
        payload = _read_json_object()
        result = self._runtime.submit_phase10_campaign(
            config_path=Path(str(payload["config_path"])),
            kind=str(payload.get("kind", "phase10_native")),
        )
        return _json_response(result)

    @cherrypy.expose  # type: ignore[misc]
    def label(self) -> bytes:
        _require_method("POST")
        payload = _read_json_object()
        result = self._runtime.submit_label_campaign(
            config_path=Path(str(payload["config_path"])),
            kind=str(payload.get("kind", "label_pgn_corpus")),
        )
        return _json_response(result)

    @cherrypy.expose  # type: ignore[misc]
    def idle_phase10(self) -> bytes:
        _require_method("POST")
        payload = _read_json_object()
        result = self._runtime.submit_idle_phase10_campaign(
            config_path=Path(str(payload["config_path"])),
            kind=str(payload.get("kind", "phase10_idle_artifacts")),
        )
        return _json_response(result)


def _render_status_page() -> str:
    runtime_paths = {
        "start": "/api/v1/runtime/start",
        "stop": "/api/v1/runtime/stop",
        "pause": "/api/v1/runtime/pause",
        "resume": "/api/v1/runtime/resume",
        "reconcile": "/api/v1/runtime/reconcile",
        "requeue": "/api/v1/runtime/requeue_expired",
    }
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>EngineKonzept Master Control</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f1e8;
      --panel: #fffdf8;
      --ink: #1f1a17;
      --muted: #6a625a;
      --line: #d7cdbd;
      --accent: #8a3b12;
      --accent-2: #c86c24;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #efe6d5 0%, var(--bg) 38%, #efe8db 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      gap: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px 18px;
      box-shadow: 0 10px 30px rgba(44, 31, 20, 0.07);
    }}
    h1, h2 {{
      margin: 0 0 12px 0;
      font-family: "IBM Plex Serif", "Georgia", serif;
    }}
    h1 {{ font-size: 32px; }}
    h2 {{ font-size: 20px; }}
    .toolbar {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    button {{
      border: 1px solid var(--accent);
      background: linear-gradient(180deg, var(--accent-2), var(--accent));
      color: white;
      border-radius: 10px;
      padding: 8px 12px;
      cursor: pointer;
      font-weight: 600;
    }}
    button.secondary {{
      background: white;
      color: var(--accent);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .runtime-status {{
      font-size: 15px;
      line-height: 1.5;
    }}
    .muted {{
      color: var(--muted);
    }}
    .settings-block {{
      border-top: 1px solid var(--line);
      padding-top: 12px;
      margin-top: 12px;
    }}
    .settings-form {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
      align-items: end;
      margin-bottom: 10px;
    }}
    label {{
      display: grid;
      gap: 4px;
      font-size: 13px;
      color: var(--muted);
    }}
    input, select, textarea {{
      font: inherit;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      color: var(--ink);
    }}
    input[type="checkbox"] {{
      width: 18px;
      height: 18px;
      padding: 0;
    }}
    textarea {{
      min-height: 180px;
      font-family: "IBM Plex Mono", monospace;
    }}
    pre {{
      margin: 0;
      font-size: 12px;
      white-space: pre-wrap;
      word-break: break-word;
      background: #f7f3eb;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
      overflow: auto;
      max-height: 480px;
    }}
    .inline-check {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--ink);
    }}
    #flash {{
      min-height: 20px;
      color: var(--accent);
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h1>Master Control</h1>
      <div class="toolbar">
        <button onclick="postAction('{runtime_paths["start"]}')">Start Loop</button>
        <button onclick="postAction('{runtime_paths["pause"]}')">Pause</button>
        <button onclick="postAction('{runtime_paths["resume"]}')">Resume</button>
        <button onclick="postAction('{runtime_paths["reconcile"]}')">Reconcile Now</button>
        <button onclick="postAction('{runtime_paths["requeue"]}')">Requeue Expired</button>
        <button class="secondary" onclick="postAction('{runtime_paths["stop"]}')">Stop Loop</button>
        <button class="secondary" onclick="refreshAll()">Refresh</button>
      </div>
      <div id="flash"></div>
      <div id="runtime" class="runtime-status muted">loading...</div>
    </section>

    <section class="grid">
      <section class="panel">
        <h2>Settings</h2>
        <div id="settings"></div>
      </section>
      <section class="panel">
        <h2>Spec Patch</h2>
        <form onsubmit="return submitPatch(this)">
          <label>JSON patch
            <textarea name="patch" id="patch"></textarea>
          </label>
          <div class="toolbar">
            <button type="submit">Apply Patch</button>
          </div>
        </form>
      </section>
    </section>

    <section class="grid">
      <section class="panel">
        <h2>Master Summary</h2>
        <pre id="summary"></pre>
      </section>
      <section class="panel">
        <h2>DB Snapshot</h2>
        <pre id="status"></pre>
      </section>
    </section>
  </main>

  <script>
    function pretty(value) {{
      return JSON.stringify(value, null, 2);
    }}

    function flash(message, isError=false) {{
      const node = document.getElementById('flash');
      node.textContent = message;
      node.style.color = isError ? '#9f1d1d' : '#8a3b12';
    }}

    async function callJson(path, method='GET', body=null) {{
      const options = {{
        method,
        headers: {{}},
      }};
      if (body !== null) {{
        options.headers['Content-Type'] = 'application/json';
        options.body = JSON.stringify(body);
      }}
      const response = await fetch(path, options);
      const text = await response.text();
      const payload = text ? JSON.parse(text) : null;
      if (!response.ok) {{
        throw new Error(payload && payload.error ? payload.error : text || response.statusText);
      }}
      return payload;
    }}

    async function postAction(path) {{
      try {{
        await callJson(path, 'POST', {{}});
        flash('Action finished.');
        await refreshAll();
      }} catch (error) {{
        flash(error.message, true);
      }}
    }}

    function renderRuntime(runtime) {{
      const lines = [
        `loop_running: ${{runtime.loop_running}}`,
        `paused: ${{runtime.paused}}`,
        `cycle_count: ${{runtime.cycle_count}}`,
        `poll_interval_seconds: ${{runtime.poll_interval_seconds}}`,
        `spec_path: ${{runtime.spec_path}}`,
        `output_root: ${{runtime.output_root}}`,
        `last_reconcile_started_at: ${{runtime.last_reconcile_started_at}}`,
        `last_reconcile_finished_at: ${{runtime.last_reconcile_finished_at}}`,
      ];
      if (runtime.last_error) {{
        lines.push(`last_error: ${{pretty(runtime.last_error)}}`);
      }}
      document.getElementById('runtime').textContent = lines.join('\\n');
    }}

    function renderNamedCollection(collectionKey, entries, title) {{
      if (!entries || !entries.length) {{
        return `<div class="settings-block"><h3>${{title}}</h3><div class="muted">none</div></div>`;
      }}
      const parts = [`<div class="settings-block"><h3>${{title}}</h3>`];
      for (const entry of entries) {{
        const encodedName = encodeURIComponent(entry.name);
        if (collectionKey === 'lineages') {{
          parts.push(`
            <form class="settings-form" onsubmit="return submitNamedSpecForm('${{collectionKey}}', '${{encodedName}}', this)">
              <label>${{entry.name}} <span class="inline-check"><input type="checkbox" name="enabled" ${{entry.enabled ? 'checked' : ''}}> enabled</span></label>
              <label>max_generations<input type="number" name="max_generations" value="${{entry.max_generations}}"></label>
              <label>on_accept
                <select name="on_accept">
                  <option value="continue_training" ${{entry.on_accept === 'continue_training' ? 'selected' : ''}}>continue_training</option>
                  <option value="stop" ${{entry.on_accept === 'stop' ? 'selected' : ''}}>stop</option>
                </select>
              </label>
              <label>on_reject
                <select name="on_reject">
                  <option value="stop" ${{entry.on_reject === 'stop' ? 'selected' : ''}}>stop</option>
                  <option value="restart_from_seed" ${{entry.on_reject === 'restart_from_seed' ? 'selected' : ''}}>restart_from_seed</option>
                </select>
              </label>
              <label>min_verify_top1_accuracy<input type="number" step="0.0001" name="min_verify_top1_accuracy" value="${{entry.promotion_thresholds.min_verify_top1_accuracy ?? ''}}"></label>
              <label>min_verify_top3_accuracy<input type="number" step="0.0001" name="min_verify_top3_accuracy" value="${{entry.promotion_thresholds.min_verify_top3_accuracy ?? ''}}"></label>
              <label>min_arena_score_rate<input type="number" step="0.0001" name="min_arena_score_rate" value="${{entry.promotion_thresholds.min_arena_score_rate ?? ''}}"></label>
              <label>safe_score_rate<input type="number" step="0.0001" name="safe_score_rate" value="${{entry.arena_progression.safe_score_rate}}"></label>
              <button type="submit">Save</button>
            </form>`);
        }} else {{
          parts.push(`
            <form class="settings-form" onsubmit="return submitNamedSpecForm('${{collectionKey}}', '${{encodedName}}', this)">
              <label>${{entry.name}} <span class="inline-check"><input type="checkbox" name="enabled" ${{entry.enabled ? 'checked' : ''}}> enabled</span></label>
              <button type="submit">Save</button>
            </form>`);
        }}
      }}
      parts.push('</div>');
      return parts.join('');
    }}

    function renderSettings(spec) {{
      const root = document.getElementById('settings');
      root.innerHTML = `
        <form class="settings-form" onsubmit="return submitPollForm(this)">
          <label>poll_interval_seconds<input type="number" step="0.1" name="poll_interval_seconds" value="${{spec.poll_interval_seconds}}"></label>
          <button type="submit">Save Poll Interval</button>
        </form>
        ${{renderNamedCollection('lineages', spec.lineages || [], 'Lineages')}}
        ${{renderNamedCollection('label_jobs', spec.label_jobs || [], 'Label Jobs')}}
        ${{renderNamedCollection('idle_phase10_jobs', spec.idle_phase10_jobs || [], 'Idle Phase10 Jobs')}}`;
      document.getElementById('patch').value = pretty(spec);
    }}

    async function submitPollForm(form) {{
      try {{
        await callJson('/api/v1/spec/patch', 'POST', {{
          poll_interval_seconds: Number(form.elements.poll_interval_seconds.value),
        }});
        flash('Poll interval updated.');
        await refreshAll();
      }} catch (error) {{
        flash(error.message, true);
      }}
      return false;
    }}

    function optionalNumber(value) {{
      return value === '' ? null : Number(value);
    }}

    async function submitNamedSpecForm(collectionKey, encodedName, form) {{
      const payload = {{
        enabled: form.elements.enabled.checked,
      }};
      if (collectionKey === 'lineages') {{
        payload.max_generations = Number(form.elements.max_generations.value);
        payload.on_accept = form.elements.on_accept.value;
        payload.on_reject = form.elements.on_reject.value;
        payload.promotion_thresholds = {{
          min_verify_top1_accuracy: optionalNumber(form.elements.min_verify_top1_accuracy.value),
          min_verify_top3_accuracy: optionalNumber(form.elements.min_verify_top3_accuracy.value),
          min_arena_score_rate: optionalNumber(form.elements.min_arena_score_rate.value),
        }};
        payload.arena_progression = {{
          safe_score_rate: Number(form.elements.safe_score_rate.value),
        }};
      }}
      try {{
        await callJson(`/api/v1/spec/${{collectionKey}}/${{encodedName}}`, 'POST', payload);
        flash('Settings updated.');
        await refreshAll();
      }} catch (error) {{
        flash(error.message, true);
      }}
      return false;
    }}

    async function submitPatch(form) {{
      try {{
        const patch = JSON.parse(form.elements.patch.value);
        await callJson('/api/v1/spec/patch', 'POST', patch);
        flash('Patch applied.');
        await refreshAll();
      }} catch (error) {{
        flash(error.message, true);
      }}
      return false;
    }}

    async function refreshAll() {{
      try {{
        const data = await callJson('/api/v1/bootstrap?limit=40');
        renderRuntime(data.runtime);
        renderSettings(data.spec);
        document.getElementById('summary').textContent = pretty(data.summary);
        document.getElementById('status').textContent = pretty(data.status);
      }} catch (error) {{
        flash(error.message, true);
      }}
    }}

    refreshAll();
    setInterval(refreshAll, 15000);
  </script>
</body>
</html>
"""


def _json_response(payload: Any) -> bytes:
    assert cherrypy is not None
    cherrypy.response.headers["Content-Type"] = "application/json; charset=utf-8"
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _html_response(payload: str) -> bytes:
    assert cherrypy is not None
    cherrypy.response.headers["Content-Type"] = "text/html; charset=utf-8"
    return payload.encode("utf-8")


def _text_response(payload: str) -> bytes:
    assert cherrypy is not None
    cherrypy.response.headers["Content-Type"] = "text/plain; charset=utf-8"
    return payload.encode("utf-8")


def _read_json_object() -> dict[str, Any]:
    assert cherrypy is not None
    raw = cherrypy.request.body.read()
    if not raw:
        return {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise cherrypy.HTTPError(400, f"invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise cherrypy.HTTPError(400, "expected a JSON object body")
    return payload


def _parse_limit(raw_limit: str) -> int:
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise cherrypy.HTTPError(400, f"invalid limit: {raw_limit}") from exc
    if limit <= 0:
        raise cherrypy.HTTPError(400, "limit must be positive")
    return limit


def _require_method(expected_method: str) -> None:
    assert cherrypy is not None
    if cherrypy.request.method != expected_method:
        raise cherrypy.HTTPError(405, f"expected {expected_method}")


def _require_cherrypy() -> None:
    if not hasattr(cherrypy, "config"):
        raise RuntimeError(
            "CherryPy is not installed. Install it before using the HTTP master server."
        )
