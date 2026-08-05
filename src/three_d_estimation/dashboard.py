"""Browser dashboard and URL serving for online surface MLE progress."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Mapping

import numpy as np

from .types import MLEEstimate


DASHBOARD_DATA_FILENAME = "dashboard_data.json"
DASHBOARD_INDEX_FILENAME = "index.html"
DEFAULT_DASHBOARD_PORT = 8878

_HTTP_SERVERS: dict[tuple[str, int], ThreadingHTTPServer] = {}
_HTTP_SERVER_ROOTS: dict[tuple[str, int], Path] = {}
_HTTP_THREADS: list[threading.Thread] = []


class _QuietHandler(SimpleHTTPRequestHandler):
    """Serve static dashboard files without per-request terminal noise."""

    def log_message(self, format: str, *args: object) -> None:
        """Suppress the standard request log."""
        del format, args


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Durably replace one dashboard artifact."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Dashboard staging file exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _tcp_port_is_open(host: str, port: int) -> bool:
    """Return whether a local TCP endpoint is already accepting connections."""
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        with socket.create_connection((connect_host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _default_public_host() -> str:
    """Return a likely browser-reachable host for dashboard URLs."""
    configured = os.environ.get(
        "MLE_DASHBOARD_PUBLIC_HOST",
        os.environ.get("CUI_SPLIT_VIEW_PUBLIC_HOST"),
    )
    if configured:
        return configured
    try:
        output = subprocess.check_output(
            ["hostname", "-I"],
            text=True,
            timeout=0.2,
        )
        candidates = [value for value in output.split() if value]
        for candidate in candidates:
            if candidate.startswith("100."):
                return candidate
        for candidate in candidates:
            if not candidate.startswith(("127.", "172.")):
                return candidate
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.settimeout(0.2)
            probe.connect(("8.8.8.8", 80))
            candidate = str(probe.getsockname()[0])
            if candidate and not candidate.startswith("127."):
                return candidate
    except OSError:
        pass
    return "127.0.0.1"


def _available_dashboard_port(host: str, requested_port: int) -> int:
    """Return the first locally available dashboard port at or above a request."""
    for candidate in range(int(requested_port), min(int(requested_port) + 100, 65536)):
        if (host, candidate) not in _HTTP_SERVERS and not _tcp_port_is_open(
            host, candidate
        ):
            return candidate
    raise OSError(
        f"No dashboard port is available in {requested_port}.."
        f"{min(int(requested_port) + 99, 65535)}."
    )


def _dashboard_browser_url(public_host: str, port: int) -> str:
    """Return one explicit browser URL, including IPv6 brackets and index path."""
    display_host = str(public_host).strip()
    if not display_host:
        raise ValueError("Dashboard public host must be nonempty.")
    if "://" in display_host or "/" in display_host:
        raise ValueError("Dashboard public host must be a host name or IP address.")
    if ":" in display_host and not display_host.startswith("["):
        display_host = f"[{display_host}]"
    return f"http://{display_host}:{int(port)}/index.html"


def ensure_dashboard_server(
    output_dir: str | Path,
    *,
    host: str = "0.0.0.0",
    port: int = DEFAULT_DASHBOARD_PORT,
    public_host: str | None = None,
) -> str:
    """Start or reuse a persistent static server and return its browser URL."""
    root = Path(output_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dashboard output directory does not exist: {root}")
    parsed_port = int(port)
    if not 1 <= parsed_port <= 65535:
        raise ValueError("Dashboard port must lie in 1..65535.")
    display_host = (
        _default_public_host()
        if public_host is None and host in {"0.0.0.0", "::"}
        else str(public_host or host)
    )
    requested_key = (host, parsed_port)
    if requested_key in _HTTP_SERVERS and _HTTP_SERVER_ROOTS.get(requested_key) == root:
        return _dashboard_browser_url(display_host, parsed_port)
    selected_port = _available_dashboard_port(host, parsed_port)
    url = _dashboard_browser_url(display_host, selected_port)
    key = (host, selected_port)

    log_path = root / f"dashboard_server_{selected_port}.log"
    pid_path = root / f"dashboard_server_{selected_port}.pid"
    try:
        with log_path.open("ab") as log_handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "http.server",
                    str(selected_port),
                    "--bind",
                    host,
                    "--directory",
                    root.as_posix(),
                ],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
        for _ in range(20):
            if _tcp_port_is_open(host, selected_port):
                return url
            if process.poll() is not None:
                break
            time.sleep(0.05)
    except OSError:
        pass

    handler = partial(_QuietHandler, directory=root.as_posix())
    server = ThreadingHTTPServer((host, selected_port), handler)
    thread = threading.Thread(
        target=server.serve_forever,
        name="online-mle-dashboard",
        daemon=True,
    )
    thread.start()
    _HTTP_SERVERS[key] = server
    _HTTP_SERVER_ROOTS[key] = root
    _HTTP_THREADS.append(thread)
    return url


def _finite_float(value: object, *, fallback: float = 0.0) -> float:
    """Return a finite JSON float for dashboard display."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return parsed if np.isfinite(parsed) else float(fallback)


def _hotspot_payload(estimate: MLEEstimate) -> list[dict[str, object]]:
    """Return compact finite hotspot rows from estimate diagnostics."""
    raw = estimate.diagnostics.get("hotspot_clusters", [])
    if not isinstance(raw, list):
        return []
    hotspots: list[dict[str, object]] = []
    for value in raw:
        if not isinstance(value, Mapping):
            continue
        centroid = np.asarray(value.get("centroid_xyz", ()), dtype=float)
        if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
            continue
        hotspots.append(
            {
                "isotope": str(value.get("isotope", "unknown")),
                "cluster_id": int(value.get("cluster_id", len(hotspots))),
                "centroid_xyz": centroid.tolist(),
                "integrated_strength_cps_1m": _finite_float(
                    value.get("integrated_strength_cps_1m", 0.0)
                ),
                "peak_density_cps_1m_m2": _finite_float(
                    value.get("peak_density_cps_1m_m2", 0.0)
                ),
                "surface_kinds": [
                    str(item) for item in (value.get("surface_kinds") or [])
                ],
            }
        )
    return hotspots


def _dashboard_payload(
    estimate: MLEEstimate | None,
    state: Mapping[str, object],
) -> dict[str, object]:
    """Build the browser's estimator-only live data contract."""
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": str(state.get("status", "starting")),
        "run_id": state.get("run_id"),
        "mode": state.get("mode"),
        "isotopes": list(state.get("isotopes", [])),
        "record_count": int(state.get("record_count", 0)),
        "latest_step_id": state.get("latest_step_id"),
        "latest_station_id": state.get("latest_station_id"),
        "station_reports": list(state.get("station_reports", [])),
        "planning": state.get("latest_planning"),
        "summary": None,
        "patches": [],
        "density_by_isotope": {},
        "detector_positions_xyz": [],
        "hotspots": [],
    }
    if estimate is None:
        return payload
    diagnostics = estimate.diagnostics
    payload["summary"] = {
        "objective": _finite_float(estimate.objective_value),
        "poisson_deviance": _finite_float(estimate.poisson_deviance),
        "iterations": int(estimate.iterations),
        "converged": bool(estimate.converged),
        "patch_count": len(estimate.patches),
        "residual_l2": _finite_float(diagnostics.get("residual_l2", 0.0)),
    }
    payload["patches"] = [
        {
            "patch_id": patch.patch_id,
            "centroid_xyz": patch.centroid_xyz.tolist(),
            "area_m2": float(patch.area_m2),
            "surface_kind": patch.surface_kind,
            "object_id": patch.object_id,
        }
        for patch in estimate.patches
    ]
    payload["density_by_isotope"] = {
        isotope: estimate.density_by_isotope[index].tolist()
        for index, isotope in enumerate(estimate.isotope_names)
    }
    positions = diagnostics.get("detector_positions_xyz", [])
    position_array = np.asarray(positions, dtype=float)
    if (
        position_array.ndim == 2
        and position_array.shape[1:] == (3,)
        and np.all(np.isfinite(position_array))
    ):
        payload["detector_positions_xyz"] = position_array.tolist()
    payload["hotspots"] = _hotspot_payload(estimate)
    return payload


class OnlineMLEDashboard:
    """Publish a self-refreshing estimator-only browser workspace."""

    def __init__(self, output_dir: str | Path) -> None:
        """Create static dashboard assets in an online result directory."""
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.output_dir / DASHBOARD_INDEX_FILENAME
        self.data_path = self.output_dir / DASHBOARD_DATA_FILENAME
        _write_bytes_atomic(self.index_path, _DASHBOARD_HTML.encode("utf-8"))

    def publish(
        self,
        estimate: MLEEstimate | None,
        state: Mapping[str, object],
    ) -> None:
        """Atomically publish the latest browser data snapshot."""
        payload = _dashboard_payload(estimate, state)
        encoded = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        _write_bytes_atomic(self.data_path, encoded)


_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>Radiation Surface MLE — Live</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #090b0d;
      --surface: #0e1114;
      --line: #252a2f;
      --muted: #8b959e;
      --text: #f2f5f7;
      --accent: #40e0d0;
      --danger: #ff756d;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; background: var(--bg); color: var(--text); }
    body { font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    button { font: inherit; }
    .topbar {
      height: 70px; display: flex; align-items: center; justify-content: space-between;
      padding: 0 28px; border-bottom: 1px solid var(--line); background: rgba(9,11,13,.94);
    }
    .brand { display: flex; align-items: baseline; gap: 13px; letter-spacing: -.02em; }
    .brand strong { font-size: 18px; font-weight: 650; }
    .brand span { color: var(--muted); font-size: 12px; letter-spacing: .08em; text-transform: uppercase; }
    .run-state { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 12px; }
    .pulse { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 0 rgba(64,224,208,.45); animation: pulse 2s infinite; }
    .pulse.finalized { animation: none; box-shadow: none; }
    @keyframes pulse { 70% { box-shadow: 0 0 0 8px rgba(64,224,208,0); } }
    .workspace { height: calc(100svh - 70px); display: grid; grid-template-columns: minmax(0, 1fr) 350px; }
    .stage { position: relative; min-width: 0; overflow: hidden; background: radial-gradient(circle at 50% 48%, #151b1f 0, var(--bg) 64%); }
    #surface { width: 100%; height: 100%; display: block; }
    .stage-head { position: absolute; inset: 24px 26px auto 26px; display: flex; align-items: flex-start; justify-content: space-between; pointer-events: none; }
    .stage-title h1 { margin: 0; font-size: clamp(23px, 3vw, 42px); font-weight: 560; letter-spacing: -.045em; }
    .stage-title p { margin: 7px 0 0; color: var(--muted); max-width: 470px; }
    .tabs { display: flex; gap: 5px; pointer-events: auto; }
    .tabs button { border: 0; color: var(--muted); background: transparent; padding: 7px 10px; cursor: pointer; border-bottom: 1px solid transparent; }
    .tabs button.active { color: var(--text); border-color: var(--accent); }
    .legend { position: absolute; left: 28px; bottom: 25px; color: var(--muted); font-size: 11px; letter-spacing: .02em; }
    .legend i { display: inline-block; width: 108px; height: 3px; margin: 0 8px; vertical-align: middle; background: linear-gradient(90deg, #263137, var(--accent)); }
    .inspector { overflow: auto; border-left: 1px solid var(--line); background: var(--surface); padding: 25px 24px 32px; }
    .section { padding: 0 0 24px; margin: 0 0 24px; border-bottom: 1px solid var(--line); }
    .section:last-child { border-bottom: 0; }
    .eyebrow { color: var(--muted); font-size: 10px; letter-spacing: .14em; text-transform: uppercase; margin: 0 0 12px; }
    .run-id { font-size: 15px; overflow-wrap: anywhere; }
    .metric { display: grid; grid-template-columns: 1fr auto; align-items: baseline; padding: 8px 0; }
    .metric span { color: var(--muted); }
    .metric strong { font-variant-numeric: tabular-nums; font-weight: 520; }
    .timeline { list-style: none; margin: 0; padding: 0; }
    .timeline li { position: relative; display: grid; grid-template-columns: 18px 1fr auto; gap: 8px; align-items: center; min-height: 35px; color: var(--muted); }
    .timeline li::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }
    .timeline li:not(:last-child)::after { content: ""; position: absolute; left: 2px; top: 21px; bottom: -12px; width: 1px; background: var(--line); }
    .timeline b { color: var(--text); font-weight: 500; }
    .hotspot { padding: 10px 0; border-top: 1px solid var(--line); }
    .hotspot:first-of-type { border-top: 0; }
    .hotspot-top { display: flex; justify-content: space-between; gap: 12px; }
    .hotspot small { color: var(--muted); }
    .hotspot strong { font-weight: 520; font-variant-numeric: tabular-nums; }
    .empty { color: var(--muted); padding: 6px 0; }
    .fade { animation: settle .25s ease both; }
    @keyframes settle { from { opacity: .55; transform: translateY(2px); } }
    @media (max-width: 860px) {
      .topbar { padding: 0 18px; }
      .workspace { height: auto; grid-template-columns: 1fr; }
      .stage { height: 62svh; min-height: 430px; }
      .inspector { border-left: 0; border-top: 1px solid var(--line); }
      .stage-head { inset: 18px 18px auto; flex-direction: column; gap: 15px; }
    }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; } }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand"><strong>Radiation Surface MLE</strong><span>online estimator</span></div>
    <div class="run-state"><i id="pulse" class="pulse"></i><span id="status">STARTING</span><span>·</span><span id="freshness">waiting for data</span></div>
  </header>
  <main class="workspace">
    <section class="stage">
      <canvas id="surface"></canvas>
      <div class="stage-head">
        <div class="stage-title"><h1>Surface intensity</h1><p>Station-complete, all-history maximum likelihood estimate. Detector path is shown in white.</p></div>
        <nav id="tabs" class="tabs" aria-label="Isotope"></nav>
      </div>
      <div class="legend">0 <i></i> peak density</div>
    </section>
    <aside class="inspector">
      <section class="section"><p class="eyebrow">Run</p><div id="runId" class="run-id">—</div></section>
      <section class="section"><p class="eyebrow">Current fit</p><div id="metrics"></div></section>
      <section class="section"><p class="eyebrow">Recommended next action</p><div id="nextAction"></div></section>
      <section class="section"><p class="eyebrow">Station history</p><ol id="timeline" class="timeline"></ol></section>
      <section class="section"><p class="eyebrow">Hotspots</p><div id="hotspots"></div></section>
    </aside>
  </main>
  <script>
    const state = { data: null, isotope: null, updatedAt: 0 };
    const canvas = document.getElementById('surface');
    const ctx = canvas.getContext('2d');
    const esc = value => String(value ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const fmt = value => Number.isFinite(Number(value)) ? Number(value).toLocaleString(undefined, {maximumFractionDigits: 3}) : '—';
    function project(point, bounds, width, height) {
      const [x,y,z] = point; const [minX,maxX,minY,maxY,minZ,maxZ] = bounds;
      const sx = (x-minX)/Math.max(maxX-minX, 1e-9)-.5;
      const sy = (y-minY)/Math.max(maxY-minY, 1e-9)-.5;
      const sz = (z-minZ)/Math.max(maxZ-minZ, 1e-9);
      return [width*.5 + (sx-sy)*width*.43, height*.66 + (sx+sy)*height*.20 - sz*height*.42];
    }
    function draw() {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width*ratio)); canvas.height = Math.max(1, Math.floor(rect.height*ratio));
      ctx.setTransform(ratio,0,0,ratio,0,0); ctx.clearRect(0,0,rect.width,rect.height);
      const data = state.data; if (!data || !data.patches.length) {
        ctx.fillStyle='#7f8a92'; ctx.font='13px system-ui'; ctx.textAlign='center'; ctx.fillText('Waiting for first station-complete MLE fit', rect.width/2, rect.height/2); return;
      }
      const plan=data.planning&&data.planning.selected_action; const plannedPose=plan&&plan.detector_pose_xyz;
      const points = data.patches.map(p => p.centroid_xyz).concat(data.detector_positions_xyz || []).concat(plannedPose?[plannedPose]:[]);
      const axis = i => points.map(p => Number(p[i]));
      const pad=.05; let bounds=[Math.min(...axis(0)),Math.max(...axis(0)),Math.min(...axis(1)),Math.max(...axis(1)),Math.min(...axis(2)),Math.max(...axis(2))];
      if (bounds[0]===bounds[1]) { bounds[0]-=1; bounds[1]+=1; } if (bounds[2]===bounds[3]) { bounds[2]-=1; bounds[3]+=1; } if (bounds[4]===bounds[5]) bounds[5]+=1;
      bounds=[bounds[0]-pad,bounds[1]+pad,bounds[2]-pad,bounds[3]+pad,bounds[4],bounds[5]+pad];
      const density = data.density_by_isotope[state.isotope] || []; const peak=Math.max(...density,1e-12);
      const rows=data.patches.map((p,i)=>({p,i,depth:p.centroid_xyz[0]+p.centroid_xyz[1]})).sort((a,b)=>a.depth-b.depth);
      ctx.strokeStyle='rgba(255,255,255,.08)'; ctx.lineWidth=1;
      for (const z of [0,.5,1]) { const a=project([bounds[0],bounds[2],bounds[4]+z*(bounds[5]-bounds[4])],bounds,rect.width,rect.height); const b=project([bounds[1],bounds[3],bounds[4]+z*(bounds[5]-bounds[4])],bounds,rect.width,rect.height); ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke(); }
      for (const row of rows) { const value=Number(density[row.i]||0); const normalized=Math.sqrt(Math.max(0,value/peak)); const [x,y]=project(row.p.centroid_xyz,bounds,rect.width,rect.height); const radius=1.8+normalized*7.5; ctx.beginPath();ctx.arc(x,y,radius,0,Math.PI*2);ctx.fillStyle=`rgba(64,224,208,${.12+normalized*.82})`;ctx.fill(); }
      const path=data.detector_positions_xyz||[]; if(path.length){ctx.beginPath();path.forEach((p,i)=>{const q=project(p,bounds,rect.width,rect.height);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.strokeStyle='rgba(255,255,255,.68)';ctx.lineWidth=1.4;ctx.stroke();for(const p of path){const q=project(p,bounds,rect.width,rect.height);ctx.beginPath();ctx.arc(...q,2.4,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();}}
      if(plannedPose){const [x,y]=project(plannedPose,bounds,rect.width,rect.height);ctx.save();ctx.translate(x,y);ctx.rotate(Math.PI/4);ctx.fillStyle='#40e0d0';ctx.fillRect(-5,-5,10,10);ctx.restore();}
    }
    function render(data) {
      state.data=data; if(!state.isotope || !data.isotopes.includes(state.isotope)) state.isotope=data.isotopes[0]||null;
      document.getElementById('status').textContent=String(data.status||'starting').toUpperCase(); document.getElementById('pulse').classList.toggle('finalized',data.status==='finalized');
      document.getElementById('runId').textContent=data.run_id||'—';
      const s=data.summary||{}; const rows=[['Records',data.record_count],['Latest station',data.latest_station_id],['Surface patches',s.patch_count],['Objective',fmt(s.objective)],['Poisson deviance',fmt(s.poisson_deviance)],['Iterations',s.iterations],['Converged',s.converged===true?'yes':s.converged===false?'no':'—']];
      document.getElementById('metrics').innerHTML=rows.map(([k,v])=>`<div class="metric"><span>${esc(k)}</span><strong>${esc(v)}</strong></div>`).join('');
      const plan=data.planning&&data.planning.selected_action;document.getElementById('nextAction').innerHTML=plan?`<div class="metric"><span>Pose xyz</span><strong>${plan.detector_pose_xyz.map(fmt).join(' / ')}</strong></div><div class="metric"><span>Fe/Pb pair IDs</span><strong>${plan.shield_pair_ids.map(esc).join(' → ')}</strong></div><div class="metric"><span>Live time</span><strong>${plan.live_time_s_by_view.map(fmt).join(' / ')} s</strong></div><div class="metric"><span>Information gain</span><strong>${fmt(plan.information_gain_nats)} nat</strong></div><div class="metric"><span>Expected counts</span><strong>${plan.expected_total_counts_by_view.map(fmt).join(' / ')}</strong></div>`:'<div class="empty">Waiting for runtime candidates</div>';
      const tabs=document.getElementById('tabs');tabs.innerHTML=data.isotopes.map(iso=>`<button class="${iso===state.isotope?'active':''}" data-iso="${esc(iso)}">${esc(iso)}</button>`).join('');tabs.querySelectorAll('button').forEach(b=>b.onclick=()=>{state.isotope=b.dataset.iso;render(data)});
      const timeline=(data.station_reports||[]).slice(-8);document.getElementById('timeline').innerHTML=timeline.length?timeline.map(item=>`<li><b>Station ${esc(item.station_id)}</b><span>step ${esc(item.data_cutoff_step)}</span></li>`).join(''):'<li class="empty">No completed station yet</li>';
      const hotspots=(data.hotspots||[]).filter(h=>!state.isotope||h.isotope===state.isotope);document.getElementById('hotspots').innerHTML=hotspots.length?hotspots.slice(0,8).map(h=>`<div class="hotspot fade"><div class="hotspot-top"><span>${esc(h.isotope)} · ${esc((h.surface_kinds||[]).join(', '))}</span><strong>${fmt(h.integrated_strength_cps_1m)} cps</strong></div><small>xyz ${h.centroid_xyz.map(fmt).join(' / ')}</small></div>`).join(''):'<div class="empty">No active hotspot cluster</div>';
      state.updatedAt=Date.now(); document.getElementById('freshness').textContent='updated now'; draw();
    }
    async function refresh(){try{const response=await fetch(`dashboard_data.json?t=${Date.now()}`,{cache:'no-store'});if(!response.ok)throw new Error(response.status);render(await response.json())}catch(error){document.getElementById('freshness').textContent='reconnecting';}}
    setInterval(()=>{if(state.updatedAt){const seconds=Math.floor((Date.now()-state.updatedAt)/1000);document.getElementById('freshness').textContent=seconds<3?'updated now':`updated ${seconds}s ago`; }},1000);
    window.addEventListener('resize',draw); refresh(); setInterval(refresh,2000);
  </script>
</body>
</html>
"""


__all__ = [
    "DASHBOARD_DATA_FILENAME",
    "DASHBOARD_INDEX_FILENAME",
    "DEFAULT_DASHBOARD_PORT",
    "OnlineMLEDashboard",
    "ensure_dashboard_server",
]
