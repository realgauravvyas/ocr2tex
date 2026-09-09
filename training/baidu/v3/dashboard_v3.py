r"""Tiny read-only local dashboard for the v3 training/generation run.

Pure stdlib (http.server) -- no new dependencies to install. Polls the JSON
files v3 already writes (training_metrics.json, checkpoint_ptr.json,
probe/probe_results.json, bench/*_progress.json) and renders them as an
auto-refreshing page. It never writes anything -- read-only over files that
already live under D:\Claude Code\BaiduOCR-v3.

Run:
    python dashboard_v3.py
    -> open http://localhost:8787

--port to change the port if 8787 is taken.
"""
import json, time, argparse, http.server, socketserver, threading
from pathlib import Path

V3 = Path(__file__).resolve().parent
TRAIN_METRICS = V3 / "out" / "baidu-ocr-math-v3" / "training_metrics.json"
CKPT_PTR = V3 / "out" / "baidu-ocr-math-v3" / "checkpoint_ptr.json"
PROBE = V3 / "probe" / "probe_results.json"
BENCH = V3 / "bench"

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>BaiduOCR v3</title>
<style>
  /* Whole-page theme driven by training phase (see JS setPhase()) -- the point
     is a glance at the tab/window tells you the state, not just a small dot.
     Defaults to yellow (loading/unknown) until the first /data.json lands. */
  :root{--accent:#fbbf24;--accent-dim:#fbbf2433;--accent-bg:#1a160a}
  html.phase-green{--accent:#4ade80;--accent-dim:#4ade8033;--accent-bg:#0c1a10}
  html.phase-yellow{--accent:#fbbf24;--accent-dim:#fbbf2433;--accent-bg:#1a160a}
  html.phase-red{--accent:#f87171;--accent-dim:#f8717133;--accent-bg:#1a0d0d}
  body{font-family:-apple-system,Segoe UI,Arial,sans-serif;color:#e6edf3;margin:0;padding:24px;
       background:radial-gradient(1200px 500px at 50% -100px,var(--accent-bg),#0b0f14 70%);
       border-top:6px solid var(--accent);transition:background .6s,border-color .6s}
  h1{font-size:18px;color:var(--accent);margin:0 0 4px;transition:color .6s}
  .sub{color:#8b98a5;font-size:12px;margin-bottom:20px}
  .banner{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
          color:var(--accent);background:var(--accent-dim);border:1px solid var(--accent);
          border-radius:6px;padding:4px 10px;margin-bottom:16px;transition:color .6s,background .6s,border-color .6s}
  .card{background:#111820;border:1px solid var(--accent-dim);border-radius:10px;padding:16px 20px;margin-bottom:16px;transition:border-color .6s}
  .card h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:#8b98a5;margin:0 0 12px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
  .stat{background:#0b0f14;border-radius:8px;padding:10px 12px}
  .stat .v{font-size:20px;font-weight:600;color:#eaf6ff}
  .stat .l{font-size:11px;color:#8b98a5;margin-top:2px}
  .bar{height:8px;background:#1f2937;border-radius:4px;overflow:hidden;margin-top:10px}
  .bar>div{height:100%;background:linear-gradient(90deg,#3b82f6,#22d3ee)}
  table{width:100%;border-collapse:collapse;font-size:12px}
  td,th{padding:4px 8px;text-align:right;border-bottom:1px solid #1f2937}
  th{color:#8b98a5;font-weight:500}
  td:first-child,th:first-child{text-align:left}
  .phase-green{color:#4ade80}.phase-yellow{color:#fbbf24}.phase-red{color:#f87171}
  .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;vertical-align:1px}
  .dot-green{background:#4ade80;box-shadow:0 0 8px #4ade80aa}
  .dot-yellow{background:#fbbf24;box-shadow:0 0 8px #fbbf24aa}
  .dot-red{background:#f87171;box-shadow:0 0 8px #f87171aa}
  canvas{width:100%;height:120px;background:#0b0f14;border-radius:8px}
  .muted{color:#8b98a5;font-size:12px}
</style></head><body>
<h1>BaiduOCR v3</h1>
<div class="sub">D:\\Claude Code\\BaiduOCR-v3 &middot; auto-refreshes every 3s &middot; read-only</div>
<div id="banner" class="banner">loading&hellip;</div>
<div id="root"></div>
<script>
function fmt(x, d) { return (x===null||x===undefined) ? '-' : (typeof x==='number' ? x.toFixed(d===undefined?4:d) : x); }
function bar(pct){ return `<div class="bar"><div style="width:${Math.max(0,Math.min(100,pct||0))}%"></div></div>`; }

function setPhase(phase, label) {
  // Whole-page tint (html.phase-*, defined in <style>) so the state is visible
  // at a glance -- title bar/tab, background, borders, headings all shift, not
  // just a status dot buried in a card.
  document.documentElement.className = 'phase-' + (phase || 'yellow');
  const b = document.getElementById('banner');
  if (b) b.textContent = label;
  document.title = (phase === 'red' ? '\\u{1F534} ' : phase === 'green' ? '\\u{1F7E2} ' : '\\u{1F7E1} ') + 'BaiduOCR v3';
}

async function tick() {
  let d;
  try { d = await (await fetch('/data.json')).json(); } catch(e) { setPhase('red', 'DASHBOARD UNREACHABLE'); return; }
  const t = d.train || {};
  const root = document.getElementById('root');
  let html = '';

  const phase = t.phase || 'red';
  setPhase(phase, `${(t.status||'unknown').toUpperCase()} \\u2014 ${t.phase_reason||''}`);

  html += '<div class="card"><h2>Training</h2>';
  if (!t.found) {
    html += '<div class="muted">no training_metrics.json yet -- has run_v3.ps1 train started?</div>';
  } else {
    const st = t.status || 'unknown';
    html += `<div class="grid">`
      + `<div class="stat"><div class="v phase-${phase}"><span class="dot dot-${phase}"></span>${st}</div>`
      + `<div class="l">${t.phase_reason || 'status'}</div></div>`
      + `<div class="stat"><div class="v">${t.step||0} / ${t.max_steps||'?'}</div><div class="l">step</div></div>`
      + `<div class="stat"><div class="v">${fmt(t.percent,1)}%</div><div class="l">progress</div></div>`
      + `<div class="stat"><div class="v">${fmt(t.loss,4)}</div><div class="l">train loss</div></div>`
      + `<div class="stat"><div class="v">${fmt(t.eval_cer,4)}</div><div class="l">decoded CER @ step ${t.last_eval_step||'-'}${t.last_eval_step&&t.step?` (${t.step-t.last_eval_step} steps ago)`:''}</div></div>`
      + `<div class="stat"><div class="v">${fmt(t.best_cer,4)}</div><div class="l">best decoded CER</div></div>`
      + `<div class="stat"><div class="v">${t.next_eval_step||'-'}</div><div class="l">next CER eval (in ${t.steps_to_next_eval!=null?t.steps_to_next_eval:'-'} steps)</div></div>`
      + `<div class="stat"><div class="v">${fmt(t.val_loss,4)}</div><div class="l">val loss (teacher-forced)</div></div>`
      + `<div class="stat"><div class="v">${fmt(t.eval_struct,1)}%</div><div class="l">last struct%%</div></div>`
      + `<div class="stat"><div class="v">${t.eta_s ? Math.round(t.eta_s/3600*10)/10+'h' : '-'}</div><div class="l">ETA</div></div>`
      + `<div class="stat"><div class="v">${t.grad_norm!=null?fmt(t.grad_norm,2):'-'}</div><div class="l">grad norm</div></div>`
      + `<div class="stat"><div class="v">${t.lr!=null?t.lr.toExponential(2):'-'}</div><div class="l">lr</div></div>`
      + `<div class="stat"><div class="v">${t.skipped_long||0}</div><div class="l">targets skipped (too long)</div></div>`
      + `</div>` + bar(t.percent);
    if (t.error) html += `<div class="muted status-error" style="margin-top:10px">error: ${t.error}</div>`;
    const cs = t.cer_series || [];
    if (cs.length) {
      const chips = cs.map(([s,v],i) => {
        let arrow = '';
        if (i > 0) { const d = v - cs[i-1][1]; arrow = d < 0 ? ` <span style="color:#4ade80">&darr;${Math.abs(d).toFixed(4)}</span>` : ` <span style="color:#f87171">&uarr;${d.toFixed(4)}</span>`; }
        return `<span style="display:inline-block;background:#0b0f14;border-radius:6px;padding:4px 10px;margin:3px 4px 0 0">step ${s}: <b>${v.toFixed(4)}</b>${arrow}</span>`;
      }).join('');
      html += `<div style="margin-top:12px"><div class="l" style="color:#8b98a5;font-size:11px;margin-bottom:4px">DECODED CER HISTORY (only updates every ${(t.config&&t.config.metric_eval_steps)||400} steps &mdash; a flat value between evals is stale, not stalled)</div>${chips}</div>`;
    }
    html += `<canvas id="lc"></canvas>`;
  }
  html += '</div>';

  html += '<div class="card"><h2>Checkpoint</h2>';
  if (d.ckpt) {
    const age = d.ckpt.saved_at ? Math.round((Date.now()/1000 - d.ckpt.saved_at)) : null;
    html += `<div class="grid">`
      + `<div class="stat"><div class="v">${d.ckpt.slot}</div><div class="l">active slot</div></div>`
      + `<div class="stat"><div class="v">${d.ckpt.step}</div><div class="l">saved at step</div></div>`
      + `<div class="stat"><div class="v">${age!=null?age+'s ago':'-'}</div><div class="l">last save</div></div>`
      + `</div><div class="muted" style="margin-top:8px">power-safe: this pointer only moves after a save fully completes -- a crash mid-write is always ignored on resume.</div>`;
  } else {
    html += '<div class="muted">no checkpoint saved yet</div>';
  }
  html += '</div>';

  if (d.probe) {
    html += '<div class="card"><h2>Probe (decode-fix A/B on v1 adapter)</h2><table><tr><th>variant</th><th>CER</th><th>struct%</th><th>complete%</th></tr>';
    for (const [k,v] of Object.entries(d.probe.summary||{})) {
      html += `<tr><td>${k}</td><td>${fmt(v.mean_cer,4)}</td><td>${fmt(v.struct_pct,1)}</td><td>${fmt(v.complete_pct,1)}</td></tr>`;
    }
    html += '</table></div>';
  }

  if (d.bench && d.bench.length) {
    html += '<div class="card"><h2>Generation (test-set benchmark)</h2>';
    for (const b of d.bench) {
      html += `<div class="grid" style="margin-bottom:8px"><div class="stat"><div class="v">${b.label}</div><div class="l">label</div></div>`
        + `<div class="stat"><div class="v">${b.done||0}/${b.total||'?'}</div><div class="l">pages</div></div>`
        + `<div class="stat"><div class="v">${b.status||'-'}</div><div class="l">status</div></div>`
        + `<div class="stat"><div class="v">${b.truncated||0}</div><div class="l">truncated so far</div></div>`
        + `<div class="stat"><div class="v">${b.eta_h!=null?b.eta_h+'h':'-'}</div><div class="l">ETA</div></div></div>`
        + bar(b.pct);
    }
    html += '</div>';
  }

  root.innerHTML = html;

  // loss/CER sparkline from history
  const hist = (t.history||[]);
  const c = document.getElementById('lc');
  if (c && hist.length) {
    const ctx = c.getContext('2d');
    c.width = c.clientWidth; c.height = 120;
    const losses = hist.filter(h=>h.loss!=undefined).map(h=>h.loss);
    if (losses.length > 1) {
      const max = Math.max(...losses), min = Math.min(...losses);
      ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 1.5; ctx.beginPath();
      losses.forEach((v,i) => {
        const x = i/(losses.length-1)*c.width;
        const y = c.height - ((v-min)/((max-min)||1))*c.height*0.85 - 5;
        i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
      });
      ctx.stroke();
    }
  }
}
tick(); setInterval(tick, 3000);
</script>
</body></html>"""


def _read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def _phase(t):
    """green = actively stepping, yellow = loading/warmup, red = not actually
    progressing (error, or claims 'running' but hasn't written metrics in a
    while).

    The naive version of this (color = status field verbatim) is what let a
    real 2-hour GPU-contention stall sit there reading "running" the whole
    time -- status said running because nothing ever set it to anything else;
    the process just wasn't making progress. So "green" here requires BOTH
    status=='running' AND a metrics write recently enough to prove it's alive.

    The one legitimate long gap is a decode-eval itself: eval_cer() runs
    24 greedy decodes (observed 1400-1650s each) before the caller writes
    metrics again, so nothing is wrong for up to ~30 minutes there. STALE_S is
    set above the worst observed eval duration with headroom, not at a tight
    per-step interval, so real evals don't falsely read as stopped.
    """
    STALE_S = 2400  # 40 min: longest real eval so far is ~28 min
    if not t.get("found"):
        return "red", "not started"
    status = t.get("status", "unknown")
    if status == "error":
        return "red", "error"
    if status in ("done", "smoke_ok"):
        return "green", status
    if status == "loading":
        return "yellow", "loading"
    if status == "running":
        age = time.time() - (t.get("updated_at") or 0)
        if age > STALE_S:
            return "red", f"no update in {int(age//60)}min -- likely stalled or crashed"
        step, warm = t.get("step") or 0, t.get("warmup_steps_computed") or 0
        if step < warm:
            return "yellow", f"warmup ({step}/{warm})"
        return "green", "running"
    return "red", status


def build_data():
    t = _read_json(TRAIN_METRICS) or {}
    t["found"] = TRAIN_METRICS.exists()
    # Warmup step count isn't stored directly -- recompute it the same way
    # train_baidu_v3.py does, so the warmup/running boundary above is exact.
    cfg = t.get("config") or {}
    total = t.get("max_steps") or 0
    ws = int(cfg.get("warmup_steps") or 0)
    if total:
        warm_n = ws if ws > 0 else max(20, int(0.05 * total))
        t["warmup_steps_computed"] = min(warm_n, max(1, total - 1))
    t["phase"], t["phase_reason"] = _phase(t)
    # Decode-evals only run every metric_eval_steps (~5.5h apart), so the CER on
    # screen is STALE between them, not stagnant. Without this, a flat number for
    # hours reads as "the model stopped improving" -- surface the staleness.
    cfg = t.get("config") or {}
    ev = int(cfg.get("metric_eval_steps") or 0)
    step = int(t.get("step") or 0)
    hist = t.get("history") or []
    ecs = [x for x in hist if "eval_cer" in x]
    t["eval_count"] = len(ecs)
    t["last_eval_step"] = ecs[-1]["step"] if ecs else None
    t["cer_series"] = [(x["step"], x["eval_cer"]) for x in ecs]
    if len(ecs) >= 2:
        prev, last = ecs[-2]["eval_cer"], ecs[-1]["eval_cer"]
        t["cer_delta"] = round(last - prev, 4)
        t["cer_delta_pct"] = round(100.0 * (last - prev) / prev, 1) if prev else None
    if ev and step:
        t["next_eval_step"] = ((step // ev) + 1) * ev
        t["steps_to_next_eval"] = t["next_eval_step"] - step
    ckpt = _read_json(CKPT_PTR)
    probe = _read_json(PROBE)
    bench = []
    if BENCH.exists():
        for p in sorted(BENCH.glob("*_progress.json")):
            d = _read_json(p) or {}
            d.setdefault("label", p.stem.replace("_progress", ""))
            bench.append(d)
    return {"train": t, "ckpt": ckpt, "probe": probe, "bench": bench}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass  # keep the console quiet

    def do_GET(self):
        if self.path.startswith("/data.json"):
            body = json.dumps(build_data()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    with socketserver.TCPServer(("127.0.0.1", args.port), Handler) as httpd:
        print(f"BaiduOCR v3 dashboard: http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
