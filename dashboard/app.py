import os
import logging
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

log = logging.getLogger("dashboard")
app = FastAPI()

_db     = None
_config = None

@app.get("/", response_class=HTMLResponse)
async def dashboard():
  try:
    stats   = await _db.get_stats()
    open_   = await _db.get_open_positions()
    closed  = await _db.get_closed_positions(limit=20)
    signals = await _db.get_recent_signals(limit=10)

    start = float(os.getenv("BANKROLL","1000"))
    roi   = ((stats["bankroll"]-start)/start*100)
    total = stats["wins"]+stats["losses"]
    wr    = round(stats["wins"]/total*100,1) if total>0 else 0

    mode = "Simulation" if (_config or {}).get("SIMULATION", True) else "Live"

    def pc(v): return "#3B82F6" if v>=0 else "#EF4444"

    open_rows = "".join([f"""<tr>
        <td class="q">{p['question'][:70]}...</td>
        <td class="{'yes' if p['side']=='YES' else 'no'}">{p['side']}</td>
        <td class="num">{p['side_price']*100:.1f}&#162;</td>
        <td class="num">{(p.get('current_price') or p['side_price'])*100:.1f}&#162;</td>
        <td class="num" style="color:{pc(p.get('unrealized_pnl',0))}">{p.get('unrealized_pnl',0):+.2f}$</td>
        <td class="num ev">+{p['ev']*100:.1f}%</td>
        <td class="num kl">{p['kl']:.3f}</td>
        <td class="num">${p['stake_amt']:.2f}</td>
        <td><a href="{p['url']}" target="_blank" class="link-arrow">&#8599;</a></td>
    </tr>""" for p in open_]) or '<tr><td colspan="9" class="empty">Нет открытых позиций</td></tr>'

    closed_rows = "".join([f"""<tr>
        <td class="q">{t['question'][:60]}...</td>
        <td class="{'yes' if t['side']=='YES' else 'no'}">{t['side']}</td>
        <td class="num">{t['side_price']*100:.1f}&#162;</td>
        <td>{t.get('outcome','?')}</td>
        <td class="num" style="color:{pc(t['pnl'])}">{t['pnl']:+.2f}$</td>
        <td><span class="badge {'win' if t['result']=='WIN' else 'loss'}">{t['result']}</span></td>
        <td class="num ev">+{t['ev']*100:.1f}%</td>
    </tr>""" for t in reversed(closed)]) or '<tr><td colspan="7" class="empty">Нет сделок</td></tr>'

    sig_rows = "".join([f"""<tr>
        <td class="q">{s['question'][:60]}...</td>
        <td class="{'yes' if s['side']=='YES' else 'no'}">{s['side']}</td>
        <td class="num">{s['p_market']*100:.1f}&#162;</td>
        <td class="num">{s['p_final']*100:.1f}&#162;</td>
        <td class="num ev">+{s['ev']*100:.1f}%</td>
        <td class="num kl">{s['kl']:.3f}</td>
        <td class="source">{s.get('source','math')}</td>
    </tr>""" for s in signals]) or '<tr><td colspan="7" class="empty">Нет сигналов</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quant Engine v3</title>
<!-- auto-refresh disabled -->
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{
  background:#111827;
  color:#E5E7EB;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  font-size:14px;
  line-height:1.5;
  -webkit-font-smoothing:antialiased;
}}
.container{{
  max-width:1400px;
  margin:0 auto;
  padding:24px 32px;
}}
.header{{
  display:flex;
  align-items:center;
  justify-content:space-between;
  margin-bottom:28px;
  padding-bottom:20px;
  border-bottom:1px solid #1F2937;
}}
.header-left{{display:flex;align-items:center;gap:12px}}
.header h1{{
  color:#F9FAFB;
  font-size:20px;
  font-weight:600;
  letter-spacing:-0.02em;
}}
.status-dot{{
  width:8px;height:8px;
  border-radius:50%;
  background:#10B981;
  box-shadow:0 0 0 3px rgba(16,185,129,0.15);
  flex-shrink:0;
}}
.header-right{{
  color:#6B7280;
  font-size:13px;
}}
.grid{{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
  gap:16px;
  margin-bottom:28px;
}}
.card{{
  background:#1F2937;
  border:1px solid #374151;
  border-radius:12px;
  padding:20px;
  transition:border-color 0.15s ease;
}}
.card:hover{{border-color:#4B5563}}
.card .label{{
  color:#9CA3AF;
  font-size:12px;
  font-weight:500;
  letter-spacing:0.05em;
  text-transform:uppercase;
  margin-bottom:8px;
}}
.card .value{{
  font-family:'SF Mono',SFMono-Regular,ui-monospace,Menlo,Monaco,Consolas,monospace;
  font-size:28px;
  font-weight:700;
  letter-spacing:-0.02em;
}}
.card .sub{{
  color:#6B7280;
  font-size:12px;
  margin-top:6px;
}}
.panel{{
  background:#1F2937;
  border:1px solid #374151;
  border-radius:12px;
  margin-bottom:20px;
  overflow:hidden;
}}
.panel-header{{
  padding:16px 20px;
  border-bottom:1px solid #374151;
  display:flex;
  align-items:center;
  gap:8px;
}}
.panel-header h2{{
  color:#D1D5DB;
  font-size:13px;
  font-weight:600;
  letter-spacing:0.03em;
  text-transform:uppercase;
}}
.panel-header .count{{
  background:#374151;
  color:#9CA3AF;
  font-size:11px;
  font-weight:500;
  padding:2px 8px;
  border-radius:10px;
}}
table{{width:100%;border-collapse:collapse}}
th{{
  color:#6B7280;
  text-align:left;
  padding:10px 16px;
  font-size:11px;
  font-weight:500;
  letter-spacing:0.05em;
  text-transform:uppercase;
  background:#1a2332;
  border-bottom:1px solid #374151;
}}
td{{
  padding:12px 16px;
  border-bottom:1px solid rgba(55,65,81,0.5);
  vertical-align:middle;
  font-size:13px;
}}
tr:nth-child(even) td{{background:rgba(17,24,39,0.3)}}
tr:hover td{{background:rgba(55,65,81,0.3)}}
.empty{{
  text-align:center;
  color:#4B5563;
  padding:32px;
  font-style:italic;
}}
.num{{
  font-family:'SF Mono',SFMono-Regular,ui-monospace,Menlo,Monaco,Consolas,monospace;
  font-size:13px;
  font-variant-numeric:tabular-nums;
}}
.yes{{color:#3B82F6;font-weight:600}}
.no{{color:#EF4444;font-weight:600}}
.ev{{color:#3B82F6}}
.kl{{color:#F59E0B}}
.q{{color:#9CA3AF;font-size:12px;max-width:280px;line-height:1.4}}
.source{{
  color:#6B7280;
  font-size:12px;
  background:#374151;
  padding:2px 8px;
  border-radius:6px;
  display:inline-block;
}}
.badge{{
  padding:3px 10px;
  border-radius:6px;
  font-size:11px;
  font-weight:600;
  letter-spacing:0.02em;
}}
.badge.win{{background:rgba(59,130,246,0.12);color:#60A5FA}}
.badge.loss{{background:rgba(239,68,68,0.12);color:#F87171}}
a.link-arrow{{
  color:#6B7280;
  text-decoration:none;
  font-size:16px;
  transition:color 0.15s;
}}
a.link-arrow:hover{{color:#3B82F6}}
.footer{{
  margin-top:32px;
  padding-top:20px;
  border-top:1px solid #1F2937;
  text-align:center;
  color:#4B5563;
  font-size:12px;
}}
</style></head><body>
<div class="container">

<div class="header">
  <div class="header-left">
    <span class="status-dot"></span>
    <h1>Quant Engine v3</h1>
  </div>
  <div class="header-right">
    <a href="/analytics" style="color:#6B7280;text-decoration:none;margin-right:16px">Analytics</a>
    {datetime.now(timezone.utc).strftime('%d.%m.%Y %H:%M UTC')}
  </div>
</div>

<div class="grid">
  <div class="card">
    <div class="label">Bankroll</div>
    <div class="value num" style="color:{pc(roi)}">${stats['bankroll']:.2f}</div>
    <div class="sub">Start: ${start:.0f}</div>
  </div>
  <div class="card">
    <div class="label">ROI</div>
    <div class="value num" style="color:{pc(roi)}">{roi:+.2f}%</div>
    <div class="sub">P&amp;L: {stats['total_pnl']:+.2f}$</div>
  </div>
  <div class="card">
    <div class="label">Win Rate</div>
    <div class="value num" style="color:{'#3B82F6' if wr>=50 else '#EF4444'}">{wr}%</div>
    <div class="sub">{stats['wins']}W / {stats['losses']}L / {total} total</div>
  </div>
  <div class="card">
    <div class="label">Avg EV</div>
    <div class="value num" style="color:#3B82F6">+{stats['avg_ev']*100:.1f}%</div>
    <div class="sub">Avg Kelly: {stats['avg_kelly']*100:.1f}%</div>
  </div>
  <div class="card">
    <div class="label">Открытые</div>
    <div class="value num" style="color:#F59E0B">{len(open_)}</div>
    <div class="sub">Max: {os.getenv('MAX_OPEN','5')}</div>
  </div>
</div>

<div class="panel">
  <div class="panel-header">
    <h2>Открытые позиции</h2>
    <span class="count">{len(open_)}</span>
  </div>
  <table>
    <tr><th>Вопрос</th><th>Side</th><th>Вход</th><th>Сейчас</th><th>uPnL</th><th>EV</th><th>KL</th><th>Ставка</th><th></th></tr>
    {open_rows}
  </table>
</div>

<div class="panel">
  <div class="panel-header">
    <h2>Последние сигналы</h2>
    <span class="count">{len(signals)}</span>
  </div>
  <table>
    <tr><th>Вопрос</th><th>Side</th><th>Рынок</th><th>pTrue</th><th>EV</th><th>KL</th><th>Источник</th></tr>
    {sig_rows}
  </table>
</div>

<div class="panel">
  <div class="panel-header">
    <h2>История</h2>
    <span class="count">{len(closed)}</span>
  </div>
  <table>
    <tr><th>Вопрос</th><th>Side</th><th>Вход</th><th>Исход</th><th>P&amp;L</th><th>Итог</th><th>EV</th></tr>
    {closed_rows}
  </table>
</div>

<div class="footer">Quant Engine v3 &middot; {mode} Mode</div>

</div>
</body></html>"""
  except Exception as e:
    log.error(f"[DASHBOARD] Render error: {e}", exc_info=True)
    return HTMLResponse(f"<h1>Dashboard Error</h1><pre>{e}</pre>", status_code=500)

@app.get("/analytics", response_class=HTMLResponse)
async def analytics():
  try:
    data = await _db.get_analytics()
    stats = await _db.get_stats()
    start = float(os.getenv("BANKROLL","1000"))
    roi   = ((stats["bankroll"]-start)/start*100)
    total = stats["wins"]+stats["losses"]
    wr    = round(stats["wins"]/total*100,1) if total>0 else 0

    def pc(v): return "#3B82F6" if v>=0 else "#EF4444"
    def wr_color(w, t): return "#3B82F6" if t>0 and w/t>=0.5 else "#EF4444" if t>0 else "#6B7280"

    # Theme table
    theme_rows = "".join([f"""<tr>
        <td>{r['theme']}</td>
        <td class="num">{r['total']}</td>
        <td class="num" style="color:{wr_color(r['wins'],r['total'])}">{r['wins']}/{r['total']} ({round(r['wins']/r['total']*100) if r['total']>0 else 0}%)</td>
        <td class="num" style="color:{pc(float(r['avg_pnl']))}">{float(r['avg_pnl']):+.2f}$</td>
    </tr>""" for r in data["by_theme"]]) or '<tr><td colspan="4" class="empty">Нет данных</td></tr>'

    # Source table
    source_rows = "".join([f"""<tr>
        <td>{r['source'] or 'unknown'}</td>
        <td class="num">{r['total']}</td>
        <td class="num" style="color:{wr_color(r['wins'],r['total'])}">{r['wins']}/{r['total']} ({round(r['wins']/r['total']*100) if r['total']>0 else 0}%)</td>
        <td class="num" style="color:{pc(float(r['avg_pnl']))}">{float(r['avg_pnl']):+.2f}$</td>
    </tr>""" for r in data["by_source"]]) or '<tr><td colspan="4" class="empty">Нет данных</td></tr>'

    # Side table
    side_rows = "".join([f"""<tr>
        <td class="{'yes' if r['side']=='YES' else 'no'}">{r['side']}</td>
        <td class="num">{r['total']}</td>
        <td class="num" style="color:{wr_color(r['wins'],r['total'])}">{r['wins']}/{r['total']} ({round(r['wins']/r['total']*100) if r['total']>0 else 0}%)</td>
        <td class="num" style="color:{pc(float(r['avg_pnl']))}">{float(r['avg_pnl']):+.2f}$</td>
    </tr>""" for r in data["by_side"]]) or '<tr><td colspan="4" class="empty">Нет данных</td></tr>'

    # Close reason
    reason_rows = "".join([f"""<tr>
        <td>{r['reason']}</td>
        <td class="num">{r['total']}</td>
        <td class="num" style="color:{pc(float(r['avg_pnl']))}">{float(r['avg_pnl']):+.2f}$</td>
    </tr>""" for r in data["by_reason"]]) or '<tr><td colspan="3" class="empty">Нет данных</td></tr>'

    # Calibration
    cal_rows = "".join([f"""<tr>
        <td>{r['bucket']}</td>
        <td class="num">{r['total']}</td>
        <td class="num">{float(r['avg_predicted'])*100:.1f}%</td>
        <td class="num" style="color:{pc(float(r['actual_wr'])-float(r['avg_predicted']))}">{float(r['actual_wr'])*100:.1f}%</td>
        <td class="num" style="color:{pc(float(r['actual_wr'])-float(r['avg_predicted']))}">{(float(r['actual_wr'])-float(r['avg_predicted']))*100:+.1f}%</td>
    </tr>""" for r in data["calibration"]]) or '<tr><td colspan="5" class="empty">Нет данных</td></tr>'

    # Daily PnL
    daily_rows = "".join([f"""<tr>
        <td>{r['day']}</td>
        <td class="num">{r['trades']}</td>
        <td class="num">{r['wins']}/{r['trades']} ({round(r['wins']/r['trades']*100) if r['trades']>0 else 0}%)</td>
        <td class="num" style="color:{pc(float(r['pnl']))}">{float(r['pnl']):+.2f}$</td>
    </tr>""" for r in data["daily_pnl"]]) or '<tr><td colspan="4" class="empty">Нет данных</td></tr>'

    # EV accuracy
    ev_pred = data["ev_predicted"]*100
    ev_act  = data["ev_actual"]*100

    return f"""<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Analytics &mdash; Quant Engine v3</title>
<!-- auto-refresh disabled -->
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#111827;color:#E5E7EB;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}}
.container{{max-width:1400px;margin:0 auto;padding:24px 32px}}
.header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;padding-bottom:20px;border-bottom:1px solid #1F2937}}
.header h1{{color:#F9FAFB;font-size:20px;font-weight:600;letter-spacing:-0.02em}}
.header a{{color:#6B7280;text-decoration:none;font-size:13px}}
.header a:hover{{color:#3B82F6}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:28px}}
.card{{background:#1F2937;border:1px solid #374151;border-radius:12px;padding:20px}}
.card .label{{color:#9CA3AF;font-size:12px;font-weight:500;letter-spacing:0.05em;text-transform:uppercase;margin-bottom:8px}}
.card .value{{font-family:'SF Mono',SFMono-Regular,ui-monospace,Menlo,Monaco,Consolas,monospace;font-size:28px;font-weight:700;letter-spacing:-0.02em}}
.card .sub{{color:#6B7280;font-size:12px;margin-top:6px}}
.row{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}}
@media(max-width:900px){{.row{{grid-template-columns:1fr}}}}
.panel{{background:#1F2937;border:1px solid #374151;border-radius:12px;overflow:hidden}}
.panel-header{{padding:16px 20px;border-bottom:1px solid #374151}}
.panel-header h2{{color:#D1D5DB;font-size:13px;font-weight:600;letter-spacing:0.03em;text-transform:uppercase}}
table{{width:100%;border-collapse:collapse}}
th{{color:#6B7280;text-align:left;padding:10px 16px;font-size:11px;font-weight:500;letter-spacing:0.05em;text-transform:uppercase;background:#1a2332;border-bottom:1px solid #374151}}
td{{padding:12px 16px;border-bottom:1px solid rgba(55,65,81,0.5);font-size:13px}}
tr:nth-child(even) td{{background:rgba(17,24,39,0.3)}}
tr:hover td{{background:rgba(55,65,81,0.3)}}
.num{{font-family:'SF Mono',SFMono-Regular,ui-monospace,Menlo,Monaco,Consolas,monospace;font-size:13px;font-variant-numeric:tabular-nums}}
.yes{{color:#3B82F6;font-weight:600}}.no{{color:#EF4444;font-weight:600}}
.empty{{text-align:center;color:#4B5563;padding:32px;font-style:italic}}
.footer{{margin-top:32px;padding-top:20px;border-top:1px solid #1F2937;text-align:center;color:#4B5563;font-size:12px}}
</style></head><body>
<div class="container">

<div class="header">
  <h1>Analytics</h1>
  <a href="/">&larr; Dashboard</a>
</div>

<div class="grid">
  <div class="card">
    <div class="label">Win Rate</div>
    <div class="value num" style="color:{'#3B82F6' if wr>=50 else '#EF4444'}">{wr}%</div>
    <div class="sub">{stats['wins']}W / {stats['losses']}L</div>
  </div>
  <div class="card">
    <div class="label">EV Predicted</div>
    <div class="value num" style="color:#3B82F6">+{ev_pred:.1f}%</div>
    <div class="sub">Avg predicted EV at entry</div>
  </div>
  <div class="card">
    <div class="label">EV Actual</div>
    <div class="value num" style="color:{pc(ev_act)}">{ev_act:+.1f}%</div>
    <div class="sub">Avg real return per trade</div>
  </div>
  <div class="card">
    <div class="label">Avg Lifetime</div>
    <div class="value num" style="color:#F59E0B">{data['avg_lifetime_hours']:.1f}h</div>
    <div class="sub">Avg position duration</div>
  </div>
</div>

<div class="row">
  <div class="panel">
    <div class="panel-header"><h2>Win Rate by Theme</h2></div>
    <table><tr><th>Theme</th><th>Trades</th><th>W/L (WR)</th><th>Avg PnL</th></tr>{theme_rows}</table>
  </div>
  <div class="panel">
    <div class="panel-header"><h2>Win Rate by Source</h2></div>
    <table><tr><th>Source</th><th>Trades</th><th>W/L (WR)</th><th>Avg PnL</th></tr>{source_rows}</table>
  </div>
</div>

<div class="row">
  <div class="panel">
    <div class="panel-header"><h2>Win Rate by Side</h2></div>
    <table><tr><th>Side</th><th>Trades</th><th>W/L (WR)</th><th>Avg PnL</th></tr>{side_rows}</table>
  </div>
  <div class="panel">
    <div class="panel-header"><h2>Close Reason</h2></div>
    <table><tr><th>Reason</th><th>Count</th><th>Avg PnL</th></tr>{reason_rows}</table>
  </div>
</div>

<div class="panel" style="margin-bottom:20px">
  <div class="panel-header"><h2>Calibration (Predicted vs Actual)</h2></div>
  <table><tr><th>Bucket</th><th>Trades</th><th>Avg Predicted</th><th>Actual WR</th><th>Bias</th></tr>{cal_rows}</table>
</div>

<div class="panel">
  <div class="panel-header"><h2>Daily P&amp;L</h2></div>
  <table><tr><th>Date</th><th>Trades</th><th>W/L (WR)</th><th>P&amp;L</th></tr>{daily_rows}</table>
</div>

<div class="footer"><a href="/" style="color:#6B7280;text-decoration:none">Quant Engine v3</a> &middot; Analytics</div>

</div></body></html>"""
  except Exception as e:
    log.error(f"[DASHBOARD] Analytics error: {e}", exc_info=True)
    return HTMLResponse(f"<h1>Analytics Error</h1><pre>{e}</pre>", status_code=500)

@app.get("/api")
async def api_stats():
    try:
        stats  = await _db.get_stats()
        open_  = await _db.get_open_positions()
        closed = await _db.get_closed_positions(limit=5)
        return JSONResponse({"stats": stats, "open": len(open_), "recent": len(closed)})
    except Exception as e:
        log.warning(f"[DASHBOARD] API error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

async def start_dashboard(db, config: dict):
    global _db, _config
    _db     = db
    _config = config
    port    = int(os.getenv("PORT","3000"))
    cfg     = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server  = uvicorn.Server(cfg)
    log.info(f"[DASHBOARD] http://localhost:{port}")
    await server.serve()
