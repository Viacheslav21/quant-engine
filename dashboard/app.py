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
    stats   = await _db.get_stats()
    open_   = await _db.get_open_positions()
    closed  = await _db.get_closed_positions(limit=20)
    signals = await _db.get_recent_signals(limit=10)

    start = float(os.getenv("BANKROLL","1000"))
    roi   = ((stats["bankroll"]-start)/start*100)
    total = stats["wins"]+stats["losses"]
    wr    = round(stats["wins"]/total*100,1) if total>0 else 0

    def pc(v): return "#00ff88" if v>=0 else "#ff4444"

    open_rows = "".join([f"""<tr>
        <td class="q">{p['question'][:70]}...</td>
        <td class="{'yes' if p['side']=='YES' else 'no'}">{p['side']}</td>
        <td>{p['side_price']*100:.1f}¢</td>
        <td>{(p.get('current_price') or p['side_price'])*100:.1f}¢</td>
        <td style="color:{pc(p.get('unrealized_pnl',0))}">{p.get('unrealized_pnl',0):+.2f}$</td>
        <td class="ev">+{p['ev']*100:.1f}%</td>
        <td class="kl">{p['kl']:.3f}</td>
        <td>${p['stake_amt']:.2f}</td>
        <td><a href="{p['url']}" target="_blank">→</a></td>
    </tr>""" for p in open_]) or '<tr><td colspan="9" class="empty">Нет открытых позиций</td></tr>'

    closed_rows = "".join([f"""<tr>
        <td class="q">{t['question'][:60]}...</td>
        <td class="{'yes' if t['side']=='YES' else 'no'}">{t['side']}</td>
        <td>{t['side_price']*100:.1f}¢</td>
        <td>{t.get('outcome','?')}</td>
        <td style="color:{pc(t['pnl'])}">{t['pnl']:+.2f}$</td>
        <td><span class="badge {'win' if t['result']=='WIN' else 'loss'}">{t['result']}</span></td>
        <td class="ev">+{t['ev']*100:.1f}%</td>
    </tr>""" for t in reversed(closed)]) or '<tr><td colspan="7" class="empty">Нет сделок</td></tr>'

    sig_rows = "".join([f"""<tr>
        <td class="q">{s['question'][:60]}...</td>
        <td class="{'yes' if s['side']=='YES' else 'no'}">{s['side']}</td>
        <td>{s['p_market']*100:.1f}¢</td>
        <td>{s['p_final']*100:.1f}¢</td>
        <td class="ev">+{s['ev']*100:.1f}%</td>
        <td class="kl">{s['kl']:.3f}</td>
        <td>{s.get('source','math')}</td>
    </tr>""" for s in signals]) or '<tr><td colspan="7" class="empty">Нет сигналов</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>⚡ Quant Engine v3</title>
<meta http-equiv="refresh" content="15">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#080810;color:#c0c0d0;font-family:'Courier New',monospace;padding:20px;font-size:13px}}
h1{{color:#00ff88;font-size:20px;letter-spacing:3px;margin-bottom:4px}}
.sub{{color:#333;font-size:11px;margin-bottom:20px}}.sub span{{color:#00ff88}}
.formula{{background:#0a0a14;border:1px solid #1a1a2e;border-radius:4px;padding:8px 12px;color:#444;font-size:10px;margin-bottom:16px;letter-spacing:1px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:20px}}
.card{{background:#0d0d1a;border:1px solid #1a1a2e;border-radius:8px;padding:14px}}
.card .l{{color:#444;font-size:10px;letter-spacing:2px;margin-bottom:6px}}
.card .v{{font-size:22px;font-weight:bold}}
.card .s{{color:#333;font-size:10px;margin-top:4px}}
.panel{{background:#0d0d1a;border:1px solid #1a1a2e;border-radius:8px;padding:16px;margin-bottom:16px}}
.panel h2{{color:#444;font-size:11px;letter-spacing:3px;margin-bottom:14px;border-bottom:1px solid #1a1a2e;padding-bottom:8px}}
table{{width:100%;border-collapse:collapse}}
th{{color:#333;text-align:left;padding:6px 8px;font-size:10px;letter-spacing:1px;border-bottom:1px solid #1a1a2e}}
td{{padding:8px;border-bottom:1px solid #0a0a14;vertical-align:middle}}
tr:hover td{{background:#0a0a14}}
.empty{{text-align:center;color:#222;padding:24px}}
.yes{{color:#00ff88;font-weight:bold}}.no{{color:#ff4444;font-weight:bold}}
.ev{{color:#4a9eff}}.kl{{color:#f0a500}}
.q{{color:#888;font-size:11px;max-width:250px}}
.badge{{padding:2px 8px;border-radius:3px;font-size:10px}}
.badge.win{{background:#0a2a14;color:#00ff88}}.badge.loss{{background:#2a0a0a;color:#ff4444}}
a{{color:#4a9eff;text-decoration:none}}
.live{{color:#00ff88;font-size:10px;animation:blink 1s infinite}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:0.3}}}}
</style></head><body>
<h1>⚡ QUANT ENGINE v3</h1>
<p class="sub">Math-first · News Monitor · Self-learning · <span class="live">● LIVE</span> · <span>{datetime.now(timezone.utc).strftime('%d.%m.%Y %H:%M UTC')}</span></p>
<div class="formula">EV=p×odds−(1−p) | Kelly=(b×p−q)/b×0.25 | KL=Σ P×log(P/Q) | Prospect: w(p)=p^γ/(p^γ+(1−p)^γ)^(1/γ)</div>
<div class="grid">
  <div class="card"><div class="l">BANKROLL</div><div class="v" style="color:{pc(roi)}">${stats['bankroll']:.2f}</div><div class="s">Start: ${start:.0f}</div></div>
  <div class="card"><div class="l">ROI</div><div class="v" style="color:{pc(roi)}">{roi:+.2f}%</div><div class="s">P&L: {stats['total_pnl']:+.2f}$</div></div>
  <div class="card"><div class="l">WIN RATE</div><div class="v" style="color:{'#00ff88' if wr>=50 else '#ff4444'}">{wr}%</div><div class="s">✅{stats['wins']} / ❌{stats['losses']} / 📋{total}</div></div>
  <div class="card"><div class="l">AVG EV</div><div class="v" style="color:#4a9eff">+{stats['avg_ev']*100:.1f}%</div><div class="s">Avg Kelly: {stats['avg_kelly']*100:.1f}%</div></div>
  <div class="card"><div class="l">OPEN</div><div class="v" style="color:#f0a500">{len(open_)}</div><div class="s">Max: {os.getenv('MAX_OPEN','5')}</div></div>
</div>
<div class="panel"><h2>📌 ОТКРЫТЫЕ ПОЗИЦИИ</h2><table>
  <tr><th>Вопрос</th><th>Side</th><th>Вход</th><th>Сейчас</th><th>uPnL</th><th>EV</th><th>KL</th><th>Ставка</th><th>→</th></tr>
  {open_rows}
</table></div>
<div class="panel"><h2>📡 ПОСЛЕДНИЕ СИГНАЛЫ</h2><table>
  <tr><th>Вопрос</th><th>Side</th><th>Рынок</th><th>pTrue</th><th>EV</th><th>KL</th><th>Источник</th></tr>
  {sig_rows}
</table></div>
<div class="panel"><h2>📜 ИСТОРИЯ</h2><table>
  <tr><th>Вопрос</th><th>Side</th><th>Вход</th><th>Исход</th><th>P&L</th><th>Итог</th><th>EV</th></tr>
  {closed_rows}
</table></div>
</body></html>"""

@app.get("/api")
async def api_stats():
    stats  = await _db.get_stats()
    open_  = await _db.get_open_positions()
    closed = await _db.get_closed_positions(limit=5)
    return JSONResponse({"stats": stats, "open": len(open_), "recent": len(closed)})

async def start_dashboard(db, config: dict):
    global _db, _config
    _db     = db
    _config = config
    port    = int(os.getenv("PORT","3000"))
    cfg     = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server  = uvicorn.Server(cfg)
    log.info(f"[DASHBOARD] http://localhost:{port}")
    await server.serve()
