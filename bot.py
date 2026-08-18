"""Smart Signals watcher -> Telegram.

Polls Binance for every symbol in watchlist.txt, evaluates the Smart Signals
entry conditions on the most recently CLOSED bar, and sends a Telegram message
when a new signal appears. Already-alerted bars are remembered in state.json so
running this every 15 minutes never double-sends.

The signal logic is the same one validated against the TradingView dashboard
(54 vs 55 trades, PF 1.67 vs 1.76 on PAXGUSDT.P 4h).

Usage:
    python bot.py            # one pass, alert on new signals
    python bot.py --test     # send a test message, then one pass
    python bot.py --dry      # evaluate and print, send nothing
    python bot.py --status   # report current state of every symbol, no alerts
"""
import json, os, sys, time, urllib.parse, urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pine as P

HERE = os.path.dirname(os.path.abspath(__file__))
_cfg_path = os.path.join(HERE, "config.json")
if not os.path.exists(_cfg_path):          # CI: real config is gitignored
    _cfg_path = os.path.join(HERE, "config.example.json")
CFG = json.load(open(_cfg_path))
# Environment overrides so secrets never live in the repo (GitHub Actions).
if os.environ.get("TELEGRAM_TOKEN"):
    CFG["telegram_token"] = os.environ["TELEGRAM_TOKEN"]
if os.environ.get("TELEGRAM_CHAT_ID"):
    CFG["telegram_chat_id"] = os.environ["TELEGRAM_CHAT_ID"]
if os.environ.get("MAX_BAR_AGE_MIN"):
    CFG["max_bar_age_min"] = float(os.environ["MAX_BAR_AGE_MIN"])
STATE_PATH = os.path.join(HERE, "state.json")
LOG_PATH = os.path.join(HERE, "bot.log")

BASE = ("https://fapi.binance.com/fapi/v1/klines" if CFG.get("market") == "futures"
        else "https://api.binance.com/api/v3/klines")


def log(msg):
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z  {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------- watchlist
def load_watchlist():
    out = []
    for raw in open(os.path.join(HERE, "watchlist.txt"), encoding="utf-8"):
        s = raw.split("#")[0].strip()
        if not s:
            continue
        # accept TradingView formats: BINANCE:WLDUSDT.P / BYBIT:WLDUSDT
        if ":" in s:
            s = s.split(":", 1)[1]
        s = s.replace(".P", "").replace("PERP", "").strip().upper()
        if s:
            out.append(s)
    return list(dict.fromkeys(out))


# ----------------------------------------------------------------- data
_BYBIT_ITV = {"15m": "15", "30m": "30", "1h": "60", "2h": "120", "4h": "240",
              "6h": "360", "12h": "720", "1d": "D"}
_OKX_ITV = {"15m": "15m", "30m": "30m", "1h": "1H", "2h": "2H", "4h": "4H",
            "6h": "6H", "12h": "12H", "1d": "1D"}

SOURCE_USED = {}      # symbol -> which venue actually answered


def _get(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": "signalbot/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _binance(symbol, interval, limit):
    k = _get(f"{BASE}?symbol={symbol}&interval={interval}&limit={limit}")
    df = pd.DataFrame(k, columns=["t", "o", "h", "l", "c", "v", "ct", "q", "n", "tb", "tq", "ig"])
    return df[["t", "o", "h", "l", "c", "v"]]


def _bybit(symbol, interval, limit):
    itv = _BYBIT_ITV[interval]
    d = _get(f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}"
             f"&interval={itv}&limit={min(limit, 1000)}")
    rows = d["result"]["list"][::-1]          # Bybit returns newest first
    return pd.DataFrame([r[:6] for r in rows], columns=["t", "o", "h", "l", "c", "v"])


def _okx(symbol, interval, limit):
    inst = symbol.replace("USDT", "") + "-USDT-SWAP"
    d = _get(f"https://www.okx.com/api/v5/market/candles?instId={inst}"
             f"&bar={_OKX_ITV[interval]}&limit={min(limit, 300)}")
    rows = d["data"][::-1]                    # OKX returns newest first
    return pd.DataFrame([r[:6] for r in rows], columns=["t", "o", "h", "l", "c", "v"])


def fetch(symbol, interval, limit=400):
    """Binance first (it is what the strategy was tested on), then Bybit, then OKX.

    GitHub-hosted runners sit on US IPs and Binance answers them with HTTP 451,
    so a fallback is required for this to work anywhere but a local machine.
    Bybit lists the same USDT perpetuals under the same tickers, so its bars are
    the closest substitute.
    """
    last_err = None
    for name, fn in (("binance", _binance), ("bybit", _bybit), ("okx", _okx)):
        for attempt in range(3):
            try:
                df = fn(symbol, interval, limit)
                if df is None or len(df) < 60:
                    raise ValueError(f"only {0 if df is None else len(df)} bars")
                df = df.astype({"t": "int64", "o": float, "h": float,
                                "l": float, "c": float, "v": float})
                df = df.drop_duplicates("t").sort_values("t").reset_index(drop=True)
                SOURCE_USED[symbol] = name
                return df
            except Exception as e:
                last_err = f"{name}: {type(e).__name__} {e}"
                code = getattr(e, "code", None)
                if code in (451, 403, 401):
                    break               # geo-blocked, no point retrying this venue
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last_err)


# ----------------------------------------------------------------- signal
def evaluate(df):
    """Replay the script's full position state machine over history.

    The raw entry condition being true is NOT a signal: the script only prints
    an entry when it is FLAT (`canSignal = ... and posDir == 0`). A position
    stays open until its stop or target is hit, and every raw condition that
    fires while a trade is running is ignored. Evaluating the condition alone
    produces false alerts, so the state has to be carried bar by bar.
    """
    c = df.c.to_numpy(float); h = df.h.to_numpy(float)
    l = df.l.to_numpy(float); v = df.v.to_numpy(float)
    e50 = P.ema(c, 50); e200 = P.ema(c, 200)
    ml, ms, _ = P.macd(c, 12, 26, 9)
    rsi = P.rsi(c, 14)
    _, _, adx = P.dmi(h, l, c, 14, 14)
    atr = P.atr(h, l, c, int(CFG["atr_len"]))
    vema = P.ema(v, 20)
    fifty = np.full(len(c), 50.0)
    cu = P.crossover(ml, ms) | P.crossover(rsi, fifty)
    cd = P.crossunder(ml, ms) | P.crossunder(rsi, fifty)

    trend = adx >= float(CFG["adx_min"])
    if CFG.get("trend_filter", True):
        up = (c > e200) & (e50 > e200)
        dn = (c < e200) & (e50 < e200)
    else:
        up = np.ones(len(c), bool); dn = np.ones(len(c), bool)
    volok = v > vema
    buy = trend & up & (rsi > 50) & (rsi < 75) & (ml > ms) & volok & cu
    sell = trend & dn & (rsi < 50) & (rsi > 25) & (ml < ms) & volok & cd

    sl_m = float(CFG["sl_atr"]); tp_m = float(CFG["tp_atr"])
    cooldown = int(CFG.get("cooldown_bars", 1))
    last = len(c) - 1

    pos = 0; entry = sl = tp = 0.0; pbar = -1; lastsig = -10**6
    entry_bar = None
    event = None            # what happened on the LAST closed bar
    for i in range(len(c)):
        if np.isnan(atr[i]) or np.isnan(e200[i]):
            continue
        # exit check first, exactly as the script orders it
        if pos != 0 and i > pbar:
            hit = 0; lvl = 0.0
            if pos == 1:
                if l[i] <= sl: hit, lvl = -1, sl
                elif h[i] >= tp: hit, lvl = 1, tp
            else:
                if h[i] >= sl: hit, lvl = -1, sl
                elif l[i] <= tp: hit, lvl = 1, tp
            if hit != 0:
                if i == last:
                    event = dict(kind="EXIT", side="LONG" if pos == 1 else "SHORT",
                                 price=lvl, hit="TARGET" if hit == 1 else "STOP",
                                 entry=entry,
                                 pnl=((lvl-entry)/entry if pos == 1 else (entry-lvl)/entry)*100)
                pos = 0; entry_bar = None
        # entry only when flat
        if pos == 0 and (i - lastsig) >= cooldown and (buy[i] or sell[i]):
            pos = 1 if buy[i] else -1
            entry = float(c[i]); pbar = i; lastsig = i; entry_bar = i
            sl = entry - sl_m*atr[i] if pos == 1 else entry + sl_m*atr[i]
            tp = entry + tp_m*atr[i] if pos == 1 else entry - tp_m*atr[i]
            if i == last:
                event = dict(kind="ENTRY", side="LONG" if pos == 1 else "SHORT",
                             price=entry, sl=sl, tp=tp)

    open_pos = None
    if pos != 0:
        open_pos = dict(side="LONG" if pos == 1 else "SHORT", entry=entry, sl=sl, tp=tp,
                        bars_held=last - entry_bar if entry_bar is not None else None,
                        unreal=((c[last]-entry)/entry if pos == 1 else (entry-c[last])/entry)*100)

    i = last
    tf_ms = {"1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
             "12h": 43200, "1d": 86400, "15m": 900, "30m": 1800}.get(CFG["timeframe"], 14400)*1000
    bar_close_ms = int(df.t.iloc[i]) + tf_ms
    age_min = (time.time()*1000 - bar_close_ms)/60000.0
    return dict(
        event=event, open_pos=open_pos, bar_age_min=age_min,
        bar_time=int(df.t.iloc[i]), close=float(c[i]),
        atr=float(atr[i]) if not np.isnan(atr[i]) else 0.0,
        raw_cond="LONG" if buy[i] else ("SHORT" if sell[i] else None),
        rsi=float(rsi[i]), adx=float(adx[i]),
        macd_above=bool(ml[i] > ms[i]),
        above_ema200=bool(c[i] > e200[i]),
        vol_ok=bool(volok[i]),
    )


# ----------------------------------------------------------------- telegram
def send(text):
    url = f"https://api.telegram.org/bot{CFG['telegram_token']}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": CFG["telegram_chat_id"],
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=25) as r:
                res = json.load(r)
            if res.get("ok"):
                return True
            log(f"telegram rejected: {res}")
            return False
        except Exception as e:
            if attempt == 2:
                log(f"telegram send failed: {e}")
                return False
            time.sleep(2 * (attempt + 1))


def fmt_price(x):
    if x >= 1000: return f"{x:,.2f}"
    if x >= 1:    return f"{x:.4f}".rstrip("0").rstrip(".")
    return f"{x:.8f}".rstrip("0").rstrip(".")


def build_entry_message(sym, s, ev):
    arrow = "🟢 LONG" if ev["side"] == "LONG" else "🔴 SHORT"
    when = datetime.fromtimestamp(s["bar_time"] / 1000, timezone.utc)
    px = ev["price"]
    risk = abs(px - ev["sl"]) / px * 100
    reward = abs(ev["tp"] - px) / px * 100
    return (
        f"<b>{arrow}  {sym}</b>  ·  {CFG['timeframe']}\n"
        f"<code>─────────────────────</code>\n"
        f"Entry  <b>{fmt_price(px)}</b>\n"
        f"Stop   {fmt_price(ev['sl'])}   ({risk:.2f}%)\n"
        f"Target {fmt_price(ev['tp'])}   ({reward:.2f}%)\n"
        f"<code>─────────────────────</code>\n"
        f"RSI {s['rsi']:.1f} · ADX {s['adx']:.1f} · ATR {fmt_price(s['atr'])}\n"
        f"bar closed {when:%Y-%m-%d %H:%M} UTC"
    )


def build_exit_message(sym, s, ev):
    icon = "🎯" if ev["hit"] == "TARGET" else "🛑"
    when = datetime.fromtimestamp(s["bar_time"] / 1000, timezone.utc)
    sign = "+" if ev["pnl"] >= 0 else ""
    return (
        f"<b>{icon} {ev['hit']} HIT  {sym}</b>  ·  {CFG['timeframe']}\n"
        f"<code>─────────────────────</code>\n"
        f"{ev['side']} closed at <b>{fmt_price(ev['price'])}</b>\n"
        f"from {fmt_price(ev['entry'])}   <b>{sign}{ev['pnl']:.2f}%</b>\n"
        f"bar closed {when:%Y-%m-%d %H:%M} UTC"
    )


# ----------------------------------------------------------------- state
def load_state():
    if os.path.exists(STATE_PATH):
        try:
            return json.load(open(STATE_PATH))
        except Exception:
            pass
    return {}


def save_state(st):
    json.dump(st, open(STATE_PATH, "w"), indent=1)


# ----------------------------------------------------------------- main
def main():
    args = set(sys.argv[1:])
    dry = "--dry" in args
    status_only = "--status" in args

    if "--test" in args:
        ok = send("✅ <b>Smart Signals bot connected</b>\nWatching "
                  f"{len(load_watchlist())} symbols on {CFG['timeframe']}.")
        log(f"test message sent: {ok}")

    syms = load_watchlist()
    state = load_state()
    fired, errors, rows = [], [], []

    for sym in syms:
        try:
            df = fetch(sym, CFG["timeframe"])
            if len(df) < 220:
                errors.append(f"{sym}: only {len(df)} bars"); continue
            if not CFG.get("alert_on_open_bar", False):
                df = df.iloc[:-1]          # drop the still-forming bar
            s = evaluate(df.reset_index(drop=True))
            rows.append((sym, s))
            if status_only:
                continue
            ev = s["event"]
            # A bar that closed hours ago must not be re-announced. This makes the
            # bot safe on stateless runners where the dedupe cache may be cold.
            max_age = float(CFG.get("max_bar_age_min", 0) or 0)
            if ev and max_age > 0 and s["bar_age_min"] > max_age:
                log(f"  {sym}: skipping stale {ev['kind']} "
                    f"(bar closed {s['bar_age_min']:.0f} min ago > {max_age:.0f})")
                ev = None
            if ev and (ev["kind"] == "ENTRY" or CFG.get("alert_on_exit", True)):
                key = f"{sym}:{s['bar_time']}:{ev['kind']}"
                if state.get(key):
                    continue               # already alerted for this bar
                msg = (build_entry_message(sym, s, ev) if ev["kind"] == "ENTRY"
                       else build_exit_message(sym, s, ev))
                if dry:
                    log(f"[dry] would alert {sym} {ev['kind']} {ev['side']} @ {ev['price']}")
                else:
                    if send(msg):
                        state[key] = True
                        log(f"ALERT {sym} {ev['kind']} {ev['side']} @ {ev['price']}")
                fired.append((sym, ev["kind"]))
        except Exception as e:
            errors.append(f"{sym}: {type(e).__name__} {e}")

    if status_only:
        print(f"\n{'symbol':<14}{'this bar':<13}{'open position':<36}{'close':>12}{'RSI':>7}{'ADX':>7}")
        print("-" * 92)
        for sym, s in rows:
            ev = s["event"]
            what = f"{ev['kind']} {ev['side']}" if ev else "-"
            op = s["open_pos"]
            held = (f"{op['side']} from {fmt_price(op['entry'])} "
                    f"({op['unreal']:+.2f}%, {op['bars_held']}b)") if op else "flat"
            print(f"{sym:<14}{what:<13}{held:<36}{fmt_price(s['close']):>12}"
                  f"{s['rsi']:>7.1f}{s['adx']:>7.1f}")
    else:
        if not dry:
            save_state(state)
        elif not os.path.exists(STATE_PATH):
            save_state(state)      # keep the CI cache step happy on a dry run
        srcs = {}
        for v in SOURCE_USED.values():
            srcs[v] = srcs.get(v, 0) + 1
        src_txt = ", ".join(f"{k}:{v}" for k, v in sorted(srcs.items())) or "none"
        log(f"scanned {len(rows)}/{len(syms)} symbols via [{src_txt}], "
            f"{len(fired)} new signal(s)" + (f", {len(errors)} error(s)" if errors else ""))
    for e in errors:
        log("  " + e)


if __name__ == "__main__":
    main()
