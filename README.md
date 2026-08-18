# Smart Signals → Telegram

Scans a Binance watchlist on the 4h timeframe for Smart Signals entries and exits
and sends them to Telegram. The signal logic is a port of the Pine script,
validated against its TradingView dashboard (54 vs 55 trades, PF 1.67 vs 1.76 on
PAXGUSDT.P 4h).

## What it alerts on

An entry is only reported when the strategy would actually be **flat** — the raw
condition firing while a trade is already open is not a signal, which is exactly
what the chart shows. Exits are reported when the stop or target is hit.

## Settings (`config.json`)

| key | default | meaning |
|---|---|---|
| `timeframe` | `4h` | only 4h was profitable out-of-sample; 15m was not |
| `adx_min` | `8.5` | ADX threshold |
| `trend_filter` | `true` | require the EMA 50/200 stack |
| `atr_len` | `7` | ATR period for stop/target |
| `sl_atr` / `tp_atr` | `2.0` / `4.0` | stop and target in ATR multiples |
| `max_bar_age_min` | `75` | never announce a bar that closed longer ago than this |

Expected volume on the shipped 10-coin watchlist: **~0.8 entries/day**, about one
every 1.2 days, or ~1.6 messages/day counting exits.

## Run locally

```bash
pip install -r requirements.txt
python bot.py --status   # what every coin is doing now
python bot.py --dry      # evaluate, send nothing
python bot.py            # one pass, send new signals
```

## Run on GitHub Actions

1. Push this folder as the repository root.
2. Settings → Secrets and variables → Actions → add `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID`.
3. The workflow runs every 15 minutes, and can be started by hand from the Actions tab.

`config.json` is gitignored; CI falls back to `config.example.json` plus the two secrets.

Note that GitHub delays scheduled runs when busy and disables schedules after 60
days without repository activity. The `max_bar_age_min` guard means a late run
either alerts once or not at all — it never replays a stale signal.

## Watchlist

`watchlist.txt`, one symbol per line. TradingView names work too
(`BINANCE:WLDUSDT.P` → `WLDUSDT`). Trailing `# comments` are ignored.
