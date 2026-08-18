"""Pine Script v5 built-in functions, ported to numpy/numba with Pine's exact
seeding and na-propagation semantics. Shared by every strategy port."""
import numpy as np
from numba import njit

NAN = np.nan

# ---------------- moving averages ----------------
def sma(x, n):
    x = np.asarray(x, float)
    if n <= 1: return x.copy()
    out = np.full(len(x), NAN)
    c = np.cumsum(np.nan_to_num(x))
    valid = ~np.isnan(x)
    cv = np.cumsum(valid)
    for i in range(n-1, len(x)):
        if cv[i] - (cv[i-n] if i >= n else 0) == n:
            out[i] = (c[i] - (c[i-n] if i >= n else 0)) / n
    return out

@njit(cache=True)
def _first_valid(x):
    for i in range(len(x)):
        if not np.isnan(x[i]): return i
    return -1

@njit(cache=True)
def ema(x, n):
    """Pine ta.ema: seeded with the SMA of the first n non-na values.
    Leading na values are skipped, as Pine does, rather than poisoning the series."""
    N = len(x); out = np.full(N, NAN)
    if N < n or n < 1: return out
    f = _first_valid(x)
    if f < 0 or f+n > N: return out
    s = 0.0
    for i in range(f, f+n): s += x[i]
    out[f+n-1] = s/n
    a = 2.0/(n+1.0)
    for i in range(f+n, N):
        v = x[i]
        out[i] = out[i-1] if np.isnan(v) else a*v + (1-a)*out[i-1]
    return out

@njit(cache=True)
def rma(x, n):
    """Pine ta.rma / Wilder smoothing, seeded with the SMA of the first n non-na values."""
    N = len(x); out = np.full(N, NAN)
    if N < n or n < 1: return out
    f = _first_valid(x)
    if f < 0 or f+n > N: return out
    s = 0.0
    for i in range(f, f+n): s += x[i]
    out[f+n-1] = s/n
    for i in range(f+n, N):
        v = x[i]
        out[i] = out[i-1] if np.isnan(v) else (out[i-1]*(n-1) + v)/n
    return out

@njit(cache=True)
def wma(x, n):
    N = len(x); out = np.full(N, NAN)
    den = n*(n+1)/2.0
    for i in range(n-1, N):
        s = 0.0; bad = False
        for j in range(n):
            v = x[i-j]
            if np.isnan(v): bad = True; break
            s += v*(n-j)
        if not bad: out[i] = s/den
    return out

def hma(x, n):
    """ta.hma = wma(2*wma(x, n/2) - wma(x, n), sqrt(n))"""
    x = np.asarray(x, float)
    h = int(n//2); s = int(np.sqrt(n))
    raw = 2*wma(x, max(h,1)) - wma(x, n)
    return wma(np.nan_to_num(raw, nan=NAN), max(s,1))

@njit(cache=True)
def linreg(x, n, offset):
    """ta.linreg: value of the least-squares line over the last n bars, at `offset`."""
    N = len(x); out = np.full(N, NAN)
    sx = 0.0; sxx = 0.0
    for j in range(n):
        sx += j; sxx += j*j
    for i in range(n-1, N):
        sy = 0.0; sxy = 0.0; bad = False
        for j in range(n):
            v = x[i-j]
            if np.isnan(v): bad = True; break
            sy += v; sxy += j*v
        if bad: continue
        den = n*sxx - sx*sx
        if den == 0: continue
        b = (n*sxy - sx*sy)/den
        a = (sy - b*sx)/n
        out[i] = a + b*offset
    return out

# ---------------- dispersion ----------------
@njit(cache=True)
def stdev(x, n):
    """ta.stdev uses the population (biased) deviation."""
    N = len(x); out = np.full(N, NAN)
    for i in range(n-1, N):
        s = 0.0; bad = False
        for j in range(n):
            v = x[i-j]
            if np.isnan(v): bad = True; break
            s += v
        if bad: continue
        m = s/n; q = 0.0
        for j in range(n):
            d = x[i-j]-m; q += d*d
        out[i] = np.sqrt(q/n)
    return out

@njit(cache=True)
def dev(x, n):
    """ta.dev: mean absolute deviation from the SMA."""
    N = len(x); out = np.full(N, NAN)
    for i in range(n-1, N):
        s = 0.0; bad = False
        for j in range(n):
            v = x[i-j]
            if np.isnan(v): bad = True; break
            s += v
        if bad: continue
        m = s/n; q = 0.0
        for j in range(n): q += abs(x[i-j]-m)
        out[i] = q/n
    return out

# ---------------- range / volatility ----------------
def tr(h, l, c, handle_na=True):
    N = len(c); out = np.empty(N)
    out[0] = h[0]-l[0]
    out[1:] = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    return out

def atr(h, l, c, n):
    return rma(tr(h, l, c), n)

# ---------------- extremes ----------------
@njit(cache=True)
def highest(x, n):
    N = len(x); out = np.full(N, NAN)
    for i in range(n-1, N):
        m = -1e308
        for j in range(n):
            if x[i-j] > m: m = x[i-j]
        out[i] = m
    return out

@njit(cache=True)
def lowest(x, n):
    N = len(x); out = np.full(N, NAN)
    for i in range(n-1, N):
        m = 1e308
        for j in range(n):
            if x[i-j] < m: m = x[i-j]
        out[i] = m
    return out

@njit(cache=True)
def highestbars(x, n):
    """Offset (<=0) to the highest value over the last n bars."""
    N = len(x); out = np.full(N, NAN)
    for i in range(n-1, N):
        m = -1e308; k = 0
        for j in range(n):
            if x[i-j] > m: m = x[i-j]; k = j
        out[i] = -k
    return out

@njit(cache=True)
def lowestbars(x, n):
    N = len(x); out = np.full(N, NAN)
    for i in range(n-1, N):
        m = 1e308; k = 0
        for j in range(n):
            if x[i-j] < m: m = x[i-j]; k = j
        out[i] = -k
    return out

# ---------------- state / comparison ----------------
@njit(cache=True)
def barssince(cond):
    N = len(cond); out = np.full(N, NAN); last = -1
    for i in range(N):
        if cond[i]: last = i
        if last >= 0: out[i] = i-last
    return out

def crossover(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    out = np.zeros(len(a), bool)
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out

def crossunder(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    out = np.zeros(len(a), bool)
    out[1:] = (a[1:] < b[1:]) & (a[:-1] >= b[:-1])
    return out

def cross(a, b):
    return crossover(a, b) | crossunder(a, b)

def change(x, n=1):
    x = np.asarray(x, float); out = np.full(len(x), NAN)
    out[n:] = x[n:]-x[:-n]
    return out

def rising(x, n):
    x = np.asarray(x, float); out = np.zeros(len(x), bool)
    for i in range(n, len(x)):
        ok = True
        for j in range(n):
            if not (x[i-j] > x[i-j-1]): ok = False; break
        out[i] = ok
    return out

def falling(x, n):
    x = np.asarray(x, float); out = np.zeros(len(x), bool)
    for i in range(n, len(x)):
        ok = True
        for j in range(n):
            if not (x[i-j] < x[i-j-1]): ok = False; break
        out[i] = ok
    return out

@njit(cache=True)
def percentrank(x, n):
    N = len(x); out = np.full(N, NAN)
    for i in range(n, N):
        c = 0
        for j in range(i-n, i):
            if x[j] <= x[i]: c += 1
        out[i] = 100.0*c/n
    return out

@njit(cache=True)
def valuewhen(cond, src, occ):
    N = len(cond); out = np.full(N, NAN)
    for i in range(N):
        seen = 0
        for j in range(i, -1, -1):
            if cond[j]:
                if seen == occ: out[i] = src[j]; break
                seen += 1
    return out

# ---------------- oscillators ----------------
def rsi(x, n):
    x = np.asarray(x, float); d = np.zeros(len(x)); d[1:] = x[1:]-x[:-1]
    up = rma(np.maximum(d, 0.0), n); dn = rma(np.maximum(-d, 0.0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(dn == 0, np.inf, up/dn)
        out = 100 - 100/(1+rs)
    return np.where(dn == 0, 100.0, np.where(up == 0, 0.0, out))

def macd(x, fast, slow, sig):
    m = ema(np.asarray(x, float), fast) - ema(np.asarray(x, float), slow)
    s = ema(np.nan_to_num(m, nan=NAN), sig)
    return m, s, m-s

def stoch(c, h, l, n):
    hh = highest(np.asarray(h, float), n); ll = lowest(np.asarray(l, float), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100*(np.asarray(c, float)-ll)/(hh-ll)

def cci(h, l, c, n):
    tp = (np.asarray(h,float)+np.asarray(l,float)+np.asarray(c,float))/3
    m = sma(tp, n); d = dev(tp, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (tp-m)/(0.015*d)

def dmi(h, l, c, n, adx_n):
    """Pine ta.dmi -> (+DI, -DI, ADX)."""
    h = np.asarray(h,float); l = np.asarray(l,float); c = np.asarray(c,float)
    up = np.zeros(len(c)); dn = np.zeros(len(c))
    up[1:] = h[1:]-h[:-1]; dn[1:] = l[:-1]-l[1:]
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    trur = rma(tr(h, l, c), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100*rma(plus, n)/trur
        mdi = 100*rma(minus, n)/trur
        dx = 100*np.abs(pdi-mdi)/(pdi+mdi)
    return pdi, mdi, rma(np.nan_to_num(dx, nan=0.0), adx_n)

def bb(x, n, mult):
    b = sma(x, n); d = mult*stdev(np.asarray(x, float), n)
    return b+d, b, b-d

@njit(cache=True)
def sar(h, l, start, inc, maxv):
    """ta.sar - parabolic SAR."""
    N = len(h); out = np.full(N, NAN)
    if N < 2: return out
    up = h[1] > h[0]
    af = start
    ep = h[1] if up else l[1]
    s = l[0] if up else h[0]
    out[1] = s
    for i in range(2, N):
        s = s + af*(ep-s)
        if up:
            if l[i-1] < s: s = l[i-1]
            if l[i-2] < s: s = l[i-2]
            if h[i] > ep:
                ep = h[i]; af = min(af+inc, maxv)
            if l[i] < s:
                up = False; s = ep; ep = l[i]; af = start
        else:
            if h[i-1] > s: s = h[i-1]
            if h[i-2] > s: s = h[i-2]
            if l[i] < ep:
                ep = l[i]; af = min(af+inc, maxv)
            if h[i] > s:
                up = True; s = ep; ep = h[i]; af = start
        out[i] = s
    return out

@njit(cache=True)
def supertrend_src(src, c, atr_, factor):
    """TradingView's built-in SuperTrend, transcribed from the reference Pine
    implementation these scripts embed. direction -1 = up-trend.
    `src` is the band centre (hl2 for ta.supertrend, close for some scripts)."""
    N = len(c); st = np.full(N, NAN); d = np.zeros(N, np.int64)
    prev_up = np.nan; prev_dn = np.nan; prev_st = np.nan
    for i in range(N):
        if np.isnan(atr_[i]):
            d[i] = 1
            continue
        up = src[i] + factor*atr_[i]
        dn = src[i] - factor*atr_[i]
        pdn = 0.0 if np.isnan(prev_dn) else prev_dn
        pup = 0.0 if np.isnan(prev_up) else prev_up
        if not (dn > pdn or (i > 0 and c[i-1] < pdn)): dn = pdn
        if not (up < pup or (i > 0 and c[i-1] > pup)): up = pup
        if i == 0 or np.isnan(atr_[i-1]):
            dirn = 1
        elif prev_st == pup:
            dirn = -1 if c[i] > up else 1
        else:
            dirn = 1 if c[i] < dn else -1
        st[i] = dn if dirn == -1 else up
        d[i] = dirn
        prev_up = up; prev_dn = dn; prev_st = st[i]
    return st, d

def supertrend(h, l, c, factor, atrlen, src=None):
    """ta.supertrend(factor, atrPeriod) -> (line, direction)."""
    h = np.asarray(h, float); l = np.asarray(l, float); c = np.asarray(c, float)
    s = (h+l)/2 if src is None else np.asarray(src, float)
    return supertrend_src(s, c, atr(h, l, c, atrlen), factor)

def nz(x, rep=0.0):
    return np.nan_to_num(np.asarray(x, float), nan=rep, posinf=rep, neginf=rep)

@njit(cache=True)
def pivothigh(x, left, right):
    """ta.pivothigh: confirmed `right` bars later, reported on the confirming bar."""
    N = len(x); out = np.full(N, NAN)
    for i in range(left, N-right):
        v = x[i]; ok = True
        for j in range(1, left+1):
            if x[i-j] >= v: ok = False; break
        if ok:
            for j in range(1, right+1):
                if x[i+j] > v: ok = False; break
        if ok: out[i+right] = v
    return out

@njit(cache=True)
def pivotlow(x, left, right):
    N = len(x); out = np.full(N, NAN)
    for i in range(left, N-right):
        v = x[i]; ok = True
        for j in range(1, left+1):
            if x[i-j] <= v: ok = False; break
        if ok:
            for j in range(1, right+1):
                if x[i+j] < v: ok = False; break
        if ok: out[i+right] = v
    return out

@njit(cache=True)
def jma(src, length, phase, power, volfactor):
    """Jurik-style MA as transcribed from the JMA Quantum Edge script."""
    N = len(src); out = np.full(N, NAN)
    phaseRatio = 0.5 if phase < -100 else (2.5 if phase > 100 else (phase/100.0+1.5))
    beta = 0.45*(length-1)/(0.45*(length-1)+2)
    e0 = 0.0; e1 = 0.0; e2 = 0.0
    for i in range(N):
        vf = volfactor[i]
        if np.isnan(vf): vf = 1.0
        alpha = beta**(power*vf)
        s = src[i]
        e0 = (1-alpha)*s + alpha*e0
        e1 = (s-e0)*(1-beta) + beta*e1
        tmp = ((e0 + phaseRatio*e1 - e2)*(1-alpha)**2) + (alpha**2)*e2
        prev_e2 = e2
        e2 = tmp
        out[i] = tmp + prev_e2
    return out
