#!/usr/bin/env python3
"""
Nifty 50 Options Strategy Dashboard — GitHub Pages Generator
Aurora Borealis Theme · v19.1 · Smart Dynamic PoP Engine + Intraday P&L Simulator
- PoP now reflects: Market Bias + Support/Resistance + Max CE/PE OI walls + PCR
- lotSize fixed to 65
- Strategies ranked by smart PoP — highest PoP = best trade right now
- FIXED v18.1: All strategy legs now show actual strike prices (3-4 leg strategies)
- FIXED v18.2: Gauges now show OI CHANGE data (chg_bull_force / chg_bear_force)
- FIXED v18.3: Silent background auto-refresh — no flicker, no layout shift
               Works on both file:// and http:// protocols using hidden iframe trick
- FIXED v18.4: Holiday-aware expiry — if Tuesday is NSE holiday, expiry moves to
               previous trading day (Monday, then Friday if Monday also holiday)
- NEW v18.5: Intraday P&L Simulator — per-strategy Today P&L (Delta+Theta+Vega)
             3-tab panel: Scenarios Table · Greeks Breakdown · Live Slider

pip install curl_cffi pandas numpy yfinance pytz scipy
"""

import os, json, time, warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from curl_cffi import requests as curl_requests
import yfinance as yf
from math import log, sqrt, exp
from scipy.stats import norm as _norm
import pytz

warnings.filterwarnings("ignore")

IST = pytz.timezone("Asia/Kolkata")

def now_ist():
    return datetime.now(IST)

def today_ist():
    return now_ist().date()

def ist_weekday():
    return today_ist().weekday()

def ist_timestamp_str():
    return now_ist().strftime("%d-%b-%Y %H:%M IST")


# =================================================================
#  NSE MARKET HOLIDAYS 2026
#  Source: NSE India official holiday list
#  If Tuesday expiry falls on a holiday → move to Monday
#  If Monday also holiday → move to Friday
# =================================================================

NSE_HOLIDAYS_2026 = {
    # date-string: description
    "15-Jan-2026": "Municipal Corporation Election - Maharashtra",
    "26-Jan-2026": "Republic Day",
    "03-Mar-2026": "Holi",
    "26-Mar-2026": "Shri Ram Navami",
    "31-Mar-2026": "Shri Mahavir Jayanti",
    "03-Apr-2026": "Good Friday",
    "14-Apr-2026": "Dr. Baba Saheb Ambedkar Jayanti",
    "01-May-2026": "Maharashtra Day",
    "28-May-2026": "Bakri Id",
    "26-Jun-2026": "Muharram",
    "14-Sep-2026": "Ganesh Chaturthi",
    "02-Oct-2026": "Mahatma Gandhi Jayanti",
    "20-Oct-2026": "Dussehra",
    "10-Nov-2026": "Diwali-Balipratipada",
    "24-Nov-2026": "Prakash Gurpurb Sri Guru Nanak Dev",
    "25-Dec-2026": "Christmas",
}

# Convert to a set of date objects for fast lookup
_HOLIDAY_DATES_2026 = set()
for _ds in NSE_HOLIDAYS_2026:
    try:
        _HOLIDAY_DATES_2026.add(datetime.strptime(_ds, "%d-%b-%Y").date())
    except Exception:
        pass


def is_nse_holiday(dt):
    """Return True if the given date is an NSE trading holiday or weekend."""
    if dt.weekday() >= 5:   # Saturday=5, Sunday=6
        return True
    return dt in _HOLIDAY_DATES_2026


def get_prev_trading_day(dt):
    """Return the nearest previous trading day (not holiday, not weekend)."""
    candidate = dt - timedelta(days=1)
    for _ in range(10):           # safety: max 10 look-back days
        if not is_nse_holiday(candidate):
            return candidate
        candidate -= timedelta(days=1)
    return candidate              # fallback (should never reach here)


# =================================================================
#  SECTION 1 -- NSE OPTION CHAIN FETCHER
# =================================================================

class NSEOptionChain:
    def __init__(self):
        self.symbol = "NIFTY"

    def _make_session(self):
        headers = {
            "authority": "www.nseindia.com",
            "accept": "application/json, text/plain, */*",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "referer": "https://www.nseindia.com/option-chain",
            "accept-language": "en-US,en;q=0.9",
        }
        session = curl_requests.Session()
        try:
            session.get("https://www.nseindia.com/", headers=headers, impersonate="chrome", timeout=15)
            time.sleep(1.5)
            session.get("https://www.nseindia.com/option-chain", headers=headers, impersonate="chrome", timeout=15)
            time.sleep(1)
        except Exception as e:
            print(f"  WARNING  Session warm-up: {e}")
        return session, headers

    def _current_or_next_tuesday_ist(self):
        """
        Find the current/next Tuesday and apply holiday adjustment:
        - If Tuesday is an NSE holiday  → move to previous trading day
        - Prints a clear log of the adjustment made
        """
        today  = today_ist()
        wd     = today.weekday()        # Mon=0 … Sun=6

        # ── Find the target Tuesday ──────────────────────────────
        if wd == 1:                     # today IS Tuesday
            target_tuesday = today
        elif wd < 1:                    # Sunday/Monday — next day is Tuesday
            days_ahead = 1 - wd
            target_tuesday = today + timedelta(days=days_ahead)
        else:                           # Wed–Sat — skip to next week's Tuesday
            days_ahead = (8 - wd)       # e.g. Wed(2): 8-2=6 days → next Tue
            target_tuesday = today + timedelta(days=days_ahead)

        # ── Holiday adjustment ────────────────────────────────────
        if is_nse_holiday(target_tuesday):
            reason = NSE_HOLIDAYS_2026.get(
                target_tuesday.strftime("%d-%b-%Y"), "Holiday/Weekend"
            )
            adjusted = get_prev_trading_day(target_tuesday)
            print(
                f"  [Holiday] {target_tuesday.strftime('%d-%b-%Y')} is '{reason}'. "
                f"Expiry moved to {adjusted.strftime('%d-%b-%Y')} ({adjusted.strftime('%A')})"
            )
            expiry_date = adjusted
        else:
            expiry_date = target_tuesday

        result = expiry_date.strftime("%d-%b-%Y")
        print(f"  Computed expiry (IST, holiday-adjusted): {result}")
        return result

    def _fetch_available_expiries(self, session, headers):
        """Fallback: fetch actual expiry list from NSE and pick nearest upcoming."""
        try:
            url = f"https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={self.symbol}"
            resp = session.get(url, headers=headers, impersonate="chrome", timeout=20)
            if resp.status_code == 200:
                expiries = resp.json().get("records", {}).get("expiryDates", [])
                if expiries:
                    today = today_ist()
                    for exp_str in expiries:
                        try:
                            exp_dt = datetime.strptime(exp_str, "%d-%b-%Y").date()
                            if exp_dt >= today:
                                print(f"  Fallback expiry from NSE API: {exp_str}")
                                return exp_str
                        except Exception:
                            continue
                    return expiries[0]
        except Exception as e:
            print(f"  WARNING  Expiry fetch: {e}")
        return None

    def _fetch_for_expiry(self, session, headers, expiry):
        api_url = (
            f"https://www.nseindia.com/api/option-chain-v3"
            f"?type=Indices&symbol={self.symbol}&expiry={expiry}"
        )
        for attempt in range(1, 3):
            try:
                resp = session.get(api_url, headers=headers, impersonate="chrome", timeout=30)
                if resp.status_code != 200:
                    time.sleep(2)
                    continue
                json_data  = resp.json()
                data       = json_data.get("records", {}).get("data", [])
                if not data:
                    return None
                underlying = json_data.get("records", {}).get("underlyingValue", 0)
                atm_strike = round(underlying / 50) * 50
                lower_bound = underlying - 500
                upper_bound = underlying + 500
                rows = []
                for item in data:
                    strike = item.get("strikePrice")
                    if strike is None or not (lower_bound <= strike <= upper_bound):
                        continue
                    ce = item.get("CE", {})
                    pe = item.get("PE", {})
                    rows.append({
                        "Strike":       strike,
                        "CE_LTP":       ce.get("lastPrice", 0),
                        "CE_OI":        ce.get("openInterest", 0),
                        "CE_Vol":       ce.get("totalTradedVolume", 0),
                        "PE_LTP":       pe.get("lastPrice", 0),
                        "PE_OI":        pe.get("openInterest", 0),
                        "PE_Vol":       pe.get("totalTradedVolume", 0),
                        "CE_OI_Change": ce.get("changeinOpenInterest", 0),
                        "PE_OI_Change": pe.get("changeinOpenInterest", 0),
                        "CE_IV":        ce.get("impliedVolatility", 0),
                        "PE_IV":        pe.get("impliedVolatility", 0),
                        "CE_Delta":     ce.get("delta", 0),
                        "PE_Delta":     pe.get("delta", 0),
                        "CE_Theta":     ce.get("theta", 0),
                        "PE_Theta":     pe.get("theta", 0),
                        "CE_Gamma":     ce.get("gamma", 0),
                        "PE_Gamma":     pe.get("gamma", 0),
                        "CE_Vega":      ce.get("vega", 0),
                        "PE_Vega":      pe.get("vega", 0),
                    })
                df = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
                print(f"    OK {len(df)} strikes | Spot={underlying:.0f} ATM={atm_strike}")
                return {"expiry": expiry, "df": df, "underlying": underlying, "atm_strike": atm_strike}
            except Exception as e:
                print(f"    FAIL Attempt {attempt}: {e}")
                time.sleep(2)
        return None

    def fetch_multiple_expiries(self, session, headers, n=7):
        """Fetch next n expiry dates directly from NSE API (includes weekly,
           monthly, and quarterly expiries exactly as NSE lists them)."""

        # ── Step 1: Get actual expiry list directly from NSE ───────
        expiry_list = []
        try:
            url = f"https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={self.symbol}"
            resp = session.get(url, headers=headers, impersonate="chrome", timeout=20)
            if resp.status_code == 200:
                all_exp = resp.json().get("records", {}).get("expiryDates", [])
                today = today_ist()
                for exp_str in all_exp:
                    try:
                        exp_dt = datetime.strptime(exp_str, "%d-%b-%Y").date()
                        if exp_dt >= today:
                            expiry_list.append(exp_str)
                        if len(expiry_list) >= n:
                            break
                    except Exception:
                        continue
                print(f"  Expiry list from NSE API ({len(expiry_list)} expiries): {expiry_list}")
            else:
                print(f"  WARNING: NSE expiry list returned status {resp.status_code}")
        except Exception as e:
            print(f"  WARNING expiry list fetch: {e}")

        # Fallback: if NSE API failed, generate Tuesdays as before
        if not expiry_list:
            print("  Falling back to generated Tuesday expiry list...")
            today = today_ist()
            wd = today.weekday()
            days_to_tue = 1 - wd if wd <= 1 else 8 - wd
            candidate = today + timedelta(days=days_to_tue)
            attempts = 0
            while len(expiry_list) < n and attempts < 30:
                adjusted = get_prev_trading_day(candidate) if is_nse_holiday(candidate) else candidate
                exp_str = adjusted.strftime("%d-%b-%Y")
                if exp_str not in expiry_list:
                    expiry_list.append(exp_str)
                candidate += timedelta(days=7)
                attempts += 1

        print(f"  Final expiry list: {expiry_list}")

        # ── Step 2: Fetch option chain for each expiry ─────────────
        results = {}
        for exp in expiry_list:
            print(f"    Fetching expiry: {exp}")
            data = self._fetch_for_expiry(session, headers, exp)
            if data:
                results[exp] = data
                print(f"      OK: {exp}")
            else:
                print(f"      SKIP: {exp} not found on NSE (may be monthly/not listed yet)")
            time.sleep(0.8)

        print(f"  Successfully fetched {len(results)} of {len(expiry_list)} expiries")
        return results, expiry_list

    def fetch(self):
        session, headers = self._make_session()

        # ── Step 1: compute holiday-adjusted expiry ───────────────
        expiry = self._current_or_next_tuesday_ist()

        # ── Step 2: try to fetch with computed expiry ─────────────
        result = self._fetch_for_expiry(session, headers, expiry)

        # ── Step 3: fallback — ask NSE for actual expiry list ─────
        if result is None:
            print(f"  Computed expiry {expiry} not found on NSE. Trying API fallback...")
            real_expiry = self._fetch_available_expiries(session, headers)
            if real_expiry and real_expiry != expiry:
                result = self._fetch_for_expiry(session, headers, real_expiry)

        if result is None:
            print("  ERROR: Option chain fetch failed for all expiries.")
        # Also capture full expiry list for dropdown
        self._cached_expiry_list = []
        try:
            url = f"https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={self.symbol}"
            resp = session.get(url, headers=headers, impersonate="chrome", timeout=20)
            if resp.status_code == 200:
                all_exp = resp.json().get("records", {}).get("expiryDates", [])
                today = today_ist()
                for exp_str in all_exp:
                    try:
                        exp_dt = datetime.strptime(exp_str, "%d-%b-%Y").date()
                        if exp_dt >= today:
                            self._cached_expiry_list.append(exp_str)
                            if len(self._cached_expiry_list) >= 7:
                                break
                    except Exception:
                        continue
                print(f"  Expiry list fetched: {self._cached_expiry_list}")
        except Exception as e:
            print(f"  WARNING expiry list: {e}")
        return result, session, headers


# =================================================================
#  SECTION 1B -- INDIA VIX FETCHER
# =================================================================

def fetch_india_vix(nse_session=None, nse_headers=None):
    if nse_session and nse_headers:
        try:
            resp = nse_session.get(
                "https://www.nseindia.com/api/allIndices",
                headers=nse_headers, impersonate="chrome", timeout=15)
            if resp.status_code == 200:
                for item in resp.json().get("data", []):
                    if "INDIA VIX" in item.get("index", "").upper():
                        last  = float(item.get("last", 0))
                        prev  = float(item.get("previousClose", last))
                        chg   = round(last - prev, 2)
                        chg_p = round((chg / prev * 100), 2) if prev else 0
                        return {"value": round(last,2), "prev_close": round(prev,2),
                                "change": chg, "change_pct": chg_p,
                                "high": float(item.get("high", last)),
                                "low":  float(item.get("low",  last)), "status": "live"}
        except Exception as e:
            print(f"  WARNING VIX source1: {e}")

    try:
        hist = yf.Ticker("^INDIAVIX").history(period="2d")
        if not hist.empty:
            last  = float(hist.iloc[-1]["Close"])
            prev  = float(hist.iloc[-2]["Close"]) if len(hist) > 1 else last
            chg   = round(last - prev, 2)
            chg_p = round((chg / prev * 100), 2) if prev else 0
            return {"value": round(last,2), "prev_close": round(prev,2),
                    "change": chg, "change_pct": chg_p,
                    "high": float(hist.iloc[-1].get("High", last)),
                    "low":  float(hist.iloc[-1].get("Low",  last)), "status": "live"}
    except Exception as e:
        print(f"  WARNING VIX source3 (yfinance): {e}")
    return None


def vix_label(v):
    if   v < 12:  return "Extremely Low",  "#00c896", "rgba(0,200,150,.12)",  "rgba(0,200,150,.35)",  "Sell Premium"
    elif v < 15:  return "Low",            "#4de8b8", "rgba(0,200,150,.08)",  "rgba(0,200,150,.25)",  "Sell Premium"
    elif v < 20:  return "Normal",         "#6480ff", "rgba(100,128,255,.1)", "rgba(100,128,255,.3)", "Balanced"
    elif v < 25:  return "Elevated",       "#ffd166", "rgba(255,209,102,.1)", "rgba(255,209,102,.3)", "Buy Premium"
    elif v < 30:  return "High",           "#ff9a3c", "rgba(255,154,60,.1)",  "rgba(255,154,60,.3)",  "Buy Straddles"
    else:         return "Extreme Fear",   "#ff6b6b", "rgba(255,107,107,.1)", "rgba(255,107,107,.3)", "Buy Puts/Hedge"


# =================================================================
#  SECTION 1C -- BLACK-SCHOLES CALCULATOR
# =================================================================

def _bs_greeks(S, K, T, r, sigma, option_type="CE"):
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        nd1   = _norm.pdf(d1)
        gamma = nd1 / (S * sigma * np.sqrt(T))
        vega  = (S * nd1 * np.sqrt(T)) / 100
        if option_type == "CE":
            delta        = _norm.cdf(d1)
            theta_annual = (-(S * nd1 * sigma) / (2 * np.sqrt(T))
                            - r * K * np.exp(-r * T) * _norm.cdf(d2))
        else:
            delta        = _norm.cdf(d1) - 1
            theta_annual = (-(S * nd1 * sigma) / (2 * np.sqrt(T))
                            + r * K * np.exp(-r * T) * _norm.cdf(-d2))
        return {
            "delta": round(float(delta), 4),
            "gamma": round(float(gamma), 6),
            "theta": round(float(theta_annual / 365.0), 4),
            "vega":  round(float(vega), 4),
        }
    except Exception:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}


def _days_to_expiry_ist(expiry_str):
    try:
        today   = today_ist()
        exp_dt  = datetime.strptime(expiry_str, "%d-%b-%Y").date()
        days    = (exp_dt - today).days
        return max(days, 1)
    except Exception:
        return 7


def _compute_greeks_for_row(r, spot, expiry_str, risk_free=0.065, vix=18.0):
    T = _days_to_expiry_ist(expiry_str) / 365.0
    K = float(r["Strike"])
    ce_iv_nse = float(r.get("CE_IV", 0) or 0)
    pe_iv_nse = float(r.get("PE_IV", 0) or 0)
    ce_iv = ce_iv_nse if ce_iv_nse > 0.5 else (pe_iv_nse if pe_iv_nse > 0.5 else vix)
    pe_iv = pe_iv_nse if pe_iv_nse > 0.5 else (ce_iv_nse if ce_iv_nse > 0.5 else vix)
    ce_g = _bs_greeks(spot, K, T, risk_free, ce_iv / 100.0, "CE")
    pe_g = _bs_greeks(spot, K, T, risk_free, pe_iv / 100.0, "PE")
    return ce_g, pe_g


def extract_atm_greeks(df, atm_strike, underlying=None, expiry_str="", vix=18.0):
    spot = underlying or float(atm_strike)
    greeks_rows = []
    for _, r in df.iterrows():
        strike    = int(r["Strike"])
        is_atm    = strike == int(atm_strike)
        ce_iv_raw = round(float(r.get("CE_IV",  0) or 0), 2)
        pe_iv_raw = round(float(r.get("PE_IV",  0) or 0), 2)
        ce_ltp    = round(float(r.get("CE_LTP", 0) or 0), 2)
        pe_ltp    = round(float(r.get("PE_LTP", 0) or 0), 2)
        ce_g, pe_g = _compute_greeks_for_row(r, spot, expiry_str, vix=vix)
        greeks_rows.append({
            "strike":   strike,
            "is_atm":   is_atm,
            "ce_iv":    ce_iv_raw,
            "pe_iv":    pe_iv_raw,
            "ce_delta": ce_g["delta"],
            "pe_delta": pe_g["delta"],
            "ce_theta": ce_g["theta"],
            "pe_theta": pe_g["theta"],
            "ce_gamma": ce_g["gamma"],
            "pe_gamma": pe_g["gamma"],
            "ce_vega":  ce_g["vega"],
            "pe_vega":  pe_g["vega"],
            "ce_ltp":   ce_ltp,
            "pe_ltp":   pe_ltp,
        })
    greeks_rows.sort(key=lambda x: x["strike"])
    atm_row = next((g for g in greeks_rows if g["is_atm"]),
                   greeks_rows[len(greeks_rows)//2] if greeks_rows else {})
    atm_idx = next((i for i, g in enumerate(greeks_rows) if g["is_atm"]),
                   len(greeks_rows)//2)
    lo = max(0, atm_idx - 2)
    hi = min(len(greeks_rows), atm_idx + 3)
    table_5 = greeks_rows[lo:hi]
    return {
        "atm_greeks":   atm_row,
        "greeks_table": table_5,
        "all_strikes":  greeks_rows,
    }


# =================================================================
#  SECTION 2 -- OPTION CHAIN ANALYSIS
# =================================================================

def analyze_option_chain(oc_data, vix=18.0):
    if not oc_data:
        return None
    df           = oc_data["df"]
    total_ce_oi  = df["CE_OI"].sum()
    total_pe_oi  = df["PE_OI"].sum()
    total_ce_vol = df["CE_Vol"].sum()
    total_pe_vol = df["PE_Vol"].sum()
    pcr_oi       = total_pe_oi  / total_ce_oi  if total_ce_oi  > 0 else 0
    pcr_vol      = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0
    ce_chg       = int(df["CE_OI_Change"].sum())
    pe_chg       = int(df["PE_OI_Change"].sum())
    net_chg      = pe_chg + ce_chg

    if   ce_chg > 0 and pe_chg < 0:
        oi_dir, oi_sig, oi_icon, oi_cls = "Strong Bearish", "Call Build-up + Put Unwinding", "RED",    "bearish"
    elif ce_chg < 0 and pe_chg > 0:
        oi_dir, oi_sig, oi_icon, oi_cls = "Strong Bullish", "Put Build-up + Call Unwinding", "GREEN",  "bullish"
    elif ce_chg > 0 and pe_chg > 0:
        if   pe_chg > ce_chg * 1.5:
            oi_dir, oi_sig, oi_icon, oi_cls = "Bullish",            "Put Build-up Dominant",      "GREEN",  "bullish"
        elif ce_chg > pe_chg * 1.5:
            oi_dir, oi_sig, oi_icon, oi_cls = "Bearish",            "Call Build-up Dominant",     "RED",    "bearish"
        else:
            oi_dir, oi_sig, oi_icon, oi_cls = "Neutral (High Vol)", "Both Calls & Puts Building", "YELLOW", "neutral"
    elif ce_chg < 0 and pe_chg < 0:
        oi_dir, oi_sig, oi_icon, oi_cls = "Neutral (Unwinding)", "Both Calls & Puts Unwinding", "YELLOW", "neutral"
    else:
        if   net_chg > 0: oi_dir, oi_sig, oi_icon, oi_cls = "Moderately Bullish", "Net Put Accumulation",  "GREEN",  "bullish"
        elif net_chg < 0: oi_dir, oi_sig, oi_icon, oi_cls = "Moderately Bearish", "Net Call Accumulation", "RED",    "bearish"
        else:              oi_dir, oi_sig, oi_icon, oi_cls = "Neutral",            "Balanced OI Changes",   "YELLOW", "neutral"

    max_ce_row   = df.loc[df["CE_OI"].idxmax()]
    max_pe_row   = df.loc[df["PE_OI"].idxmax()]
    df["pain"]   = abs(df["CE_OI"] - df["PE_OI"])
    max_pain_row = df.loc[df["pain"].idxmin()]
    top_ce = df.nlargest(5, "CE_OI")[["Strike", "CE_OI", "CE_LTP"]].to_dict("records")
    top_pe = df.nlargest(5, "PE_OI")[["Strike", "PE_OI", "PE_LTP"]].to_dict("records")

    strikes_data = []
    for _, row in df.iterrows():
        strikes_data.append({
            "strike": int(row["Strike"]),
            "ce_ltp": float(row["CE_LTP"]),
            "pe_ltp": float(row["PE_LTP"]),
            "ce_iv":  float(row.get("CE_IV", 15)),
            "pe_iv":  float(row.get("PE_IV", 15)),
            "ce_oi":  int(row["CE_OI"]),
            "pe_oi":  int(row["PE_OI"]),
        })

    raw_total = int(total_pe_oi) + int(total_ce_oi)
    raw_total = raw_total if raw_total > 0 else 1
    bull_pct  = round(int(total_pe_oi) / raw_total * 100)
    bear_pct  = 100 - bull_pct

    if   pcr_oi > 1.5:
        raw_oi_dir = "STRONG BULLISH"; raw_oi_sig = "Heavy Put Writing — Strong Support Floor"; raw_oi_cls = "bullish"
    elif pcr_oi > 1.2:
        raw_oi_dir = "BULLISH";        raw_oi_sig = "Put OI > Call OI — Bulls in Control";      raw_oi_cls = "bullish"
    elif pcr_oi < 0.5:
        raw_oi_dir = "STRONG BEARISH"; raw_oi_sig = "Heavy Call Writing — Strong Resistance Cap"; raw_oi_cls = "bearish"
    elif pcr_oi < 0.7:
        raw_oi_dir = "BEARISH";        raw_oi_sig = "Call OI > Put OI — Bears in Control";      raw_oi_cls = "bearish"
    else:
        if int(total_pe_oi) >= int(total_ce_oi):
            raw_oi_dir = "CAUTIOUSLY BULLISH"; raw_oi_sig = "Balanced OI — Slight Put Dominance"; raw_oi_cls = "bullish"
        else:
            raw_oi_dir = "CAUTIOUSLY BEARISH"; raw_oi_sig = "Balanced OI — Slight Call Dominance"; raw_oi_cls = "bearish"

    chg_bull_force = (abs(pe_chg) if pe_chg > 0 else 0) + (abs(ce_chg) if ce_chg < 0 else 0)
    chg_bear_force = (abs(ce_chg) if ce_chg > 0 else 0) + (abs(pe_chg) if pe_chg < 0 else 0)

    chg_total = chg_bull_force + chg_bear_force
    chg_total = chg_total if chg_total > 0 else 1
    chg_bull_pct = round(chg_bull_force / chg_total * 100)
    chg_bear_pct = 100 - chg_bull_pct

    atm_strike = oc_data["atm_strike"]
    greeks = extract_atm_greeks(df, atm_strike,
                                underlying=oc_data["underlying"],
                                expiry_str=oc_data["expiry"],
                                vix=vix)

    # Merge BS-computed greeks into strikes_data so JS STRIKE_MAP has delta/theta/vega
    _g_by_strike = {g["strike"]: g for g in greeks["all_strikes"]}
    for _sd in strikes_data:
        _g = _g_by_strike.get(_sd["strike"], {})
        _sd["ce_delta"] = round(_g.get("ce_delta", 0.50),  4)
        _sd["pe_delta"] = round(_g.get("pe_delta", -0.50), 4)
        _sd["ce_theta"] = round(_g.get("ce_theta", 0.0),   4)
        _sd["pe_theta"] = round(_g.get("pe_theta", 0.0),   4)
        _sd["ce_vega"]  = round(_g.get("ce_vega",  0.0),   4)
        _sd["pe_vega"]  = round(_g.get("pe_vega",  0.0),   4)

    return {
        "expiry":          oc_data["expiry"],
        "underlying":      oc_data["underlying"],
        "atm_strike":      atm_strike,
        "pcr_oi":          round(pcr_oi, 3),
        "pcr_vol":         round(pcr_vol, 3),
        "total_ce_oi":     int(total_ce_oi),
        "total_pe_oi":     int(total_pe_oi),
        "max_ce_strike":   int(max_ce_row["Strike"]),
        "max_ce_oi":       int(max_ce_row["CE_OI"]),
        "max_pe_strike":   int(max_pe_row["Strike"]),
        "max_pe_oi":       int(max_pe_row["PE_OI"]),
        "max_pain":        int(max_pain_row["Strike"]),
        "ce_chg":          ce_chg,
        "pe_chg":          pe_chg,
        "net_chg":         net_chg,
        "oi_dir":          oi_dir,
        "oi_sig":          oi_sig,
        "oi_icon":         oi_icon,
        "oi_cls":          oi_cls,
        "top_ce":          top_ce,
        "top_pe":          top_pe,
        "strikes_data":    strikes_data,
        "bull_pct":        bull_pct,
        "bear_pct":        bear_pct,
        "bull_force":      int(total_pe_oi),
        "bear_force":      int(total_ce_oi),
        "raw_oi_dir":      raw_oi_dir,
        "raw_oi_sig":      raw_oi_sig,
        "raw_oi_cls":      raw_oi_cls,
        "chg_bull_force":  chg_bull_force,
        "chg_bear_force":  chg_bear_force,
        "chg_bull_pct":    chg_bull_pct,
        "chg_bear_pct":    chg_bear_pct,
        "atm_greeks":      greeks["atm_greeks"],
        "greeks_table":    greeks["greeks_table"],
        "all_strikes":     greeks["all_strikes"],
    }


# =================================================================
#  SECTION 3 -- TECHNICAL ANALYSIS
# =================================================================

def get_technical_data():
    try:
        nifty = yf.Ticker("^NSEI")
        df    = nifty.history(period="1y")
        if df.empty:
            return None
        df["SMA_20"]  = df["Close"].rolling(20).mean()
        df["SMA_50"]  = df["Close"].rolling(50).mean()
        df["SMA_200"] = df["Close"].rolling(200).mean()
        delta = df["Close"].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"]    = 100 - (100 / (1 + gain / loss))
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"]   = df["EMA_12"] - df["EMA_26"]
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        latest = df.iloc[-1]
        cp     = latest["Close"]

        s1 = s2 = r1 = r2 = None
        try:
            df_1h = nifty.history(interval="1h", period="60d")
            if not df_1h.empty:
                recent_1h = df_1h.tail(120)
                highs = sorted(recent_1h["High"].values)
                lows  = sorted(recent_1h["Low"].values)
                res_c = [h for h in highs if cp < h <= cp + 200]
                sup_c = [l for l in lows  if cp - 200 <= l < cp]
                if len(res_c) >= 4:
                    r1 = round(float(np.percentile(res_c, 40)) / 25) * 25
                    r2 = round(float(np.percentile(res_c, 80)) / 25) * 25
                if len(sup_c) >= 4:
                    s1 = round(float(np.percentile(sup_c, 70)) / 25) * 25
                    s2 = round(float(np.percentile(sup_c, 20)) / 25) * 25
                if r1 and r1 <= cp:         r1 = round((cp + 50) / 25) * 25
                if r2 and r1 and r2 <= r1:  r2 = r1 + 75
                if s1 and s1 >= cp:         s1 = round((cp - 50) / 25) * 25
                if s2 and s1 and s2 >= s1:  s2 = s1 - 75
        except Exception as e:
            print(f"  WARNING 1H data: {e}")

        recent_d   = df.tail(60)
        resistance = r1 if r1 else recent_d["High"].quantile(0.90)
        support    = s1 if s1 else recent_d["Low"].quantile(0.10)
        strong_res = r2 if r2 else resistance + 100
        strong_sup = s2 if s2 else support - 100

        df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
        hv = df["log_ret"].tail(20).std() * np.sqrt(252) * 100

        return {
            "price":       cp,
            "sma20":       latest["SMA_20"],
            "sma50":       latest["SMA_50"],
            "sma200":      latest["SMA_200"],
            "rsi":         latest["RSI"],
            "macd":        latest["MACD"],
            "signal_line": latest["Signal"],
            "support":     support,
            "resistance":  resistance,
            "strong_sup":  strong_sup,
            "strong_res":  strong_res,
            "hv":          round(float(hv), 2),
        }
    except Exception as e:
        print(f"  ERROR Technical: {e}")
        return None


# =================================================================
#  SECTION 4 -- MARKET DIRECTION SCORING
# =================================================================

def compute_market_direction(tech, oc_analysis):
    if not tech:
        return {"bias": "UNKNOWN", "confidence": "LOW", "bull": 0, "bear": 0, "diff": 0, "bias_cls": "neutral"}

    cp   = tech["price"]
    bull = bear = 0

    for sma in ["sma20", "sma50", "sma200"]:
        if cp > tech[sma]: bull += 1
        else:               bear += 1

    rsi = tech["rsi"]
    if   rsi > 70: bear += 1
    elif rsi < 30: bull += 2

    if tech["macd"] > tech["signal_line"]: bull += 1
    else:                                   bear += 1

    if oc_analysis:
        pcr = oc_analysis["pcr_oi"]
        if   pcr > 1.2: bull += 2
        elif pcr < 0.7: bear += 2
        mp = oc_analysis["max_pain"]
        if   cp > mp + 100: bear += 1
        elif cp < mp - 100: bull += 1

    diff = bull - bear

    if   diff >= 3:  bias, bias_cls = "BULLISH",  "bullish"; confidence = "HIGH" if diff >= 4 else "MEDIUM"
    elif diff <= -3: bias, bias_cls = "BEARISH",  "bearish"; confidence = "HIGH" if diff <= -4 else "MEDIUM"
    else:            bias, bias_cls = "SIDEWAYS", "neutral"; confidence = "MEDIUM"

    return {"bias": bias, "bias_cls": bias_cls, "confidence": confidence,
            "bull": bull, "bear": bear, "diff": diff}


# =================================================================
#  SECTION 5 -- HELPERS
# =================================================================

def _cls_color(cls):
    return "#00c896" if cls == "bullish" else ("#ff6b6b" if cls == "bearish" else "#6480ff")

def _cls_bg(cls):
    return ("rgba(0,200,150,.08)"   if cls == "bullish" else
            "rgba(255,107,107,.08)" if cls == "bearish" else "rgba(100,128,255,.08)")

def _cls_bdr(cls):
    return ("rgba(0,200,150,.22)"   if cls == "bullish" else
            "rgba(255,107,107,.22)" if cls == "bearish" else "rgba(100,128,255,.22)")

def _fmt_oi(n):
    abs_n = abs(n); sign = "+" if n > 0 else ("-" if n < 0 else "")
    if abs_n >= 1_00_00_000:  return f"{sign}{abs_n / 1_00_00_000:.1f}Cr"
    if abs_n >= 1_00_000:     return f"{sign}{abs_n / 1_00_000:.0f}L"
    if abs_n >= 1_000:        return f"{sign}{abs_n / 1_000:.0f}K"
    return f"{n:+,}"

def _fmt_chg_oi(n):
    if abs(n) >= 1_000_000: return f"{'+' if n > 0 else ''}{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:     return f"{'+' if n > 0 else ''}{n / 1_000:.0f}K"
    return f"{n:+,}"


# =================================================================
#  SECTION 5A -- OPTION GREEKS PANEL
# =================================================================

def _delta_bar_html(delta_val, is_ce=True):
    pct = abs(delta_val) * 100
    col = "#00c896" if is_ce else "#ff6b6b"
    return (
        f'<div style="display:flex;align-items:center;gap:5px;">'
        f'<div style="width:34px;height:3px;background:rgba(255,255,255,.10);border-radius:2px;overflow:hidden;">'
        f'<div style="width:{pct:.0f}%;height:100%;background:{col};border-radius:2px;"></div></div>'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:15.9px;font-weight:700;color:{col};">'
        f'{delta_val:+.3f}</span></div>'
    )


def build_greeks_sidebar_html(oc_analysis):
    if not oc_analysis:
        return '<div style="padding:14px 12px;font-size:15.9px;color:rgba(255,255,255,.3);text-align:center;">Greeks unavailable.</div>'

    g    = oc_analysis.get("atm_greeks", {})
    atm  = oc_analysis.get("atm_strike", 0)
    exp  = oc_analysis.get("expiry", "N/A")
    all_rows = oc_analysis.get("all_strikes", oc_analysis.get("greeks_table", []))

    if not g:
        return '<div style="padding:14px 12px;font-size:15.9px;color:rgba(255,255,255,.3);text-align:center;">Greeks not computed yet.</div>'

    ce_iv    = g.get("ce_iv",    15.0)
    pe_iv    = g.get("pe_iv",    15.0)
    ce_delta = g.get("ce_delta", 0.5)
    pe_delta = g.get("pe_delta", -0.5)
    ce_theta = g.get("ce_theta", 0.0)
    pe_theta = g.get("pe_theta", 0.0)
    ce_vega  = g.get("ce_vega",  0.0)
    pe_vega  = g.get("pe_vega",  0.0)
    ce_ltp   = g.get("ce_ltp",   0.0)
    pe_ltp   = g.get("pe_ltp",   0.0)

    iv_skew  = round(pe_iv - ce_iv, 2)
    skew_col = "#ff6b6b" if iv_skew > 1.5 else ("#00c896" if iv_skew < -1.5 else "#6480ff")
    skew_txt = f"PE Skew +{iv_skew:.1f}" if iv_skew > 0 else f"CE Skew {iv_skew:.1f}"

    iv_avg    = (ce_iv + pe_iv) / 2
    iv_pct    = min(100, max(0, (iv_avg / 60) * 100))
    iv_col    = "#ff6b6b" if iv_avg > 25 else "#ffd166" if iv_avg > 18 else "#00c896"
    iv_regime = "High IV · Buy Premium" if iv_avg > 25 else "Normal IV · Balanced" if iv_avg > 15 else "Low IV · Sell Premium"

    def tfmt(t): return f"&#8377;{abs(t):.2f}" if abs(t) >= 0.01 else f"{t:.4f}"
    def vfmt(v): return f"{v:.4f}" if abs(v) >= 0.0001 else "&#8212;"

    otm_ce_opts = ""
    atm_opt     = ""
    otm_pe_opts = ""
    for row in sorted(all_rows, key=lambda x: x["strike"], reverse=True):
        s      = int(row["strike"])
        is_atm = row["is_atm"]
        dist   = abs(s - atm) // 50
        if is_atm:
            label   = f"\u2605  ATM  \u20b9{s:,}"
            atm_opt = f'<option value="{s}" selected>{label}</option>\n'
        elif s > atm:
            label       = f"\u25b2  CE+{dist}  \u20b9{s:,}"
            otm_ce_opts += f'<option value="{s}">{label}</option>\n'
        else:
            label       = f"\u25bc  PE-{dist}  \u20b9{s:,}"
            otm_pe_opts += f'<option value="{s}">{label}</option>\n'

    dropdown_options = (
        f'<optgroup label="\u2500 OTM CALLS (CE) \u2500">{otm_ce_opts}</optgroup>'
        f'<optgroup label="\u2500 ATM \u2500">{atm_opt}</optgroup>'
        f'<optgroup label="\u2500 OTM PUTS (PE) \u2500">{otm_pe_opts}</optgroup>'
    )

    return f"""
<div class="greeks-panel" id="greeksPanelInner">
  <div class="greeks-title">
    &#9652; OPTION GREEKS
    <span class="greeks-expiry-tag">{exp}</span>
  </div>
  <div class="greeks-strike-wrap">
    <select class="greeks-strike-select" id="greeksStrikeSelect"
            onchange="greeksUpdateStrike(this.value)">
      {dropdown_options}
    </select>
  </div>
  <div class="greeks-atm-badge" id="greeksAtmBadge">
    <span style="font-size:12.3px;font-weight:700;color:rgba(138,160,255,.9);" id="greeksStrikeTypeLabel">ATM</span>
    <span class="greeks-atm-strike" id="greeksStrikeLabel">&#8377;{atm:,}</span>
    <span style="font-size:11.6px;color:rgba(255,255,255,.2);">|</span>
    <span style="font-size:12.3px;color:rgba(0,200,220,.8);" id="greeksCeLtp">CE &#8377;{ce_ltp:.1f}</span>
    <span style="font-size:11.6px;color:rgba(255,255,255,.25);">/</span>
    <span style="font-size:12.3px;color:rgba(255,107,107,.8);" id="greeksPeLtp">PE &#8377;{pe_ltp:.1f}</span>
  </div>
  <div class="greeks-row">
    <div style="display:flex;flex-direction:column;">
      <span class="greek-name">&#916; Delta</span>
      <span class="greek-sub">CE / PE</span>
    </div>
    <div style="display:flex;flex-direction:column;gap:3px;align-items:flex-end;" id="greeksDeltaWrap">
      {_delta_bar_html(ce_delta, True)}
      {_delta_bar_html(pe_delta, False)}
    </div>
  </div>
  <div class="greeks-row">
    <div style="display:flex;flex-direction:column;">
      <span class="greek-name">&#963; IV</span>
      <span class="greek-sub" id="greeksSkewLbl" style="color:{skew_col};font-weight:700;">{skew_txt}</span>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:3px;">
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:12.3px;color:rgba(0,200,220,.85);">CE</span>
        <span style="font-family:'DM Mono',monospace;font-size:18.8px;font-weight:700;color:#00c8e0;" id="greeksIvCe">{ce_iv:.1f}%</span>
      </div>
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:12.3px;color:rgba(255,144,144,.85);">PE</span>
        <span style="font-family:'DM Mono',monospace;font-size:18.8px;font-weight:700;color:#ff9090;" id="greeksIvPe">{pe_iv:.1f}%</span>
      </div>
    </div>
  </div>
  <div class="greeks-row">
    <div style="display:flex;flex-direction:column;">
      <span class="greek-name">&#920; Theta</span>
      <span class="greek-sub">per day</span>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:3px;">
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:12.3px;color:rgba(0,200,220,.85);">CE</span>
        <span style="font-family:'DM Mono',monospace;font-size:15.9px;font-weight:700;color:#ff9090;" id="greeksThetaCe">{tfmt(ce_theta)}</span>
      </div>
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:12.3px;color:rgba(255,144,144,.85);">PE</span>
        <span style="font-family:'DM Mono',monospace;font-size:15.9px;font-weight:700;color:#ff9090;" id="greeksThetaPe">{tfmt(pe_theta)}</span>
      </div>
    </div>
  </div>
  <div class="greeks-row">
    <div style="display:flex;flex-direction:column;">
      <span class="greek-name">&#957; Vega</span>
      <span class="greek-sub">per 1% IV</span>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:3px;">
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:12.3px;color:rgba(0,200,220,.85);">CE</span>
        <span style="font-family:'DM Mono',monospace;font-size:15.9px;font-weight:700;color:#8aa0ff;" id="greeksVegaCe">{vfmt(ce_vega)}</span>
      </div>
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:12.3px;color:rgba(255,144,144,.85);">PE</span>
        <span style="font-family:'DM Mono',monospace;font-size:15.9px;font-weight:700;color:#8aa0ff;" id="greeksVegaPe">{vfmt(pe_vega)}</span>
      </div>
    </div>
  </div>
  <div class="iv-bar-wrap">
    <span class="iv-bar-label">IV Avg</span>
    <div class="iv-bar-track">
      <div class="iv-bar-fill" id="greeksIvBar"
           style="width:{iv_pct:.1f}%;background:{iv_col};box-shadow:0 0 6px {iv_col}88;"></div>
    </div>
    <span class="iv-bar-num" id="greeksIvAvg" style="color:{iv_col};">{iv_avg:.1f}%</span>
  </div>
  <div style="font-size:12.3px;text-align:center;margin-top:6px;font-weight:700;letter-spacing:.5px;color:{iv_col};"
       id="greeksIvRegime">{iv_regime}</div>
</div>
"""


def build_greeks_table_html(oc_analysis):
    if not oc_analysis:
        return ""
    rows = oc_analysis.get("greeks_table", [])
    atm  = oc_analysis.get("atm_strike", 0)
    exp  = oc_analysis.get("expiry", "N/A")
    spot = oc_analysis.get("underlying", 0)
    if not rows:
        return ""

    def atm_tag_html(is_atm):
        if not is_atm: return ""
        return '<span style="font-size:11.6px;background:rgba(100,128,255,.25);color:#8aa0ff;padding:1px 5px;border-radius:4px;margin-left:4px;font-weight:700;">ATM</span>'

    def delta_iv_row(g):
        is_atm   = g["is_atm"]
        row_cls  = 'style="background:rgba(100,128,255,.06);border-left:3px solid rgba(100,128,255,.45);"' if is_atm else ""
        sc       = 'style="color:#8aa0ff;"' if is_atm else ""
        ce_d_col = "#00c896" if g["ce_delta"] >= 0.5 else "#ffd166" if g["ce_delta"] >= 0.35 else "#ff9090"
        pe_d_col = "#ff6b6b" if g["pe_delta"] <= -0.5 else "#ffd166" if g["pe_delta"] <= -0.35 else "#4de8b8"
        ce_iv_c  = "#ff6b6b" if g["ce_iv"] > 25 else "#ffd166" if g["ce_iv"] > 18 else "#00c896"
        pe_iv_c  = "#ff6b6b" if g["pe_iv"] > 25 else "#ffd166" if g["pe_iv"] > 18 else "#00c896"
        return (
            f'<div class="greeks-tbl-row" {row_cls}>'
            f'<div class="greeks-tbl-strike" {sc}>&#8377;{g["strike"]:,}{atm_tag_html(is_atm)}</div>'
            f'<div class="greeks-tbl-cell" style="color:{ce_d_col};">{g["ce_delta"]:+.3f}</div>'
            f'<div class="greeks-tbl-cell" style="color:{pe_d_col};">{g["pe_delta"]:+.3f}</div>'
            f'<div class="greeks-tbl-cell" style="color:{ce_iv_c};">{g["ce_iv"]:.1f}%</div>'
            f'<div class="greeks-tbl-cell" style="color:{pe_iv_c};">{g["pe_iv"]:.1f}%</div>'
            f'</div>'
        )

    def theta_vega_row(g):
        is_atm   = g["is_atm"]
        row_cls  = 'style="background:rgba(100,128,255,.06);border-left:3px solid rgba(100,128,255,.45);"' if is_atm else ""
        sc       = 'style="color:#8aa0ff;"' if is_atm else ""
        tfmt = lambda t: f"&#8377;{abs(t):.2f}" if abs(t) >= 0.01 else f"{t:.4f}"
        vfmt = lambda v: f"{v:.4f}" if abs(v) >= 0.0001 else "&#8212;"
        return (
            f'<div class="greeks-tbl-row" {row_cls}>'
            f'<div class="greeks-tbl-strike" {sc}>&#8377;{g["strike"]:,}{atm_tag_html(is_atm)}</div>'
            f'<div class="greeks-tbl-cell" style="color:#ff9090;">{tfmt(g["ce_theta"])}</div>'
            f'<div class="greeks-tbl-cell" style="color:#ff9090;">{tfmt(g["pe_theta"])}</div>'
            f'<div class="greeks-tbl-cell" style="color:#8aa0ff;">{vfmt(g["ce_vega"])}</div>'
            f'<div class="greeks-tbl-cell" style="color:#8aa0ff;">{vfmt(g["pe_vega"])}</div>'
            f'</div>'
        )

    dv_rows = "".join(delta_iv_row(g) for g in rows)
    tv_rows = "".join(theta_vega_row(g) for g in rows)

    return f"""
<div class="greeks-table-section" id="greeksTable">
  <div class="sec-title" style="color:#8aa0ff;border-color:rgba(100,128,255,.18);">
    OPTION GREEKS &nbsp;&middot;&nbsp; ATM &#177;2 STRIKES
    <span class="sec-sub">Live from NSE &middot; {exp} &middot; Spot &#8377;{spot:,.0f}</span>
  </div>
  <div class="greeks-table-wrap">
    <div>
      <div style="font-size:13.8px;font-weight:700;color:rgba(100,128,255,.75);margin-bottom:10px;letter-spacing:1.5px;text-transform:uppercase;display:flex;align-items:center;gap:8px;">
        <span style="width:3px;height:14px;background:#6480ff;border-radius:2px;display:inline-block;"></span>
        &#916; DELTA &amp; &#963; IMPLIED VOLATILITY
      </div>
      <div class="greeks-tbl">
        <div class="greeks-tbl-head">
          <div class="greeks-tbl-head-label" style="text-align:left;">Strike</div>
          <div class="greeks-tbl-head-label" style="color:rgba(0,200,150,.65);">CE &#916;</div>
          <div class="greeks-tbl-head-label" style="color:rgba(255,107,107,.65);">PE &#916;</div>
          <div class="greeks-tbl-head-label" style="color:rgba(0,200,220,.65);">CE IV</div>
          <div class="greeks-tbl-head-label" style="color:rgba(255,144,144,.65);">PE IV</div>
        </div>
        {dv_rows}
      </div>
    </div>
    <div>
      <div style="font-size:13.8px;font-weight:700;color:rgba(255,107,107,.75);margin-bottom:10px;letter-spacing:1.5px;text-transform:uppercase;display:flex;align-items:center;gap:8px;">
        <span style="width:3px;height:14px;background:#ff6b6b;border-radius:2px;display:inline-block;"></span>
        &#920; THETA (Day Decay) &amp; &#957; VEGA
      </div>
      <div class="greeks-tbl">
        <div class="greeks-tbl-head">
          <div class="greeks-tbl-head-label" style="text-align:left;">Strike</div>
          <div class="greeks-tbl-head-label" style="color:rgba(0,200,220,.65);">CE &#920;</div>
          <div class="greeks-tbl-head-label" style="color:rgba(255,144,144,.65);">PE &#920;</div>
          <div class="greeks-tbl-head-label" style="color:rgba(138,160,255,.65);">CE &#957;</div>
          <div class="greeks-tbl-head-label" style="color:rgba(138,160,255,.65);">PE &#957;</div>
        </div>
        {tv_rows}
      </div>
    </div>
  </div>
</div>
"""


# =================================================================
#  SECTION 5B -- HERO
# =================================================================

def build_dual_gauge_hero(oc, tech, md, ts):
    if oc:
        chg_bull = oc["chg_bull_force"]; chg_bear = oc["chg_bear_force"]
        bull_pct = oc["chg_bull_pct"]; bear_pct = oc["chg_bear_pct"]; pcr = oc["pcr_oi"]
        oi_dir = oc["raw_oi_dir"]; oi_sig = oc["raw_oi_sig"]; oi_cls = oc["raw_oi_cls"]
        bull_label = _fmt_chg_oi(chg_bull); bear_label = _fmt_chg_oi(chg_bear)
        expiry = oc["expiry"]; underlying = oc["underlying"]; atm = oc["atm_strike"]; max_pain = oc["max_pain"]
    else:
        chg_bull = chg_bear = 0; bull_pct = bear_pct = 50; pcr = 1.0
        oi_sig = "NSE data unavailable"; oi_dir = "UNKNOWN"; oi_cls = "neutral"
        expiry = "N/A"; underlying = 0; atm = 0; max_pain = 0
        bull_label = "N/A"; bear_label = "N/A"

    cp = tech["price"] if tech else 0
    bias = md["bias"]; conf = md["confidence"]; bull_sc = md["bull"]; bear_sc = md["bear"]; diff = md["diff"]
    dir_col = _cls_color(oi_cls)
    pcr_col = "#00c896" if pcr > 1.2 else ("#ff6b6b" if pcr < 0.7 else "#6480ff")
    b_col = _cls_color(md.get("bias_cls", "neutral"))
    b_bg = _cls_bg(md.get("bias_cls", "neutral")); b_bdr = _cls_bdr(md.get("bias_cls", "neutral"))
    C = 194.8
    def clamp(v, lo=10, hi=97): return max(lo, min(hi, v))
    bull_offset = C * (1 - clamp(bull_pct) / 100); bear_offset = C * (1 - clamp(bear_pct) / 100)
    oi_bar_w = clamp(bull_pct); bear_bar_w = clamp(bear_pct)
    b_arrow = "▲" if bias == "BULLISH" else ("▼" if bias == "BEARISH" else "◆")
    glow_rgb = ("0,200,150" if dir_col == "#00c896" else "255,107,107" if dir_col == "#ff6b6b" else "100,128,255")

    return f"""
<div class="hero" id="heroWidget">
  <div class="h-gauges">
    <div class="gauge-wrap">
      <svg width="76" height="76" viewBox="0 0 76 76">
        <circle cx="38" cy="38" r="31" fill="none" stroke="rgba(255,255,255,.18)" stroke-width="6"/>
        <circle cx="38" cy="38" r="31" fill="none" stroke="url(#bull-g)" stroke-width="6"
          stroke-linecap="round" stroke-dasharray="{C}" stroke-dashoffset="{bull_offset:.1f}"
          style="transform:rotate(-90deg);transform-origin:38px 38px;transition:stroke-dashoffset 1s ease;"/>
        <defs><linearGradient id="bull-g" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#00c896"/><stop offset="100%" stop-color="#4de8b8"/>
        </linearGradient></defs>
      </svg>
      <div class="gauge-inner">
        <div class="g-val" style="color:#00c896;">{bull_label}</div>
        <div class="g-lbl">CHG BULL</div>
      </div>
    </div>
    <div class="gauge-sep"></div>
    <div class="gauge-wrap">
      <svg width="76" height="76" viewBox="0 0 76 76">
        <circle cx="38" cy="38" r="31" fill="none" stroke="rgba(255,255,255,.18)" stroke-width="6"/>
        <circle cx="38" cy="38" r="31" fill="none" stroke="url(#bear-g)" stroke-width="6"
          stroke-linecap="round" stroke-dasharray="{C}" stroke-dashoffset="{bear_offset:.1f}"
          style="transform:rotate(-90deg);transform-origin:38px 38px;transition:stroke-dashoffset 1s ease;"/>
        <defs><linearGradient id="bear-g" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#ff6b6b"/><stop offset="100%" stop-color="#ff9090"/>
        </linearGradient></defs>
      </svg>
      <div class="gauge-inner">
        <div class="g-val" style="color:#ff6b6b;">{bear_label}</div>
        <div class="g-lbl">CHG BEAR</div>
      </div>
    </div>
  </div>
  <div class="h-mid">
    <div class="h-eyebrow">OI NET SIGNAL · {expiry} · SPOT ₹{underlying:,.0f}</div>
    <div class="h-signal" style="color:{dir_col};text-shadow:0 0 20px rgba({glow_rgb},.6),0 0 40px rgba({glow_rgb},.3);font-size:31.9px;font-weight:900;letter-spacing:1px;">{oi_dir}</div>
    <div class="h-sub">{oi_sig} · PCR <span style="color:{pcr_col};font-weight:700;">{pcr:.3f}</span></div>
    <div class="h-divider"></div>
    <div class="pill-row">
      <div class="pill-dot" style="background:#00c896;box-shadow:0 0 5px rgba(0,200,150,.5);"></div>
      <div class="pill-lbl">BULL STRENGTH</div>
      <div class="pill-track"><div class="pill-fill" style="width:{oi_bar_w}%;background:linear-gradient(90deg,#00c896,#4de8b8);"></div></div>
      <div class="pill-num" style="color:#00c896;">{bull_pct}%</div>
    </div>
    <div class="pill-row">
      <div class="pill-dot" style="background:#ff6b6b;box-shadow:0 0 5px rgba(255,107,107,.4);"></div>
      <div class="pill-lbl">BEAR STRENGTH</div>
      <div class="pill-track"><div class="pill-fill" style="width:{bear_bar_w}%;background:linear-gradient(90deg,#ff6b6b,#ff9090);"></div></div>
      <div class="pill-num" style="color:#ff6b6b;">{bear_pct}%</div>
    </div>
  </div>
  <div class="h-stats">
    <div class="h-stat-row">
      <div class="h-stat"><div class="h-stat-lbl">NIFTY SPOT</div><div class="h-stat-val" style="color:rgba(255,255,255,.85);">&#8377;{cp:,.2f}</div></div>
      <div class="h-stat"><div class="h-stat-lbl">ATM STRIKE</div><div class="h-stat-val" style="color:#00c896;">&#8377;{atm:,}</div></div>
      <div class="h-stat"><div class="h-stat-lbl">EXPIRY</div><div class="h-stat-val" style="color:#00c8e0;">{expiry}</div></div>
      <div class="h-stat"><div class="h-stat-lbl">PCR (OI)</div><div class="h-stat-val" style="color:{pcr_col};">{pcr:.3f}</div></div>
      <div class="h-stat"><div class="h-stat-lbl">MAX PAIN</div><div class="h-stat-val" style="color:#ffd166;">&#8377;{max_pain:,}</div></div>
    </div>
    <div class="h-stat-bottom">
      <div class="h-bias-row">
        <span class="h-chip" style="background:{b_bg};color:{b_col};border:1px solid {b_bdr};">{b_arrow}&nbsp;{bias}</span>
        <span class="h-chip" style="background:rgba(255,209,102,.1);color:#ffd166;border:1px solid rgba(255,209,102,.3);">{conf}&nbsp;CONF</span>
        <span class="h-score">Bull&nbsp;{bull_sc} · Bear&nbsp;{bear_sc} · Diff&nbsp;{diff:+d}</span>
      </div>
      <div class="h-ts" id="lastUpdatedTs">{ts}</div>
    </div>
  </div>
</div>
"""


# =================================================================
#  SECTION 5C -- OI DASHBOARD
# =================================================================

def build_oi_html(oc):
    ce  = oc["ce_chg"]; pe = oc["pe_chg"]
    expiry = oc["expiry"]; oi_dir = oc["oi_dir"]; oi_sig = oc["oi_sig"]; oi_cls = oc["oi_cls"]
    pcr = oc["pcr_oi"]; total_ce = oc["total_ce_oi"]; total_pe = oc["total_pe_oi"]
    max_ce_s = oc["max_ce_strike"]; max_pe_s = oc["max_pe_strike"]; max_pain = oc["max_pain"]
    underlying = oc["underlying"]

    dir_col = _cls_color(oi_cls); dir_bg = _cls_bg(oi_cls); dir_bdr = _cls_bdr(oi_cls)
    pcr_col = "#00c896" if pcr > 1.2 else ("#ff6b6b" if pcr < 0.7 else "#6480ff")

    ce_col   = "#00c896" if ce < 0 else "#ff6b6b"
    ce_label = "Call Unwinding ↓ (Bullish)" if ce < 0 else "Call Build-up ↑ (Bearish)"
    ce_fmt   = _fmt_chg_oi(ce)
    pe_col   = "#00c896" if pe > 0 else "#ff6b6b"
    pe_label = "Put Build-up ↑ (Bullish)" if pe > 0 else "Put Unwinding ↓ (Bearish)"
    pe_fmt   = _fmt_chg_oi(pe)

    bull_force = (abs(pe) if pe > 0 else 0) + (abs(ce) if ce < 0 else 0)
    bear_force = (abs(ce) if ce > 0 else 0) + (abs(pe) if pe < 0 else 0)
    net_diff       = bull_force - bear_force
    net_is_bullish = net_diff >= 0
    net_col        = "#00c896" if net_is_bullish else "#ff6b6b"
    net_label      = "Net Bullish Flow" if net_is_bullish else "Net Bearish Flow"
    net_fmt        = _fmt_chg_oi(net_diff)

    total_abs  = abs(ce) + abs(pe) or 1
    ce_pct     = round(abs(ce) / total_abs * 100)
    ce_bullish = ce < 0
    ce_pct_display = f"+{ce_pct}%" if ce_bullish else f"−{ce_pct}%"
    ce_bar_col = "#00c896" if ce_bullish else "#ff6b6b"
    pe_pct     = round(abs(pe) / total_abs * 100)
    pe_bullish = pe > 0
    pe_pct_display = f"+{pe_pct}%" if pe_bullish else f"−{pe_pct}%"
    pe_bar_col = "#00c896" if pe_bullish else "#ff6b6b"

    total_f    = bull_force + bear_force or 1
    bull_pct   = round(bull_force / total_f * 100)
    bear_pct   = 100 - bull_pct
    net_pct    = round(abs(net_diff) / total_f * 100) if total_f > 0 else 0
    net_pct    = max(5, min(95, net_pct))
    net_bar_col     = "#00c896" if net_is_bullish else "#ff6b6b"
    net_pct_display = f"+{net_pct}%" if net_is_bullish else f"−{net_pct}%"

    dir_card = f"""
<div style="display:flex;align-items:stretch;border:1px solid {dir_bdr};border-radius:14px;
  background:{dir_bg};overflow:hidden;margin-bottom:16px;">
  <div style="padding:18px 24px;min-width:200px;border-right:1px solid rgba(255,255,255,.07);
    display:flex;flex-direction:column;justify-content:center;flex-shrink:0;">
    <div style="font-size:12.3px;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,.28);margin-bottom:7px;">OI CHANGE DIRECTION</div>
    <div style="font-size:30.4px;font-weight:700;color:{dir_col};line-height:1.1;margin-bottom:5px;">{oi_dir}</div>
    <div style="font-size:15.2px;color:{dir_col};opacity:.7;">{oi_sig}</div>
    <div style="margin-top:10px;font-family:'DM Mono',monospace;font-size:14.5px;color:rgba(255,255,255,.3);">PCR &nbsp;<span style="color:{pcr_col};font-weight:700;">{pcr:.3f}</span></div>
  </div>
  <div style="display:flex;flex:1;align-items:stretch;">
    <div style="flex:1;display:flex;flex-direction:column;justify-content:center;padding:16px 20px;border-right:1px solid rgba(255,255,255,.05);gap:5px;">
      <div style="font-size:12.3px;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,.28);white-space:nowrap;">CE OI Change</div>
      <div style="font-family:'DM Mono',monospace;font-size:31.9px;font-weight:700;color:{ce_col};line-height:1;">{ce_fmt}</div>
      <div style="font-size:14.5px;color:rgba(255,255,255,.3);white-space:nowrap;">{ce_label}</div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:3px;">
        <div style="flex:1;height:5px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;"><div style="width:{ce_pct}%;height:100%;border-radius:3px;background:{ce_bar_col};"></div></div>
        <div style="font-family:'DM Mono',monospace;font-size:14.5px;font-weight:700;color:{ce_bar_col};min-width:38px;text-align:right;">{ce_pct_display}</div>
      </div>
    </div>
    <div style="flex:1;display:flex;flex-direction:column;justify-content:center;padding:16px 20px;border-right:1px solid rgba(255,255,255,.05);gap:5px;">
      <div style="font-size:12.3px;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,.28);white-space:nowrap;">PE OI Change</div>
      <div style="font-family:'DM Mono',monospace;font-size:31.9px;font-weight:700;color:{pe_col};line-height:1;">{pe_fmt}</div>
      <div style="font-size:14.5px;color:rgba(255,255,255,.3);white-space:nowrap;">{pe_label}</div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:3px;">
        <div style="flex:1;height:5px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;"><div style="width:{pe_pct}%;height:100%;border-radius:3px;background:{pe_bar_col};"></div></div>
        <div style="font-family:'DM Mono',monospace;font-size:14.5px;font-weight:700;color:{pe_bar_col};min-width:38px;text-align:right;">{pe_pct_display}</div>
      </div>
    </div>
    <div style="flex:1;display:flex;flex-direction:column;justify-content:center;padding:16px 20px;gap:5px;">
      <div style="font-size:12.3px;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,.28);white-space:nowrap;">Net OI Change</div>
      <div style="font-family:'DM Mono',monospace;font-size:31.9px;font-weight:700;color:{net_col};line-height:1;">{net_fmt}</div>
      <div style="font-size:14.5px;color:rgba(255,255,255,.3);white-space:nowrap;">{net_label}</div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:3px;">
        <div style="flex:1;height:5px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;"><div style="width:{net_pct}%;height:100%;border-radius:3px;background:{net_bar_col};box-shadow:0 0 8px {net_bar_col}66;"></div></div>
        <div style="font-family:'DM Mono',monospace;font-size:14.5px;font-weight:700;color:{net_bar_col};min-width:38px;text-align:right;">{net_pct_display}</div>
      </div>
    </div>
  </div>
</div>"""

    snapshot_table = (
        f'<div class="oi-ticker-table">'
        f'<div class="oi-ticker-hdr" style="background:rgba(100,128,255,.05);border-bottom:1px solid rgba(100,128,255,.1);">'
        f'<div class="oi-ticker-hdr-label" style="color:rgba(100,128,255,.8);">&#9632; OI SNAPSHOT</div>'
        f'<div class="oi-ticker-hdr-cell">Total CE OI</div><div class="oi-ticker-hdr-cell">Total PE OI</div>'
        f'<div class="oi-ticker-hdr-cell">PCR (OI)</div><div class="oi-ticker-hdr-cell">Max CE</div>'
        f'<div class="oi-ticker-hdr-cell">Max PE</div></div>'
        f'<div class="oi-ticker-row">'
        f'<div class="oi-ticker-metric">Snapshot</div>'
        f'<div class="oi-ticker-cell" style="color:#ff6b6b;font-family:\'DM Mono\',monospace;font-weight:700;font-size:21.8px;">{total_ce:,}</div>'
        f'<div class="oi-ticker-cell" style="color:#00c896;font-family:\'DM Mono\',monospace;font-weight:700;font-size:21.8px;">{total_pe:,}</div>'
        f'<div class="oi-ticker-cell" style="color:{pcr_col};font-family:\'DM Mono\',monospace;font-weight:700;font-size:21.8px;">{pcr:.3f}</div>'
        f'<div class="oi-ticker-cell" style="color:#ff6b6b;font-family:\'DM Mono\',monospace;font-weight:700;font-size:21.8px;">&#8377;{max_ce_s:,}</div>'
        f'<div class="oi-ticker-cell" style="color:#00c896;font-family:\'DM Mono\',monospace;font-weight:700;font-size:21.8px;">&#8377;{max_pe_s:,}</div>'
        f'</div>'
        f'<div style="display:flex;align-items:center;justify-content:space-between;padding:10px 18px;border-top:1px solid rgba(255,255,255,.04);flex-wrap:wrap;gap:10px;">'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<span style="font-size:13px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.3);">MAX PAIN</span>'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:26.1px;font-weight:700;color:#6480ff;">&#8377;{max_pain:,}</span>'
        f'<span style="font-size:14.5px;color:rgba(100,128,255,.6);">Option writers\' target</span></div>'
        f'</div></div>'
    )

    return (
        f'<div class="section"><div class="sec-title">OPEN INTEREST DASHBOARD'
        f'<span class="sec-sub">Spot ±500 pts · Expiry: {expiry} · Spot: &#8377;{underlying:,.2f}</span></div>'
        + dir_card + snapshot_table + "</div>"
    )


def build_key_levels_html(tech, oc):
    cp = tech["price"]; ss = tech["strong_sup"]; s1 = tech["support"]
    r1 = tech["resistance"]; sr = tech["strong_res"]; rng = sr - ss or 1
    def pct(v): return round(max(3, min(97, (v - ss) / rng * 100)), 1)
    cp_pct = pct(cp); pts_r = int(r1 - cp); pts_s = int(cp - s1)
    mp_html = ""
    if oc:
        mp_p = pct(oc["max_pain"]); mp = oc["max_pain"]
        mp_html = (f'<div class="kl-node" style="left:{mp_p}%;top:0;transform:translateX(-50%);">'
                   f'<div class="kl-dot" style="background:#6480ff;box-shadow:0 0 8px rgba(100,128,255,.5);margin:0 auto 4px;"></div>'
                   f'<div class="kl-lbl" style="color:#6480ff;">Max Pain</div>'
                   f'<div class="kl-val" style="color:#8aa0ff;">&#8377;{mp:,}</div></div>')
    return (
        f'<div class="section"><div class="sec-title">KEY LEVELS'
        f'<span class="sec-sub">1H Candles · Last 120 bars · Rounded to 25</span></div>'
        f'<div class="kl-zone-labels"><span style="color:#00c896;">SUPPORT ZONE</span><span style="color:#ff6b6b;">RESISTANCE ZONE</span></div>'
        f'<div style="position:relative;height:58px;">'
        f'<div class="kl-node" style="left:3%;bottom:0;transform:translateX(-50%);"><div class="kl-lbl" style="color:#00a07a;">Strong Sup</div><div class="kl-val" style="color:#00c896;">&#8377;{ss:,.0f}</div><div class="kl-dot" style="background:#00a07a;margin:5px auto 0;"></div></div>'
        f'<div class="kl-node" style="left:22%;bottom:0;transform:translateX(-50%);"><div class="kl-lbl" style="color:#00c896;">Support</div><div class="kl-val" style="color:#4de8b8;">&#8377;{s1:,.0f}</div><div class="kl-dot" style="background:#00c896;box-shadow:0 0 8px rgba(0,200,150,.5);margin:5px auto 0;"></div></div>'
        f'<div style="position:absolute;left:{cp_pct}%;bottom:6px;transform:translateX(-50%);background:linear-gradient(90deg,#00c896,#6480ff);color:#fff;font-size:15.9px;font-weight:700;padding:3px 14px;border-radius:20px;white-space:nowrap;box-shadow:0 2px 14px rgba(0,200,150,.35);z-index:10;">NOW &#8377;{cp:,.0f}</div>'
        f'<div class="kl-node" style="left:75%;bottom:0;transform:translateX(-50%);"><div class="kl-lbl" style="color:#ff6b6b;">Resistance</div><div class="kl-val" style="color:#ff9090;">&#8377;{r1:,.0f}</div><div class="kl-dot" style="background:#ff6b6b;box-shadow:0 0 8px rgba(255,107,107,.5);margin:5px auto 0;"></div></div>'
        f'<div class="kl-node" style="left:95%;bottom:0;transform:translateX(-50%);"><div class="kl-lbl" style="color:#cc4040;">Strong Res</div><div class="kl-val" style="color:#ff6b6b;">&#8377;{sr:,.0f}</div><div class="kl-dot" style="background:#cc4040;margin:5px auto 0;"></div></div>'
        f'</div>'
        f'<div class="kl-gradient-bar"><div class="kl-price-tick" style="left:{cp_pct}%;"></div></div>'
        f'<div style="position:relative;height:54px;">{mp_html}</div>'
        f'<div class="kl-dist-row">'
        f'<div class="kl-dist-box" style="border-color:rgba(255,107,107,.18);"><span style="color:var(--muted);">To Resistance</span><span style="color:#ff6b6b;font-weight:700;">+{pts_r:,} pts</span></div>'
        f'<div class="kl-dist-box" style="border-color:rgba(0,200,150,.18);"><span style="color:var(--muted);">To Support</span><span style="color:#00c896;font-weight:700;">-{pts_s:,} pts</span></div>'
        f'</div></div>'
    )


def build_strikes_html(oc):
    if not oc or (not oc["top_ce"] and not oc["top_pe"]): return ""
    def ce_rows(rows):
        o = ""
        for i, r in enumerate(rows, 1):
            o += (f"<tr><td>{i}</td><td><b>&#8377;{int(r['Strike']):,}</b></td>"
                  f"<td>{int(r['CE_OI']):,}</td>"
                  f"<td style=\"color:#00c8e0;font-weight:700;\">&#8377;{r['CE_LTP']:.2f}</td></tr>")
        return o
    def pe_rows(rows):
        o = ""
        for i, r in enumerate(rows, 1):
            o += (f"<tr><td>{i}</td><td><b>&#8377;{int(r['Strike']):,}</b></td>"
                  f"<td>{int(r['PE_OI']):,}</td>"
                  f"<td style=\"color:#ff6b6b;font-weight:700;\">&#8377;{r['PE_LTP']:.2f}</td></tr>")
        return o
    return (
        f'<div class="section"><div class="sec-title">TOP 5 STRIKES BY OPEN INTEREST'
        f'<span class="sec-sub">Spot ±500 pts · Top 5 CE + PE</span></div>'
        f'<div class="strikes-wrap">'
        f'<div><div class="strikes-head" style="color:#00c8e0;">▲ CALL Options (CE)</div>'
        f'<table class="s-table"><thead><tr><th>#</th><th>Strike</th><th>OI</th><th>LTP</th></tr></thead>'
        f'<tbody>{ce_rows(oc["top_ce"])}</tbody></table></div>'
        f'<div><div class="strikes-head" style="color:#ff6b6b;">▼ PUT Options (PE)</div>'
        f'<table class="s-table"><thead><tr><th>#</th><th>Strike</th><th>OI</th><th>LTP</th></tr></thead>'
        f'<tbody>{pe_rows(oc["top_pe"])}</tbody></table></div>'
        f'</div></div>'
    )


# =================================================================
#  SECTION 6 -- STRATEGIES WITH SMART POP ENGINE
# =================================================================

def make_payoff_svg(shape, bull_color="#00c896", bear_color="#ff6b6b"):
    w, h = 80, 50; mid = h // 2; pad = 8
    def pts(*coords): return " ".join(f"{x},{y}" for x, y in coords)
    shapes = {
        "long_call":         {"profit": [(pad,mid),(40,mid),(72,h-pad)],           "loss": [(pad,mid),(40,mid)]},
        "short_put":         {"profit": [(pad,h-pad),(40,mid),(72,mid)],           "loss": [(pad,mid),(40,mid)]},
        "bull_call_spread":  {"profit": [(pad,mid),(30,mid),(55,h-pad),(72,h-pad)],"loss": [(pad,mid),(30,mid)]},
        "bull_put_spread":   {"profit": [(pad,h-pad),(30,h-pad),(55,mid),(72,mid)],"loss": [(pad,mid),(72,mid)]},
        "call_ratio_back":   {"profit": [(pad,h-pad),(25,mid),(50,mid),(72,h-pad)],"loss": [(pad,mid),(25,mid),(50,mid)]},
        "long_synthetic":    {"profit": [(pad,mid),(72,h-pad)],                    "loss": [(pad,pad),(40,mid)]},
        "range_forward":     {"profit": [(pad,mid),(30,mid),(55,h-pad),(72,h-pad)],"loss": [(pad,pad),(30,pad),(55,mid)]},
        "bull_butterfly":    {"profit": [(pad,mid),(36,h-pad),(54,mid)],           "loss": [(pad,mid),(36,mid),(54,mid),(72,mid)]},
        "bull_condor":       {"profit": [(pad,mid),(28,mid),(36,h-pad),(50,h-pad),(58,mid),(72,mid)],"loss": [(pad,mid),(72,mid)]},
        "short_call":        {"profit": [(pad,mid),(40,mid),(72,pad)],             "loss": [(pad,mid),(40,mid)]},
        "long_put":          {"profit": [(pad,h-pad),(40,mid),(72,mid)],           "loss": [(pad,mid),(40,mid)]},
        "bear_call_spread":  {"profit": [(pad,h-pad),(30,h-pad),(55,mid),(72,mid)],"loss": [(pad,mid),(72,mid)]},
        "bear_put_spread":   {"profit": [(pad,mid),(30,mid),(55,h-pad),(72,h-pad)],"loss": [(pad,mid),(30,mid)]},
        "put_ratio_back":    {"profit": [(pad,h-pad),(25,mid),(50,mid),(72,pad)],  "loss": [(pad,mid),(25,mid),(50,mid)]},
        "short_synthetic":   {"profit": [(pad,mid),(72,pad)],                     "loss": [(pad,h-pad),(40,mid)]},
        "risk_reversal":     {"profit": [(pad,h-pad),(36,mid),(72,pad)],           "loss": [(pad,mid),(36,mid)]},
        "bear_butterfly":    {"profit": [(pad,mid),(36,pad),(54,mid)],             "loss": [(pad,mid),(36,mid),(54,mid),(72,mid)]},
        "bear_condor":       {"profit": [(pad,mid),(28,mid),(36,pad),(50,pad),(58,mid),(72,mid)],"loss": [(pad,mid),(72,mid)]},
        "long_straddle":     {"profit": [(pad,h-pad),(36,mid),(54,mid),(72,h-pad)],"loss": [(pad,mid),(36,mid),(54,mid),(72,mid)]},
        "short_straddle":    {"profit": [(pad,mid),(36,mid),(54,mid),(72,mid)],    "loss": [(pad,h-pad),(36,mid),(54,mid),(72,h-pad)]},
        "long_strangle":     {"profit": [(pad,h-pad),(30,mid),(50,mid),(72,h-pad)],"loss": [(pad,mid),(30,mid),(50,mid),(72,mid)]},
        "short_strangle":    {"profit": [(pad,mid),(30,mid),(50,mid),(72,mid)],    "loss": [(pad,h-pad),(30,mid),(50,mid),(72,h-pad)]},
        "jade_lizard":       {"profit": [(pad,pad),(30,mid),(55,mid),(72,mid)],    "loss": [(pad,mid),(30,mid)]},
        "reverse_jade":      {"profit": [(pad,mid),(30,mid),(55,h-pad),(72,h-pad)],"loss": [(pad,pad),(30,mid)]},
        "call_ratio_spread": {"profit": [(pad,mid),(36,h-pad),(72,mid)],           "loss": [(pad,mid),(36,mid),(72,pad)]},
        "put_ratio_spread":  {"profit": [(pad,mid),(36,h-pad),(72,mid)],           "loss": [(pad,pad),(36,mid),(72,mid)]},
        "batman":            {"profit": [(pad,mid),(20,h-pad),(36,mid),(54,mid),(68,h-pad),(72,mid)],"loss": [(pad,mid),(72,mid)]},
        "long_iron_fly":     {"profit": [(pad,pad),(36,mid),(54,mid),(72,pad)],    "loss": [(pad,mid),(36,h-pad),(54,h-pad),(72,mid)]},
        "short_iron_fly":    {"profit": [(pad,mid),(36,h-pad),(54,h-pad),(72,mid)],"loss": [(pad,pad),(36,mid),(54,mid),(72,pad)]},
        "double_fly":        {"profit": [(pad,mid),(20,h-pad),(36,mid),(54,mid),(68,h-pad),(72,mid)],"loss": [(pad,mid),(72,mid)]},
        "long_iron_condor":  {"profit": [(pad,pad),(24,mid),(36,mid),(54,mid),(66,pad),(72,pad)],   "loss": [(pad,mid),(24,mid),(66,mid),(72,mid)]},
        "short_iron_condor": {"profit": [(pad,mid),(24,h-pad),(36,h-pad),(54,h-pad),(66,mid),(72,mid)],"loss": [(pad,pad),(72,pad)]},
        "double_condor":     {"profit": [(pad,mid),(20,h-pad),(36,mid),(54,mid),(68,h-pad),(72,mid)],"loss": [(pad,mid),(72,mid)]},
        "call_calendar":     {"profit": [(pad,mid),(36,h-pad),(54,mid),(72,mid)],  "loss": [(pad,mid),(36,mid),(54,mid),(72,mid)]},
        "put_calendar":      {"profit": [(pad,mid),(36,h-pad),(54,mid),(72,mid)],  "loss": [(pad,mid),(36,mid),(54,mid),(72,mid)]},
        "diagonal_calendar": {"profit": [(pad,mid),(30,h-pad),(55,mid),(72,mid)],  "loss": [(pad,mid),(30,mid),(55,mid),(72,mid)]},
        "call_butterfly":    {"profit": [(pad,mid),(36,h-pad),(54,mid),(72,mid)],  "loss": [(pad,mid),(72,mid)]},
        "put_butterfly":     {"profit": [(pad,mid),(36,h-pad),(54,mid),(72,mid)],  "loss": [(pad,mid),(72,mid)]},
    }
    s = shapes.get(shape, {"profit": [(pad,mid),(72,mid)], "loss": []})
    profit_pts = pts(*s["profit"]); loss_pts = pts(*s["loss"]) if s["loss"] else ""
    def area(coords, is_p):
        if not coords: return ""
        col = bull_color if is_p else bear_color
        d = f"M {coords[0][0]},{mid} " + " ".join(f"L {x},{y}" for x, y in coords) + f" L {coords[-1][0]},{mid} Z"
        return f'<path d="{d}" fill="{col}" fill-opacity="0.18"/>'
    svg = (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
           f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="rgba(255,255,255,.15)" stroke-width="1"/>'
           f'<line x1="{pad}" y1="{mid}" x2="{w-pad}" y2="{mid}" stroke="rgba(255,255,255,.15)" stroke-width="1"/>'
           f'{area(s["profit"],True)}{area(s["loss"],False)}'
           f'<polyline points="{profit_pts}" fill="none" stroke="{bull_color}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>')
    if loss_pts:
        svg += f'<polyline points="{loss_pts}" fill="none" stroke="{bear_color}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round" stroke-dasharray="3,2"/>'
    return svg + '</svg>'


STRATEGIES_DATA = {
    "bullish": [
        {"name":"Long Call","shape":"long_call","risk":"Limited","reward":"Unlimited","legs":"BUY CALL (ATM)","desc":"Buy a call option. Profits as market rises above strike. Risk is limited to premium paid."},
        {"name":"Short Put","shape":"short_put","risk":"Moderate","reward":"Limited","legs":"SELL PUT (OTM)","desc":"Sell a put option below market. Collect premium. Profit if market stays above strike."},
        {"name":"Bull Call Spread","shape":"bull_call_spread","risk":"Limited","reward":"Limited","legs":"BUY CALL (Low) · SELL CALL (High)","desc":"Buy lower call, sell higher call. Reduces cost; caps profit at upper strike."},
        {"name":"Bull Put Spread","shape":"bull_put_spread","risk":"Limited","reward":"Limited","legs":"SELL PUT (High) · BUY PUT (Low)","desc":"Sell higher put, buy lower put. Credit received upfront. Profit if market stays above higher strike."},
        {"name":"Call Ratio Back Spread","shape":"call_ratio_back","risk":"Limited","reward":"Unlimited","legs":"SELL 1 CALL (Low) · BUY 2 CALLS (High)","desc":"Sell fewer calls, buy more higher calls. Benefits from a big upside move."},
        {"name":"Long Synthetic","shape":"long_synthetic","risk":"High","reward":"Unlimited","legs":"BUY CALL (ATM) · SELL PUT (ATM)","desc":"Replicates owning the underlying. Unlimited profit potential with high risk."},
        {"name":"Range Forward","shape":"range_forward","risk":"Limited","reward":"Limited","legs":"BUY CALL (High) · SELL PUT (Low)","desc":"Collar-like structure. Profit in a range. Used to hedge existing positions."},
        {"name":"Bull Butterfly","shape":"bull_butterfly","risk":"Limited","reward":"Limited","legs":"BUY Low CALL · SELL 2 Mid CALL · BUY High CALL","desc":"Max profit at middle strike. Low cost strategy for moderate bullish view."},
        {"name":"Bull Condor","shape":"bull_condor","risk":"Limited","reward":"Limited","legs":"BUY Low · SELL Mid-Low · SELL Mid-High · BUY High","desc":"Four-leg bullish strategy. Profit in a range above current price."},
    ],
    "bearish": [
        {"name":"Short Call","shape":"short_call","risk":"Unlimited","reward":"Limited","legs":"SELL CALL (ATM/OTM)","desc":"Sell a call option above market. Collect premium. Profit if market falls or stays below strike."},
        {"name":"Long Put","shape":"long_put","risk":"Limited","reward":"High","legs":"BUY PUT (ATM)","desc":"Buy a put option. Profits as market falls below strike. Risk is limited to premium paid."},
        {"name":"Bear Call Spread","shape":"bear_call_spread","risk":"Limited","reward":"Limited","legs":"SELL CALL (Low) · BUY CALL (High)","desc":"Sell lower call, buy higher call. Credit received. Profit if market stays below lower strike."},
        {"name":"Bear Put Spread","shape":"bear_put_spread","risk":"Limited","reward":"Limited","legs":"BUY PUT (High) · SELL PUT (Low)","desc":"Buy higher put, sell lower put. Cheaper bearish bet with capped profit."},
        {"name":"Put Ratio Back Spread","shape":"put_ratio_back","risk":"Limited","reward":"High","legs":"SELL 1 PUT (High) · BUY 2 PUTS (Low)","desc":"Sell fewer puts, buy more lower puts. Benefits from a big downside move."},
        {"name":"Short Synthetic","shape":"short_synthetic","risk":"High","reward":"High","legs":"SELL CALL (ATM) · BUY PUT (ATM)","desc":"Replicates shorting the underlying. Profit as market falls. High risk."},
        {"name":"Risk Reversal","shape":"risk_reversal","risk":"High","reward":"High","legs":"BUY PUT (Low) · SELL CALL (High)","desc":"Protect downside while giving up upside. Common hedging structure."},
        {"name":"Bear Butterfly","shape":"bear_butterfly","risk":"Limited","reward":"Limited","legs":"BUY Low PUT · SELL 2 Mid PUT · BUY High PUT","desc":"Max profit at middle strike. Low cost strategy for moderate bearish view."},
        {"name":"Bear Condor","shape":"bear_condor","risk":"Limited","reward":"Limited","legs":"BUY High · SELL Mid-High · SELL Mid-Low · BUY Low","desc":"Four-leg bearish strategy. Profit in a range below current price."},
    ],
    "nondirectional": [
        {"name":"Long Straddle","shape":"long_straddle","risk":"Limited","reward":"Unlimited","legs":"BUY CALL (ATM) + BUY PUT (ATM)","desc":"Buy both ATM call and put. Profit from big move in either direction. Best before events."},
        {"name":"Short Straddle","shape":"short_straddle","risk":"Unlimited","reward":"Limited","legs":"SELL CALL (ATM) + SELL PUT (ATM)","desc":"Sell both ATM call and put. Profit from low volatility. High risk unlimited loss."},
        {"name":"Long Strangle","shape":"long_strangle","risk":"Limited","reward":"Unlimited","legs":"BUY OTM CALL + BUY OTM PUT","desc":"Buy OTM call and put. Cheaper than straddle. Needs bigger move to profit."},
        {"name":"Short Strangle","shape":"short_strangle","risk":"Unlimited","reward":"Limited","legs":"SELL OTM CALL + SELL OTM PUT","desc":"Sell OTM call and put. Wider profit range than short straddle. Still high risk."},
        {"name":"Jade Lizard","shape":"jade_lizard","risk":"Limited","reward":"Limited","legs":"SELL OTM PUT + SELL CALL SPREAD","desc":"No upside risk. Collect premium. Bearish but risk-defined."},
        {"name":"Reverse Jade Lizard","shape":"reverse_jade","risk":"Limited","reward":"Limited","legs":"SELL OTM CALL + SELL PUT SPREAD","desc":"No downside risk. Collect premium. Bullish but risk-defined."},
        {"name":"Call Ratio Spread","shape":"call_ratio_spread","risk":"Unlimited","reward":"Limited","legs":"BUY 1 CALL (Low) · SELL 2 CALLS (High)","desc":"Sell more calls than bought. Credit or debit. Risk if big upside move occurs."},
        {"name":"Put Ratio Spread","shape":"put_ratio_spread","risk":"Unlimited","reward":"Limited","legs":"BUY 1 PUT (High) · SELL 2 PUTS (Low)","desc":"Sell more puts than bought. Risk if big downside move occurs."},
        {"name":"Batman Strategy","shape":"batman","risk":"Limited","reward":"Limited","legs":"BUY 2 CALLS + SELL 4 CALLS + BUY 2 CALLS","desc":"Double butterfly. Two profit peaks. Complex strategy for range-bound markets."},
        {"name":"Long Iron Fly","shape":"long_iron_fly","risk":"Limited","reward":"Limited","legs":"BUY CALL · BUY PUT · SELL ATM CALL · SELL ATM PUT","desc":"Debit iron fly. Profit from a big move. Max loss if price stays at ATM."},
        {"name":"Short Iron Fly","shape":"short_iron_fly","risk":"Limited","reward":"Limited","legs":"SELL CALL · SELL PUT · BUY OTM CALL · BUY OTM PUT","desc":"Credit iron fly. Max profit at ATM. Common non-directional strategy."},
        {"name":"Double Fly","shape":"double_fly","risk":"Limited","reward":"Limited","legs":"TWO BUTTERFLY SPREADS","desc":"Two butterfly spreads at different strikes. Two profit peaks."},
        {"name":"Long Iron Condor","shape":"long_iron_condor","risk":"Limited","reward":"Limited","legs":"BUY CALL SPREAD + BUY PUT SPREAD","desc":"Debit condor. Profit from a big move. Opposite of short iron condor."},
        {"name":"Short Iron Condor","shape":"short_iron_condor","risk":"Limited","reward":"Limited","legs":"SELL CALL SPREAD + SELL PUT SPREAD","desc":"Collect premium from both sides. Profit if price stays in a range."},
        {"name":"Double Condor","shape":"double_condor","risk":"Limited","reward":"Limited","legs":"TWO CONDOR SPREADS","desc":"Two condor spreads. Wider profit range. Complex multi-leg strategy."},
        {"name":"Call Calendar","shape":"call_calendar","risk":"Limited","reward":"Limited","legs":"SELL NEAR-TERM CALL · BUY FAR-TERM CALL","desc":"Profit from time decay difference. Best when price stays near strike."},
        {"name":"Put Calendar","shape":"put_calendar","risk":"Limited","reward":"Limited","legs":"SELL NEAR-TERM PUT · BUY FAR-TERM PUT","desc":"Profit from time decay. Best when price stays near strike on expiry."},
        {"name":"Diagonal Calendar","shape":"diagonal_calendar","risk":"Limited","reward":"Limited","legs":"SELL NEAR CALL/PUT · BUY FAR DIFF STRIKE","desc":"Calendar spread with different strikes. Combines time and price movement."},
        {"name":"Call Butterfly","shape":"call_butterfly","risk":"Limited","reward":"Limited","legs":"BUY Low CALL · SELL 2 Mid CALL · BUY High CALL","desc":"Max profit at middle strike using calls only. Low net debit strategy."},
        {"name":"Put Butterfly","shape":"put_butterfly","risk":"Limited","reward":"Limited","legs":"BUY High PUT · SELL 2 Mid PUT · BUY Low PUT","desc":"Max profit at middle strike using puts only. Low net debit strategy."},
    ],
}


def build_strategies_html(oc_analysis, tech=None, md=None, multi_expiry_analyzed=None, expiry_list=None):
    spot       = oc_analysis["underlying"]   if oc_analysis else 23000
    atm        = oc_analysis["atm_strike"]   if oc_analysis else 23000
    pcr        = oc_analysis["pcr_oi"]       if oc_analysis else 1.0
    mp         = oc_analysis["max_pain"]     if oc_analysis else 23000
    max_ce_s   = oc_analysis["max_ce_strike"] if oc_analysis else atm + 200
    max_pe_s   = oc_analysis["max_pe_strike"] if oc_analysis else atm - 200
    strikes_json = json.dumps(oc_analysis.get("strikes_data", [])) if oc_analysis else "[]"

    # Build multi-expiry data for dropdown
    all_expiry_js = {}
    if multi_expiry_analyzed and expiry_list:
        for exp in expiry_list:
            oc_e = multi_expiry_analyzed.get(exp)
            if not oc_e:
                continue
            all_expiry_js[exp] = {
                "spot":        round(oc_e["underlying"], 2),
                "atm":         oc_e["atm_strike"],
                "pcr":         round(oc_e["pcr_oi"], 3),
                "maxCeStrike": oc_e["max_ce_strike"],
                "maxPeStrike": oc_e["max_pe_strike"],
                "support":     round(tech["support"], 2) if tech else spot - 150,
                "resistance":  round(tech["resistance"], 2) if tech else spot + 150,
                "strongSup":   round(tech["strong_sup"], 2) if tech else spot - 300,
                "strongRes":   round(tech["strong_res"], 2) if tech else spot + 300,
                "strikes":     oc_e.get("strikes_data", []),
            }
    all_expiry_json = json.dumps(all_expiry_js)
    expiry_opts_html = ""
    if expiry_list:
        first_with_data = True
        for exp in expiry_list:
            has_data = exp in (all_expiry_js or {})
            sel = "selected" if (has_data and first_with_data) else ""
            if has_data and first_with_data:
                first_with_data = False
            # Show all expiries; disable ones with no data
            if has_data:
                expiry_opts_html += f'<option value="{exp}" {sel}>{exp}</option>\n'
            else:
                expiry_opts_html += f'<option value="{exp}" disabled style="color:rgba(255,255,255,0.3);">{exp} (no data)</option>\n'
    else:
        expiry_opts_html = f'<option value="">{oc_analysis["expiry"] if oc_analysis else "N/A"}</option>'
    
    support     = tech["support"]    if tech else spot - 150
    resistance  = tech["resistance"] if tech else spot + 150
    strong_sup  = tech["strong_sup"] if tech else spot - 300
    strong_res  = tech["strong_res"] if tech else spot + 300

    bias        = md["bias"]       if md else "SIDEWAYS"
    conf        = md["confidence"] if md else "MEDIUM"
    bull_sc     = md["bull"]       if md else 4
    bear_sc     = md["bear"]       if md else 4

    def render_cards(strats, cat):
        cards = ""
        for idx, s in enumerate(strats):
            svg  = make_payoff_svg(s["shape"])
            rc   = "#00c896" if s["risk"]   in ("Limited","Low") else ("#ff6b6b" if s["risk"] in ("Unlimited","High") else "#6480ff")
            rwc  = "#00c896" if s["reward"] == "Unlimited" else "#6480ff"
            cid  = f"sc_{cat}_{idx}"
            cards += (
                f'<div class="sc-card" data-cat="{cat}" data-shape="{s["shape"]}" '
                f'data-name="{s["name"]}" data-legs="{s["legs"]}" '
                f'data-risk="{s["risk"]}" data-reward="{s["reward"]}" '
                f'data-margin-mult="{s.get("margin_mult",1.0)}" data-lot-size="{s.get("lot_size",65)}" id="{cid}">'
                f'<div class="sc-pop-badge" id="pop_{cid}">—%</div>'
                f'<div class="sc-svg">{svg}</div>'
                f'<div class="sc-body">'
                f'<div class="sc-name">{s["name"]}</div>'
                f'<div class="sc-legs">{s["legs"]}</div>'
                f'<div class="sc-tags">'
                f'<span class="sc-tag" style="color:{rc};border-color:{rc}40;">Risk: {s["risk"]}</span>'
                f'<span class="sc-tag" style="color:{rwc};border-color:{rwc}40;">Reward: {s["reward"]}</span>'
                f'</div></div>'
                f'<div class="sc-detail" id="detail_{cid}">'
                f'<div class="sc-desc">{s["desc"]}</div>'
                f'<div class="sc-metrics-live" id="metrics_{cid}">'
                f'<div class="sc-loading">&#9685; Calculating metrics...</div>'
                f'</div></div></div>'
            )
        return cards

    bull_cards = render_cards(STRATEGIES_DATA["bullish"],       "bullish")
    bear_cards = render_cards(STRATEGIES_DATA["bearish"],       "bearish")
    nd_cards   = render_cards(STRATEGIES_DATA["nondirectional"],"nondirectional")

    return f"""
<div class="section" id="strat">
  <div class="sec-title">STRATEGIES REFERENCE
    <span class="sec-sub">Smart PoP · Live S/R + OI Walls + Market Bias · Click to expand</span>
  </div>

  <div id="smartPopLegend" style="
    background:linear-gradient(135deg,rgba(100,128,255,.08),rgba(0,200,150,.06));
    border:1px solid rgba(100,128,255,.2);border-radius:14px;padding:14px 18px;
    margin-bottom:18px;display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start;">
    <div style="flex:0 0 auto;">
      <div style="font-size:13px;font-weight:700;letter-spacing:2px;color:#8aa0ff;text-transform:uppercase;margin-bottom:8px;">&#9889; SMART POP ENGINE</div>
      <div style="font-size:14.5px;color:rgba(255,255,255,.45);line-height:1.8;">
        PoP = Base 50% + Bias Alignment + S/R Zone + OI Walls + PCR Weight
      </div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;flex:1;">
      <div id="legendBias" style="background:rgba(0,0,0,.2);border-radius:8px;padding:8px 12px;min-width:130px;">
        <div style="font-size:11.6px;color:rgba(255,255,255,.3);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;">A · BIAS</div>
        <div id="legendBiasVal" style="font-family:'DM Mono',monospace;font-size:17.4px;font-weight:700;color:#8aa0ff;">—</div>
      </div>
      <div id="legendSR" style="background:rgba(0,0,0,.2);border-radius:8px;padding:8px 12px;min-width:160px;">
        <div style="font-size:11.6px;color:rgba(255,255,255,.3);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;">B · S/R ZONE</div>
        <div id="legendSRVal" style="font-family:'DM Mono',monospace;font-size:17.4px;font-weight:700;color:#ffd166;">—</div>
      </div>
      <div id="legendOI" style="background:rgba(0,0,0,.2);border-radius:8px;padding:8px 12px;min-width:160px;">
        <div style="font-size:11.6px;color:rgba(255,255,255,.3);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;">C · OI WALLS</div>
        <div id="legendOIVal" style="font-family:'DM Mono',monospace;font-size:17.4px;font-weight:700;color:#00c8e0;">—</div>
      </div>
      <div id="legendPCR" style="background:rgba(0,0,0,.2);border-radius:8px;padding:8px 12px;min-width:120px;">
        <div style="font-size:11.6px;color:rgba(255,255,255,.3);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;">D · PCR</div>
        <div id="legendPCRVal" style="font-family:'DM Mono',monospace;font-size:17.4px;font-weight:700;color:#00c896;">—</div>
      </div>
      <div id="legendRec" style="background:rgba(0,200,150,.06);border:1px solid rgba(0,200,150,.2);border-radius:8px;padding:8px 12px;min-width:180px;">
        <div style="font-size:11.6px;color:rgba(0,200,150,.6);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;">&#9733; TOP STRATEGY</div>
        <div id="legendRecVal" style="font-size:15.9px;font-weight:700;color:#00c896;">Calculating...</div>
      </div>
    </div>
  </div>

  <div class="sc-tabs">
    <button class="sc-tab active" onclick="filterStrat('bullish',this)"
      style="border-color:#00c896;color:#00c896;background:rgba(0,200,150,.12);">
      &#9650; BULLISH <span class="sc-cnt" style="background:#00c896;">9</span>
    </button>
    <button class="sc-tab" onclick="filterStrat('bearish',this)"
      style="border-color:rgba(255,255,255,.15);color:rgba(255,255,255,.5);">
      &#9660; BEARISH <span class="sc-cnt" style="background:#ff6b6b;">9</span>
    </button>
    <button class="sc-tab" onclick="filterStrat('nondirectional',this)"
      style="border-color:rgba(255,255,255,.15);color:rgba(255,255,255,.5);">
      &#8596; NON-DIRECTIONAL <span class="sc-cnt" style="background:#6480ff;">20</span>
    </button>
    <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
      <span style="font-size:13px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
                   color:rgba(255,209,102,.7);">&#128197; EXPIRY DATE</span>
      <select id="expiryDropdown" onchange="switchExpiry(this.value)"
        style="appearance:none;-webkit-appearance:none;
               background:linear-gradient(135deg,rgba(245,197,24,.12),rgba(200,155,10,.06));
               border:1px solid rgba(245,197,24,.45);border-radius:8px;
               color:#ffd166;font-family:'DM Mono',monospace;font-size:15.9px;font-weight:700;
               padding:7px 28px 7px 12px;cursor:pointer;outline:none;letter-spacing:.5px;">
        {expiry_opts_html}
      </select>
    </div>
  </div>
  <div class="sc-grid" id="sc-grid">
    {bull_cards}{bear_cards}{nd_cards}
  </div>
</div>

<script>
const OC={{
  spot:        {spot:.2f},
  atm:         {atm},
  pcr:         {pcr:.3f},
  maxPain:     {mp},
  maxCeStrike: {max_ce_s},
  maxPeStrike: {max_pe_s},
  support:     {support:.2f},
  resistance:  {resistance:.2f},
  strongSup:   {strong_sup:.2f},
  strongRes:   {strong_res:.2f},
  bias:        "{bias}",
  biasConf:    "{conf}",
  bullScore:   {bull_sc},
  bearScore:   {bear_sc},
  strikes:     {strikes_json},
  lotSize:     65
}};

const STRIKE_MAP={{}};
OC.strikes.forEach(s=>{{ STRIKE_MAP[s.strike]=s; }});

function smartPoP(shape, cat) {{
  const spot=OC.spot, pcr=OC.pcr;
  const sup=OC.support, res=OC.resistance;
  const ssup=OC.strongSup, sres=OC.strongRes;
  const maxCE=OC.maxCeStrike, maxPE=OC.maxPeStrike;
  const bias=OC.bias, conf=OC.biasConf;
  const rangeSize = res - sup || 200;
  const confMult = conf==="HIGH" ? 1.25 : conf==="LOW" ? 0.6 : 1.0;
  let biasAdj = 0;
  if (cat === "bullish") {{ biasAdj = bias==="BULLISH" ? 15 : bias==="BEARISH" ? -15 : 0; }}
  else if (cat === "bearish") {{ biasAdj = bias==="BEARISH" ? 15 : bias==="BULLISH" ? -15 : 0; }}
  else {{ biasAdj = bias==="SIDEWAYS" ? 8 : (OC.bullScore===OC.bearScore ? 5 : -5); }}
  biasAdj = biasAdj * confMult;
  let srAdj = 0;
  const distToSup = spot - sup; const distToRes = res - spot;
  if (cat === "bullish") {{
    if (distToSup >= 0 && distToSup <= rangeSize * 0.25) {{ srAdj = 10; }}
    else if (distToSup >= 0 && distToSup <= rangeSize * 0.5) {{ srAdj = 5; }}
    else if (distToRes >= 0 && distToRes <= rangeSize * 0.2) {{ srAdj = -10; }}
    else if (spot > res) {{ srAdj = -8; }} else {{ srAdj = 2; }}
  }} else if (cat === "bearish") {{
    if (distToRes >= 0 && distToRes <= rangeSize * 0.25) {{ srAdj = 10; }}
    else if (distToRes >= 0 && distToRes <= rangeSize * 0.5) {{ srAdj = 5; }}
    else if (distToSup >= 0 && distToSup <= rangeSize * 0.2) {{ srAdj = -10; }}
    else if (spot < sup) {{ srAdj = -8; }} else {{ srAdj = 2; }}
  }} else {{
    const midRange = (sup + res) / 2; const distFromMid = Math.abs(spot - midRange); const halfRange = rangeSize / 2;
    if (distFromMid <= halfRange * 0.3) {{ srAdj = 10; }} else if (distFromMid <= halfRange * 0.6) {{ srAdj = 5; }} else {{ srAdj = -5; }}
  }}
  srAdj = srAdj * confMult;
  let oiAdj = 0;
  const distAboveMaxPE = spot - maxPE; const distBelowMaxCE = maxCE - spot;
  if (cat === "bullish") {{
    if (distAboveMaxPE > 0 && distAboveMaxPE < 150) {{ oiAdj += 8; }} else if (distAboveMaxPE > 150) {{ oiAdj += 4; }} else {{ oiAdj -= 8; }}
    if (distBelowMaxCE > 200) {{ oiAdj += 5; }} else if (distBelowMaxCE < 100) {{ oiAdj -= 7; }}
  }} else if (cat === "bearish") {{
    if (distBelowMaxCE > 0 && distBelowMaxCE < 150) {{ oiAdj += 8; }} else if (distBelowMaxCE > 150) {{ oiAdj += 4; }} else {{ oiAdj -= 8; }}
    if (distAboveMaxPE > 200) {{ oiAdj += 5; }} else if (distAboveMaxPE < 100) {{ oiAdj -= 7; }}
  }} else {{
    if (distAboveMaxPE > 0 && distBelowMaxCE > 0) {{
      const oiRange = maxCE - maxPE || 200; const midOI = (maxPE + maxCE) / 2; const distFromOIMid = Math.abs(spot - midOI);
      if (distFromOIMid < oiRange * 0.3) {{ oiAdj = 10; }} else {{ oiAdj = 4; }}
    }} else {{ oiAdj = -5; }}
  }}
  let pcrAdj = 0;
  if (cat === "bullish") {{ pcrAdj = pcr > 1.5 ? 8 : pcr > 1.2 ? 6 : pcr > 1.0 ? 3 : pcr < 0.7 ? -8 : pcr < 0.9 ? -4 : 0; }}
  else if (cat === "bearish") {{ pcrAdj = pcr < 0.5 ? 8 : pcr < 0.7 ? 6 : pcr < 0.9 ? 3 : pcr > 1.3 ? -8 : pcr > 1.1 ? -4 : 0; }}
  else {{ pcrAdj = (pcr >= 0.85 && pcr <= 1.15) ? 6 : (pcr >= 0.7 && pcr <= 1.3) ? 3 : -4; }}
  let stratAdj = 0;
  if (shape.includes('spread') || shape.includes('condor') || shape.includes('butterfly')) {{ stratAdj = 2; }}
  if (shape === 'short_straddle' || shape === 'short_strangle') {{ stratAdj = bias === 'SIDEWAYS' ? 8 : -10; }}
  if (shape === 'long_straddle' || shape === 'long_strangle') {{ stratAdj = bias === 'SIDEWAYS' ? -8 : 8; }}
  if ((shape === 'short_iron_condor' || shape === 'short_iron_fly') && bias === 'SIDEWAYS') {{ stratAdj = 10; }}
  const rawPoP = 50 + biasAdj + srAdj + oiAdj + pcrAdj + stratAdj;
  return {{ pop: Math.min(95, Math.max(5, Math.round(rawPoP))), biasAdj: Math.round(biasAdj), srAdj: Math.round(srAdj), oiAdj: Math.round(oiAdj), pcrAdj: Math.round(pcrAdj), stratAdj: Math.round(stratAdj) }};
}}

function normCDF(x) {{
  const a1=0.254829592,a2=-0.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=0.3275911;
  const sign=x<0?-1:1; x=Math.abs(x);
  const t=1/(1+p*x); const y=1-(((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*Math.exp(-x*x);
  return 0.5*(1+sign*y);
}}
function getATMLTP(type) {{
  const row=STRIKE_MAP[OC.atm]||OC.strikes.reduce((b,s)=>Math.abs(s.strike-OC.atm)<Math.abs(b.strike-OC.atm)?s:b,OC.strikes[0]||{{strike:OC.atm,ce_ltp:0,pe_ltp:0,ce_iv:15,pe_iv:15}});
  return type==='ce'?row.ce_ltp:row.pe_ltp;
}}
function getOTM(type,offset) {{
  const t=type==='ce'?OC.atm+offset*50:OC.atm-offset*50;
  const row=STRIKE_MAP[t]||OC.strikes.reduce((b,s)=>Math.abs(s.strike-t)<Math.abs(b.strike-t)?s:b,OC.strikes[0]||{{strike:OC.atm,ce_ltp:0,pe_ltp:0,ce_iv:15,pe_iv:15}});
  return {{strike:row.strike||t,ltp:type==='ce'?row.ce_ltp:row.pe_ltp,iv:type==='ce'?row.ce_iv:row.pe_iv}};
}}

function getGreeks(type, strike) {{
  // Returns {{delta, theta, vega}} for a given strike and option type (ce/pe)
  // Uses STRIKE_MAP which now includes BS-computed greeks from Python
  const row = STRIKE_MAP[strike] || OC.strikes.reduce(
    (b, s) => Math.abs(s.strike - strike) < Math.abs(b.strike - strike) ? s : b,
    OC.strikes[0] || {{strike: OC.atm, ce_delta:0.5, pe_delta:-0.5, ce_theta:-0.1, pe_theta:-0.1, ce_vega:0.05, pe_vega:0.05}}
  );
  if (type === 'ce') return {{ delta: row.ce_delta||0.5,  theta: row.ce_theta||0, vega: row.ce_vega||0  }};
  else               return {{ delta: row.pe_delta||-0.5, theta: row.pe_theta||0, vega: row.pe_vega||0 }};
}}

function calcMetrics(shape, smartPop) {{
  const spot   = OC.spot, atm = OC.atm;
  const lotSz  = OC.lotSize;
  const ce_atm = getATMLTP('ce'), pe_atm = getATMLTP('pe');

  const co1 = getOTM('ce', 1), co2 = getOTM('ce', 2), co3 = getOTM('ce', 3);
  const po1 = getOTM('pe', 1), po2 = getOTM('pe', 2), po3 = getOTM('pe', 3);

  const ceWing1 = co1.strike - atm;
  const peWing1 = atm - po1.strike;

  // Greeks helpers
  const gCeAtm = getGreeks('ce', atm);
  const gPeAtm = getGreeks('pe', atm);
  const gCo1   = getGreeks('ce', co1.strike);
  const gCo2   = getGreeks('ce', co2.strike);
  const gCo3   = getGreeks('ce', co3.strike);
  const gPo1   = getGreeks('pe', po1.strike);
  const gPo2   = getGreeks('pe', po2.strike);
  const gPo3   = getGreeks('pe', po3.strike);

  let pop = smartPop || 50;
  let mp = 0, ml = 0, be = [], nc = 0, margin = 0, rrRatio = 0;
  let ltpParts = [];
  // Net greeks for intraday simulator (per lot)
  let netDelta = 0, netTheta = 0, netVega = 0;

  switch (shape) {{

    case 'long_call': {{
      const p = ce_atm || 150;
      mp = 999999; ml = p * lotSz; be = [atm + p];
      nc = -p * lotSz; margin = p * lotSz;
      ltpParts = [{{ l: 'BUY CE \u20b9' + atm.toLocaleString('en-IN'), v: p, c: '#00c8e0' }}];
      netDelta = gCeAtm.delta * lotSz; netTheta = gCeAtm.theta * lotSz; netVega = gCeAtm.vega * lotSz;
      break;
    }}
    case 'long_put': {{
      const p = pe_atm || 150;
      mp = 999999; ml = p * lotSz; be = [atm - p];
      nc = -p * lotSz; margin = p * lotSz;
      ltpParts = [{{ l: 'BUY PE \u20b9' + atm.toLocaleString('en-IN'), v: p, c: '#ff9090' }}];
      netDelta = gPeAtm.delta * lotSz; netTheta = gPeAtm.theta * lotSz; netVega = gPeAtm.vega * lotSz;
      break;
    }}
    case 'short_put': {{
      const p = pe_atm || 150;
      mp = p * lotSz; ml = (atm - p) * lotSz; be = [atm - p];
      nc = p * lotSz; margin = atm * lotSz * 0.15;
      rrRatio = ((atm - p) / p).toFixed(2);
      ltpParts = [{{ l: 'SELL PE \u20b9' + atm.toLocaleString('en-IN'), v: p, c: '#ff9090' }}];
      netDelta = -gPeAtm.delta * lotSz; netTheta = -gPeAtm.theta * lotSz; netVega = -gPeAtm.vega * lotSz;
      break;
    }}
    case 'short_call': {{
      const p = ce_atm || 150;
      mp = p * lotSz; ml = 999999; be = [atm + p];
      nc = p * lotSz; margin = atm * lotSz * 0.15;
      ltpParts = [{{ l: 'SELL CE \u20b9' + atm.toLocaleString('en-IN'), v: p, c: '#00c8e0' }}];
      netDelta = -gCeAtm.delta * lotSz; netTheta = -gCeAtm.theta * lotSz; netVega = -gCeAtm.vega * lotSz;
      break;
    }}

    case 'bull_call_spread': {{
      const bp = ce_atm || 150, sp = co1.ltp || 80;
      const nd = bp - sp, sw = ceWing1;
      mp = (sw - nd) * lotSz; ml = nd * lotSz; be = [atm + nd];
      nc = -nd * lotSz; margin = nd * lotSz;
      rrRatio = ((sw - nd) / nd).toFixed(2);
      ltpParts = [
        {{ l: 'BUY CE \u20b9' + atm.toLocaleString('en-IN'), v: bp, c: '#00c8e0' }},
        {{ l: 'SELL CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: sp, c: '#00c896' }}
      ];
      netDelta=(gCeAtm.delta - gCo1.delta)*lotSz; netTheta=(gCeAtm.theta - gCo1.theta)*lotSz; netVega=(gCeAtm.vega - gCo1.vega)*lotSz;
      break;
    }}
    case 'bull_put_spread': {{
      const sp = pe_atm || 150, bp = po1.ltp || 80;
      const nc2 = sp - bp, sw = peWing1;
      mp = nc2 * lotSz; ml = (sw - nc2) * lotSz; be = [atm - nc2];
      nc = nc2 * lotSz; margin = sw * lotSz;
      rrRatio = (nc2 / (sw - nc2)).toFixed(2);
      ltpParts = [
        {{ l: 'SELL PE \u20b9' + atm.toLocaleString('en-IN'), v: sp, c: '#00c896' }},
        {{ l: 'BUY PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: bp, c: '#ff9090' }}
      ];
      netDelta=(-gPeAtm.delta + gPo1.delta)*lotSz; netTheta=(-gPeAtm.theta + gPo1.theta)*lotSz; netVega=(-gPeAtm.vega + gPo1.vega)*lotSz;
      break;
    }}
    case 'bear_call_spread': {{
      const sp = ce_atm || 150, bp = co1.ltp || 80;
      const nc2 = sp - bp, sw = ceWing1;
      mp = nc2 * lotSz; ml = (sw - nc2) * lotSz; be = [atm + nc2];
      nc = nc2 * lotSz; margin = sw * lotSz;
      rrRatio = (nc2 / (sw - nc2)).toFixed(2);
      ltpParts = [
        {{ l: 'SELL CE \u20b9' + atm.toLocaleString('en-IN'), v: sp, c: '#00c896' }},
        {{ l: 'BUY CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: bp, c: '#00c8e0' }}
      ];
      netDelta=(-gCeAtm.delta + gCo1.delta)*lotSz; netTheta=(-gCeAtm.theta + gCo1.theta)*lotSz; netVega=(-gCeAtm.vega + gCo1.vega)*lotSz;
      break;
    }}
    case 'bear_put_spread': {{
      const bp = pe_atm || 150, sp = po1.ltp || 80;
      const nd = bp - sp, sw = peWing1;
      mp = (sw - nd) * lotSz; ml = nd * lotSz; be = [atm - nd];
      nc = -nd * lotSz; margin = nd * lotSz;
      rrRatio = ((sw - nd) / nd).toFixed(2);
      ltpParts = [
        {{ l: 'BUY PE \u20b9' + atm.toLocaleString('en-IN'), v: bp, c: '#ff9090' }},
        {{ l: 'SELL PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: sp, c: '#00c896' }}
      ];
      netDelta=(gPeAtm.delta - gPo1.delta)*lotSz; netTheta=(gPeAtm.theta - gPo1.theta)*lotSz; netVega=(gPeAtm.vega - gPo1.vega)*lotSz;
      break;
    }}

    case 'long_straddle': {{
      const cp2 = ce_atm || 150, pp = pe_atm || 150, tp = cp2 + pp;
      mp = 999999; ml = tp * lotSz; be = [atm - tp, atm + tp];
      nc = -tp * lotSz; margin = tp * lotSz;
      ltpParts = [
        {{ l: 'BUY CE \u20b9' + atm.toLocaleString('en-IN'), v: cp2, c: '#00c8e0' }},
        {{ l: 'BUY PE \u20b9' + atm.toLocaleString('en-IN'), v: pp, c: '#ff9090' }}
      ];
      netDelta=(gCeAtm.delta + gPeAtm.delta)*lotSz; netTheta=(gCeAtm.theta + gPeAtm.theta)*lotSz; netVega=(gCeAtm.vega + gPeAtm.vega)*lotSz;
      break;
    }}
    case 'short_straddle': {{
      const cp2 = ce_atm || 150, pp = pe_atm || 150, tp = cp2 + pp;
      mp = tp * lotSz; ml = 999999; be = [atm - tp, atm + tp];
      nc = tp * lotSz; margin = Math.round(0.155 * OC.spot * OC.lotSize); // SPAN straddle: both ATM naked ~15.5% notional
      ltpParts = [
        {{ l: 'SELL CE \u20b9' + atm.toLocaleString('en-IN'), v: cp2, c: '#00c8e0' }},
        {{ l: 'SELL PE \u20b9' + atm.toLocaleString('en-IN'), v: pp, c: '#ff9090' }}
      ];
      netDelta=-(gCeAtm.delta + gPeAtm.delta)*lotSz; netTheta=-(gCeAtm.theta + gPeAtm.theta)*lotSz; netVega=-(gCeAtm.vega + gPeAtm.vega)*lotSz;
      break;
    }}
    case 'long_strangle': {{
      const cp2 = co1.ltp || 100, pp = po1.ltp || 100, tp = cp2 + pp;
      mp = 999999; ml = tp * lotSz;
      be = [po1.strike - tp, co1.strike + tp];
      nc = -tp * lotSz; margin = tp * lotSz;
      ltpParts = [
        {{ l: 'BUY CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: cp2, c: '#00c8e0' }},
        {{ l: 'BUY PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: pp, c: '#ff9090' }}
      ];
      netDelta=(gCo1.delta + gPo1.delta)*lotSz; netTheta=(gCo1.theta + gPo1.theta)*lotSz; netVega=(gCo1.vega + gPo1.vega)*lotSz;
      break;
    }}
    case 'short_strangle': {{
      const cp2 = co1.ltp || 100, pp = po1.ltp || 100, tp = cp2 + pp;
      mp = tp * lotSz; ml = 999999;
      be = [po1.strike - tp, co1.strike + tp];
      nc = tp * lotSz; margin = Math.round((Math.max(0.117*OC.spot,0.075*co1.strike)*OC.lotSize > Math.max(0.117*OC.spot,0.075*po1.strike)*OC.lotSize ? Math.max(0.117*OC.spot,0.075*co1.strike)*OC.lotSize*0.85 + Math.max(0.117*OC.spot,0.075*po1.strike)*OC.lotSize*0.15 : Math.max(0.117*OC.spot,0.075*po1.strike)*OC.lotSize*0.85 + Math.max(0.117*OC.spot,0.075*co1.strike)*OC.lotSize*0.15)); // SPAN strangle: 85/15 netting
      ltpParts = [
        {{ l: 'SELL CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: cp2, c: '#00c8e0' }},
        {{ l: 'SELL PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: pp, c: '#ff9090' }}
      ];
      netDelta=-(gCo1.delta + gPo1.delta)*lotSz; netTheta=-(gCo1.theta + gPo1.theta)*lotSz; netVega=-(gCo1.vega + gPo1.vega)*lotSz;
      break;
    }}

    case 'short_iron_fly': {{
      const cp2 = ce_atm || 150, pp = pe_atm || 150;
      const wc = co1.ltp || 80, wp = po1.ltp || 80;
      const nc2 = cp2 + pp - wc - wp;
      mp = nc2 * lotSz; ml = (ceWing1 - nc2) * lotSz;
      be = [atm - nc2, atm + nc2];
      nc = nc2 * lotSz; margin = ceWing1 * lotSz * 2;
      rrRatio = (nc2 / (ceWing1 - nc2)).toFixed(2);
      ltpParts = [
        {{ l: 'SELL CE \u20b9' + atm.toLocaleString('en-IN'), v: cp2, c: '#00c8e0' }},
        {{ l: 'SELL PE \u20b9' + atm.toLocaleString('en-IN'), v: pp, c: '#ff9090' }},
        {{ l: 'BUY CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: wc, c: '#00c8e0' }},
        {{ l: 'BUY PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: wp, c: '#ff9090' }}
      ];
      netDelta=-(gCeAtm.delta+gPeAtm.delta-gCo1.delta-gPo1.delta)*lotSz; netTheta=-(gCeAtm.theta+gPeAtm.theta-gCo1.theta-gPo1.theta)*lotSz; netVega=-(gCeAtm.vega+gPeAtm.vega-gCo1.vega-gPo1.vega)*lotSz;
      break;
    }}
    case 'long_iron_fly': {{
      const cp2 = ce_atm || 150, pp = pe_atm || 150;
      const wc = co1.ltp || 80, wp = po1.ltp || 80;
      const nd = wc + wp - cp2 - pp;
      mp = (ceWing1 - Math.abs(nd)) * lotSz; ml = Math.abs(nd) * lotSz;
      be = [atm - Math.abs(nd), atm + Math.abs(nd)];
      nc = -Math.abs(nd) * lotSz; margin = Math.abs(nd) * lotSz;
      rrRatio = ((ceWing1 - Math.abs(nd)) / Math.abs(nd)).toFixed(2);
      ltpParts = [
        {{ l: 'BUY CE \u20b9' + atm.toLocaleString('en-IN'), v: cp2, c: '#00c8e0' }},
        {{ l: 'BUY PE \u20b9' + atm.toLocaleString('en-IN'), v: pp, c: '#ff9090' }},
        {{ l: 'SELL CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: wc, c: '#00c8e0' }},
        {{ l: 'SELL PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: wp, c: '#ff9090' }}
      ];
      netDelta=(gCeAtm.delta+gPeAtm.delta-gCo1.delta-gPo1.delta)*lotSz; netTheta=(gCeAtm.theta+gPeAtm.theta-gCo1.theta-gPo1.theta)*lotSz; netVega=(gCeAtm.vega+gPeAtm.vega-gCo1.vega-gPo1.vega)*lotSz;
      break;
    }}
    case 'short_iron_condor': {{
      const sc = co1.ltp || 100, bc = co2.ltp || 50;
      const sp = po1.ltp || 100, bp = po2.ltp || 50;
      const nc2 = sc - bc + sp - bp;
      mp = nc2 * lotSz; ml = (ceWing1 - nc2) * lotSz;
      be = [po1.strike - nc2, co1.strike + nc2];
      nc = nc2 * lotSz; margin = ceWing1 * lotSz * 2;
      rrRatio = (nc2 / (ceWing1 - nc2)).toFixed(2);
      ltpParts = [
        {{ l: 'SELL CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: sc, c: '#00c8e0' }},
        {{ l: 'BUY CE \u20b9' + co2.strike.toLocaleString('en-IN'), v: bc, c: '#00c8e0' }},
        {{ l: 'SELL PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: sp, c: '#ff9090' }},
        {{ l: 'BUY PE \u20b9' + po2.strike.toLocaleString('en-IN'), v: bp, c: '#ff9090' }}
      ];
      netDelta=-(gCo1.delta+gPo1.delta-gCo2.delta-gPo2.delta)*lotSz; netTheta=-(gCo1.theta+gPo1.theta-gCo2.theta-gPo2.theta)*lotSz; netVega=-(gCo1.vega+gPo1.vega-gCo2.vega-gPo2.vega)*lotSz;
      break;
    }}
    case 'long_iron_condor': {{
      const sc = co1.ltp || 100, bc = co2.ltp || 50;
      const sp = po1.ltp || 100, bp = po2.ltp || 50;
      const nd = bc - sc + bp - sp;
      mp = (ceWing1 - Math.abs(nd)) * lotSz; ml = Math.abs(nd) * lotSz;
      be = [po1.strike - Math.abs(nd), co1.strike + Math.abs(nd)];
      nc = nd * lotSz; margin = Math.abs(nd) * lotSz;
      rrRatio = ((ceWing1 - Math.abs(nd)) / Math.abs(nd)).toFixed(2);
      ltpParts = [
        {{ l: 'SELL CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: sc, c: '#00c8e0' }},
        {{ l: 'BUY CE \u20b9' + co2.strike.toLocaleString('en-IN'), v: bc, c: '#00c8e0' }},
        {{ l: 'SELL PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: sp, c: '#ff9090' }},
        {{ l: 'BUY PE \u20b9' + po2.strike.toLocaleString('en-IN'), v: bp, c: '#ff9090' }}
      ];
      netDelta=(gCo1.delta+gPo1.delta-gCo2.delta-gPo2.delta)*lotSz; netTheta=(gCo1.theta+gPo1.theta-gCo2.theta-gPo2.theta)*lotSz; netVega=(gCo1.vega+gPo1.vega-gCo2.vega-gPo2.vega)*lotSz;
      break;
    }}

    case 'call_butterfly':
    case 'bull_butterfly': {{
      const lp = ce_atm || 150, mid = co1.ltp || 80, hp = co2.ltp || 40;
      const nd = lp - 2 * mid + hp;
      mp = (ceWing1 - nd) * lotSz; ml = nd * lotSz;
      be = [atm + nd, co2.strike - nd];
      nc = -nd * lotSz; margin = nd * lotSz;
      rrRatio = ((ceWing1 - nd) / nd).toFixed(2);
      ltpParts = [
        {{ l: 'BUY CE \u20b9' + atm.toLocaleString('en-IN'), v: lp, c: '#00c8e0' }},
        {{ l: 'SELL 2x CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: mid, c: '#00c896' }},
        {{ l: 'BUY CE \u20b9' + co2.strike.toLocaleString('en-IN'), v: hp, c: '#00c8e0' }}
      ];
      netDelta=(gCeAtm.delta - 2*gCo1.delta + gCo2.delta)*lotSz; netTheta=(gCeAtm.theta - 2*gCo1.theta + gCo2.theta)*lotSz; netVega=(gCeAtm.vega - 2*gCo1.vega + gCo2.vega)*lotSz;
      break;
    }}
    case 'put_butterfly':
    case 'bear_butterfly': {{
      const hp = pe_atm || 150, mid = po1.ltp || 80, lp = po2.ltp || 40;
      const nd = hp - 2 * mid + lp;
      mp = (peWing1 - nd) * lotSz; ml = nd * lotSz;
      be = [po2.strike + nd, atm - nd];
      nc = -nd * lotSz; margin = nd * lotSz;
      rrRatio = ((peWing1 - nd) / nd).toFixed(2);
      ltpParts = [
        {{ l: 'BUY PE \u20b9' + atm.toLocaleString('en-IN'), v: hp, c: '#ff9090' }},
        {{ l: 'SELL 2x PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: mid, c: '#00c896' }},
        {{ l: 'BUY PE \u20b9' + po2.strike.toLocaleString('en-IN'), v: lp, c: '#ff9090' }}
      ];
      netDelta=(gPeAtm.delta - 2*gPo1.delta + gPo2.delta)*lotSz; netTheta=(gPeAtm.theta - 2*gPo1.theta + gPo2.theta)*lotSz; netVega=(gPeAtm.vega - 2*gPo1.vega + gPo2.vega)*lotSz;
      break;
    }}

    case 'call_ratio_back': {{
      const sp = ce_atm || 150, bp = co1.ltp || 80, nd = 2 * bp - sp;
      mp = 999999; ml = nd > 0 ? nd * lotSz : 0;
      be = [co1.strike + Math.abs(nd)];
      nc = -nd * lotSz; margin = Math.round((0.117 - 0.01) * OC.spot * OC.lotSize); // SPAN: naked(11.7%) - 1 BUY hedge(1%)
      ltpParts = [
        {{ l: 'SELL CE \u20b9' + atm.toLocaleString('en-IN'), v: sp, c: '#00c896' }},
        {{ l: 'BUY 2x CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: bp, c: '#00c8e0' }}
      ];
      netDelta=(-gCeAtm.delta + 2*gCo1.delta)*lotSz; netTheta=(-gCeAtm.theta + 2*gCo1.theta)*lotSz; netVega=(-gCeAtm.vega + 2*gCo1.vega)*lotSz;
      break;
    }}
    case 'put_ratio_back': {{
      const sp = pe_atm || 150, bp = po1.ltp || 80, nd = 2 * bp - sp;
      mp = 999999; ml = nd > 0 ? nd * lotSz : 0;
      be = [po1.strike - Math.abs(nd)];
      nc = -nd * lotSz; margin = Math.round((0.117 - 0.01) * OC.spot * OC.lotSize); // SPAN: naked(11.7%) - 1 BUY hedge(1%)
      ltpParts = [
        {{ l: 'SELL PE \u20b9' + atm.toLocaleString('en-IN'), v: sp, c: '#00c896' }},
        {{ l: 'BUY 2x PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: bp, c: '#ff9090' }}
      ];
      netDelta=(-gPeAtm.delta + 2*gPo1.delta)*lotSz; netTheta=(-gPeAtm.theta + 2*gPo1.theta)*lotSz; netVega=(-gPeAtm.vega + 2*gPo1.vega)*lotSz;
      break;
    }}

    case 'call_ratio_spread': {{
      const bp = ce_atm || 150, sp = co1.ltp || 80;
      const nc2 = 2 * sp - bp;
      const maxProfitPts = ceWing1 + nc2;
      mp = maxProfitPts * lotSz; ml = 999999;
      be = nc2 >= 0
        ? [2 * co1.strike - atm + nc2]
        : [atm - nc2, 2 * co1.strike - atm + nc2];
      nc = nc2 * lotSz; margin = co1.strike * lotSz * 0.15;
      ltpParts = [
        {{ l: 'BUY CE \u20b9' + atm.toLocaleString('en-IN'), v: bp, c: '#00c8e0' }},
        {{ l: 'SELL 2x CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: sp, c: '#00c896' }}
      ];
      netDelta=(gCeAtm.delta - 2*gCo1.delta)*lotSz; netTheta=(gCeAtm.theta - 2*gCo1.theta)*lotSz; netVega=(gCeAtm.vega - 2*gCo1.vega)*lotSz;
      break;
    }}
    case 'put_ratio_spread': {{
      const bp = pe_atm || 150, sp = po1.ltp || 80;
      const nc2 = 2 * sp - bp;
      const maxProfitPts = peWing1 + nc2;
      mp = maxProfitPts * lotSz; ml = 999999;
      be = nc2 >= 0
        ? [2 * po1.strike - atm - nc2]
        : [2 * po1.strike - atm - nc2, atm + nc2];
      nc = nc2 * lotSz; margin = po1.strike * lotSz * 0.15;
      ltpParts = [
        {{ l: 'BUY PE \u20b9' + atm.toLocaleString('en-IN'), v: bp, c: '#ff9090' }},
        {{ l: 'SELL 2x PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: sp, c: '#00c896' }}
      ];
      netDelta=(gPeAtm.delta - 2*gPo1.delta)*lotSz; netTheta=(gPeAtm.theta - 2*gPo1.theta)*lotSz; netVega=(gPeAtm.vega - 2*gPo1.vega)*lotSz;
      break;
    }}

    case 'long_synthetic': {{
      const cp2 = ce_atm || 150, pp = pe_atm || 150, nd = cp2 - pp;
      mp = 999999; ml = 999999;
      be = [atm + nd]; nc = -Math.abs(nd) * lotSz;
      margin = atm * lotSz * 0.30;
      ltpParts = [
        {{ l: 'BUY CE \u20b9' + atm.toLocaleString('en-IN'), v: cp2, c: '#00c8e0' }},
        {{ l: 'SELL PE \u20b9' + atm.toLocaleString('en-IN'), v: pp, c: '#ff9090' }}
      ];
      netDelta=(gCeAtm.delta - gPeAtm.delta)*lotSz; netTheta=(gCeAtm.theta - gPeAtm.theta)*lotSz; netVega=(gCeAtm.vega - gPeAtm.vega)*lotSz;
      break;
    }}
    case 'short_synthetic': {{
      const cp2 = ce_atm || 150, pp = pe_atm || 150, nc2 = cp2 - pp;
      mp = 999999; ml = 999999;
      be = [atm + nc2]; nc = Math.abs(nc2) * lotSz;
      margin = atm * lotSz * 0.30;
      ltpParts = [
        {{ l: 'SELL CE \u20b9' + atm.toLocaleString('en-IN'), v: cp2, c: '#00c8e0' }},
        {{ l: 'BUY PE \u20b9' + atm.toLocaleString('en-IN'), v: pp, c: '#ff9090' }}
      ];
      netDelta=(-gCeAtm.delta + gPeAtm.delta)*lotSz; netTheta=(-gCeAtm.theta + gPeAtm.theta)*lotSz; netVega=(-gCeAtm.vega + gPeAtm.vega)*lotSz;
      break;
    }}

    case 'risk_reversal': {{
      const bp = po1.ltp || 100, sc = co1.ltp || 100;
      const nd = bp - sc;
      mp = 999999; ml = 999999;
      be = [po1.strike - Math.abs(nd), co1.strike + Math.abs(nd)];
      nc = -nd * lotSz; margin = atm * lotSz * 0.20;
      ltpParts = [
        {{ l: 'BUY PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: bp, c: '#ff9090' }},
        {{ l: 'SELL CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: sc, c: '#00c8e0' }}
      ];
      netDelta=(gPo1.delta - gCo1.delta)*lotSz; netTheta=(gPo1.theta - gCo1.theta)*lotSz; netVega=(gPo1.vega - gCo1.vega)*lotSz;
      break;
    }}
    case 'range_forward': {{
      const bc = co1.ltp || 100, sp = po1.ltp || 100;
      const nd = bc - sp;
      mp = 999999; ml = 999999;
      be = [po1.strike - Math.abs(nd), co1.strike + Math.abs(nd)];
      nc = -nd * lotSz; margin = atm * lotSz * 0.20;
      ltpParts = [
        {{ l: 'BUY CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: bc, c: '#00c8e0' }},
        {{ l: 'SELL PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: sp, c: '#ff9090' }}
      ];
      netDelta=(gCo1.delta - gPo1.delta)*lotSz; netTheta=(gCo1.theta - gPo1.theta)*lotSz; netVega=(gCo1.vega - gPo1.vega)*lotSz;
      break;
    }}

    case 'jade_lizard': {{
      const pp = po1.ltp || 100, cs = co1.ltp || 80, cb = co2.ltp || 40;
      const nc2 = pp + cs - cb;
      mp = nc2 * lotSz; ml = (po1.strike - nc2) * lotSz;
      be = [po1.strike - nc2];
      nc = nc2 * lotSz; margin = po1.strike * lotSz * 0.15;
      ltpParts = [
        {{ l: 'SELL PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: pp, c: '#ff9090' }},
        {{ l: 'SELL CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: cs, c: '#00c8e0' }},
        {{ l: 'BUY CE \u20b9' + co2.strike.toLocaleString('en-IN'), v: cb, c: '#00c8e0' }}
      ];
      netDelta=(-gPo1.delta - gCo1.delta + gCo2.delta)*lotSz; netTheta=(-gPo1.theta - gCo1.theta + gCo2.theta)*lotSz; netVega=(-gPo1.vega - gCo1.vega + gCo2.vega)*lotSz;
      break;
    }}
    case 'reverse_jade': {{
      const cp2 = co1.ltp || 100, ps = po1.ltp || 80, pb = po2.ltp || 40;
      const nc2 = cp2 + ps - pb;
      mp = nc2 * lotSz; ml = (po1.strike - nc2) * lotSz;
      be = [po1.strike - nc2];
      nc = nc2 * lotSz; margin = co1.strike * lotSz * 0.15;
      ltpParts = [
        {{ l: 'SELL CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: cp2, c: '#00c8e0' }},
        {{ l: 'SELL PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: ps, c: '#ff9090' }},
        {{ l: 'BUY PE \u20b9' + po2.strike.toLocaleString('en-IN'), v: pb, c: '#ff9090' }}
      ];
      netDelta=(-gCo1.delta - gPo1.delta + gPo2.delta)*lotSz; netTheta=(-gCo1.theta - gPo1.theta + gPo2.theta)*lotSz; netVega=(-gCo1.vega - gPo1.vega + gPo2.vega)*lotSz;
      break;
    }}

    case 'bull_condor': {{
      const s1 = ce_atm || 150;
      const s2 = co1.ltp || 100;
      const s3 = co2.ltp || 60;
      const s4 = co3.ltp || 30;
      const nd = s1 - s2 - s3 + s4;
      mp = (ceWing1 - nd) * lotSz; ml = nd * lotSz;
      be = [atm + nd, co3.strike - nd];
      nc = -nd * lotSz; margin = nd * lotSz;
      rrRatio = ((ceWing1 - nd) / nd).toFixed(2);
      ltpParts = [
        {{ l: 'BUY CE \u20b9'  + atm.toLocaleString('en-IN'),        v: s1, c: '#00c8e0' }},
        {{ l: 'SELL CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: s2, c: '#00c896' }},
        {{ l: 'SELL CE \u20b9' + co2.strike.toLocaleString('en-IN'), v: s3, c: '#00c896' }},
        {{ l: 'BUY CE \u20b9'  + co3.strike.toLocaleString('en-IN'), v: s4, c: '#00c8e0' }}
      ];
      netDelta=(gCeAtm.delta - gCo1.delta - gCo2.delta + gCo3.delta)*lotSz; netTheta=(gCeAtm.theta - gCo1.theta - gCo2.theta + gCo3.theta)*lotSz; netVega=(gCeAtm.vega - gCo1.vega - gCo2.vega + gCo3.vega)*lotSz;
      break;
    }}
    case 'bear_condor': {{
      const s1 = pe_atm || 150;
      const s2 = po1.ltp || 100;
      const s3 = po2.ltp || 60;
      const s4 = po3.ltp || 30;
      const nd = s1 - s2 - s3 + s4;
      mp = (peWing1 - nd) * lotSz; ml = nd * lotSz;
      be = [po3.strike + nd, atm - nd];
      nc = -nd * lotSz; margin = nd * lotSz;
      rrRatio = ((peWing1 - nd) / nd).toFixed(2);
      ltpParts = [
        {{ l: 'BUY PE \u20b9'  + atm.toLocaleString('en-IN'),        v: s1, c: '#ff9090' }},
        {{ l: 'SELL PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: s2, c: '#00c896' }},
        {{ l: 'SELL PE \u20b9' + po2.strike.toLocaleString('en-IN'), v: s3, c: '#00c896' }},
        {{ l: 'BUY PE \u20b9'  + po3.strike.toLocaleString('en-IN'), v: s4, c: '#ff9090' }}
      ];
      netDelta=(gPeAtm.delta - gPo1.delta - gPo2.delta + gPo3.delta)*lotSz; netTheta=(gPeAtm.theta - gPo1.theta - gPo2.theta + gPo3.theta)*lotSz; netVega=(gPeAtm.vega - gPo1.vega - gPo2.vega + gPo3.vega)*lotSz;
      break;
    }}

    case 'batman': {{
      const s1 = ce_atm || 150, s2 = co1.ltp || 80, s3 = co2.ltp || 40;
      const ndPerUnit = s1 - 2 * s2 + s3;
      const totalNd = 2 * ndPerUnit;
      mp = 2 * (ceWing1 - ndPerUnit) * lotSz; ml = totalNd * lotSz;
      be = [atm + ndPerUnit, co2.strike - ndPerUnit];
      nc = -totalNd * lotSz; margin = totalNd * lotSz;
      ltpParts = [
        {{ l: 'BUY 2x CE \u20b9' + atm.toLocaleString('en-IN'), v: s1, c: '#00c8e0' }},
        {{ l: 'SELL 4x CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: s2, c: '#00c896' }},
        {{ l: 'BUY 2x CE \u20b9' + co2.strike.toLocaleString('en-IN'), v: s3, c: '#00c8e0' }}
      ];
      netDelta=2*(gCeAtm.delta - 2*gCo1.delta + gCo2.delta)*lotSz; netTheta=2*(gCeAtm.theta - 2*gCo1.theta + gCo2.theta)*lotSz; netVega=2*(gCeAtm.vega - 2*gCo1.vega + gCo2.vega)*lotSz;
      break;
    }}

    case 'double_fly': {{
      const cNd = ce_atm - 2 * co1.ltp + (co2.ltp || 40);
      const pNd = pe_atm - 2 * po1.ltp + (po2.ltp || 40);
      const totalNd = cNd + pNd;
      mp = ((ceWing1 - cNd) + (peWing1 - pNd)) * lotSz;
      ml = totalNd * lotSz;
      be = [atm - totalNd * 0.5, atm + totalNd * 0.5];
      nc = -totalNd * lotSz; margin = totalNd * lotSz;
      ltpParts = [
        {{ l: 'CALL FLY: BUY/SELL/BUY CE', v: ce_atm || 150, c: '#00c8e0' }},
        {{ l: 'PUT FLY:  BUY/SELL/BUY PE', v: pe_atm || 150, c: '#ff9090' }}
      ];
      netDelta=((gCeAtm.delta-2*gCo1.delta+gCo2.delta)+(gPeAtm.delta-2*gPo1.delta+gPo2.delta))*lotSz; netTheta=((gCeAtm.theta-2*gCo1.theta+gCo2.theta)+(gPeAtm.theta-2*gPo1.theta+gPo2.theta))*lotSz; netVega=((gCeAtm.vega-2*gCo1.vega+gCo2.vega)+(gPeAtm.vega-2*gPo1.vega+gPo2.vega))*lotSz;
      break;
    }}

    case 'double_condor': {{
      const cNd = ce_atm - co1.ltp - co2.ltp + (co3.ltp || 30);
      const pNd = pe_atm - po1.ltp - po2.ltp + (po3.ltp || 30);
      const totalNd = cNd + pNd;
      mp = ((ceWing1 - cNd) + (peWing1 - pNd)) * lotSz;
      ml = totalNd * lotSz;
      be = [atm - totalNd, atm + totalNd];
      nc = -totalNd * lotSz; margin = totalNd * lotSz;
      ltpParts = [
        {{ l: 'CE CONDOR \u20b9' + atm.toLocaleString('en-IN') + ' \u2192 \u20b9' + co3.strike.toLocaleString('en-IN'), v: ce_atm || 150, c: '#00c8e0' }},
        {{ l: 'PE CONDOR \u20b9' + atm.toLocaleString('en-IN') + ' \u2192 \u20b9' + po3.strike.toLocaleString('en-IN'), v: pe_atm || 150, c: '#ff9090' }}
      ];
      netDelta=((gCeAtm.delta-gCo1.delta-gCo2.delta+gCo3.delta)+(gPeAtm.delta-gPo1.delta-gPo2.delta+gPo3.delta))*lotSz; netTheta=((gCeAtm.theta-gCo1.theta-gCo2.theta+gCo3.theta)+(gPeAtm.theta-gPo1.theta-gPo2.theta+gPo3.theta))*lotSz; netVega=((gCeAtm.vega-gCo1.vega-gCo2.vega+gCo3.vega)+(gPeAtm.vega-gPo1.vega-gPo2.vega+gPo3.vega))*lotSz;
      break;
    }}

    case 'call_calendar': {{
      const farLTP  = ce_atm || 150;
      const nearLTP = Math.round(farLTP * 0.55);
      const nd = farLTP - nearLTP;
      mp = Math.round(nd * 0.80) * lotSz; ml = nd * lotSz;
      be = [atm - Math.round(nd * 0.6), atm + Math.round(nd * 0.6)];
      nc = -nd * lotSz; margin = nd * lotSz;
      ltpParts = [
        {{ l: 'SELL NEAR CE \u20b9' + atm.toLocaleString('en-IN') + ' (~est)', v: nearLTP, c: '#00c896' }},
        {{ l: 'BUY FAR CE \u20b9'   + atm.toLocaleString('en-IN'),             v: farLTP,  c: '#00c8e0' }}
      ];
      netDelta=gCeAtm.delta*0.1*lotSz; netTheta=Math.abs(gCeAtm.theta)*0.45*lotSz; netVega=gCeAtm.vega*0.3*lotSz;
      break;
    }}
    case 'put_calendar': {{
      const farLTP  = pe_atm || 150;
      const nearLTP = Math.round(farLTP * 0.55);
      const nd = farLTP - nearLTP;
      mp = Math.round(nd * 0.80) * lotSz; ml = nd * lotSz;
      be = [atm - Math.round(nd * 0.6), atm + Math.round(nd * 0.6)];
      nc = -nd * lotSz; margin = nd * lotSz;
      ltpParts = [
        {{ l: 'SELL NEAR PE \u20b9' + atm.toLocaleString('en-IN') + ' (~est)', v: nearLTP, c: '#00c896' }},
        {{ l: 'BUY FAR PE \u20b9'   + atm.toLocaleString('en-IN'),             v: farLTP,  c: '#ff9090' }}
      ];
      netDelta=gPeAtm.delta*0.1*lotSz; netTheta=Math.abs(gPeAtm.theta)*0.45*lotSz; netVega=gPeAtm.vega*0.3*lotSz;
      break;
    }}
    case 'diagonal_calendar': {{
      const farLTP  = ce_atm || 150;
      const nearLTP = Math.round((co1.ltp || 80) * 0.55);
      const nd = farLTP - nearLTP;
      mp = Math.round(nd * 0.70) * lotSz; ml = nd * lotSz;
      be = [atm + Math.round(nd * 0.25), atm + Math.round(nd * 1.4)];
      nc = -nd * lotSz; margin = nd * lotSz;
      ltpParts = [
        {{ l: 'SELL NEAR CE \u20b9' + co1.strike.toLocaleString('en-IN') + ' (~est)', v: nearLTP, c: '#00c896' }},
        {{ l: 'BUY FAR CE \u20b9'   + atm.toLocaleString('en-IN'),                   v: farLTP,  c: '#00c8e0' }}
      ];
      netDelta=(gCeAtm.delta - gCo1.delta*0.5)*0.1*lotSz; netTheta=Math.abs(gCeAtm.theta)*0.35*lotSz; netVega=(gCeAtm.vega-gCo1.vega*0.5)*0.3*lotSz;
      break;
    }}

    default: {{
      const p = ce_atm || 150;
      mp = p * lotSz * 0.5; ml = p * lotSz * 0.3;
      be = [atm]; nc = -p * 0.3 * lotSz;
      margin = p * lotSz; rrRatio = 1.5;
      ltpParts = [{{ l: 'ATM \u20b9' + atm.toLocaleString('en-IN'), v: p, c: '#00c8e0' }}];
    }}
  }}

  const beStr     = be.map(v => '\u20b9' + Math.round(v).toLocaleString('en-IN')).join(' / ');
  const mpStr     = mp === 999999 ? 'Unlimited' : '\u20b9' + Math.round(mp).toLocaleString('en-IN');
  const mlStr     = ml === 999999 ? 'Unlimited' : '\u20b9' + Math.round(ml).toLocaleString('en-IN');
  const ncStr     = (nc >= 0 ? '+ ' : '- ') + '\u20b9' + Math.abs(Math.round(nc)).toLocaleString('en-IN');
  const marginStr = '\u20b9' + Math.round(margin).toLocaleString('en-IN');
  const rrStr     = rrRatio === 0 ? '\u221e' : ('1:' + Math.abs(rrRatio));
  const mpPct     = mp === 999999 ? '\u221e' : (ml > 0 ? (mp / ml * 100).toFixed(0) + '%' : '\u2014');
  const ltpStr    = ltpParts.map(x =>
    `<span style="display:inline-flex;align-items:center;gap:4px;margin-bottom:2px;">
      <span style="font-size:12.3px;color:rgba(255,255,255,.35);">${{x.l}}</span>
      <span style="font-family:'DM Mono',monospace;font-weight:700;color:${{x.c}};">\u20b9${{x.v.toFixed(2)}}</span>
    </span>`
  ).join('<br>');
  const strikeStr = 'ATM \u20b9' + atm.toLocaleString('en-IN');
  return {{pop,mpStr,mlStr,rrStr,beStr,ncStr,marginStr,mpPct,strikeStr,ltpStr,
           mpRaw:mp,mlRaw:ml,ncRaw:Math.round(nc),ncPositive:nc>=0,
           netDelta:Math.round(netDelta*100)/100,
           netTheta:Math.round(netTheta*100)/100,
           netVega:Math.round(netVega*100)/100,
           mlRawVal:ml}};
}}

function renderMetrics(m, scoreBreakdown) {{
  const pc=m.pop>=70?'#00c896':(m.pop>=55?'#4de8b8':m.pop>=45?'#6480ff':'#ff6b6b');
  const nc=m.ncPositive?'#00c896':'#ff6b6b';
  const sbHtml = scoreBreakdown ? `
    <div style="background:rgba(100,128,255,.06);border-top:1px solid rgba(100,128,255,.1);padding:8px 12px;">
      <div style="font-size:11.6px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(100,128,255,.6);margin-bottom:6px;">PoP BREAKDOWN</div>
      <div style="display:flex;flex-wrap:wrap;gap:6px;">
        <span style="font-size:13px;background:rgba(0,0,0,.2);padding:2px 8px;border-radius:6px;color:rgba(255,255,255,.5);">Base <b style="color:#fff;">50%</b></span>
        <span style="font-size:13px;background:rgba(0,0,0,.2);padding:2px 8px;border-radius:6px;color:rgba(255,255,255,.5);">Bias <b style="color:${{scoreBreakdown.biasAdj>=0?'#00c896':'#ff6b6b'}};">${{scoreBreakdown.biasAdj>=0?'+':''}}${{scoreBreakdown.biasAdj}}</b></span>
        <span style="font-size:13px;background:rgba(0,0,0,.2);padding:2px 8px;border-radius:6px;color:rgba(255,255,255,.5);">S/R <b style="color:${{scoreBreakdown.srAdj>=0?'#00c896':'#ff6b6b'}};">${{scoreBreakdown.srAdj>=0?'+':''}}${{scoreBreakdown.srAdj}}</b></span>
        <span style="font-size:13px;background:rgba(0,0,0,.2);padding:2px 8px;border-radius:6px;color:rgba(255,255,255,.5);">OI <b style="color:${{scoreBreakdown.oiAdj>=0?'#00c896':'#ff6b6b'}};">${{scoreBreakdown.oiAdj>=0?'+':''}}${{scoreBreakdown.oiAdj}}</b></span>
        <span style="font-size:13px;background:rgba(0,0,0,.2);padding:2px 8px;border-radius:6px;color:rgba(255,255,255,.5);">PCR <b style="color:${{scoreBreakdown.pcrAdj>=0?'#00c896':'#ff6b6b'}};">${{scoreBreakdown.pcrAdj>=0?'+':''}}${{scoreBreakdown.pcrAdj}}</b></span>
        <span style="font-size:13px;background:rgba(0,0,0,.2);padding:2px 8px;border-radius:6px;color:rgba(255,255,255,.5);">Strat <b style="color:${{scoreBreakdown.stratAdj>=0?'#00c896':'#ff6b6b'}};">${{scoreBreakdown.stratAdj>=0?'+':''}}${{scoreBreakdown.stratAdj}}</b></span>
      </div>
    </div>` : '';
  return `<div class="metric-row metric-strike"><span class="metric-lbl">Strike Price</span>
    <span class="metric-val" style="color:#ffd166;font-size:15.9px;text-align:right;">${{m.strikeStr}}</span></div>
    <div class="metric-row" style="background:rgba(0,200,220,.04);border-bottom:1px solid rgba(0,200,220,.10);">
      <span class="metric-lbl" style="color:rgba(0,200,220,.7);">LTP (per leg)</span>
      <span class="metric-val" style="text-align:right;line-height:1.6;display:flex;flex-direction:column;align-items:flex-end;">${{m.ltpStr}}</span>
    </div>
    <div class="metric-row"><span class="metric-lbl">Prob. of Profit</span>
    <span class="metric-val" style="color:${{pc}};font-weight:800;font-size:21.8px;">${{m.pop}}%</span></div>
    <div class="metric-row"><span class="metric-lbl">Max. Profit</span>
    <span class="metric-val" style="color:#00c896;">${{m.mpStr}} <small style="opacity:.5;">${{m.mpPct}}</small></span></div>
    <div class="metric-row"><span class="metric-lbl">Max. Loss</span>
    <span class="metric-val" style="color:#ff6b6b;">${{m.mlStr}}</span></div>
    <div class="metric-row"><span class="metric-lbl">Max RR Ratio</span>
    <span class="metric-val" style="color:#6480ff;">${{m.rrStr}}</span></div>
    <div class="metric-row"><span class="metric-lbl">Breakevens</span>
    <span class="metric-val" style="color:#00c8e0;font-size:15.9px;">${{m.beStr}}</span></div>
    <div class="metric-row"><span class="metric-lbl">Net Credit / Debit</span>
    <span class="metric-val" style="color:${{nc}};">${{m.ncStr}}</span></div>
    <div class="metric-row" style="border-bottom:none;"><span class="metric-lbl">Est. Margin/Premium</span>
    <span class="metric-val" style="color:#8aa0ff;">${{m.marginStr}}</span></div>
    ${{sbHtml}}
    ${{buildIntradaySim(m)}}`;
}}

// ── Intraday P&L Simulator ───────────────────────────────────────────────────
function buildIntradaySim(m) {{
  const lotSz   = OC.lotSize;
  const maxL    = m.mlRawVal === 999999 ? null : m.mlRawVal;
  const maxP    = m.mpRaw    === 999999 ? null : m.mpRaw;
  const nd      = m.netDelta;   // ₹ per 1 point Nifty move
  const nt      = m.netTheta;   // ₹ per day (negative = decay cost)
  const nv      = m.netVega;    // ₹ per 1% IV change

  const moves = [-300,-200,-150,-100,-50,0,50,100,150,200,300];

  function calcPnl(movePts) {{
    let pnl = nd * movePts + nt;
    if (maxL !== null) pnl = Math.max(-maxL, pnl);
    if (maxP !== null) pnl = Math.min(maxP * 0.9, pnl);
    return Math.round(pnl);
  }}

  const flatPnl  = calcPnl(0);
  const flatCol  = flatPnl >= 0 ? '#00c896' : '#ff6b6b';
  const ntCol    = nt >= 0 ? '#00c896' : '#ff6b6b';
  const ntSign   = nt >= 0 ? '+' : '';
  const ndStr    = (nd >= 0 ? '+' : '') + '\u20b9' + Math.abs(nd).toFixed(2);
  const ntStr    = ntSign + '\u20b9' + Math.abs(Math.round(nt));
  const nvStr    = (nv >= 0 ? '+' : '') + '\u20b9' + Math.abs(nv).toFixed(2);

  let tRows = '';
  moves.forEach(mv => {{
    const pnl    = calcPnl(mv);
    const col    = pnl > 100 ? '#00c896' : pnl > 0 ? '#4de8b8' : pnl > -200 ? '#ffd166' : '#ff6b6b';
    const mvcol  = mv > 0 ? '#00c896' : mv < 0 ? '#ff6b6b' : '#6480ff';
    const mvbg   = mv > 0 ? 'rgba(0,200,150,.12)' : mv < 0 ? 'rgba(255,107,107,.12)' : 'rgba(100,128,255,.12)';
    const mvlbl  = mv > 0 ? '+' + mv : mv === 0 ? 'Flat' : String(mv);
    const pctmp  = maxP ? ((pnl / maxP) * 100).toFixed(0) + '%' : '—';
    const isFlat = mv === 0;
    const rowBg  = isFlat ? 'background:rgba(245,197,24,.05);' : '';
    tRows += `<tr style="${{rowBg}}">
      <td style="padding:5px 6px;white-space:nowrap;">
        <span style="font-size:14.5px;font-weight:700;padding:2px 8px;border-radius:5px;background:${{mvbg}};color:${{mvcol}};white-space:nowrap;display:inline-block;">${{mvlbl}}${{mv!==0?'p':''}}</span>
      </td>
      <td style="padding:5px 6px;color:rgba(255,255,255,.4);font-family:'DM Mono',monospace;font-size:14.5px;white-space:nowrap;text-align:right;">${{(OC.spot+mv).toLocaleString('en-IN')}}</td>
      <td style="padding:5px 6px;font-family:'DM Mono',monospace;font-weight:700;font-size:15.9px;color:${{col}};white-space:nowrap;text-align:right;">${{pnl>=0?'+':''}}\u20b9${{Math.abs(pnl).toLocaleString('en-IN')}}</td>
      <td style="padding:5px 6px;font-size:13.8px;font-weight:600;color:${{col}};text-align:center;white-space:nowrap;opacity:.75;">${{pctmp}}</td>
    </tr>`;
  }});

  // Slider range: spot ± 400 in steps of 25
  const slMin = Math.round((OC.spot - 400) / 25) * 25;
  const slMax = Math.round((OC.spot + 400) / 25) * 25;
  const simId = 'sim_' + Math.random().toString(36).slice(2,7);

  return `
<div style="border-top:2px solid rgba(255,209,102,.22);background:linear-gradient(135deg,rgba(245,197,24,.04),rgba(200,155,10,.02));" onclick="event.stopPropagation()">
  <!-- Sub-tabs -->
  <div style="display:flex;border-bottom:1px solid rgba(255,255,255,.06);">
    <button class="sim-tab active" id="${{simId}}_t1"
      onclick="simTab('${{simId}}','t1')" style="flex:1;padding:8px 4px;font-family:'DM Mono',monospace;font-size:13px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;cursor:pointer;border:none;border-bottom:2px solid #f5c518;color:#f5c518;background:rgba(245,197,24,.07);transition:all .2s;">📊 SCENARIOS</button>
    <button class="sim-tab" id="${{simId}}_t2"
      onclick="simTab('${{simId}}','t2')" style="flex:1;padding:8px 4px;font-family:'DM Mono',monospace;font-size:13px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;cursor:pointer;border:none;border-bottom:2px solid transparent;color:rgba(255,255,255,.3);background:transparent;transition:all .2s;">🔬 GREEKS</button>
    <button class="sim-tab" id="${{simId}}_t3"
      onclick="simTab('${{simId}}','t3')" style="flex:1;padding:8px 4px;font-family:'DM Mono',monospace;font-size:13px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;cursor:pointer;border:none;border-bottom:2px solid transparent;color:rgba(255,255,255,.3);background:transparent;transition:all .2s;">🎚 SLIDER</button>
  </div>

  <!-- TAB 1: Scenarios -->
  <div id="${{simId}}_c1">
    <div style="padding:9px 12px 6px;display:flex;align-items:center;justify-content:space-between;">
      <div style="font-size:13px;font-weight:700;letter-spacing:1.5px;color:rgba(255,209,102,.8);text-transform:uppercase;">📅 TODAY'S P&amp;L SCENARIOS</div>
      <div style="font-size:12.3px;color:rgba(255,255,255,.25);">Delta×move + Theta · 1 day</div>
    </div>
    <div style="padding:0 10px 10px;">
      <table style="width:100%;border-collapse:collapse;">
        <thead><tr style="background:rgba(255,255,255,.05);">
          <th style="padding:5px 6px;font-size:11.6px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.3);text-align:left;border-bottom:1px solid rgba(255,255,255,.07);">MOVE</th>
          <th style="padding:5px 6px;font-size:11.6px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.3);text-align:right;border-bottom:1px solid rgba(255,255,255,.07);">SPOT</th>
          <th style="padding:5px 6px;font-size:11.6px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.3);text-align:right;border-bottom:1px solid rgba(255,255,255,.07);">TODAY P&amp;L</th>
          <th style="padding:5px 6px;font-size:11.6px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.3);text-align:center;border-bottom:1px solid rgba(255,255,255,.07);">vs MAX</th>
        </tr></thead>
        <tbody>${{tRows}}</tbody>
      </table>
    </div>
    <div style="margin:0 12px 12px;padding:8px 10px;background:rgba(255,107,107,.06);border:1px solid rgba(255,107,107,.15);border-radius:8px;font-size:13.8px;color:rgba(255,150,150,.7);line-height:1.6;">
      ⏱ P&amp;L = <b>Delta×move</b> + <b>Theta (1 day)</b>. Actual exit P&amp;L may differ due to IV changes. Max profit of ${{m.mpStr}} is achievable at expiry only.
    </div>
  </div>

  <!-- TAB 2: Greeks Breakdown -->
  <div id="${{simId}}_c2" style="display:none;">
    <div style="padding:9px 12px 8px;display:flex;align-items:center;justify-content:space-between;">
      <div style="font-size:13px;font-weight:700;letter-spacing:1.5px;color:rgba(255,209,102,.8);text-transform:uppercase;">🔬 NET GREEKS (per lot)</div>
      <div style="font-size:11.6px;color:rgba(255,255,255,.25);letter-spacing:.5px;">values per lot · today</div>
    </div>
    <!-- Greeks: compact single-line rows -->
    <div style="padding:0 10px 10px;display:flex;flex-direction:column;gap:5px;">

      <!-- Delta row -->
      <div style="display:flex;align-items:center;gap:10px;background:rgba(0,200,150,.07);border:1px solid rgba(0,200,150,.18);border-radius:9px;padding:8px 12px;">
        <div style="width:24px;height:24px;border-radius:6px;background:rgba(0,200,150,.18);border:1px solid rgba(0,200,150,.3);display:flex;align-items:center;justify-content:center;font-size:18.8px;font-weight:700;color:#00c896;flex-shrink:0;">Δ</div>
        <div style="flex:1;min-width:0;">
          <span style="font-size:13px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:rgba(0,200,150,.8);">DELTA</span>
          <span style="font-size:12.3px;color:rgba(255,255,255,.3);margin-left:6px;white-space:nowrap;">per 1pt move</span>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:21.8px;font-weight:700;color:${{nd>=0?'#00c896':'#ff6b6b'}};white-space:nowrap;flex-shrink:0;">${{ndStr}}</div>
      </div>

      <!-- Theta row -->
      <div style="display:flex;align-items:center;gap:10px;background:rgba(255,107,107,.07);border:1px solid rgba(255,107,107,.18);border-radius:9px;padding:8px 12px;">
        <div style="width:24px;height:24px;border-radius:6px;background:rgba(255,107,107,.18);border:1px solid rgba(255,107,107,.3);display:flex;align-items:center;justify-content:center;font-size:18.8px;font-weight:700;color:#ff9090;flex-shrink:0;">Θ</div>
        <div style="flex:1;min-width:0;">
          <span style="font-size:13px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:rgba(255,150,150,.8);">THETA</span>
          <span style="font-size:12.3px;color:rgba(255,255,255,.3);margin-left:6px;white-space:nowrap;">decay / day</span>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:21.8px;font-weight:700;color:${{ntCol}};white-space:nowrap;flex-shrink:0;">${{ntSign}}\u20b9${{Math.abs(Math.round(nt))}}</div>
      </div>

      <!-- Vega row -->
      <div style="display:flex;align-items:center;gap:10px;background:rgba(138,160,255,.07);border:1px solid rgba(138,160,255,.18);border-radius:9px;padding:8px 12px;">
        <div style="width:24px;height:24px;border-radius:6px;background:rgba(138,160,255,.18);border:1px solid rgba(138,160,255,.3);display:flex;align-items:center;justify-content:center;font-size:18.8px;font-weight:700;color:#8aa0ff;flex-shrink:0;">ν</div>
        <div style="flex:1;min-width:0;">
          <span style="font-size:13px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:rgba(138,160,255,.8);">VEGA</span>
          <span style="font-size:12.3px;color:rgba(255,255,255,.3);margin-left:6px;white-space:nowrap;">per 1% IV</span>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:21.8px;font-weight:700;color:#8aa0ff;white-space:nowrap;flex-shrink:0;">${{nvStr}}</div>
      </div>

    </div>
    <div style="height:1px;background:rgba(255,255,255,.05);margin:0 10px 4px;"></div>
    <div style="padding:10px 10px 6px;">
      <div style="font-size:11.6px;color:rgba(255,255,255,.25);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px;">TODAY'S P&amp;L IF MARKET IS FLAT</div>
      <div style="display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:8px;padding:12px 10px;background:rgba(0,0,0,.25);border-radius:10px;border:1px solid rgba(255,255,255,.06);">
        <div style="text-align:center;">
          <div style="font-size:11.6px;color:rgba(255,255,255,.3);letter-spacing:1px;text-transform:uppercase;margin-bottom:5px;">THETA DRAG</div>
          <div style="font-family:'DM Mono',monospace;font-size:26.1px;font-weight:700;color:${{ntCol}};line-height:1;">${{ntSign}}\u20b9${{Math.abs(Math.round(nt)).toLocaleString('en-IN')}}</div>
        </div>
        <div style="font-size:23.2px;color:rgba(255,255,255,.15);text-align:center;">=</div>
        <div style="text-align:center;">
          <div style="font-size:11.6px;color:rgba(255,255,255,.3);letter-spacing:1px;text-transform:uppercase;margin-bottom:5px;">FLAT P&amp;L TODAY</div>
          <div style="font-family:'DM Mono',monospace;font-size:26.1px;font-weight:700;color:${{flatCol}};line-height:1;">${{flatPnl>=0?'+':''}}\u20b9${{Math.abs(flatPnl).toLocaleString('en-IN')}}</div>
        </div>
      </div>
      <div style="margin-top:10px;font-size:13.8px;color:rgba(255,255,255,.35);line-height:1.7;">
        ${{nt < 0 ? '🔴 <b style="color:rgba(255,150,150,.8);">Theta negative</b> — you pay \u20b9' + Math.abs(Math.round(nt)).toLocaleString("en-IN") + '/day for holding. Need Nifty to move <b style="color:#ffd166;">' + Math.ceil(Math.abs(nt)/Math.abs(nd||1)) + ' pts</b> just to break even today.' : '🟢 <b style="color:rgba(0,200,150,.8);">Theta positive</b> — you earn \u20b9' + Math.abs(Math.round(nt)).toLocaleString("en-IN") + '/day time decay. Premium selling strategy benefits from flat market.'}}
      </div>
    </div>
  </div>

  <!-- TAB 3: Live Slider -->
  <div id="${{simId}}_c3" style="display:none;">
    <div style="padding:9px 12px 4px;">
      <div style="font-size:13px;font-weight:700;letter-spacing:1.5px;color:rgba(255,209,102,.8);text-transform:uppercase;">🎚 LIVE SCENARIO SLIDER</div>
    </div>
    <div style="padding:6px 12px 10px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:5px;font-size:13px;">
        <span style="color:rgba(255,255,255,.3);">\u20b9${{slMin.toLocaleString('en-IN')}}</span>
        <span id="${{simId}}_slv" style="font-family:'DM Mono',monospace;font-size:15.9px;font-weight:700;color:#f5c518;background:rgba(245,197,24,.12);border:1px solid rgba(245,197,24,.3);border-radius:5px;padding:1px 8px;">Spot: \u20b9${{OC.spot.toLocaleString('en-IN')}}</span>
        <span style="color:rgba(255,255,255,.3);">\u20b9${{slMax.toLocaleString('en-IN')}}</span>
      </div>
      <input type="range" id="${{simId}}_sl" min="${{slMin}}" max="${{slMax}}" value="${{OC.spot}}" step="25"
        style="width:100%;height:4px;border-radius:2px;outline:none;border:none;-webkit-appearance:none;cursor:pointer;background:linear-gradient(90deg,#f5c518 50%,rgba(255,255,255,.1) 50%);"
        onclick="event.stopPropagation()" onmousedown="event.stopPropagation()" ontouchstart="event.stopPropagation()" oninput="simSlide('${{simId}}', this.value, ${{slMin}}, ${{slMax}}, ${{nd}}, ${{nt}}, ${{maxL===null?'null':maxL}}, ${{maxP===null?'null':maxP}})">
    </div>
    <div id="${{simId}}_sr" style="padding:4px 12px 14px;text-align:center;">
      <div style="background:rgba(0,0,0,.3);border-radius:12px;padding:14px;border:1px solid rgba(255,255,255,.07);">
        <div style="font-size:12.3px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,.25);margin-bottom:6px;">ESTIMATED EXIT P&amp;L TODAY</div>
        <div id="${{simId}}_spnl" style="font-family:'DM Mono',monospace;font-size:43.5px;font-weight:700;color:${{ntCol}};">${{flatPnl>=0?'+':''}}\u20b9${{Math.abs(flatPnl).toLocaleString('en-IN')}}</div>
        <div id="${{simId}}_snote" style="font-size:13.8px;color:rgba(255,255,255,.3);margin-top:4px;">Flat market — theta drag only</div>
        <div style="display:flex;gap:12px;justify-content:center;margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,.05);">
          <div style="text-align:center;">
            <div style="font-size:11.6px;color:rgba(255,255,255,.3);letter-spacing:1px;text-transform:uppercase;margin-bottom:2px;">Delta P&amp;L</div>
            <div id="${{simId}}_sdelta" style="font-family:'DM Mono',monospace;font-size:17.4px;font-weight:700;color:#00c896;">\u20b90</div>
          </div>
          <div style="text-align:center;">
            <div style="font-size:11.6px;color:rgba(255,255,255,.3);letter-spacing:1px;text-transform:uppercase;margin-bottom:2px;">Theta Cost</div>
            <div style="font-family:'DM Mono',monospace;font-size:17.4px;font-weight:700;color:${{ntCol}};">${{ntSign}}\u20b9${{Math.abs(Math.round(nt)).toLocaleString('en-IN')}}</div>
          </div>
          <div style="text-align:center;">
            <div style="font-size:11.6px;color:rgba(255,255,255,.3);letter-spacing:1px;text-transform:uppercase;margin-bottom:2px;">% of Max</div>
            <div id="${{simId}}_spct" style="font-family:'DM Mono',monospace;font-size:17.4px;font-weight:700;color:#ffd166;">—</div>
          </div>
        </div>
      </div>
    </div>
  </div>

</div>`;
}}

function simTab(simId, tab) {{
  ['t1','t2','t3'].forEach(t => {{
    const btn = document.getElementById(simId + '_' + t);
    const con = document.getElementById(simId + '_c' + t.slice(1));
    if (!btn || !con) return;
    const isActive = t === tab;
    con.style.display = isActive ? 'block' : 'none';
    btn.style.borderBottomColor = isActive ? '#f5c518' : 'transparent';
    btn.style.color = isActive ? '#f5c518' : 'rgba(255,255,255,.3)';
    btn.style.background = isActive ? 'rgba(245,197,24,.07)' : 'transparent';
  }});
}}

function simSlide(simId, val, slMin, slMax, nd, nt, maxL, maxP) {{
  const spot = parseInt(val);
  const move = spot - OC.spot;
  const pct  = ((val - slMin) / (slMax - slMin) * 100);
  const sl   = document.getElementById(simId + '_sl');
  if (sl) sl.style.background = `linear-gradient(90deg,#f5c518 ${{pct}}%,rgba(255,255,255,.1) ${{pct}}%)`;
  const slv = document.getElementById(simId + '_slv');
  if (slv) slv.textContent = 'Spot: \u20b9' + spot.toLocaleString('en-IN');
  let pnl = nd * move + nt;
  if (maxL !== null) pnl = Math.max(-maxL, pnl);
  if (maxP !== null) pnl = Math.min(maxP * 0.9, pnl);
  pnl = Math.round(pnl);
  const col = pnl > 100 ? '#00c896' : pnl > 0 ? '#4de8b8' : pnl > -200 ? '#ffd166' : '#ff6b6b';
  const pEl = document.getElementById(simId + '_spnl');
  if (pEl) {{ pEl.textContent = (pnl>=0?'+':'') + '\u20b9' + Math.abs(pnl).toLocaleString('en-IN'); pEl.style.color = col; }}
  const nEl = document.getElementById(simId + '_snote');
  if (nEl) nEl.textContent = move > 0 ? `Nifty up ${{move}} pts` : move < 0 ? `Nifty down ${{Math.abs(move)}} pts` : 'Flat market — theta drag only';
  const deltaPnl = Math.round(nd * move);
  const dEl = document.getElementById(simId + '_sdelta');
  if (dEl) {{ dEl.textContent = (deltaPnl>=0?'+':'') + '\u20b9' + Math.abs(deltaPnl).toLocaleString('en-IN'); dEl.style.color = deltaPnl >= 0 ? '#00c896' : '#ff6b6b'; }}
  const pctEl = document.getElementById(simId + '_spct');
  if (pctEl) {{ const pPct = maxP ? ((pnl / maxP) * 100).toFixed(1) + '%' : '—'; pctEl.textContent = pPct; pctEl.style.color = pnl >= 0 ? '#00c896' : '#ff9090'; }}
}}

function popBadgeStyle(pop) {{
  if(pop>=70) return 'background:rgba(0,200,150,.25);color:#00c896;border-color:rgba(0,200,150,.5);font-weight:800;';
  if(pop>=60) return 'background:rgba(77,232,184,.2);color:#4de8b8;border-color:rgba(77,232,184,.4);';
  if(pop>=50) return 'background:rgba(100,128,255,.2);color:#8aa0ff;border-color:rgba(100,128,255,.4);';
  return 'background:rgba(255,107,107,.2);color:#ff6b6b;border-color:rgba(255,107,107,.4);';
}}

function initAllCards() {{
  let topPop=0, topName='', topCat='';
  const bullEx = smartPoP('bull_put_spread','bullish');
  const el_b = document.getElementById('legendBiasVal');
  if(el_b) {{ el_b.textContent=OC.bias+' ('+OC.biasConf+')'; el_b.style.color=OC.bias==='BULLISH'?'#00c896':OC.bias==='BEARISH'?'#ff6b6b':'#6480ff'; }}
  const srPts = bullEx.srAdj;
  const el_sr = document.getElementById('legendSRVal');
  if(el_sr) {{ const srLabel = srPts>5?'Near Support ✓':srPts<-5?'Near Resistance ✗':'Mid Range'; el_sr.textContent=srLabel+' ('+(srPts>=0?'+':'')+srPts+')'; el_sr.style.color=srPts>=0?'#00c896':'#ff6b6b'; }}
  const oiPts = bullEx.oiAdj;
  const el_oi = document.getElementById('legendOIVal');
  if(el_oi) {{ const oiLabel = OC.spot>OC.maxPeStrike?'Above PE Wall ✓':'Below PE Wall ✗'; el_oi.textContent=oiLabel+' ('+(oiPts>=0?'+':'')+oiPts+')'; el_oi.style.color=oiPts>=0?'#00c896':'#ff6b6b'; }}
  const el_pcr = document.getElementById('legendPCRVal');
  if(el_pcr) {{ const pcrLabel = OC.pcr>1.2?'Bullish PCR ':OC.pcr<0.8?'Bearish PCR ':'Neutral PCR '; el_pcr.textContent=pcrLabel+OC.pcr.toFixed(3); el_pcr.style.color=OC.pcr>1.2?'#00c896':OC.pcr<0.8?'#ff6b6b':'#6480ff'; }}
  document.querySelectorAll('.sc-card').forEach(card=>{{
    const shape=card.dataset.shape, cat=card.dataset.cat;
    const badge=document.getElementById('pop_'+card.id);
    try {{
      const result = smartPoP(shape, cat);
      const m=calcMetrics(shape, result.pop);
      card.dataset.pop=result.pop;
      card.dataset.scoreBreakdown=JSON.stringify(result);
      if(badge) {{ badge.textContent=result.pop+'%'; badge.setAttribute('style', badge.getAttribute('style')||''); badge.setAttribute('style', popBadgeStyle(result.pop)); }}
      if(result.pop>topPop) {{ topPop=result.pop; topName=card.dataset.name; topCat=cat; }}
    }}catch(e){{card.dataset.pop=0;if(badge)badge.textContent='—%';}}
  }});
  const el_rec = document.getElementById('legendRecVal');
  if(el_rec && topName) {{
    const recCol = topCat==='bullish'?'#00c896':topCat==='bearish'?'#ff6b6b':'#6480ff';
    el_rec.innerHTML=`<span style="color:${{recCol}};">${{topName}}</span> <span style="color:rgba(255,255,255,.4);font-size:13px;">${{topPop}}% PoP</span>`;
  }}
}}

function sortGridByPoP(cat) {{
  const grid=document.getElementById('sc-grid'); if(!grid)return;
  const cards=Array.from(grid.querySelectorAll(`.sc-card[data-cat="${{cat}}"]`));
  cards.sort((a,b)=>parseInt(b.dataset.pop||0)-parseInt(a.dataset.pop||0));
  cards.forEach(c=>grid.appendChild(c));
}}

window.addEventListener('load',function(){{
  initAllCards();
  ['bullish','bearish','nondirectional'].forEach(sortGridByPoP);
  filterStrat('bullish',document.querySelector('.sc-tab'));
}});

// ── Multi-Expiry Switcher ─────────────────────────────────────
const ALL_EXPIRY_DATA = {all_expiry_json};

window.switchExpiry = function(exp) {{
  let d = ALL_EXPIRY_DATA[exp];
  if (!d) {{
    const sel = document.getElementById('expiryDropdown');
    if (sel) {{
      const opts = Array.from(sel.options);
      const curIdx = opts.findIndex(o => o.value === exp);
      for (let i = curIdx + 1; i < opts.length; i++) {{
        const nextExp = opts[i].value;
        if (ALL_EXPIRY_DATA[nextExp]) {{
          sel.value = nextExp;
          exp = nextExp;
          d = ALL_EXPIRY_DATA[nextExp];
          break;
        }}
      }}
    }}
    if (!d) return;
  }}
  // Update OC object with selected expiry's data
  OC.spot        = d.spot;
  OC.atm         = d.atm;
  OC.pcr         = d.pcr;
  OC.maxCeStrike = d.maxCeStrike;
  OC.maxPeStrike = d.maxPeStrike;
  OC.support     = d.support;
  OC.resistance  = d.resistance;
  OC.strongSup   = d.strongSup;
  OC.strongRes   = d.strongRes;
  OC.strikes     = d.strikes;
  // Rebuild strike map
  Object.keys(STRIKE_MAP).forEach(k => delete STRIKE_MAP[k]);
  OC.strikes.forEach(s => {{ STRIKE_MAP[s.strike] = s; }});
  // Collapse all expanded cards first
  document.querySelectorAll('.sc-card.expanded').forEach(c => c.classList.remove('expanded'));
  // Reset all metrics panels
  document.querySelectorAll('.sc-metrics-live').forEach(m => {{
    m.innerHTML = '<div class="sc-loading">&#9685; Calculating metrics...</div>';
  }});
  // Re-run PoP + sort
  initAllCards();
  ['bullish','bearish','nondirectional'].forEach(sortGridByPoP);
  // Flash indicator
  const sel = document.getElementById('expiryDropdown');
  if (sel) {{
    sel.style.borderColor = '#00c896';
    sel.style.boxShadow = '0 0 10px rgba(0,200,150,.3)';
    setTimeout(() => {{
      sel.style.borderColor = 'rgba(245,197,24,.45)';
      sel.style.boxShadow = 'none';
    }}, 800);
  }}
}};
</script>
"""


# =================================================================
#  SECTION 7 -- TICKER BAR
# =================================================================

def rsi_label(rsi):
    if   rsi >= 70: return "Overbought","#ff6b6b","bearish"
    elif rsi >= 60: return "Strong",    "#ffd166","neutral"
    elif rsi >= 40: return "Neutral",   "#6480ff","neutral"
    elif rsi >= 30: return "Weak",      "#ffd166","neutral"
    else:           return "Oversold",  "#00c896","bullish"

def macd_label(macd, signal):
    d = macd - signal
    if   d >  0.5: return "Bullish",       "#00c896","bullish"
    elif d >  0:   return "Mildly Bullish","#4de8b8","bullish"
    elif d > -0.5: return "Mildly Bearish","#ffd166","neutral"
    else:          return "Bearish",       "#ff6b6b","bearish"

def pcr_label(pcr):
    if   pcr > 1.3: return "Very Bullish","#00c896","bullish"
    elif pcr > 1.1: return "Bullish",     "#4de8b8","bullish"
    elif pcr > 0.9: return "Neutral",     "#6480ff","neutral"
    elif pcr > 0.7: return "Bearish",     "#ffd166","neutral"
    else:           return "Very Bearish","#ff6b6b","bearish"


def build_ticker_bar(tech, oc, vix_data):
    items = []
    if vix_data:
        v=vix_data["value"]; chg=vix_data["change"]; chg_p=vix_data["change_pct"]
        lbl,col,bg,bdr,sig=vix_label(v)
        chg_col="#ff6b6b" if chg>0 else ("#00c896" if chg<0 else "#6480ff")
        items.append(
            f'<div class="tk-item">'
            f'<span class="tk-name" style="background:rgba(0,200,220,.15);color:#00c8e0;border:1px solid rgba(0,200,220,.3);">&#9650;&nbsp;INDIA VIX</span>'
            f'<span class="tk-val" style="color:{col};">{v:.2f}</span>'
            f'<span class="tk-sub" style="color:{chg_col};">{chg:+.2f} ({chg_p:+.2f}%)</span>'
            f'<span class="tk-badge" style="background:{bg};color:{col};border:1px solid {bdr};">{lbl} · {sig}</span>'
            f'</div>'
        )
    if tech:
        rsi=tech["rsi"]; lbl,col,cls=rsi_label(rsi)
        rgb="0,200,150" if cls=="bullish" else ("255,107,107" if cls=="bearish" else "100,128,255")
        items.append(
            f'<div class="tk-item">'
            f'<span class="tk-name" style="background:rgba({rgb},.18);color:{col};border:1px solid rgba({rgb},.35);">&#9643;&nbsp;RSI (14)</span>'
            f'<span class="tk-val" style="color:{col};">{rsi:.1f}</span>'
            f'<span class="tk-badge" style="background:rgba({rgb},.1);color:{col};border:1px solid rgba({rgb},.3);">{lbl}</span>'
            f'</div>'
        )
        macd=tech["macd"]; sig2=tech["signal_line"]; lbl,col,cls=macd_label(macd,sig2)
        rgb2="0,200,150" if cls=="bullish" else ("255,107,107" if cls=="bearish" else "100,128,255")
        diff=macd-sig2
        items.append(
            f'<div class="tk-item">'
            f'<span class="tk-name" style="background:rgba({rgb2},.18);color:{col};border:1px solid rgba({rgb2},.35);">&#9654;&nbsp;MACD</span>'
            f'<span class="tk-val" style="color:{col};">{macd:.2f}</span>'
            f'<span class="tk-sub" style="color:rgba(255,255,255,.4);">Sig:&nbsp;{sig2:.2f}&nbsp;Hist:&nbsp;{diff:+.2f}</span>'
            f'<span class="tk-badge" style="background:rgba({rgb2},.1);color:{col};border:1px solid rgba({rgb2},.3);">{lbl}</span>'
            f'</div>'
        )
    if oc:
        pcr=oc["pcr_oi"]; lbl,col,cls=pcr_label(pcr)
        rgb3="0,200,150" if cls=="bullish" else ("255,107,107" if cls=="bearish" else "100,128,255")
        items.append(
            f'<div class="tk-item">'
            f'<span class="tk-name" style="background:rgba({rgb3},.18);color:{col};border:1px solid rgba({rgb3},.35);">&#9670;&nbsp;PCR (OI)</span>'
            f'<span class="tk-val" style="color:{col};">{pcr:.3f}</span>'
            f'<span class="tk-badge" style="background:rgba({rgb3},.1);color:{col};border:1px solid rgba({rgb3},.3);">{lbl}</span>'
            f'</div>'
        )
    track = "".join(items) * 3
    return f'''<div class="ticker-wrap">
  <div class="ticker-label">LIVE&nbsp;&#9654;</div>
  <div class="ticker-viewport"><div class="ticker-track" id="tkTrack">''' + track + '''</div></div>
</div>'''

# =================================================================
#  SECTION 8 -- CSS
# =================================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Mono:wght@300;400;500&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#06080f;--surf:#080b14;--card:#0c1020;
  --bdr:rgba(255,255,255,.07);--bdr2:rgba(255,255,255,.12);
  --aurora1:#00c896;--aurora2:#6480ff;--aurora3:#00c8e0;
  --bull:#00c896;--bear:#ff6b6b;--neut:#6480ff;
  --text:rgba(255,255,255,.9);--muted:rgba(255,255,255,.45);--muted2:rgba(255,255,255,.28);
  --fh:'Sora',sans-serif;--fm:'DM Mono',monospace;
  --gold:#f5c518;--gold-dim:rgba(245,197,24,.45);--gold-bg:rgba(245,197,24,.10);
}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:var(--fh);font-size:18.8px;line-height:1.6;min-height:100vh}
body::before{content:'';position:fixed;inset:0;
  background-image:
    radial-gradient(ellipse at 15% 0%,rgba(0,200,150,.10) 0%,transparent 50%),
    radial-gradient(ellipse at 85% 10%,rgba(100,128,255,.10) 0%,transparent 50%),
    radial-gradient(ellipse at 50% 90%,rgba(0,200,220,.06) 0%,transparent 50%);
  pointer-events:none;z-index:0;}
.app{position:relative;z-index:1;display:grid;grid-template-rows:auto auto auto 1fr auto;min-height:100vh}
header{display:flex;align-items:center;justify-content:space-between;padding:14px 32px;
  background:rgba(6,8,15,.85);backdrop-filter:blur(16px);
  border-bottom:1px solid rgba(255,255,255,.07);position:sticky;top:0;z-index:200;
  box-shadow:0 1px 0 rgba(0,200,150,.1)}
.logo-wrap{position:relative;height:28px;overflow:hidden;min-width:280px;}
.logo-slide{position:absolute;top:0;left:0;width:100%;font-family:var(--fh);font-size:29px;font-weight:700;
  background:linear-gradient(90deg,#00c896,#6480ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  filter:drop-shadow(0 0 12px rgba(0,200,150,.3));opacity:0;transform:translateY(20px);
  transition:opacity .5s ease, transform .5s ease;white-space:nowrap;}
.logo-slide.active{opacity:1;transform:translateY(0);}
.logo-slide.exit{opacity:0;transform:translateY(-20px);}
.hdr-meta{display:flex;align-items:center;gap:14px;font-size:15.9px;color:var(--muted);font-family:var(--fm)}
.live-dot{width:7px;height:7px;border-radius:50%;background:#00c896;box-shadow:0 0 10px #00c896;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.2}}
.refresh-countdown{display:flex;align-items:center;gap:8px;background:rgba(0,200,150,.07);
  border:1px solid rgba(0,200,150,.18);border-radius:20px;padding:4px 12px;font-family:var(--fm);font-size:15.9px;}
.countdown-arc-wrap{position:relative;width:18px;height:18px;flex-shrink:0;}
.countdown-arc-wrap svg{display:block;}
.countdown-num{font-family:var(--fm);font-size:17.4px;font-weight:700;color:#00c896;min-width:20px;text-align:center;transition:color .3s;}
.countdown-num.urgent{color:#ff6b6b;}
.countdown-num.halfway{color:#ffd166;}
.countdown-lbl{font-size:14.5px;color:rgba(255,255,255,.3);letter-spacing:.5px;}
.refresh-ring{display:none;width:14px;height:14px;border-radius:50%;border:2px solid rgba(0,200,150,.2);border-top-color:#00c896;animation:spin 0.8s linear infinite;}
.refresh-ring.active{display:inline-block;}
@keyframes spin{to{transform:rotate(360deg)}}
#refreshStatus{font-size:14.5px;color:rgba(255,255,255,.35);transition:color .3s;letter-spacing:.3px;}
#refreshStatus.updated{color:#00c896;font-weight:600;}
.hero{display:flex;align-items:stretch;background:linear-gradient(135deg,rgba(0,200,150,.055) 0%,rgba(100,128,255,.055) 100%);border-bottom:1px solid rgba(255,255,255,.07);overflow:hidden;position:relative;height:97px;}
.hero::before{content:'';position:absolute;top:-50px;left:-50px;width:200px;height:200px;border-radius:50%;background:radial-gradient(circle,rgba(0,200,150,.10),transparent 70%);pointer-events:none;}
.h-gauges{flex-shrink:0;display:flex;align-items:center;gap:10px;padding:0 16px 0 18px;}
.gauge-sep{width:1px;height:56px;background:rgba(255,255,255,.08);flex-shrink:0;}
.gauge-wrap{position:relative;width:76px;height:76px;}
.gauge-wrap svg{display:block;}
.gauge-inner{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.g-val{font-family:'DM Mono',monospace;font-size:18.8px;font-weight:700;line-height:1;}
.g-lbl{font-size:10.9px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.28);margin-top:2px;}
.h-mid{flex:1;min-width:0;display:flex;flex-direction:column;justify-content:center;padding:0 15px 0 13px;border-left:1px solid rgba(255,255,255,.05);}
.h-eyebrow{font-size:11.6px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,.22);margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.h-signal{font-size:31.9px;font-weight:900;letter-spacing:1px;line-height:1.1;margin-bottom:2px;}
.h-sub{font-size:13.8px;color:rgba(255,255,255,.32);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.h-divider{height:1px;background:rgba(255,255,255,.05);margin:5px 0;}
.pill-row{display:flex;align-items:center;gap:8px;margin-bottom:4px;}
.pill-row:last-child{margin-bottom:0;}
.pill-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.pill-lbl{font-size:11.6px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.35);width:96px;flex-shrink:0;}
.pill-track{width:120px;height:5px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;flex-shrink:0;}
.pill-fill{height:100%;border-radius:3px;}
.pill-num{font-family:'DM Mono',monospace;font-size:14.5px;font-weight:700;margin-left:8px;flex-shrink:0;}
.h-stats{flex-shrink:0;min-width:360px;display:flex;flex-direction:column;border-left:1px solid rgba(255,255,255,.07);background:rgba(255,255,255,.015);}
.h-stat-row{display:flex;align-items:stretch;flex:1;border-bottom:1px solid rgba(255,255,255,.05);}
.h-stat{flex:1;display:flex;flex-direction:column;justify-content:center;padding:5px 10px;text-align:center;border-right:1px solid rgba(255,255,255,.04);}
.h-stat:last-child{border-right:none;}
.h-stat-lbl{font-size:10.9px;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,.22);margin-bottom:3px;white-space:nowrap;}
.h-stat-val{font-family:'DM Mono',monospace;font-size:18.8px;font-weight:700;line-height:1;white-space:nowrap;}
.h-stat-bottom{display:flex;align-items:center;justify-content:space-between;padding:4px 10px;}
.h-bias-row{display:flex;align-items:center;gap:6px;}
.h-chip{font-size:13px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;padding:2px 9px;border-radius:20px;white-space:nowrap;}
.h-score{font-family:'DM Mono',monospace;font-size:11.6px;color:rgba(255,255,255,.22);letter-spacing:.5px;}
.h-ts{font-family:'DM Mono',monospace;font-size:11.6px;color:rgba(255,255,255,.18);letter-spacing:.5px;white-space:nowrap;}
.main{display:grid;grid-template-columns:268px 1fr;min-height:0}
.sidebar{background:rgba(8,11,20,.7);backdrop-filter:blur(12px);border-right:1px solid rgba(255,255,255,.06);position:sticky;top:57px;height:calc(100vh - 57px);overflow-y:auto;display:flex;flex-direction:column;}
.sidebar-sticky-top{position:sticky;top:0;z-index:50;background:rgba(8,11,20,.95);backdrop-filter:blur(16px);border-bottom:1px solid rgba(100,128,255,.15);padding-bottom:4px;}
.sidebar-scroll{flex:1;overflow-y:auto;}
.sidebar::-webkit-scrollbar{width:3px}
.sidebar::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:2px}
.sidebar-scroll::-webkit-scrollbar{width:3px}
.sidebar-scroll::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:2px}
.sb-sec{padding:16px 12px 8px}
.sb-lbl{font-size:13px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--aurora1);margin-bottom:8px;padding:0 0 0 8px;border-left:2px solid var(--aurora1)}
.sb-btn{display:flex;align-items:center;gap:8px;width:100%;padding:9px 12px;border-radius:8px;border:1px solid transparent;cursor:pointer;background:transparent;color:var(--muted);font-family:var(--fh);font-size:17.4px;text-align:left;transition:all .15s}
.sb-btn:hover{background:rgba(0,200,150,.08);color:rgba(255,255,255,.8);border-color:rgba(0,200,150,.2)}
.sb-btn.active{background:rgba(0,200,150,.1);border-color:rgba(0,200,150,.25);color:#00c896;font-weight:600}
.sb-badge{font-size:14.5px;margin-left:auto;font-weight:700}
.content{overflow-y:auto}
.section{padding:26px 28px;border-bottom:1px solid rgba(255,255,255,.05);background:transparent;position:relative}
.section:nth-child(odd){background:rgba(255,255,255,.015)}
.sec-title{font-family:var(--fh);font-size:15.9px;font-weight:700;letter-spacing:2.5px;color:var(--aurora1);text-transform:uppercase;display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:20px;padding-bottom:12px;border-bottom:1px solid rgba(0,200,150,.15)}
.sec-sub{font-size:15.9px;color:var(--muted2);font-weight:400;letter-spacing:.5px;text-transform:none;margin-left:auto}
.oi-ticker-table{border:1px solid rgba(255,255,255,.07);border-radius:14px;overflow:hidden}
.oi-ticker-hdr{display:grid;grid-template-columns:130px repeat(5,1fr);padding:9px 18px;align-items:center;gap:6px}
.oi-ticker-hdr-label{font-size:13px;font-weight:700;letter-spacing:2px;text-transform:uppercase}
.oi-ticker-hdr-cell{font-size:13px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.35);text-align:center}
.oi-ticker-row{display:grid;grid-template-columns:130px repeat(5,1fr);padding:15px 18px;border-top:1px solid rgba(255,255,255,.04);align-items:center;gap:6px;transition:background .15s}
.oi-ticker-row:hover{background:rgba(255,255,255,.03)}
.oi-ticker-metric{font-size:14.5px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:rgba(255,255,255,.35)}
.oi-ticker-cell{text-align:center}
.kl-zone-labels{display:flex;justify-content:space-between;margin-bottom:6px;font-size:15.9px;font-weight:700}
.kl-node{position:absolute;text-align:center}
.kl-lbl{font-size:14.5px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;line-height:1.3;white-space:nowrap}
.kl-val{font-size:17.4px;font-weight:700;color:rgba(255,255,255,.7);white-space:nowrap;margin-top:2px}
.kl-dot{width:11px;height:11px;border-radius:50%;border:2px solid var(--bg)}
.kl-gradient-bar{position:relative;height:6px;border-radius:3px;background:linear-gradient(90deg,#00a07a 0%,#00c896 25%,#6480ff 55%,#ff6b6b 80%,#cc4040 100%);box-shadow:0 0 12px rgba(0,200,150,.2)}
.kl-price-tick{position:absolute;top:50%;transform:translate(-50%,-50%);width:3px;height:18px;background:#fff;border-radius:2px;box-shadow:0 0 12px rgba(255,255,255,.6);z-index:10}
.kl-dist-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px}
.kl-dist-box{background:rgba(255,255,255,.03);border:1px solid;border-radius:10px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center}
.strikes-head{font-weight:700;margin-bottom:10px;font-size:18.8px}
.strikes-wrap{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.s-table{width:100%;border-collapse:collapse;border-radius:10px;overflow:hidden}
.s-table th{background:linear-gradient(90deg,rgba(0,200,150,.15),rgba(100,128,255,.15));color:rgba(255,255,255,.7);padding:10px 12px;font-size:15.9px;font-weight:600;text-align:left;letter-spacing:.5px;border-bottom:1px solid rgba(255,255,255,.08)}
.s-table td{padding:10px 12px;border-bottom:1px solid rgba(255,255,255,.05);font-size:17.4px;color:rgba(255,255,255,.8);background:rgba(255,255,255,.02)}
.s-table tr:last-child td{border-bottom:none}
.s-table tr:hover td{background:rgba(0,200,150,.05)}
.ticker-wrap{display:flex;align-items:center;background:rgba(4,6,12,.97);border-bottom:1px solid rgba(255,255,255,.07);height:46px;overflow:hidden;position:relative;z-index:190;box-shadow:0 2px 20px rgba(0,0,0,.5);}
.ticker-label{flex-shrink:0;padding:0 16px;font-family:var(--fm);font-size:13px;font-weight:700;letter-spacing:3px;color:#00c896;text-transform:uppercase;border-right:1px solid rgba(0,200,150,.2);height:100%;display:flex;align-items:center;background:rgba(0,200,150,.07);white-space:nowrap;}
.ticker-viewport{flex:1;overflow:hidden;height:100%}
.ticker-track{display:flex;align-items:center;height:100%;white-space:nowrap;animation:ticker-scroll 38s linear infinite;will-change:transform;}
.ticker-track:hover{animation-play-state:paused}
@keyframes ticker-scroll{0%{transform:translateX(0)}100%{transform:translateX(-33.333%)}}
.tk-item{display:inline-flex;align-items:center;gap:10px;padding:0 20px;height:100%;border-right:1px solid rgba(255,255,255,.04);flex-shrink:0;}
.tk-name{font-family:var(--fm);font-size:14.5px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;padding:3px 10px;border-radius:6px;white-space:nowrap;flex-shrink:0;background:rgba(255,255,255,.08);color:rgba(255,255,255,.5);border:1px solid rgba(255,255,255,.1);}
.tk-val{font-family:var(--fm);font-size:26.1px;font-weight:700;line-height:1;white-space:nowrap;}
.tk-sub{font-family:var(--fm);font-size:14.5px;color:rgba(255,255,255,.35);white-space:nowrap;}
.tk-badge{font-family:var(--fh);font-size:14.5px;font-weight:700;padding:3px 10px;border-radius:20px;white-space:nowrap;letter-spacing:.3px;}
footer{padding:16px 32px;border-top:1px solid rgba(255,255,255,.06);background:rgba(6,8,15,.9);backdrop-filter:blur(12px);display:flex;justify-content:space-between;font-size:15.9px;color:var(--muted2);font-family:var(--fm)}
.sc-tabs{display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap}
.sc-tab{padding:8px 20px;border-radius:24px;border:1px solid;cursor:pointer;font-family:var(--fh);font-size:17.4px;font-weight:600;transition:all .2s;display:flex;align-items:center;gap:8px;background:transparent}
.sc-tab:hover{opacity:.85}
.sc-cnt{font-size:14.5px;padding:1px 7px;border-radius:10px;color:#fff;font-weight:700}
.sc-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}
.sc-card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:14px;overflow:hidden;cursor:pointer;transition:all .2s;display:flex;flex-direction:column;position:relative;}
.sc-card:hover{border-color:rgba(0,200,150,.3);transform:translateY(-3px);box-shadow:0 8px 28px rgba(0,200,150,.1)}
.sc-card.hidden{display:none}
.sc-card.expanded .sc-detail{display:block}
.sc-card.expanded{border-color:rgba(0,200,150,.35);box-shadow:0 0 0 1px rgba(0,200,150,.2),0 12px 32px rgba(0,200,150,.12)}
.sc-pop-badge{position:absolute;top:8px;right:8px;font-family:'DM Mono',monospace;font-size:14.5px;font-weight:700;padding:3px 8px;border-radius:20px;border:1px solid rgba(255,255,255,.15);background:rgba(255,255,255,.08);color:rgba(255,255,255,.5);z-index:5;letter-spacing:.5px;transition:all .3s;min-width:38px;text-align:center;}
.sc-svg{display:flex;align-items:center;justify-content:center;padding:14px 0 6px;background:rgba(255,255,255,.02)}
.sc-body{padding:10px 12px 12px}
.sc-name{font-family:var(--fh);font-size:17.4px;font-weight:700;color:rgba(255,255,255,.9);margin-bottom:4px;line-height:1.3;padding-right:48px}
.sc-legs{font-family:var(--fm);font-size:13px;color:rgba(0,200,220,.7);margin-bottom:8px;letter-spacing:.3px;line-height:1.4}
.sc-tags{display:flex;flex-direction:column;gap:4px}
.sc-tag{font-size:13px;padding:2px 8px;border-radius:6px;border:1px solid;background:rgba(0,0,0,.2);display:inline-block;width:fit-content}
.sc-detail{display:none;border-top:1px solid rgba(255,255,255,.06);background:rgba(0,200,150,.03)}
.sc-desc{font-size:15.9px;color:rgba(255,255,255,.5);line-height:1.7;padding:12px 12px 8px;border-bottom:1px solid rgba(255,255,255,.05);}
.sc-metrics-live{padding:0}
.sc-loading{padding:14px 12px;font-size:15.9px;color:rgba(255,255,255,.3);text-align:center;font-family:'DM Mono',monospace}
.metric-row{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.04);transition:background .15s;}
.metric-row:hover{background:rgba(255,255,255,.03)}
.metric-strike{background:rgba(255,209,102,.04);border-bottom:1px solid rgba(255,209,102,.12) !important;}
.metric-lbl{font-size:14.5px;color:rgba(255,255,255,.35);letter-spacing:.5px;text-transform:uppercase;font-family:'DM Mono',monospace;}
.metric-val{font-family:'DM Mono',monospace;font-size:17.4px;font-weight:600;text-align:right;}
.greeks-panel{margin:10px 10px 6px;padding:14px 12px;background:linear-gradient(135deg,rgba(100,128,255,.12),rgba(0,200,220,.10));border-radius:14px;border:1px solid rgba(100,128,255,.28);box-shadow:0 4px 20px rgba(100,128,255,.1),inset 0 1px 0 rgba(255,255,255,.06);}
.greeks-title{font-size:13px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:rgba(138,160,255,1.0);margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid rgba(100,128,255,.25);display:flex;align-items:center;justify-content:space-between;}
.greeks-expiry-tag{font-size:12.3px;color:rgba(255,255,255,.5);font-weight:400;letter-spacing:.5px;text-transform:none;}
.greeks-strike-wrap{position:relative;margin-bottom:10px;}
.greeks-strike-wrap::after{content:'▼';position:absolute;right:10px;top:50%;transform:translateY(-50%);font-size:11.6px;color:var(--gold);pointer-events:none;z-index:2;}
.greeks-strike-select{width:100%;appearance:none;-webkit-appearance:none;background:linear-gradient(135deg,rgba(245,197,24,.12),rgba(200,155,10,.06));border:1px solid var(--gold-dim);border-radius:8px;color:var(--gold);font-family:'DM Mono',monospace;font-size:15.9px;font-weight:700;padding:7px 28px 7px 10px;cursor:pointer;outline:none;letter-spacing:.5px;transition:border-color .2s,background .2s,box-shadow .2s;}
.greeks-strike-select:hover{border-color:rgba(245,197,24,.75);background:linear-gradient(135deg,rgba(245,197,24,.18),rgba(200,155,10,.10));box-shadow:0 0 10px rgba(245,197,24,.18);}
.greeks-strike-select:focus{border-color:var(--gold);box-shadow:0 0 0 2px rgba(245,197,24,.25);}
.greeks-strike-select option{background:#0e1225;color:var(--gold);font-weight:700;}
.greek-name{font-family:'DM Mono',monospace;font-size:15.9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:rgba(255,255,255,.92);}
.greek-sub{font-size:11.6px;color:rgba(255,255,255,.55);margin-top:1px;}
.greeks-row{display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.06);}
.greeks-row:last-child{border-bottom:none;}
.greeks-atm-badge{display:flex;align-items:center;justify-content:center;gap:6px;background:rgba(100,128,255,.1);border:1px solid rgba(100,128,255,.25);border-radius:8px;padding:5px 8px;margin-bottom:10px;font-family:'DM Mono',monospace;font-size:15.9px;flex-wrap:wrap;}
.greeks-atm-strike{font-weight:700;color:#8aa0ff;}
.iv-bar-wrap{display:flex;align-items:center;gap:6px;margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,.06);}
.iv-bar-label{font-size:11.6px;color:rgba(255,255,255,.7);letter-spacing:1px;text-transform:uppercase;font-weight:600;width:42px;flex-shrink:0;}
.iv-bar-track{flex:1;height:4px;background:rgba(255,255,255,.08);border-radius:2px;overflow:hidden;}
.iv-bar-fill{height:100%;border-radius:2px;transition:width .6s ease;}
.iv-bar-num{font-family:'DM Mono',monospace;font-size:15.9px;font-weight:700;min-width:38px;text-align:right;}
.greeks-table-section{padding:22px 28px;border-bottom:1px solid rgba(255,255,255,.05);}
.greeks-table-wrap{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
.greeks-tbl{border:1px solid rgba(255,255,255,.07);border-radius:12px;overflow:hidden;}
.greeks-tbl-head{display:grid;grid-template-columns:90px repeat(4,1fr);background:rgba(255,255,255,.04);padding:8px 14px;border-bottom:1px solid rgba(255,255,255,.06);gap:4px;}
.greeks-tbl-head-label{font-size:12.3px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.3);text-align:center;}
.greeks-tbl-row{display:grid;grid-template-columns:90px repeat(4,1fr);padding:9px 14px;border-bottom:1px solid rgba(255,255,255,.04);align-items:center;gap:4px;transition:background .15s;}
.greeks-tbl-row:last-child{border-bottom:none;}
.greeks-tbl-row:hover{background:rgba(255,255,255,.03);}
.greeks-tbl-strike{font-family:'DM Mono',monospace;font-size:17.4px;font-weight:700;color:rgba(255,255,255,.8);}
.greeks-tbl-cell{font-family:'DM Mono',monospace;font-size:15.9px;font-weight:600;text-align:center;color:rgba(255,255,255,.65);}
/* Hidden refresh iframe — zero footprint */
#silentRefreshFrame{position:fixed;width:0;height:0;border:none;visibility:hidden;pointer-events:none;opacity:0;}
@media(max-width:1024px){
  .main{grid-template-columns:1fr}.sidebar{position:static;height:auto;border-right:none;border-bottom:1px solid rgba(255,255,255,.06)}
  .hero{height:auto;flex-wrap:wrap;}.h-gauges{padding:12px 18px;}.h-stats{min-width:100%;border-left:none;border-top:1px solid rgba(255,255,255,.07);}
  .strikes-wrap{grid-template-columns:1fr}
  .greeks-table-wrap{grid-template-columns:1fr}
  .sc-grid{grid-template-columns:repeat(auto-fill,minmax(160px,1fr))}
}
@media(max-width:640px){
  header{padding:12px 16px}.section{padding:18px 16px}
  .kl-dist-row{grid-template-columns:1fr}footer{flex-direction:column;gap:6px}
  .logo-wrap{min-width:160px;}.refresh-countdown{display:none;}
  .sc-grid{grid-template-columns:repeat(auto-fill,minmax(150px,1fr))}
}
"""

# =================================================================
#  SECTION 9 -- ANIMATED JS  (v18.3 — SILENT BACKGROUND REFRESH)
# =================================================================

ANIMATED_JS = """
<script>
// ── Logo rotator ────────────────────────────────────────────────
(function() {
  const NAMES = ['NIFTYCRAFT','Nifty Option Strategy Builder','OI Signal Dashboard','Options Analytics Hub','PCR & Max Pain Tracker'];
  const wrap = document.getElementById('logoWrap');
  if (!wrap) return;
  NAMES.forEach((name, i) => {
    const el = document.createElement('div');
    el.className = 'logo-slide' + (i === 0 ? ' active' : '');
    el.textContent = name;
    wrap.appendChild(el);
  });
  let cur = 0;
  setInterval(() => {
    const slides = wrap.querySelectorAll('.logo-slide');
    slides[cur].classList.remove('active');
    slides[cur].classList.add('exit');
    setTimeout(() => slides[cur].classList.remove('exit'), 600);
    cur = (cur + 1) % slides.length;
    slides[cur].classList.add('active');
  }, 4000);
})();

// ── Silent Background Refresh Engine (v18.3) ───────────────────
(function() {
  const TOTAL_SECS = 30;
  const R = 7, C = 2 * Math.PI * R;

  function setCountdownUI(secs) {
    const numEl = document.getElementById('cdNum');
    const arcEl = document.getElementById('cdArc');
    if (numEl) {
      numEl.textContent = secs;
      numEl.className = 'countdown-num' +
        (secs <= 5 ? ' urgent' : secs <= 15 ? ' halfway' : '');
    }
    if (arcEl) {
      arcEl.style.strokeDashoffset = (C * (1 - secs / TOTAL_SECS)).toFixed(2);
      arcEl.style.stroke = secs <= 5 ? '#ff6b6b' : secs <= 15 ? '#ffd166' : '#00c896';
    }
  }

  function showSpinner(on) {
    const ring = document.getElementById('refreshRing');
    if (ring) ring.classList.toggle('active', on);
  }

  function flashUpdated() {
    const txt = document.getElementById('refreshStatus');
    if (!txt) return;
    txt.textContent = 'Updated \u2713';
    txt.classList.add('updated');
    setTimeout(() => { txt.textContent = ''; txt.classList.remove('updated'); }, 2500);
  }

  const PATCH_IDS = [
    'heroWidget','oi','kl','strikes','greeksTable','greeksPanel','tkTrack','lastUpdatedTs'
  ];

  function microDiff(newDoc) {
    let changed = false;
    PATCH_IDS.forEach(function(id) {
      const curEl = document.getElementById(id);
      const newEl = newDoc.getElementById(id);
      if (!curEl || !newEl) return;
      if (curEl.innerHTML !== newEl.innerHTML) {
        const content = document.querySelector('.content');
        const scrollTop = content ? content.scrollTop : 0;
        curEl.innerHTML = newEl.innerHTML;
        if (content) content.scrollTop = scrollTop;
        changed = true;
      }
    });
    return changed;
  }

  let iframe = document.getElementById('silentRefreshFrame');
  if (!iframe) {
    iframe = document.createElement('iframe');
    iframe.id = 'silentRefreshFrame';
    iframe.style.cssText = 'position:fixed;width:0;height:0;border:none;' +
                            'visibility:hidden;pointer-events:none;opacity:0;' +
                            'top:-9999px;left:-9999px;';
    document.body.appendChild(iframe);
  }

  let _lastTimestamp  = null;
  let _refreshing     = false;

  function doSilentRefresh() {
    if (_refreshing) return;
    _refreshing = true;
    showSpinner(true);
    iframe.src = 'index.html?_=' + Date.now();
  }

  iframe.addEventListener('load', function() {
    if (!iframe.src || iframe.src === 'about:blank') {
      _refreshing = false; showSpinner(false); return;
    }
    try {
      const newDoc = iframe.contentDocument || iframe.contentWindow.document;
      if (!newDoc || !newDoc.body) throw new Error('empty doc');
      const newTsEl = newDoc.getElementById('lastUpdatedTs');
      const newTs   = newTsEl ? newTsEl.textContent.trim() : '';
      if (_lastTimestamp !== null && newTs === _lastTimestamp) {
        showSpinner(false); _refreshing = false; return;
      }
      _lastTimestamp = newTs;
      const changed = microDiff(newDoc);
      showSpinner(false); _refreshing = false;
      if (changed) {
        flashUpdated();
        setTimeout(function() {
          try {
            if (typeof initAllCards === 'function') {
              initAllCards();
              ['bullish','bearish','nondirectional'].forEach(function(c) {
                if (typeof sortGridByPoP === 'function') sortGridByPoP(c);
              });
            }
            if (typeof greeksUpdateStrike === 'function') {
              var sel = document.getElementById('greeksStrikeSelect');
              if (sel) greeksUpdateStrike(sel.value);
            }
          } catch(e) {}
        }, 60);
      }
    } catch(e) {
      showSpinner(false); _refreshing = false;
    }
    setTimeout(function() { try { iframe.src = 'about:blank'; } catch(e) {} }, 500);
  });

  let remaining = TOTAL_SECS;
  setCountdownUI(remaining);

  setInterval(function() {
    remaining -= 1;
    if (remaining <= 0) {
      remaining = TOTAL_SECS;
      setCountdownUI(remaining);
      doSilentRefresh();
    } else {
      setCountdownUI(remaining);
    }
  }, 1000);

  window.addEventListener('load', function() {
    setTimeout(doSilentRefresh, 2000);
  });
})();
</script>
"""


def build_greeks_script_html(oc_analysis):
    if not oc_analysis:
        return ""
    all_rows = oc_analysis.get("all_strikes", oc_analysis.get("greeks_table", []))
    atm      = int(oc_analysis.get("atm_strike", 0))
    if not all_rows:
        return ""
    strikes_json_parts = []
    for row in all_rows:
        s = int(row["strike"])
        strikes_json_parts.append(
            f'"{s}":{{'
            + f'"ce_ltp":{round(float(row["ce_ltp"]),2)},'
            + f'"pe_ltp":{round(float(row["pe_ltp"]),2)},'
            + f'"ce_delta":{round(float(row["ce_delta"]),4)},'
            + f'"pe_delta":{round(float(row["pe_delta"]),4)},'
            + f'"ce_iv":{round(float(row["ce_iv"]),2)},'
            + f'"pe_iv":{round(float(row["pe_iv"]),2)},'
            + f'"ce_theta":{round(float(row["ce_theta"]),4)},'
            + f'"pe_theta":{round(float(row["pe_theta"]),4)},'
            + f'"ce_vega":{round(float(row["ce_vega"]),4)},'
            + f'"pe_vega":{round(float(row["pe_vega"]),4)}'
            + f'}}')
    strikes_json = "{" + ",".join(strikes_json_parts) + "}"
    return f"""<script>
(function() {{
  var _gData = {strikes_json};
  var _atm   = {atm};
  function _initGreeks() {{
    var sel = document.getElementById('greeksStrikeSelect');
    if (sel) {{ greeksUpdateStrike(sel.value); }}
  }}
  window.greeksUpdateStrike = function(strike) {{
    var key = String(parseInt(strike, 10));
    var d   = _gData[key];
    if (!d) {{
      var keys = Object.keys(_gData).map(Number);
      var nearest = keys.reduce((a,b) => Math.abs(b-parseInt(strike))<Math.abs(a-parseInt(strike))?b:a, keys[0]);
      d = _gData[String(nearest)];
    }}
    if (!d) return;
    var sel  = parseInt(strike, 10);
    var dist = Math.round(Math.abs(sel - _atm) / 50);
    var lbl  = sel === _atm ? 'ATM' : (sel > _atm ? 'CE+' + dist : 'PE-' + dist);
    var e1 = document.getElementById('greeksStrikeTypeLabel'); if(e1) e1.textContent = lbl;
    var e2 = document.getElementById('greeksStrikeLabel'); if(e2) e2.innerHTML = '&#8377;' + sel.toLocaleString('en-IN');
    var e3 = document.getElementById('greeksCeLtp'); if(e3) e3.innerHTML = 'CE &#8377;' + (d.ce_ltp||0).toFixed(1);
    var e4 = document.getElementById('greeksPeLtp'); if(e4) e4.innerHTML = 'PE &#8377;' + (d.pe_ltp||0).toFixed(1);
    var ceCol='#00c896', peCol='#ff6b6b';
    var cePct=Math.min(100,Math.abs(d.ce_delta)*100).toFixed(0);
    var pePct=Math.min(100,Math.abs(d.pe_delta)*100).toFixed(0);
    var dw = document.getElementById('greeksDeltaWrap');
    if(dw) dw.innerHTML =
      '<div style="display:flex;align-items:center;gap:5px;">' +
        '<div style="width:34px;height:3px;background:rgba(255,255,255,.10);border-radius:2px;overflow:hidden;">' +
          '<div style="width:'+cePct+'%;height:100%;background:'+ceCol+';border-radius:2px;"></div></div>' +
        '<span style="font-family:DM Mono,monospace;font-size:15.9px;font-weight:700;color:'+ceCol+';">' +
             (d.ce_delta>=0?'+':'')+d.ce_delta.toFixed(3)+'</span></div>' +
      '<div style="display:flex;align-items:center;gap:5px;margin-top:3px;">' +
        '<div style="width:34px;height:3px;background:rgba(255,255,255,.10);border-radius:2px;overflow:hidden;">' +
          '<div style="width:'+pePct+'%;height:100%;background:'+peCol+';border-radius:2px;"></div></div>' +
        '<span style="font-family:DM Mono,monospace;font-size:15.9px;font-weight:700;color:'+peCol+';">' +
             (d.pe_delta>=0?'+':'')+d.pe_delta.toFixed(3)+'</span></div>';
    var ice = document.getElementById('greeksIvCe'); if(ice) ice.textContent = (d.ce_iv||0).toFixed(1)+'%';
    var ipe = document.getElementById('greeksIvPe'); if(ipe) ipe.textContent = (d.pe_iv||0).toFixed(1)+'%';
    var skew=((d.pe_iv||0)-(d.ce_iv||0)).toFixed(1);
    var skewEl=document.getElementById('greeksSkewLbl');
    if(skewEl) {{ skewEl.textContent = parseFloat(skew)>0?'PE Skew +'+skew:'CE Skew '+skew; skewEl.style.color = parseFloat(skew)>1.5?'#ff6b6b':(parseFloat(skew)<-1.5?'#00c896':'#6480ff'); }}
    function tfmt(t){{ return Math.abs(t)>=0.01?'&#8377;'+Math.abs(t).toFixed(2):t.toFixed(4); }}
    var tc = document.getElementById('greeksThetaCe'); if(tc) tc.innerHTML = tfmt(d.ce_theta||0);
    var tp = document.getElementById('greeksThetaPe'); if(tp) tp.innerHTML = tfmt(d.pe_theta||0);
    function vfmt(v){{ return Math.abs(v)>=0.0001?v.toFixed(4):'&mdash;'; }}
    var vc = document.getElementById('greeksVegaCe'); if(vc) vc.innerHTML = vfmt(d.ce_vega||0);
    var vp = document.getElementById('greeksVegaPe'); if(vp) vp.innerHTML = vfmt(d.pe_vega||0);
    var ivAvg=((d.ce_iv||0)+(d.pe_iv||0))/2;
    var ivCol=ivAvg>25?'#ff6b6b':(ivAvg>18?'#ffd166':'#00c896');
    var ivReg=ivAvg>25?'High IV \u00b7 Buy Premium':(ivAvg>15?'Normal IV \u00b7 Balanced':'Low IV \u00b7 Sell Premium');
    var ivPct=Math.min(100,Math.max(0,(ivAvg/60)*100)).toFixed(1);
    var barEl=document.getElementById('greeksIvBar');
    if(barEl) {{barEl.style.width=ivPct+'%'; barEl.style.background=ivCol; barEl.style.boxShadow='0 0 6px '+ivCol+'88';}}
    var avgEl=document.getElementById('greeksIvAvg');
    if(avgEl) {{avgEl.textContent=ivAvg.toFixed(1)+'%'; avgEl.style.color=ivCol;}}
    var regEl=document.getElementById('greeksIvRegime');
    if(regEl) {{regEl.textContent=ivReg; regEl.style.color=ivCol;}}
  }};
  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', _initGreeks);
  }} else {{
    setTimeout(_initGreeks, 80);
  }}
}})();
</script>"""


# =================================================================
#  SECTION 10 -- HTML ASSEMBLER
# =================================================================

def generate_html(tech, oc, md, ts, vix_data=None, multi_expiry_analyzed=None, expiry_list=None):
    oi_html        = build_oi_html(oc)               if oc   else ""
    kl_html        = build_key_levels_html(tech, oc) if tech else ""
    strat_html     = build_strategies_html(oc, tech, md, multi_expiry_analyzed=multi_expiry_analyzed, expiry_list=expiry_list)
    strikes_html   = build_strikes_html(oc)
    ticker_html    = build_ticker_bar(tech, oc, vix_data)
    gauge_html     = build_dual_gauge_hero(oc, tech, md, ts)
    greeks_sidebar = build_greeks_sidebar_html(oc)
    greeks_script  = build_greeks_script_html(oc)
    greeks_table   = build_greeks_table_html(oc)

    C = 2 * 3.14159 * 7
    cp    = tech["price"] if tech else 0
    bias  = md["bias"]; conf = md["confidence"]
    bull  = md["bull"]; bear  = md["bear"]; diff = md["diff"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Nifty 50 Options Dashboard v18.4</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<div class="app">
<header>
  <div class="logo-wrap" id="logoWrap"></div>
  <div class="hdr-meta">
    <div class="live-dot"></div>
    <span>NSE Options Dashboard</span>
    <span style="color:rgba(255,255,255,.15);">|</span>
    <span>{ts}</span>
    <span style="color:rgba(255,255,255,.15);">|</span>
    <div class="refresh-countdown">
      <div class="countdown-arc-wrap">
        <svg width="18" height="18" viewBox="0 0 18 18">
          <circle cx="9" cy="9" r="7" fill="none" stroke="rgba(255,255,255,.1)" stroke-width="2"/>
          <circle id="cdArc" cx="9" cy="9" r="7" fill="none" stroke="#00c896" stroke-width="2"
            stroke-linecap="round" stroke-dasharray="{C:.2f}" stroke-dashoffset="0"
            style="transform:rotate(-90deg);transform-origin:9px 9px;transition:stroke-dashoffset 1s linear,stroke .3s;"/>
        </svg>
      </div>
      <span class="countdown-num" id="cdNum">30</span>
      <span class="countdown-lbl" id="cdLbl">s</span>
      <div class="refresh-ring" id="refreshRing"></div>
      <span id="refreshStatus"></span>
    </div>
  </div>
</header>
{ticker_html}
{gauge_html}
<div class="main">
  <aside class="sidebar">
    <div class="sidebar-sticky-top">
      <div id="greeksPanel">{greeks_sidebar}</div>
    </div>
    <div class="sidebar-scroll">
    <div class="sb-sec">
      <div class="sb-lbl">LIVE ANALYSIS</div>
      <button class="sb-btn active" onclick="go('oi',this)">OI Dashboard</button>
      <button class="sb-btn"        onclick="go('greeksTable',this)">&#9652; Option Greeks</button>
      <button class="sb-btn"        onclick="go('kl',this)">Key Levels</button>
    </div>
    <div class="sb-sec">
      <div class="sb-lbl">STRATEGIES</div>
      <button class="sb-btn" onclick="go('strat',this);filterStrat('bullish',null)">&#9650; Bullish <span class="sb-badge" style="color:var(--bull);">9</span></button>
      <button class="sb-btn" onclick="go('strat',this);filterStrat('bearish',null)">&#9660; Bearish <span class="sb-badge" style="color:var(--bear);">9</span></button>
      <button class="sb-btn" onclick="go('strat',this);filterStrat('nondirectional',null)">&#8596; Non-Directional <span class="sb-badge" style="color:var(--neut);">20</span></button>
    </div>
    <div class="sb-sec">
      <div class="sb-lbl">OPTION CHAIN</div>
      <button class="sb-btn" onclick="go('strikes',this)">Top 5 Strikes</button>
    </div>
    </div>
  </aside>
  <main class="content">
    <div id="oi">{oi_html}</div>
    {greeks_table}
    <div id="kl">{kl_html}</div>
    {strat_html}
    <div id="strikes">{strikes_html}</div>
    <div class="section">
      <div style="background:rgba(100,128,255,.06);border:1px solid rgba(100,128,255,.18);
                  border-left:3px solid #6480ff;border-radius:12px;padding:16px 18px;
                  font-size:18.8px;color:rgba(255,255,255,.5);line-height:1.8;">
        <strong style="color:rgba(255,255,255,.7);">DISCLAIMER</strong><br>
        This dashboard is for EDUCATIONAL purposes only &mdash; NOT financial advice.<br>
        Smart PoP uses S/R levels, OI walls, market bias and PCR &mdash; not a guaranteed signal.<br>
        Always use stop losses. Consult a SEBI-registered investment advisor before trading.
      </div>
    </div>
  </main>
</div>
<footer>
  <span>NiftyCraft &middot; v19.1 &middot; Holiday-Aware Expiry + Intraday P&amp;L Simulator</span>
  <span>S/R + OI Walls + Bias + PCR &middot; Educational Only &middot; &copy; 2025</span>
</footer>
</div>

<iframe id="silentRefreshFrame" src="about:blank"
  style="position:fixed;width:0;height:0;border:none;visibility:hidden;
         pointer-events:none;opacity:0;top:-9999px;left:-9999px;"></iframe>

<script>
function go(id,btn){{
  const el=document.getElementById(id);
  if(el)el.scrollIntoView({{behavior:"smooth",block:"start"}});
  if(btn){{document.querySelectorAll(".sb-btn").forEach(b=>b.classList.remove("active"));btn.classList.add("active");}}
}}
function filterStrat(cat,btn){{
  document.querySelectorAll(".sc-card").forEach(c=>{{c.classList.toggle("hidden",c.dataset.cat!==cat);}});
  const colors={{bullish:"#00c896",bearish:"#ff6b6b",nondirectional:"#6480ff"}};
  const col=colors[cat]||"#00c896";
  document.querySelectorAll(".sc-tab").forEach(t=>{{t.style.borderColor="rgba(255,255,255,.15)";t.style.color="rgba(255,255,255,.5)";t.style.background="transparent";}});
  if(btn){{btn.style.borderColor=col;btn.style.color=col;btn.style.background=col+"20";}}
  else{{document.querySelectorAll(".sc-tab").forEach(t=>{{
    if((cat==="bullish"&&t.textContent.includes("BULLISH"))||(cat==="bearish"&&t.textContent.includes("BEARISH"))||(cat==="nondirectional"&&t.textContent.includes("NON")))
    {{t.style.borderColor=col;t.style.color=col;t.style.background=col+"20";}}
  }});}}
}}
document.addEventListener("click",function(e){{
  const card=e.target.closest(".sc-card");
  if(card){{
    const was=card.classList.contains("expanded");
    document.querySelectorAll(".sc-card.expanded").forEach(c=>c.classList.remove("expanded"));
    if(!was){{
      card.classList.add("expanded");
      const mel=card.querySelector('.sc-metrics-live');
      if(mel&&mel.querySelector('.sc-loading')){{
        try{{
          const shape=card.dataset.shape, cat=card.dataset.cat;
          const scoreResult=smartPoP(shape,cat);
          const m=calcMetrics(shape,scoreResult.pop);
          mel.innerHTML=renderMetrics(m, scoreResult);
        }}catch(err){{mel.innerHTML='<div class="sc-loading">Could not calculate metrics</div>';}}
      }}
    }}
  }}
}});
</script>
{greeks_script}
{ANIMATED_JS}
</body>
</html>"""


# =================================================================
#  SECTION 11 -- MAIN
# =================================================================

def main():
    ts = ist_timestamp_str()
    print("=" * 65)
    print("  NIFTY 50 OPTIONS DASHBOARD — v18.4 · Holiday-Aware Expiry")
    print(f"  {ts}")
    print(f"  IST Date: {today_ist()}  IST Weekday: {ist_weekday()}")
    print("=" * 65)

    # ── Print holiday check for current week ──────────────────────
    print("\n[0/4] Holiday Awareness Check...")
    from datetime import date as _date
    today = today_ist()
    wd    = today.weekday()
    if wd <= 1:
        days = 1 - wd
    else:
        days = (8 - wd)
    this_tue = today + timedelta(days=days) if wd != 1 else today
    if is_nse_holiday(this_tue):
        reason = NSE_HOLIDAYS_2026.get(this_tue.strftime("%d-%b-%Y"), "Holiday")
        prev_td = get_prev_trading_day(this_tue)
        print(f"  ⚠ {this_tue.strftime('%d-%b-%Y')} (Tue) = {reason}")
        print(f"  ✓ Expiry shifted to {prev_td.strftime('%d-%b-%Y')} ({prev_td.strftime('%A')})")
    else:
        print(f"  ✓ {this_tue.strftime('%d-%b-%Y')} (Tue) is a normal trading day. No holiday adjustment needed.")

    print("\n[1/4] Fetching NSE Option Chain...")
    nse = NSEOptionChain()
    oc_raw, nse_session, nse_headers = nse.fetch()

    print("\n[2/4] Fetching India VIX...")
    vix_data = fetch_india_vix(nse_session, nse_headers)
    live_vix = vix_data["value"] if vix_data else 18.0
    # Fetch all 7 expiries for dropdown
    print("\n  Fetching next 7 expiries for dropdown...")
    time.sleep(1.5)   # small gap so NSE doesn't block
    multi_expiry_raw, expiry_list = nse.fetch_multiple_expiries(nse_session, nse_headers, n=15)
    print(f"  Expiry dropdown will show: {expiry_list}")

    # Pre-analyze all expiry data
    multi_expiry_analyzed = {}
    for exp, raw in multi_expiry_raw.items():
        analyzed = analyze_option_chain(raw, vix=live_vix)
        if analyzed:
            multi_expiry_analyzed[exp] = analyzed
            print(f"    Analyzed {exp}: ATM={analyzed['atm_strike']} PCR={analyzed['pcr_oi']:.3f}")

    oc_analysis = analyze_option_chain(oc_raw, vix=live_vix) if oc_raw else None
    if oc_analysis:
        print(f"\n  OK  Spot={oc_analysis['underlying']:.2f}  ATM={oc_analysis['atm_strike']}")
        print(f"      MaxCE={oc_analysis['max_ce_strike']}  MaxPE={oc_analysis['max_pe_strike']}")
        print(f"      Expiry={oc_analysis['expiry']}  PCR={oc_analysis['pcr_oi']:.3f}")
        print(f"      CE CHG={oc_analysis['ce_chg']:+,}  PE CHG={oc_analysis['pe_chg']:+,}")
        print(f"      CHG Bull Force={oc_analysis['chg_bull_force']:,}  CHG Bear Force={oc_analysis['chg_bear_force']:,}")
        print(f"      CHG Bull%={oc_analysis['chg_bull_pct']}%  CHG Bear%={oc_analysis['chg_bear_pct']}%")

    print("\n[3/4] Fetching Technical Indicators (S/R levels)...")
    tech = get_technical_data()
    if tech:
        print(f"  Support={tech['support']:.0f}  Resistance={tech['resistance']:.0f}")
        print(f"  StrongSup={tech['strong_sup']:.0f}  StrongRes={tech['strong_res']:.0f}")

    print("\n[4/4] Scoring Market Direction...")
    md = compute_market_direction(tech, oc_analysis)
    print(f"  Bias={md['bias']}  Conf={md['confidence']}  Bull={md['bull']}  Bear={md['bear']}")

    print("\nGenerating Holiday-Aware Dashboard...")
    html = generate_html(tech, oc_analysis, md, ts, vix_data=vix_data,
                     multi_expiry_analyzed=multi_expiry_analyzed,
                     expiry_list=expiry_list)

    os.makedirs("docs", exist_ok=True)
    out = os.path.join("docs", "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved: {out}  ({len(html)/1024:.1f} KB)")

    meta = {
        "timestamp":       ts,
        "generated_at":    int(time.time()),
        "ist_date":        str(today_ist()),
        "ist_weekday":     ist_weekday(),
        "bias":            md["bias"],
        "confidence":      md["confidence"],
        "bull":            md["bull"],
        "bear":            md["bear"],
        "diff":            md["diff"],
        "price":           round(tech["price"], 2)         if tech        else None,
        "expiry":          oc_analysis["expiry"]           if oc_analysis else None,
        "pcr":             oc_analysis["pcr_oi"]           if oc_analysis else None,
        "oi_dir":          oc_analysis["oi_dir"]           if oc_analysis else None,
        "raw_oi_dir":      oc_analysis["raw_oi_dir"]       if oc_analysis else None,
        "india_vix":       vix_data["value"]               if vix_data    else None,
        "atm_strike":      oc_analysis["atm_strike"]       if oc_analysis else None,
        "max_ce":          oc_analysis["max_ce_strike"]    if oc_analysis else None,
        "max_pe":          oc_analysis["max_pe_strike"]    if oc_analysis else None,
        "support":         round(tech["support"], 0)       if tech        else None,
        "resistance":      round(tech["resistance"], 0)    if tech        else None,
        "ce_chg":          oc_analysis["ce_chg"]           if oc_analysis else None,
        "pe_chg":          oc_analysis["pe_chg"]           if oc_analysis else None,
        "chg_bull_force":  oc_analysis["chg_bull_force"]   if oc_analysis else None,
        "chg_bear_force":  oc_analysis["chg_bear_force"]   if oc_analysis else None,
        "chg_bull_pct":    oc_analysis["chg_bull_pct"]     if oc_analysis else None,
        "chg_bear_pct":    oc_analysis["chg_bear_pct"]     if oc_analysis else None,
    }
    with open(os.path.join("docs", "latest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("  Saved: docs/latest.json")
    print("\n" + "=" * 65)
    print(f"  DONE  |  v18.4 · Holiday-Aware Expiry Active")
    print(f"  Bias: {md['bias']}  |  Confidence: {md['confidence']}")
    print("  Holiday list: 2026 NSE official holidays pre-loaded")
    print("  Logic: Tuesday holiday → Monday → Friday (fallback)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
