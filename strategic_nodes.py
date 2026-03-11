"""
Nifty Options Strategy Analyzer
=================================
Fetches live NSE option chain data and generates a rich HTML dashboard.
Runs via GitHub Actions every 30 minutes → pushes index.html to gh-pages.

Dependencies:
    pip install curl_cffi pandas numpy scipy
"""

import json
import math
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm


# ── Custom JSON encoder — handles numpy bool/int/float types ─────
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def _to_json(obj):
    """Serialize to JSON safely — handles numpy scalar types."""
    return json.dumps(obj, cls=_NumpyEncoder, ensure_ascii=False)


# ── Timezone (IST) ────────────────────────────────────────────────
try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
    def today_ist():
        return datetime.now(IST).date()
    def now_ist_str():
        return datetime.now(IST).strftime("%d-%b-%Y %H:%M:%S IST")
except ImportError:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
    def today_ist():
        return datetime.now(IST).date()
    def now_ist_str():
        return datetime.now(IST).strftime("%d-%b-%Y %H:%M:%S IST")

# ── NSE Trading Holidays (hardcoded — update each year) ──────────
NSE_HOLIDAYS = {
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

_HOLIDAY_DATES = set()
for _ds in NSE_HOLIDAYS:
    try:
        _HOLIDAY_DATES.add(datetime.strptime(_ds, "%d-%b-%Y").date())
    except Exception:
        pass


# =================================================================
#  SECTION 1 -- HOLIDAY & TRADING DAY UTILITIES
# =================================================================

def is_nse_holiday(dt):
    if dt.weekday() >= 5:
        return True
    return dt in _HOLIDAY_DATES


def get_prev_trading_day(dt):
    candidate = dt - timedelta(days=1)
    for _ in range(10):
        if not is_nse_holiday(candidate):
            return candidate
        candidate -= timedelta(days=1)
    return candidate


# =================================================================
#  SECTION 2 -- NSE OPTION CHAIN FETCHER
# =================================================================

class NSEOptionChain:
    def __init__(self):
        self.symbol = "NIFTY"
        self._cached_expiry_list = []

    def _make_session(self):
        from curl_cffi import requests as curl_requests
        headers = {
            "authority":         "www.nseindia.com",
            "accept":            "application/json, text/plain, */*",
            "user-agent":        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                 "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "referer":           "https://www.nseindia.com/option-chain",
            "accept-language":   "en-US,en;q=0.9",
        }
        session = curl_requests.Session()
        try:
            session.get("https://www.nseindia.com/", headers=headers,
                        impersonate="chrome", timeout=15)
            time.sleep(1.5)
            session.get("https://www.nseindia.com/option-chain", headers=headers,
                        impersonate="chrome", timeout=15)
            time.sleep(1.0)
        except Exception as e:
            print(f"  WARNING  Session warm-up: {e}")
        return session, headers

    def _current_or_next_tuesday_ist(self):
        today = today_ist()
        wd    = today.weekday()
        if wd == 1:
            target_tuesday = today
        elif wd < 1:
            target_tuesday = today + timedelta(days=1 - wd)
        else:
            target_tuesday = today + timedelta(days=8 - wd)
        if is_nse_holiday(target_tuesday):
            reason   = NSE_HOLIDAYS.get(target_tuesday.strftime("%d-%b-%Y"), "Holiday/Weekend")
            adjusted = get_prev_trading_day(target_tuesday)
            print(f"  [Holiday] {target_tuesday.strftime('%d-%b-%Y')} is '{reason}'. "
                  f"Expiry moved to {adjusted.strftime('%d-%b-%Y')} ({adjusted.strftime('%A')})")
            expiry_date = adjusted
        else:
            expiry_date = target_tuesday
        result = expiry_date.strftime("%d-%b-%Y")
        print(f"  Computed expiry (IST, holiday-adjusted): {result}")
        return result

    def _fetch_available_expiries(self, session, headers):
        try:
            url  = (f"https://www.nseindia.com/api/option-chain-v3"
                    f"?type=Indices&symbol={self.symbol}")
            resp = session.get(url, headers=headers, impersonate="chrome", timeout=20)
            if resp.status_code == 200:
                expiries = resp.json().get("records", {}).get("expiryDates", [])
                today    = today_ist()
                for exp_str in expiries:
                    try:
                        exp_dt = datetime.strptime(exp_str, "%d-%b-%Y").date()
                        if exp_dt >= today:
                            print(f"  Fallback expiry from NSE API: {exp_str}")
                            return exp_str
                    except Exception:
                        continue
                if expiries:
                    return expiries[0]
        except Exception as e:
            print(f"  WARNING  Expiry fetch: {e}")
        return None

    def _fetch_for_expiry(self, session, headers, expiry):
        api_url = (f"https://www.nseindia.com/api/option-chain-v3"
                   f"?type=Indices&symbol={self.symbol}&expiry={expiry}")
        for attempt in range(1, 4):
            try:
                resp = session.get(api_url, headers=headers,
                                   impersonate="chrome", timeout=30)
                if resp.status_code != 200:
                    print(f"    HTTP {resp.status_code} on attempt {attempt}")
                    time.sleep(2)
                    continue
                json_data   = resp.json()
                data        = json_data.get("records", {}).get("data", [])
                if not data:
                    return None
                underlying  = json_data.get("records", {}).get("underlyingValue", 0)
                atm_strike  = round(underlying / 50) * 50
                lower_bound = underlying - 2000   # FIX: widened from 600 → 2000 so far-OTM BE targets are always in range
                upper_bound = underlying + 2000   # FIX: widened from 600 → 2000
                rows = []
                for item in data:
                    strike = item.get("strikePrice")
                    if strike is None or not (lower_bound <= strike <= upper_bound):
                        continue
                    ce = item.get("CE", {})
                    pe = item.get("PE", {})
                    rows.append({
                        "Strike":       strike,
                        "CE_LTP":       ce.get("lastPrice",            0),
                        "CE_OI":        ce.get("openInterest",         0),
                        "CE_Vol":       ce.get("totalTradedVolume",    0),
                        "CE_OI_Change": ce.get("changeinOpenInterest", 0),
                        "CE_IV":        ce.get("impliedVolatility",    0),
                        "CE_Delta":     ce.get("delta",                0),
                        "CE_Theta":     ce.get("theta",                0),
                        "CE_Gamma":     ce.get("gamma",                0),
                        "CE_Vega":      ce.get("vega",                 0),
                        "PE_LTP":       pe.get("lastPrice",            0),
                        "PE_OI":        pe.get("openInterest",         0),
                        "PE_Vol":       pe.get("totalTradedVolume",    0),
                        "PE_OI_Change": pe.get("changeinOpenInterest", 0),
                        "PE_IV":        pe.get("impliedVolatility",    0),
                        "PE_Delta":     pe.get("delta",                0),
                        "PE_Theta":     pe.get("theta",                0),
                        "PE_Gamma":     pe.get("gamma",                0),
                        "PE_Vega":      pe.get("vega",                 0),
                    })
                df = (pd.DataFrame(rows)
                        .sort_values("Strike")
                        .reset_index(drop=True))
                print(f"    OK {len(df)} strikes | Spot={underlying:.0f} ATM={atm_strike}")
                return {
                    "expiry":     expiry,
                    "df":         df,
                    "underlying": underlying,
                    "atm_strike": atm_strike,
                }
            except Exception as e:
                print(f"    FAIL Attempt {attempt}: {e}")
                time.sleep(2)
        return None

    def fetch_multiple_expiries(self, session, headers, n=7):
        expiry_list = []
        try:
            url  = (f"https://www.nseindia.com/api/option-chain-v3"
                    f"?type=Indices&symbol={self.symbol}")
            resp = session.get(url, headers=headers, impersonate="chrome", timeout=20)
            if resp.status_code == 200:
                all_exp = resp.json().get("records", {}).get("expiryDates", [])
                today   = today_ist()
                for exp_str in all_exp:
                    try:
                        exp_dt = datetime.strptime(exp_str, "%d-%b-%Y").date()
                        if exp_dt >= today:
                            expiry_list.append(exp_str)
                        if len(expiry_list) >= n:
                            break
                    except Exception:
                        continue
                print(f"  Expiry list from NSE ({len(expiry_list)}): {expiry_list}")
        except Exception as e:
            print(f"  WARNING expiry list fetch: {e}")
        if not expiry_list:
            print("  Falling back to generated Tuesday expiry list...")
            today = today_ist()
            wd    = today.weekday()
            days_to_tue = 1 - wd if wd <= 1 else 8 - wd
            candidate   = today + timedelta(days=days_to_tue)
            attempts    = 0
            while len(expiry_list) < n and attempts < 30:
                adjusted = get_prev_trading_day(candidate) if is_nse_holiday(candidate) else candidate
                exp_str  = adjusted.strftime("%d-%b-%Y")
                if exp_str not in expiry_list:
                    expiry_list.append(exp_str)
                candidate += timedelta(days=7)
                attempts  += 1
        results = {}
        for exp in expiry_list:
            print(f"    Fetching: {exp}")
            data = self._fetch_for_expiry(session, headers, exp)
            if data:
                results[exp] = data
            else:
                print(f"      SKIP: {exp}")
            time.sleep(0.8)
        print(f"  Fetched {len(results)}/{len(expiry_list)} expiries")
        return results, expiry_list

    def fetch(self):
        session, headers = self._make_session()
        expiry           = self._current_or_next_tuesday_ist()
        result           = self._fetch_for_expiry(session, headers, expiry)
        if result is None:
            print(f"  Computed expiry {expiry} not found. Trying API fallback...")
            real_expiry = self._fetch_available_expiries(session, headers)
            if real_expiry and real_expiry != expiry:
                result = self._fetch_for_expiry(session, headers, real_expiry)
        if result is None:
            print("  ERROR: Option chain fetch failed.")
        self._cached_expiry_list = []
        try:
            url  = (f"https://www.nseindia.com/api/option-chain-v3"
                    f"?type=Indices&symbol={self.symbol}")
            resp = session.get(url, headers=headers, impersonate="chrome", timeout=20)
            if resp.status_code == 200:
                all_exp = resp.json().get("records", {}).get("expiryDates", [])
                today   = today_ist()
                for exp_str in all_exp:
                    try:
                        exp_dt = datetime.strptime(exp_str, "%d-%b-%Y").date()
                        if exp_dt >= today:
                            self._cached_expiry_list.append(exp_str)
                            if len(self._cached_expiry_list) >= 7:
                                break
                    except Exception:
                        continue
        except Exception as e:
            print(f"  WARNING expiry list: {e}")
        return result, session, headers


# =================================================================
#  SECTION 3 -- BLACK-SCHOLES & GREEKS ENGINE
# =================================================================

def black_scholes(S, K, T, r, sigma, option_type="CE"):
    if T <= 0 or sigma <= 0:
        intrinsic = max(0, S - K) if option_type == "CE" else max(0, K - S)
        return {"price": intrinsic, "delta": 0, "gamma": 0, "theta": 0, "vega": 0}
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "CE":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        prob_profit = norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        prob_profit = norm.cdf(-d2)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
             - r * K * math.exp(-r * T) * (norm.cdf(d2) if option_type == "CE" else norm.cdf(-d2))) / 365
    vega  = S * norm.pdf(d1) * math.sqrt(T) / 100
    return {
        "price":       round(price, 2),
        "delta":       round(delta, 4),
        "gamma":       round(gamma, 6),
        "theta":       round(theta, 4),
        "vega":        round(vega, 4),
        "prob_profit": round(prob_profit, 4),
        "d1": d1, "d2": d2,
    }


def compute_greeks_for_chain(df, underlying, expiry_str, r=0.065):
    today   = today_ist()
    try:
        exp_dt  = datetime.strptime(expiry_str, "%d-%b-%Y").date()
        dte     = max((exp_dt - today).days, 0)
    except Exception:
        dte = 7
    T = dte / 365.0
    atm_strike  = round(underlying / 50) * 50
    atm_greeks  = {}
    all_strikes = []
    for _, row in df.iterrows():
        strike  = row["Strike"]
        is_atm  = bool(strike == atm_strike)
        ce_iv = row.get("CE_IV", 0) or 0
        pe_iv = row.get("PE_IV", 0) or 0
        if ce_iv <= 0: ce_iv = 15.0
        if pe_iv <= 0: pe_iv = 15.0
        sigma_ce = ce_iv / 100
        sigma_pe = pe_iv / 100
        ce_delta = row.get("CE_Delta", 0) or 0
        pe_delta = row.get("PE_Delta", 0) or 0
        ce_theta = row.get("CE_Theta", 0) or 0
        pe_theta = row.get("PE_Theta", 0) or 0
        ce_gamma = row.get("CE_Gamma", 0) or 0
        pe_gamma = row.get("PE_Gamma", 0) or 0
        ce_vega  = row.get("CE_Vega",  0) or 0
        pe_vega  = row.get("PE_Vega",  0) or 0
        if ce_delta == 0 and T > 0:
            bsm_ce   = black_scholes(underlying, strike, T, r, sigma_ce, "CE")
            ce_delta = bsm_ce["delta"]
            ce_theta = bsm_ce["theta"]
            ce_gamma = bsm_ce["gamma"]
            ce_vega  = bsm_ce["vega"]
        if pe_delta == 0 and T > 0:
            bsm_pe   = black_scholes(underlying, strike, T, r, sigma_pe, "PE")
            pe_delta = bsm_pe["delta"]
            pe_theta = bsm_pe["theta"]
            pe_gamma = bsm_pe["gamma"]
            pe_vega  = bsm_pe["vega"]
        bsm_pop_ce = black_scholes(underlying, strike, T, r, sigma_ce, "CE")
        bsm_pop_pe = black_scholes(underlying, strike, T, r, sigma_pe, "PE")
        ce_pop = round(bsm_pop_ce.get("prob_profit", 0.5) * 100, 1)
        pe_pop = round(bsm_pop_pe.get("prob_profit", 0.5) * 100, 1)
        strike_data = {
            "strike":   strike,
            "is_atm":   is_atm,
            "dte":      dte,
            "ce_ltp":   round(row.get("CE_LTP", 0), 2),
            "pe_ltp":   round(row.get("PE_LTP", 0), 2),
            "ce_iv":    round(ce_iv, 2),
            "pe_iv":    round(pe_iv, 2),
            "ce_delta": round(ce_delta, 4),
            "pe_delta": round(pe_delta, 4),
            "ce_theta": round(ce_theta, 4),
            "pe_theta": round(pe_theta, 4),
            "ce_gamma": round(ce_gamma, 6),
            "pe_gamma": round(pe_gamma, 6),
            "ce_vega":  round(ce_vega,  4),
            "pe_vega":  round(pe_vega,  4),
            "ce_oi":    int(row.get("CE_OI", 0)),
            "pe_oi":    int(row.get("PE_OI", 0)),
            "ce_oi_chg": int(row.get("CE_OI_Change", 0)),
            "pe_oi_chg": int(row.get("PE_OI_Change", 0)),
            "ce_vol":   int(row.get("CE_Vol", 0)),
            "pe_vol":   int(row.get("PE_Vol", 0)),
            "ce_pop":   ce_pop,
            "pe_pop":   pe_pop,
        }
        all_strikes.append(strike_data)
        if is_atm:
            atm_greeks = strike_data
    return {
        "expiry":      expiry_str,
        "underlying":  underlying,
        "atm_strike":  atm_strike,
        "dte":         dte,
        "atm_greeks":  atm_greeks,
        "all_strikes": all_strikes,
    }


# =================================================================
#  SECTION 4 -- STRATEGY ENGINE
# =================================================================

LOT_SIZE    = 65
RISK_FREE_R = 0.065


def _strategy_pop(legs, underlying, T, r=RISK_FREE_R, breakevens=None, strategy_type=None):
    be = breakevens or []
    avg_iv = sum(l["iv"] for l in legs) / len(legs)
    sigma  = avg_iv / 100
    all_sell = all(l["action"] == "sell" for l in legs)
    all_buy  = all(l["action"] == "buy"  for l in legs)
    if be:
        if strategy_type == "debit_spread":
            be_price = be[0]
            buy_leg  = next((l for l in legs if l["action"] == "buy"), legs[0])
            is_call  = buy_leg["opt_type"] == "CE"
            if is_call:
                bs  = black_scholes(underlying, be_price, T, r, sigma, "CE")
                pop = (1 - bs["prob_profit"]) * 100
            else:
                bs  = black_scholes(underlying, be_price, T, r, sigma, "PE")
                pop = bs["prob_profit"] * 100
        elif all_buy and strategy_type in ("straddle", "strangle"):
            lower_be   = be[0]
            upper_be   = be[1] if len(be) >= 2 else None
            bs_lower   = black_scholes(underlying, lower_be, T, r, sigma, "PE")
            prob_below = bs_lower["prob_profit"]
            prob_above = 0
            if upper_be:
                bs_upper   = black_scholes(underlying, upper_be, T, r, sigma, "CE")
                prob_above = 1 - bs_upper["prob_profit"]
            pop = min((prob_below + prob_above) * 100, 99)
        elif all_sell or strategy_type in ("credit_spread", "iron_condor", "iron_butterfly"):
            lower_be   = be[0]
            upper_be   = be[1] if len(be) >= 2 else None
            bs_lower   = black_scholes(underlying, lower_be, T, r, sigma, "PE")
            prob_below = bs_lower["prob_profit"]
            prob_above = 0
            if upper_be:
                bs_upper   = black_scholes(underlying, upper_be, T, r, sigma, "CE")
                prob_above = 1 - bs_upper["prob_profit"]
            pop = min((1 - prob_below - prob_above) * 100, 99)
        else:
            total = sum(
                (black_scholes(underlying, l["strike"], T, r, l["iv"]/100, l["opt_type"])
                 .get("prob_profit", 0.5))
                if l["action"] == "sell"
                else (1 - black_scholes(underlying, l["strike"], T, r, l["iv"]/100, l["opt_type"])
                      .get("prob_profit", 0.5))
                for l in legs
            )
            pop = min(total / len(legs) * 100, 99)
    else:
        total = sum(
            (black_scholes(underlying, l["strike"], T, r, l["iv"]/100, l["opt_type"])
             .get("prob_profit", 0.5))
            if l["action"] == "sell"
            else (1 - black_scholes(underlying, l["strike"], T, r, l["iv"]/100, l["opt_type"])
                  .get("prob_profit", 0.5))
            for l in legs
        )
        pop = min(total / len(legs) * 100, 99)
    return round(max(pop, 1), 1)


def _payoff_at_expiry(legs, price_range):
    payoffs = []
    for price in price_range:
        pnl = 0
        for leg in legs:
            if leg["opt_type"] == "CE":
                intrinsic = max(0, price - leg["strike"])
            else:
                intrinsic = max(0, leg["strike"] - price)
            leg_pnl = (intrinsic - leg["premium"]) if leg["action"] == "buy" else (leg["premium"] - intrinsic)
            pnl += leg_pnl
        payoffs.append(round(pnl * LOT_SIZE, 2))
    return payoffs


def build_strategies(oc_analysis, support_levels, resistance_levels, bias="neutral"):
    if not oc_analysis:
        return []
    all_strikes = oc_analysis["all_strikes"]
    underlying  = oc_analysis["underlying"]
    atm_strike  = oc_analysis["atm_strike"]
    expiry_str  = oc_analysis["expiry"]
    dte         = oc_analysis["dte"]
    T           = max(dte / 365.0, 0.001)
    strike_map = {s["strike"]: s for s in all_strikes}
    strikes    = sorted(strike_map.keys())
    def nearest(val):
        return min(strikes, key=lambda x: abs(x - val))
    def get(strike, field, default=0):
        return strike_map.get(strike, {}).get(field, default)
    # ── Support: closest level to current price (aggressive) or lowest (conservative)
    if support_levels:
        closest_support    = min(support_levels, key=lambda x: abs(x - underlying))
        conservative_support = min(support_levels)
        # Use closest if it's below spot, else fall back to conservative
        best_support = closest_support if closest_support < underlying else conservative_support
    else:
        best_support = underlying - 300

    # ── Resistance: closest level above spot, or highest as fallback
    if resistance_levels:
        above_res = [r for r in resistance_levels if r > underlying]
        best_resistance = min(above_res, key=lambda x: abs(x - underlying)) if above_res else max(resistance_levels)
    else:
        best_resistance = underlying + 300

    # ── ATR-based wing width (use ~1.5× daily ATR as minimum wing)
    atm_iv    = (get(atm_strike, "ce_iv", 15) + get(atm_strike, "pe_iv", 15)) / 2
    daily_atr = round(underlying * (atm_iv / 100) * math.sqrt(1 / 365), 0)
    atr_wing  = max(int(daily_atr * 1.5 / 50) * 50, 100)   # minimum 100 pts, rounded to 50

    sr_range  = best_resistance - best_support
    atm  = atm_strike
    s_st = nearest(best_support)
    r_st = nearest(best_resistance)
    otm_c = nearest(underlying + sr_range * 0.3)
    otm_p = nearest(underlying - sr_range * 0.3)
    far_c = nearest(best_resistance + atr_wing)
    far_p = nearest(best_support    - atr_wing)
    # (avg_support/avg_resistance aliases removed — using best_support/best_resistance directly)
    atm_iv = (get(atm, "ce_iv", 15) + get(atm, "pe_iv", 15)) / 2
    exp_move = round(underlying * (atm_iv / 100) * math.sqrt(T), 0)
    raw = []
    def make_strategy(name, legs, bias_tag, strategy_type):
        net_premium  = sum(
            (-l["premium"] if l["action"] == "buy" else l["premium"]) for l in legs
        )
        is_debit     = net_premium < 0
        net_premium  = round(net_premium, 2)
        price_range  = list(range(int(underlying - 1500), int(underlying + 1500), 25))
        payoffs      = _payoff_at_expiry(legs, price_range)
        max_profit   = max(payoffs)
        max_loss     = min(payoffs)
        if max_profit <= 0:
            return None
        breakevens = []
        for i in range(len(payoffs) - 1):
            if (payoffs[i] < 0) != (payoffs[i + 1] < 0):
                be = price_range[i] + (price_range[i + 1] - price_range[i]) * abs(payoffs[i]) / (abs(payoffs[i]) + abs(payoffs[i + 1]))
                breakevens.append(round(be, 0))
        rr_ratio = round(abs(max_profit / max_loss), 2) if max_loss != 0 else 0
        pop = _strategy_pop(legs, underlying, T,
                            breakevens=breakevens, strategy_type=strategy_type)
        rr_norm   = min(rr_ratio, 5.0) / 5.0          # normalize RR to 0-1 (cap at 5:1)
        rr_adj    = rr_norm * pop / 100                # penalise high-RR/low-PoP combos
        efficiency = min(abs(max_profit) / max(abs(max_loss), 1), 5.0) / 5.0
        score = round(
            (pop / 100) * 0.40 * 100 +
            rr_adj       * 0.35 * 100 +
            efficiency   * 0.25 * 100,
            1,
        )
        margin_est = abs(max_loss)
        return {
            "name":          name,
            "type":          strategy_type,
            "bias":          bias_tag,
            "legs":          legs,
            "net_premium":   net_premium,
            "is_debit":      is_debit,
            "max_profit":    round(max_profit, 2),
            "max_loss":      round(abs(max_loss), 2),
            "breakevens":    breakevens,
            "rr_ratio":      rr_ratio,
            "pop":           pop,
            "score":         score,
            "margin_est":    round(margin_est, 2),
            "payoffs":       payoffs,
            "price_range":   price_range,
            "expiry":        expiry_str,
            "dte":           dte,
            "exp_move":      exp_move,
        }
    if bias != "bearish":
        buy_st, sell_st = atm, r_st
        if buy_st != sell_st:
            s = make_strategy("Bull Call Spread", [
                {"action": "buy",  "strike": buy_st,  "opt_type": "CE",
                 "premium": get(buy_st,  "ce_ltp"), "iv": get(buy_st,  "ce_iv", 15)},
                {"action": "sell", "strike": sell_st, "opt_type": "CE",
                 "premium": get(sell_st, "ce_ltp"), "iv": get(sell_st, "ce_iv", 15)},
            ], "bullish", "debit_spread")
            if s: raw.append(s)
    if bias != "bullish":
        buy_st, sell_st = atm, s_st
        if buy_st != sell_st:
            s = make_strategy("Bear Put Spread", [
                {"action": "buy",  "strike": buy_st,  "opt_type": "PE",
                 "premium": get(buy_st,  "pe_ltp"), "iv": get(buy_st,  "pe_iv", 15)},
                {"action": "sell", "strike": sell_st, "opt_type": "PE",
                 "premium": get(sell_st, "pe_ltp"), "iv": get(sell_st, "pe_iv", 15)},
            ], "bearish", "debit_spread")
            if s: raw.append(s)
    if bias != "bearish":
        sell_st = s_st
        buy_st  = nearest(best_support - atr_wing)
        if sell_st != buy_st:
            s = make_strategy("Bull Put Spread", [
                {"action": "sell", "strike": sell_st, "opt_type": "PE",
                 "premium": get(sell_st, "pe_ltp"), "iv": get(sell_st, "pe_iv", 15)},
                {"action": "buy",  "strike": buy_st,  "opt_type": "PE",
                 "premium": get(buy_st,  "pe_ltp"), "iv": get(buy_st,  "pe_iv", 15)},
            ], "bullish", "credit_spread")
            if s: raw.append(s)
    if bias != "bullish":
        sell_st = r_st
        buy_st  = nearest(best_resistance + atr_wing)
        if sell_st != buy_st:
            s = make_strategy("Bear Call Spread", [
                {"action": "sell", "strike": sell_st, "opt_type": "CE",
                 "premium": get(sell_st, "ce_ltp"), "iv": get(sell_st, "ce_iv", 15)},
                {"action": "buy",  "strike": buy_st,  "opt_type": "CE",
                 "premium": get(buy_st,  "ce_ltp"), "iv": get(buy_st,  "ce_iv", 15)},
            ], "bearish", "credit_spread")
            if s: raw.append(s)
    sell_ce, buy_ce = r_st, far_c
    sell_pe, buy_pe = s_st, far_p
    if len({sell_ce, buy_ce, sell_pe, buy_pe}) == 4:
        s = make_strategy("Iron Condor", [
            {"action": "sell", "strike": sell_ce, "opt_type": "CE",
             "premium": get(sell_ce, "ce_ltp"), "iv": get(sell_ce, "ce_iv", 15)},
            {"action": "buy",  "strike": buy_ce,  "opt_type": "CE",
             "premium": get(buy_ce,  "ce_ltp"), "iv": get(buy_ce,  "ce_iv", 15)},
            {"action": "sell", "strike": sell_pe, "opt_type": "PE",
             "premium": get(sell_pe, "pe_ltp"), "iv": get(sell_pe, "pe_iv", 15)},
            {"action": "buy",  "strike": buy_pe,  "opt_type": "PE",
             "premium": get(buy_pe,  "pe_ltp"), "iv": get(buy_pe,  "pe_iv", 15)},
        ], "neutral", "iron_condor")
        if s: raw.append(s)
    wing = int(sr_range * 0.4 / 50) * 50
    wing = max(wing, 100)
    buy_c_ibf = nearest(atm + wing)
    buy_p_ibf = nearest(atm - wing)
    if len({atm, buy_c_ibf, buy_p_ibf}) == 3:
        s = make_strategy("Iron Butterfly", [
            {"action": "sell", "strike": atm,       "opt_type": "CE",
             "premium": get(atm,       "ce_ltp"), "iv": get(atm,       "ce_iv", 15)},
            {"action": "sell", "strike": atm,       "opt_type": "PE",
             "premium": get(atm,       "pe_ltp"), "iv": get(atm,       "pe_iv", 15)},
            {"action": "buy",  "strike": buy_c_ibf, "opt_type": "CE",
             "premium": get(buy_c_ibf, "ce_ltp"), "iv": get(buy_c_ibf, "ce_iv", 15)},
            {"action": "buy",  "strike": buy_p_ibf, "opt_type": "PE",
             "premium": get(buy_p_ibf, "pe_ltp"), "iv": get(buy_p_ibf, "pe_iv", 15)},
        ], "neutral", "iron_butterfly")
        if s: raw.append(s)
    s = make_strategy("Long Straddle", [
        {"action": "buy", "strike": atm, "opt_type": "CE",
         "premium": get(atm, "ce_ltp"), "iv": get(atm, "ce_iv", 15)},
        {"action": "buy", "strike": atm, "opt_type": "PE",
         "premium": get(atm, "pe_ltp"), "iv": get(atm, "pe_iv", 15)},
    ], "volatile", "straddle")
    if s: raw.append(s)
    if bias == "neutral":
        s = make_strategy("Short Straddle", [
            {"action": "sell", "strike": atm, "opt_type": "CE",
             "premium": get(atm, "ce_ltp"), "iv": get(atm, "ce_iv", 15)},
            {"action": "sell", "strike": atm, "opt_type": "PE",
             "premium": get(atm, "pe_ltp"), "iv": get(atm, "pe_iv", 15)},
        ], "neutral", "straddle")
        if s: raw.append(s)
    s = make_strategy("Long Strangle", [
        {"action": "buy", "strike": otm_c, "opt_type": "CE",
         "premium": get(otm_c, "ce_ltp"), "iv": get(otm_c, "ce_iv", 15)},
        {"action": "buy", "strike": otm_p, "opt_type": "PE",
         "premium": get(otm_p, "pe_ltp"), "iv": get(otm_p, "pe_iv", 15)},
    ], "volatile", "strangle")
    if s: raw.append(s)
    if bias == "neutral":
        s = make_strategy("Short Strangle", [
            {"action": "sell", "strike": otm_c, "opt_type": "CE",
             "premium": get(otm_c, "ce_ltp"), "iv": get(otm_c, "ce_iv", 15)},
            {"action": "sell", "strike": otm_p, "opt_type": "PE",
             "premium": get(otm_p, "pe_ltp"), "iv": get(otm_p, "pe_iv", 15)},
        ], "neutral", "strangle")
        if s: raw.append(s)
    raw.sort(key=lambda x: x["score"], reverse=True)
    return raw


# =================================================================
#  SECTION 5 -- HTML GENERATOR
# =================================================================

def build_html(all_expiry_data, expiry_list, generated_at):
    expiry_json = {}
    for exp, data in all_expiry_data.items():
        oc = compute_greeks_for_chain(data["df"], data["underlying"], exp)
        expiry_json[exp] = {
            "underlying":  data["underlying"],
            "atm_strike":  data["atm_strike"],
            "expiry":      exp,
            "dte":         oc["dte"],
            "atm_greeks":  oc["atm_greeks"],
            "all_strikes": oc["all_strikes"],
        }

    data_json     = _to_json(expiry_json)
    expiry_list_j = _to_json(expiry_list)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nifty Options Analyzer</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:         #07090f;
  --bg2:        #0d1117;
  --bg3:        #131d2b;
  --border:     #1a2535;
  --border2:    #0e1e30;
  --cyan:       #00d4ff;
  --green:      #00c896;
  --red:        #ff6b6b;
  --gold:       #ffd166;
  --purple:     #8aa0ff;
  --orange:     #ff9f43;
  --text:       #ffffff;
  --text2:      #d8eeff;
  --text3:      #b8d4e8;
  --grid:       #0b1520;
}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh;overflow-x:hidden;}}
body::before{{content:'';position:fixed;inset:0;background-image:linear-gradient(var(--grid) 1px,transparent 1px),linear-gradient(90deg,var(--grid) 1px,transparent 1px);background-size:44px 44px;pointer-events:none;z-index:0;}}
.wrap{{position:relative;z-index:1;max-width:1536px;margin:0 auto;padding:12px 60px;}}

/* ── HEADER ── */
.hdr{{display:flex;align-items:center;justify-content:space-between;padding:16px 22px;background:linear-gradient(135deg,#0d111799,#11182699);border:1px solid var(--border);border-top:2px solid var(--cyan);border-radius:12px;margin-bottom:18px;backdrop-filter:blur(12px);}}
.hdr-left{{display:flex;align-items:center;gap:14px;}}
.logo{{width:44px;height:44px;background:linear-gradient(135deg,var(--cyan),var(--purple));border-radius:10px;display:flex;align-items:center;justify-content:center;font-family:'DM Mono',monospace;font-weight:500;font-size:19px;color:#000;box-shadow:0 0 18px #00d4ff33;}}
.hdr h1{{font-size:22px;font-weight:800;letter-spacing:-.5px;}}
.hdr h1 span{{color:var(--cyan);}}
.hdr-sub{{font-size:14px;color:#d8eeff;font-family:'DM Mono',monospace;margin-top:2px;}}
.live-pill{{display:flex;align-items:center;gap:7px;background:#00c89611;border:1px solid #00c89633;border-radius:20px;padding:5px 13px;font-size:14px;font-weight:700;color:var(--green);font-family:'DM Mono',monospace;}}
.live-dot{{width:7px;height:7px;background:var(--green);border-radius:50%;animation:pulse 1.5s infinite;}}

/* ── TICKER ── */
.ticker{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px;}}
.tick-card{{background:var(--bg3);border:1px solid var(--border);border-top:2px solid var(--tc,var(--cyan));border-radius:10px;padding:14px 18px;transition:border-color .3s;}}
.tick-card:hover{{border-color:var(--tc,var(--cyan));}}
.tick-lbl{{font-size:13px;color:#d8eeff;font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:1px;}}
.tick-val{{font-size:25px;font-weight:800;font-family:'DM Mono',monospace;margin-top:3px;}}
.tick-sub{{font-size:13px;color:#b8d4e8;font-family:'DM Mono',monospace;margin-top:1px;}}

/* ── MAIN LAYOUT ── */
.main{{display:grid;grid-template-columns:385px 1fr 310px;gap:18px;margin-bottom:18px;}}

/* ── PANEL ── */
.panel{{background:var(--bg3);border:1px solid var(--border);border-radius:12px;overflow:hidden;}}
.panel-hdr{{padding:13px 18px;background:linear-gradient(90deg,#0d1a26,#0d1117);border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;}}
.panel-title{{font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:#d8eeff;display:flex;align-items:center;gap:7px;}}
.panel-body{{padding:18px;}}

/* ── MODE TOGGLE ── */
.mode-toggle{{display:grid;grid-template-columns:1fr 1fr;background:var(--bg2);border:1px solid var(--border);border-radius:9px;padding:4px;margin-bottom:16px;gap:3px;}}
.mode-btn{{padding:9px 6px;border:1px solid transparent;border-radius:7px;background:transparent;color:#b8d4e8;font-family:'DM Mono',monospace;font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;cursor:pointer;transition:all .22s;text-align:center;display:flex;align-items:center;justify-content:center;gap:5px;}}
.mode-btn:hover{{color:#d8eeff;}}
.mode-btn.active-sr{{background:linear-gradient(135deg,rgba(0,212,255,.14),rgba(138,160,255,.09));border-color:rgba(0,212,255,.35);color:var(--cyan);box-shadow:0 2px 12px rgba(0,212,255,.12);}}
.mode-btn.active-be{{background:linear-gradient(135deg,rgba(255,209,102,.14),rgba(255,159,67,.09));border-color:rgba(255,209,102,.35);color:var(--gold);box-shadow:0 2px 12px rgba(255,209,102,.12);}}
.mode-dot{{width:6px;height:6px;border-radius:50%;background:currentColor;}}

/* ── INPUTS ── */
.form-grp{{margin-bottom:14px;}}
.form-lbl{{display:block;font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#d8eeff;margin-bottom:6px;font-family:'DM Mono',monospace;}}
.inp{{width:100%;background:var(--bg2);border:1px solid var(--border);border-radius:7px;padding:9px 12px;color:var(--text);font-family:'DM Mono',monospace;font-size:16px;outline:none;transition:all .2s;}}
.inp:focus{{border-color:var(--cyan);box-shadow:0 0 0 3px #00d4ff12;}}
.sel{{width:100%;background:var(--bg2);border:1px solid var(--border);border-radius:7px;padding:9px 12px;color:var(--text);font-family:'DM Mono',monospace;font-size:15px;outline:none;cursor:pointer;appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'%3E%3Cpath fill='%2300d4ff' d='M5 7L0 2h10z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 12px center;transition:all .2s;}}
.sel:focus{{border-color:var(--cyan);box-shadow:0 0 0 3px #00d4ff12;}}
.sel option{{background:#0d1117;}}
.bias-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;}}
.bias-btn{{padding:9px 6px;border:1px solid var(--border);border-radius:7px;background:var(--bg2);color:#d8eeff;font-family:'DM Mono',monospace;font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;cursor:pointer;transition:all .2s;text-align:center;}}
.bias-btn:hover{{border-color:var(--cyan);color:var(--cyan);}}
.bias-bull{{border-color:var(--green)!important;background:#00c89611!important;color:var(--green)!important;}}
.bias-bear{{border-color:var(--red)!important;background:#ff6b6b11!important;color:var(--red)!important;}}
.bias-neut{{border-color:var(--gold)!important;background:#ffd16611!important;color:var(--gold)!important;}}
.sr-row{{display:flex;gap:7px;margin-bottom:7px;align-items:center;}}
.add-btn{{background:transparent;border:1px dashed var(--border);border-radius:7px;color:#b8d4e8;padding:7px 11px;font-size:18px;cursor:pointer;transition:all .2s;line-height:1;}}
.add-btn:hover{{border-color:var(--cyan);color:var(--cyan);}}
.rm-btn{{background:transparent;border:none;color:#b8d4e8;font-size:17px;cursor:pointer;padding:3px 7px;transition:color .2s;line-height:1;}}
.rm-btn:hover{{color:var(--red);}}
.divider{{height:1px;background:var(--border);margin:12px 0;}}
.analyze-btn{{width:100%;padding:13px;border:none;border-radius:9px;font-family:'Syne',sans-serif;font-size:17px;font-weight:800;text-transform:uppercase;letter-spacing:2px;cursor:pointer;transition:all .3s;margin-top:6px;}}
.analyze-btn.sr-btn{{background:linear-gradient(135deg,var(--cyan),var(--purple));color:#000;}}
.analyze-btn.be-btn{{background:linear-gradient(135deg,var(--gold),var(--orange));color:#000;}}
.analyze-btn:hover{{transform:translateY(-2px);box-shadow:0 8px 28px #00d4ff33;}}
.analyze-btn.be-btn:hover{{box-shadow:0 8px 28px rgba(255,209,102,.3);}}
.analyze-btn:active{{transform:translateY(0);}}

/* ── BE MODE INPUTS ── */
.be-inp-wrap{{position:relative;}}
.be-inp-icon{{position:absolute;left:12px;top:50%;transform:translateY(-50%);font-size:15px;pointer-events:none;}}
.be-inp-padded{{padding-left:30px!important;}}
.be-optional-badge{{display:inline-flex;align-items:center;gap:4px;font-size:11px;font-weight:700;color:#9dbdd8;font-family:'DM Mono',monospace;background:var(--bg2);border:1px solid var(--border2);border-radius:10px;padding:2px 7px;margin-left:8px;text-transform:uppercase;letter-spacing:.5px;vertical-align:middle;}}

/* ── BE RANGE BAR ── */
.be-rangebar-wrap{{background:var(--bg2);border:1px solid var(--border2);border-radius:8px;padding:10px 14px;margin-bottom:14px;}}
.be-rb-lbl{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#b8d4e8;font-family:'DM Mono',monospace;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;}}
.be-rb-lbl span{{color:var(--gold);font-size:13px;font-family:'DM Mono',monospace;}}
.be-rb-track{{position:relative;height:8px;background:var(--bg);border-radius:4px;overflow:visible;margin-bottom:6px;}}
.be-rb-fill{{position:absolute;top:0;height:100%;background:linear-gradient(90deg,rgba(255,107,107,.4),rgba(0,200,150,.4));border-radius:4px;transition:all .35s;}}
.be-rb-fill-lower{{position:absolute;top:0;height:100%;background:rgba(255,107,107,.4);border-radius:4px 0 0 4px;transition:all .35s;}}
.be-rb-fill-upper{{position:absolute;top:0;height:100%;background:rgba(0,200,150,.4);border-radius:0 4px 4px 0;transition:all .35s;}}
.be-rb-spot{{position:absolute;top:50%;transform:translate(-50%,-50%);width:13px;height:13px;border-radius:50%;background:var(--cyan);border:2px solid var(--bg2);box-shadow:0 0 10px rgba(0,212,255,.6);transition:left .35s;z-index:2;}}
.be-rb-marker-lo{{position:absolute;top:50%;transform:translate(-50%,-50%);width:10px;height:10px;border-radius:50%;background:var(--red);border:2px solid var(--bg2);z-index:2;transition:left .35s;}}
.be-rb-marker-hi{{position:absolute;top:50%;transform:translate(-50%,-50%);width:10px;height:10px;border-radius:50%;background:var(--green);border:2px solid var(--bg2);z-index:2;transition:left .35s;}}
.be-rb-label-row{{display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:12px;}}

/* ── BE CHIPS ── */
.be-chips{{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-bottom:14px;}}
.be-chip{{background:var(--bg2);border:1px solid var(--border2);border-radius:7px;padding:8px 10px;text-align:center;}}
.be-chip-lbl{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:#b8d4e8;font-family:'DM Mono',monospace;margin-bottom:3px;}}
.be-chip-val{{font-size:16px;font-weight:800;font-family:'DM Mono',monospace;}}

/* ── OPTION CHAIN ── */
.chain-wrap{{overflow:auto;max-height:540px;}}
.chain-side-hdr{{display:grid;grid-template-columns:1fr 92px 1fr;padding:6px 0;border-bottom:1px solid var(--border);background:linear-gradient(90deg,#0d1a26,#0d1117);position:sticky;top:0;z-index:3;}}
.chain-side-hdr .ce-hdr{{text-align:right;padding-right:10px;font-size:12px;font-weight:800;color:var(--green);letter-spacing:1.4px;text-transform:uppercase;}}
.chain-side-hdr .st-hdr{{text-align:center;font-size:12px;font-weight:800;color:#d8eeff;letter-spacing:1px;text-transform:uppercase;}}
.chain-side-hdr .pe-hdr{{text-align:left;padding-left:10px;font-size:12px;font-weight:800;color:var(--red);letter-spacing:1.4px;text-transform:uppercase;}}
.chain-col-hdr{{display:grid;grid-template-columns:1fr 92px 1fr;padding:5px 0 4px;border-bottom:1px solid var(--border2);background:#0a1218;position:sticky;top:30px;z-index:2;}}
.chain-col-hdr .ce-cols{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;padding:0 8px 0 5px;}}
.chain-col-hdr .pe-cols{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;padding:0 5px 0 8px;}}
.chain-col-hdr span{{font-size:10px;color:#9dbdd8;text-transform:uppercase;letter-spacing:.8px;font-family:'JetBrains Mono',monospace;text-align:right;}}
.chain-col-hdr .pe-cols span{{text-align:left;}}
.chain-row{{display:grid;grid-template-columns:1fr 92px 1fr;border-bottom:1px solid var(--border2);transition:background .12s;position:relative;}}
.chain-row:hover{{background:#ffffff04;}}
.chain-row.atm-row{{background:#00d4ff07;border-left:2px solid var(--cyan);}}
.chain-row.sup-row .stk-cell{{border-left:2px solid var(--green)!important;}}
.chain-row.res-row .stk-cell{{border-left:2px solid var(--red)!important;}}
.chain-row.be-lo-row .stk-cell{{border-left:2px solid var(--red)!important;border-right:2px solid var(--red)!important;}}
.chain-row.be-hi-row .stk-cell{{border-left:2px solid var(--green)!important;border-right:2px solid var(--green)!important;}}
.ce-side{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;align-items:center;padding:6px 8px 6px 5px;gap:2px;position:relative;overflow:hidden;}}
.ce-heat-bg{{position:absolute;top:0;right:0;bottom:0;background:var(--green);opacity:0.14;pointer-events:none;border-radius:2px 0 0 2px;}}
.pe-side{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;align-items:center;padding:6px 5px 6px 8px;gap:2px;position:relative;overflow:hidden;}}
.pe-heat-bg{{position:absolute;top:0;left:0;bottom:0;background:var(--red);opacity:0.14;pointer-events:none;border-radius:0 2px 2px 0;}}
.stk-cell{{display:flex;align-items:center;justify-content:center;font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;border-left:1px solid var(--border);border-right:1px solid var(--border);background:#0a1520;min-height:31px;position:relative;flex-direction:column;gap:1px;}}
.atm-tag{{background:var(--cyan);color:#000;font-size:9px;font-weight:800;padding:1px 5px;border-radius:0 0 3px 3px;position:absolute;top:0;letter-spacing:.5px;}}
.cv-ltp{{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;text-align:right;}}
.cv-iv{{font-size:11px;color:#d8eeff;font-family:'JetBrains Mono',monospace;text-align:right;}}
.cv-oi{{font-size:11px;font-family:'JetBrains Mono',monospace;color:#d8eeff;text-align:right;}}
.cv-doi{{font-size:11px;font-family:'JetBrains Mono',monospace;text-align:right;}}
.pe-side .cv-ltp,.pe-side .cv-iv,.pe-side .cv-oi,.pe-side .cv-doi{{text-align:left;}}
.ce-ltp-v{{color:var(--green);}} .pe-ltp-v{{color:var(--red);}}
.up{{color:var(--green)!important;}} .down{{color:var(--red)!important;}}

/* ── GREEKS SIDEBAR ── */
.greeks-panel{{padding:0;}}
.greeks-title{{padding:13px 14px 10px;font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:#d8eeff;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border);}}
.g-exp-tag{{font-size:12px;color:var(--cyan);background:#00d4ff12;border:1px solid #00d4ff22;padding:2px 8px;border-radius:10px;font-family:'DM Mono',monospace;}}
.g-sel-wrap{{padding:10px 12px 8px;border-bottom:1px solid var(--border);}}
.g-sel{{width:100%;background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:7px 10px;color:var(--text);font-family:'DM Mono',monospace;font-size:13px;outline:none;cursor:pointer;appearance:none;}}
.g-sel option{{background:#0d1117;}}
.g-atm-badge{{margin:8px 12px;padding:8px 10px;background:linear-gradient(135deg,#00d4ff0a,#8aa0ff0a);border:1px solid #00d4ff22;border-radius:8px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:4px;}}
.g-strike-type{{font-size:11px;font-weight:700;color:rgba(138,160,255,.9);font-family:'DM Mono',monospace;}}
.g-strike-val{{font-size:18px;font-weight:800;color:var(--cyan);font-family:'DM Mono',monospace;}}
.g-ltp-row{{display:flex;gap:6px;margin-top:2px;width:100%;}}
.g-ce-ltp{{font-size:12px;color:#00c8e0;font-family:'DM Mono',monospace;}}
.g-pe-ltp{{font-size:12px;color:#ff9090;font-family:'DM Mono',monospace;}}
.g-row{{display:flex;align-items:center;justify-content:space-between;padding:8px 12px;border-bottom:1px solid var(--border2);}}
.g-label{{display:flex;flex-direction:column;}}
.g-name{{font-size:14px;font-weight:700;color:var(--text);}}
.g-sub{{font-size:12px;color:#cce4f5;margin-top:1px;font-family:'DM Mono',monospace;}}
.g-vals{{display:flex;flex-direction:column;gap:3px;align-items:flex-end;}}
.g-ce-val{{font-family:'DM Mono',monospace;font-size:14px;font-weight:700;color:#00c8e0;}}
.g-pe-val{{font-family:'DM Mono',monospace;font-size:14px;font-weight:700;color:#ff9090;}}
.delta-bar-wrap{{display:flex;align-items:center;gap:4px;}}
.delta-bar-track{{width:32px;height:3px;background:rgba(255,255,255,.08);border-radius:2px;overflow:hidden;}}
.delta-bar-fill{{height:100%;border-radius:2px;}}
.iv-gauge-wrap{{padding:8px 12px 6px;border-bottom:1px solid var(--border2);}}
.iv-gauge-row{{display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;}}
.iv-gauge-lbl{{font-size:12px;color:#d8eeff;font-family:'DM Mono',monospace;}}
.iv-gauge-val{{font-size:14px;font-weight:700;font-family:'DM Mono',monospace;}}
.iv-gauge-track{{height:4px;background:var(--border);border-radius:2px;overflow:hidden;margin-bottom:4px;}}
.iv-gauge-fill{{height:100%;border-radius:2px;transition:width .6s;}}
.iv-regime{{font-size:8.5px;text-align:center;font-weight:700;letter-spacing:.5px;padding:4px 0 8px;font-family:'DM Mono',monospace;}}
.skew-lbl{{font-size:12px;font-weight:700;font-family:'DM Mono',monospace;}}

/* ── STRATEGIES ── */
.strat-section{{margin-bottom:18px;}}
.sec-hdr{{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;}}
.sec-title{{font-size:17px;font-weight:800;display:flex;align-items:center;gap:9px;}}
.sec-tag{{font-size:12px;font-weight:700;background:#00d4ff12;border:1px solid #00d4ff28;color:var(--cyan);padding:2px 9px;border-radius:14px;font-family:'DM Mono',monospace;letter-spacing:1px;}}
.strat-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;}}

/* ── ACCORDION CARD ── */
.strat-card{{background:var(--bg3);border:1px solid var(--border);border-radius:11px;overflow:hidden;transition:border-color .25s,box-shadow .25s;cursor:pointer;}}
.strat-card:hover{{box-shadow:0 4px 20px rgba(0,0,0,.35);border-color:var(--cc,var(--cyan));}}
.strat-card.sc-open{{border-color:var(--cc,var(--cyan));box-shadow:0 0 0 1px var(--cc,var(--cyan))22,0 8px 28px rgba(0,0,0,.4);}}

/* Collapsed header row */
.sc-header{{display:flex;align-items:center;gap:8px;padding:10px 12px;user-select:none;flex-wrap:wrap;}}
.sc-chevron{{font-size:11px;color:#6a90b8;transition:transform .25s;flex-shrink:0;margin-right:2px;}}
.strat-card.sc-open .sc-chevron{{transform:rotate(90deg);color:var(--cc,var(--cyan));}}
.sc-pill-bias{{font-size:11px;font-weight:800;font-family:'DM Mono',monospace;padding:3px 8px;border-radius:4px;letter-spacing:.8px;flex-shrink:0;}}
.sc-pill-bull{{background:rgba(0,200,150,.15);color:#00c896;border:1px solid rgba(0,200,150,.3);}}
.sc-pill-bear{{background:rgba(255,107,107,.15);color:#ff6b6b;border:1px solid rgba(255,107,107,.3);}}
.sc-pill-neut{{background:rgba(0,212,255,.12);color:#00d4ff;border:1px solid rgba(0,212,255,.25);}}
.sc-pill-volt{{background:rgba(168,130,255,.15);color:#a882ff;border:1px solid rgba(168,130,255,.3);}}
.sc-name{{font-size:15px;font-weight:800;color:#ffffff;flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.sc-header-right{{display:flex;align-items:center;gap:6px;flex-shrink:0;flex-wrap:wrap;}}
.sc-legs-mini{{display:flex;flex-wrap:wrap;gap:4px;max-width:100%;}}
.sc-leg-chip{{font-size:12px;font-weight:700;font-family:'DM Mono',monospace;padding:3px 8px;border-radius:4px;white-space:nowrap;}}
.sc-leg-chip.sell{{background:rgba(255,107,107,.15);color:#ff8080;border:1px solid rgba(255,107,107,.25);}}
.sc-leg-chip.buy{{background:rgba(0,200,150,.12);color:#00c896;border:1px solid rgba(0,200,150,.22);}}
.pop-pill{{padding:4px 10px;border-radius:14px;font-size:15px;font-weight:800;font-family:'DM Mono',monospace;white-space:nowrap;flex-shrink:0;}}
.sc-fit-badge{{font-size:12px;font-weight:800;font-family:'DM Mono',monospace;padding:3px 8px;border-radius:4px;letter-spacing:.5px;flex-shrink:0;background:rgba(255,209,102,.12);color:#ffd166;border:1px solid rgba(255,209,102,.25);}}

/* Expandable body */
.sc-body{{display:none;border-top:1px solid var(--border);}}
.strat-card.sc-open .sc-body{{display:block;}}

.sc-top{{padding:13px 15px 10px;border-bottom:1px solid var(--border);display:flex;align-items:flex-start;justify-content:space-between;}}
.sc-sub{{font-size:13px;color:#d8eeff;margin-top:2px;font-family:'DM Mono',monospace;}}
.sc-fields{{padding:10px 15px;display:grid;grid-template-columns:1fr 1fr;gap:6px 10px;border-bottom:1px solid var(--border);}}
.sc-field{{display:flex;flex-direction:column;}}
.sc-field-lbl{{font-size:12px;font-weight:700;color:#c8dff0;text-transform:uppercase;letter-spacing:.7px;font-family:'DM Mono',monospace;}}
.sc-field-val{{font-size:15px;font-weight:700;font-family:'DM Mono',monospace;margin-top:2px;}}
.sc-legs{{padding:9px 15px;background:#0d111766;display:flex;flex-wrap:wrap;gap:5px;border-bottom:1px solid var(--border);}}
.leg-tag{{border-radius:5px;padding:3px 8px;font-size:9.5px;font-weight:700;font-family:'DM Mono',monospace;}}
.leg-buy{{border:1px solid var(--green);color:var(--green);background:#00c89608;}}
.leg-sell{{border:1px solid var(--red);color:var(--red);background:#ff6b6b08;}}

/* ── BE STRATEGY CARD EXTRAS ── */
.be-mode-bar{{background:linear-gradient(90deg,rgba(255,209,102,.18),transparent);border-bottom:1px solid rgba(255,209,102,.30);padding:5px 14px;font-size:13px;font-weight:800;color:#ffd166;font-family:'DM Mono',monospace;letter-spacing:1px;display:flex;align-items:center;gap:6px;}}
.sc-legs-detail{{padding:9px 15px;background:#0d111766;border-bottom:1px solid var(--border);}}
.sc-legs-detail-title{{font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:1.2px;color:#d8eeff;font-family:'DM Mono',monospace;margin-bottom:7px;}}
.leg-detail-row{{display:flex;align-items:flex-start;gap:8px;padding:6px 9px;border-radius:7px;border:1px solid transparent;margin-bottom:5px;}}
.leg-detail-row:last-child{{margin-bottom:0;}}
.leg-detail-row.buy{{background:rgba(0,200,150,.07);border-color:rgba(0,200,150,.18);}}
.leg-detail-row.sell{{background:rgba(255,107,107,.07);border-color:rgba(255,107,107,.18);}}
.leg-detail-badge{{font-size:12px;font-weight:800;font-family:'DM Mono',monospace;padding:2px 7px;border-radius:4px;flex-shrink:0;margin-top:2px;}}
.leg-detail-badge.buy{{background:rgba(0,200,150,.2);color:var(--green);}}
.leg-detail-badge.sell{{background:rgba(255,107,107,.2);color:var(--red);}}
.leg-detail-body{{flex:1;}}
.leg-detail-main{{display:flex;align-items:center;gap:7px;font-family:'DM Mono',monospace;font-size:14px;font-weight:700;}}
.leg-ce{{color:var(--green);}} .leg-pe{{color:var(--red);}}
.leg-stk{{color:var(--cyan);}} .leg-prem{{color:#d8eeff;font-size:14px;font-weight:600;}}
.leg-why{{font-size:13px;color:#c8dff0;font-family:'DM Mono',monospace;margin-top:2px;line-height:1.5;}}

.fit-bar-wrap{{padding:8px 15px;border-bottom:1px solid var(--border);}}
.fit-bar-hdr{{display:flex;justify-content:space-between;font-size:12px;font-weight:700;font-family:'DM Mono',monospace;color:#d8eeff;text-transform:uppercase;letter-spacing:.8px;margin-bottom:5px;}}
.fit-bar-track{{height:4px;background:var(--border);border-radius:2px;overflow:hidden;}}
.fit-bar-fill{{height:100%;border-radius:2px;transition:width .8s cubic-bezier(.4,0,.2,1);}}

.sc-score{{padding:8px 15px;display:flex;align-items:center;gap:8px;}}
.score-bar-track{{flex:1;height:3px;background:var(--border);border-radius:2px;overflow:hidden;}}
.score-bar-fill{{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--cyan),var(--purple));transition:width 1s ease;}}
.score-lbl{{font-size:13px;font-weight:700;color:#d8eeff;font-family:'DM Mono',monospace;}}
.score-num{{font-size:14px;font-weight:700;color:var(--cyan);font-family:'DM Mono',monospace;}}

/* ── INTRADAY P&L SIMULATOR ── */
.intraday-sim{{border-top:2px solid rgba(255,209,102,.22);background:linear-gradient(135deg,rgba(245,197,24,.03),rgba(200,155,10,.015));}}
.sim-tabs{{display:flex;border-bottom:1px solid rgba(255,255,255,.06);}}
.sim-tab{{flex:1;padding:9px 4px;font-family:'DM Mono',monospace;font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;cursor:pointer;text-align:center;color:#d8eeff;border:none;background:transparent;transition:all .2s;border-bottom:2px solid transparent;margin-bottom:-1px;}}
.sim-tab.active{{color:#ffd166;border-bottom-color:#ffd166;background:rgba(245,197,24,.10);}}
.sim-tab:hover:not(.active){{color:rgba(255,255,255,.55);background:rgba(255,255,255,.03);}}
.sim-hdr{{display:flex;align-items:center;justify-content:space-between;padding:8px 10px 6px;}}
.sim-icon{{width:20px;height:20px;border-radius:5px;background:rgba(245,197,24,.15);border:1px solid rgba(245,197,24,.3);display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;}}
.sim-title{{font-size:14px;font-weight:800;color:#ffd166;letter-spacing:.8px;text-transform:uppercase;}}
.sim-subtitle{{font-size:13px;color:#d8eeff;margin-top:2px;}}
.sim-tbl{{width:100%;border-collapse:collapse;}}
.sim-tbl thead tr{{background:rgba(255,255,255,.03);}}
.sim-tbl th{{padding:7px 8px;font-size:12px;font-weight:800;letter-spacing:1px;text-transform:uppercase;color:#d8eeff;text-align:center;border-bottom:1px solid rgba(255,255,255,.12);}}
.sim-tbl th:first-child{{text-align:left;}}
.sim-tbl td{{padding:7px 8px;font-family:'DM Mono',monospace;font-size:14px;font-weight:600;text-align:center;border-bottom:1px solid rgba(255,255,255,.06);transition:background .12s;}}
.sim-tbl td:first-child{{text-align:left;}}
.sim-tbl tr:last-child td{{border-bottom:none;}}
.sim-tbl tr:hover td{{background:rgba(255,255,255,.025);}}
.sim-tbl tr.sim-flat td{{background:rgba(245,197,24,.06);border-left:2px solid rgba(245,197,24,.35);}}
.sim-move-lbl{{font-size:13px;font-weight:800;padding:3px 8px;border-radius:5px;display:inline-block;}}
.sim-pnl-val{{font-weight:700;font-size:14px;}}
.sim-live-pnl{{display:flex;align-items:center;justify-content:center;gap:8px;padding:10px;flex-wrap:wrap;border-bottom:1px solid rgba(255,255,255,.05);}}
.slpb{{display:flex;flex-direction:column;align-items:center;gap:2px;padding:8px 12px;border-radius:9px;min-width:90px;}}
.slpb-lbl{{font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#b8d4e8;}}
.slpb-num{{font-family:'DM Mono',monospace;font-size:18px;font-weight:700;line-height:1;}}
.slpb-sub{{font-size:12px;color:#c8dff0;}}
.cbar-row{{display:flex;align-items:center;gap:6px;margin-bottom:5px;}}
.cbar-lbl{{font-family:'DM Mono',monospace;font-size:12px;font-weight:700;width:44px;flex-shrink:0;}}
.cbar-track{{flex:1;height:4px;background:rgba(255,255,255,.07);border-radius:2px;overflow:hidden;}}
.cbar-fill{{height:100%;border-radius:2px;transition:width .4s;}}
.cbar-val{{font-family:'DM Mono',monospace;font-size:12px;font-weight:700;min-width:58px;text-align:right;}}
.sim-note{{margin:0 10px 10px;padding:8px 10px;background:rgba(255,107,107,.08);border:1px solid rgba(255,107,107,.25);border-radius:7px;font-size:13px;color:rgba(255,180,180,.95);display:flex;align-items:flex-start;gap:6px;line-height:1.6;}}
.sim-slide-wrap{{padding:0 10px 10px;}}
.sim-slide-labels{{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;}}
.sim-slide-cur{{font-family:'DM Mono',monospace;font-size:13px;font-weight:700;color:#ffd166;background:rgba(245,197,24,.1);border:1px solid rgba(245,197,24,.28);border-radius:5px;padding:2px 8px;}}
.sim-slide-edge{{font-size:13px;font-weight:600;color:#d8eeff;}}
input.sim-range{{width:100%;height:4px;border-radius:2px;outline:none;border:none;-webkit-appearance:none;cursor:pointer;background:linear-gradient(90deg,#ffd166 var(--pct,50%),rgba(255,255,255,.09) var(--pct,50%));}}
input.sim-range::-webkit-slider-thumb{{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:#ffd166;border:2px solid var(--bg);box-shadow:0 0 6px rgba(245,197,24,.45);cursor:pointer;}}

/* ── PAYOFF ── */
.payoff-wrap{{position:relative;height:320px;padding:16px;background:var(--bg2);border-radius:0 0 11px 11px;}}
canvas#payoffChart{{width:100%!important;height:288px!important;}}

/* ── EMPTY STATE ── */
.empty{{text-align:center;padding:50px 20px;color:#b8d4e8;grid-column:1/-1;}}
.empty-icon{{font-size:44px;margin-bottom:12px;opacity:.6;}}
.empty p{{font-size:14px;font-family:'DM Mono',monospace;line-height:1.8;}}

/* ── FOOTER ── */
.footer{{text-align:center;padding:16px;font-size:13px;color:#b8d4e8;font-family:'DM Mono',monospace;border-top:1px solid var(--border);margin-top:6px;}}

/* ── ANIMATIONS ── */
@keyframes pulse{{0%,100%{{opacity:1;box-shadow:0 0 0 0 #00c89655;}}50%{{opacity:.6;box-shadow:0 0 0 5px transparent;}}}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(14px);}}to{{opacity:1;transform:translateY(0);}}}}
@keyframes modeSwitch{{from{{opacity:0;transform:translateX(-6px);}}to{{opacity:1;transform:translateX(0);}}}}
.strat-card{{animation:fadeUp .35s ease both;}}
.strat-card:nth-child(2){{animation-delay:.05s;}}
.strat-card:nth-child(3){{animation-delay:.10s;}}
.strat-card:nth-child(4){{animation-delay:.15s;}}
.strat-card:nth-child(5){{animation-delay:.20s;}}
.strat-card:nth-child(6){{animation-delay:.25s;}}
.mode-anim{{animation:modeSwitch .25s ease both;}}
::-webkit-scrollbar{{width:5px;height:5px;}}
::-webkit-scrollbar-track{{background:var(--bg);}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:3px;}}

@media(max-width:1100px){{.main{{grid-template-columns:300px 1fr!important;}}.greeks-panel{{display:none;}}.ticker{{grid-template-columns:repeat(2,1fr);}}}}
@media(max-width:800px){{.main{{grid-template-columns:1fr!important;}}.ticker{{grid-template-columns:repeat(2,1fr);}}.strat-grid{{grid-template-columns:1fr!important;}}.hdr{{flex-direction:column;gap:10px;align-items:flex-start;}}.hdr>div:last-child{{align-self:flex-end;}}.wrap{{padding:10px;}}}}
@media(max-width:500px){{.ticker{{grid-template-columns:1fr 1fr;gap:8px;}}.tick-val{{font-size:19px;}}.hdr h1{{font-size:18px;}}.bias-row{{grid-template-columns:repeat(3,1fr);}}.sc-fields{{grid-template-columns:1fr 1fr;}}}}
</style>
</head>
<body>
<div class="wrap">

<!-- HEADER -->
<div class="hdr">
  <div class="hdr-left">
    <div class="logo">N⊕</div>
    <div>
      <h1>Nifty <span>Options</span> Analyzer</h1>
      <div class="hdr-sub">// NSE LIVE DATA · STRATEGY ENGINE · BLACK-SCHOLES</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:14px;">
    <div style="text-align:right;">
      <div style="font-size:12px;color:#9dbdd8;font-family:'DM Mono',monospace;">GENERATED</div>
      <div class="gen-time" style="font-size:14px;color:#d8eeff;font-family:'DM Mono',monospace;">{generated_at}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:12px;color:#9dbdd8;font-family:'DM Mono',monospace;">IST TIME</div>
      <div style="font-size:17px;font-weight:700;color:var(--cyan);font-family:'DM Mono',monospace;letter-spacing:1px;" id="istClock">--:--:--</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:12px;color:#9dbdd8;font-family:'DM Mono',monospace;">NEXT REFRESH</div>
      <div style="font-size:14px;color:var(--cyan);font-family:'DM Mono',monospace;" id="countdown">30s</div>
    </div>
    <div class="live-pill"><div class="live-dot"></div>LIVE</div>
  </div>
</div>

<!-- TICKER -->
<div class="ticker">
  <div class="tick-card" style="--tc:var(--cyan)">
    <div class="tick-lbl">NIFTY SPOT</div>
    <div class="tick-val" style="color:var(--cyan)" id="spotVal">—</div>
    <div class="tick-sub" id="spotDte">DTE: —</div>
  </div>
  <div class="tick-card" style="--tc:var(--gold)">
    <div class="tick-lbl">ATM STRIKE</div>
    <div class="tick-val" style="color:var(--gold)" id="atmVal">—</div>
    <div class="tick-sub" id="atmIv">ATM IV: —</div>
  </div>
  <div class="tick-card" style="--tc:var(--green)">
    <div class="tick-lbl">MAX CE OI</div>
    <div class="tick-val up" id="maxCeOi">—</div>
    <div class="tick-sub">Key Resistance</div>
  </div>
  <div class="tick-card" style="--tc:var(--red)">
    <div class="tick-lbl">MAX PE OI</div>
    <div class="tick-val down" id="maxPeOi">—</div>
    <div class="tick-sub">Key Support</div>
  </div>
</div>

<!-- MAIN GRID -->
<div class="main">

  <!-- ════════════════════════════════════
       INPUT PANEL — with SR / BE toggle
  ════════════════════════════════════ -->
  <div class="panel">
    <div class="panel-hdr">
      <div class="panel-title">⚙ Strategy Parameters</div>
    </div>
    <div class="panel-body">

      <!-- MARKET BIAS (always visible) -->
      <div class="form-grp">
        <label class="form-lbl">Market Bias</label>
        <div class="bias-row">
          <button class="bias-btn" id="btnBull" onclick="setBias('bullish')">🐂 Bullish</button>
          <button class="bias-btn" id="btnBear" onclick="setBias('bearish')">🐻 Bearish</button>
          <button class="bias-btn" id="btnNeut" onclick="setBias('neutral')">⚖️ Neutral</button>
        </div>
      </div>

      <div class="divider"></div>

      <!-- MODE TOGGLE -->
      <div class="form-grp" style="margin-bottom:16px;">
        <label class="form-lbl">Input Mode</label>
        <div class="mode-toggle">
          <button class="mode-btn active-sr" id="modeSR" onclick="setMode('sr')">
            <div class="mode-dot"></div> 📍 S/R Levels
          </button>
          <button class="mode-btn" id="modeBE" onclick="setMode('be')">
            <div class="mode-dot" style="opacity:.6;"></div> 🎯 Breakeven
          </button>
        </div>
      </div>

      <!-- ══════════════════════════════
           SR MODE CONTENT
      ══════════════════════════════ -->
      <div id="srContent" class="mode-anim">
        <div class="form-grp">
          <label class="form-lbl">Support Levels</label>
          <div id="supContainer">
            <div class="sr-row"><input type="number" class="inp sup-inp" placeholder="e.g. 22000"/><button class="rm-btn" onclick="rmRow(this)">✕</button></div>
          </div>
          <button class="add-btn" onclick="addLevel('sup')">+ Add Support</button>
        </div>
        <div class="form-grp">
          <label class="form-lbl">Resistance Levels</label>
          <div id="resContainer">
            <div class="sr-row"><input type="number" class="inp res-inp" placeholder="e.g. 22500"/><button class="rm-btn" onclick="rmRow(this)">✕</button></div>
          </div>
          <button class="add-btn" onclick="addLevel('res')">+ Add Resistance</button>
        </div>
        <div class="divider"></div>
        <div class="form-grp">
          <label class="form-lbl">Expiry</label>
          <select class="sel" id="expirySel" onchange="onExpiryChange()"></select>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
          <div class="form-grp"><label class="form-lbl">Lot Size</label><input type="number" class="inp" id="lotSize" value="65"/></div>
          <div class="form-grp"><label class="form-lbl">Max Capital ₹</label><input type="number" class="inp" id="maxCap" placeholder="500000"/></div>
        </div>
        <button class="analyze-btn sr-btn" onclick="analyze()">⚡ Analyze Strategies</button>
      </div>

      <!-- ══════════════════════════════
           BREAKEVEN MODE CONTENT
      ══════════════════════════════ -->
      <div id="beContent" style="display:none;" class="mode-anim">

        <!-- Range visual bar (shown when at least one BE is entered) -->
        <div class="be-rangebar-wrap" id="beRangeBarWrap" style="display:none;">
          <div class="be-rb-lbl">
            Expected Range
            <span id="beWidthLabel">—</span>
          </div>
          <div class="be-rb-track">
            <div id="beRbFillLower" class="be-rb-fill-lower" style="left:0%;width:0%;"></div>
            <div id="beRbFillUpper" class="be-rb-fill-upper" style="left:100%;width:0%;"></div>
            <div id="beRbFillBoth"  class="be-rb-fill" style="display:none;left:10%;width:80%;"></div>
            <div class="be-rb-spot" id="beRbSpot" style="left:50%;"></div>
            <div class="be-rb-marker-lo" id="beRbMarkerLo" style="display:none;"></div>
            <div class="be-rb-marker-hi" id="beRbMarkerHi" style="display:none;"></div>
          </div>
          <div class="be-rb-label-row">
            <span id="beRbLoLbl" style="color:var(--red);font-family:'DM Mono',monospace;font-size:12px;">—</span>
            <span style="color:var(--cyan);font-family:'DM Mono',monospace;font-size:12px;" id="beRbSpotLbl">SPOT —</span>
            <span id="beRbHiLbl" style="color:var(--green);font-family:'DM Mono',monospace;font-size:12px;">—</span>
          </div>
        </div>

        <!-- BE LOWER — optional -->
        <div class="form-grp">
          <label class="form-lbl">
            🔴 BE Lower (Support Floor)
            <span class="be-optional-badge">optional</span>
          </label>
          <div class="be-inp-wrap">
            <span class="be-inp-icon">⬇</span>
            <input type="number" class="inp be-inp-padded" id="beLower"
                   placeholder="e.g. 24000 — leave blank to skip"
                   oninput="updateBEVisual()"/>
          </div>
          <div style="font-size:12px;color:#9dbdd8;font-family:'DM Mono',monospace;margin-top:4px;" id="beLowerHint">
            "Nifty will NOT close below this at expiry"
          </div>
        </div>

        <!-- BE UPPER — optional -->
        <div class="form-grp">
          <label class="form-lbl">
            🟢 BE Upper (Resistance Ceiling)
            <span class="be-optional-badge">optional</span>
          </label>
          <div class="be-inp-wrap">
            <span class="be-inp-icon">⬆</span>
            <input type="number" class="inp be-inp-padded" id="beUpper"
                   placeholder="e.g. 25500 — leave blank to skip"
                   oninput="updateBEVisual()"/>
          </div>
          <div style="font-size:12px;color:#9dbdd8;font-family:'DM Mono',monospace;margin-top:4px;" id="beUpperHint">
            "Nifty will NOT close above this at expiry"
          </div>
        </div>

        <!-- Dynamic hint based on what's entered -->
        <div id="beModeHint" style="margin-bottom:14px;padding:9px 12px;border-radius:8px;font-size:9.5px;font-family:'DM Mono',monospace;line-height:1.7;display:none;"></div>

        <!-- Info chips (shown when both BEs entered) -->
        <div class="be-chips" id="beChips" style="display:none;">
          <div class="be-chip">
            <div class="be-chip-lbl">Range Width</div>
            <div class="be-chip-val" style="color:var(--gold);" id="chipWidth">—</div>
          </div>
          <div class="be-chip">
            <div class="be-chip-lbl">Spot vs Mid</div>
            <div class="be-chip-val" style="color:var(--cyan);" id="chipMid">—</div>
          </div>
          <div class="be-chip">
            <div class="be-chip-lbl">Lower Buffer</div>
            <div class="be-chip-val" style="color:var(--red);" id="chipLoBuf">—</div>
          </div>
          <div class="be-chip">
            <div class="be-chip-lbl">Upper Buffer</div>
            <div class="be-chip-val" style="color:var(--green);" id="chipHiBuf">—</div>
          </div>
        </div>

        <div class="divider"></div>

        <div class="form-grp">
          <label class="form-lbl">Expiry</label>
          <select class="sel" id="expirySel2" onchange="onExpiryChange2()"></select>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
          <div class="form-grp"><label class="form-lbl">Lot Size</label><input type="number" class="inp" id="lotSizeBE" value="65"/></div>
          <div class="form-grp"><label class="form-lbl">Max Capital ₹</label><input type="number" class="inp" id="maxCapBE" placeholder="500000"/></div>
        </div>

        <button class="analyze-btn be-btn" onclick="analyzeBE()">🎯 Find BE Strategies</button>
      </div>

    </div>
  </div>

  <!-- OPTION CHAIN -->
  <div class="panel">
    <div class="panel-hdr">
      <div class="panel-title">📊 Live Option Chain</div>
      <div style="display:flex;align-items:center;gap:14px;">
        <span style="font-size:13px;color:#d8eeff;font-family:'JetBrains Mono',monospace;">SPOT <b style="color:var(--cyan);" id="chainSpotLbl">—</b></span>
        <span style="font-size:13px;color:#d8eeff;font-family:'JetBrains Mono',monospace;">DTE <b style="color:var(--gold);" id="chainDteLbl">—</b></span>
        <span style="font-size:13px;color:#d8eeff;font-family:'JetBrains Mono',monospace;" id="chainExpLbl"></span>
      </div>
    </div>
    <div class="chain-wrap">
      <div class="chain-side-hdr">
        <div class="ce-hdr">── CALLS (CE) ──</div>
        <div class="st-hdr">STRIKE</div>
        <div class="pe-hdr">── PUTS (PE) ──</div>
      </div>
      <div class="chain-col-hdr">
        <div class="ce-cols"><span>LTP</span><span>IV%</span><span>OI(L)</span><span>ΔOI</span></div>
        <div></div>
        <div class="pe-cols"><span>ΔOI</span><span>OI(L)</span><span>IV%</span><span>LTP</span></div>
      </div>
      <div id="chainBody"><div style="text-align:center;padding:50px;color:#9dbdd8;font-family:'JetBrains Mono',monospace;font-size:14px;">Loading…</div></div>
    </div>
  </div>

  <!-- GREEKS SIDEBAR -->
  <div class="panel greeks-panel" id="greeksPanel">
    <div class="greeks-title">▲ GREEKS <span class="g-exp-tag" id="greeksExpTag">—</span></div>
    <div class="g-sel-wrap"><select class="g-sel" id="greeksStrikeSel" onchange="updateGreeksForStrike(this.value)"></select></div>
    <div class="g-atm-badge" id="greeksAtmBadge">
      <div><div class="g-strike-type" id="greeksTypeLabel">ATM</div><div class="g-strike-val" id="greeksStrikeVal">—</div></div>
      <div class="g-ltp-row"><span class="g-ce-ltp" id="greeksCeLtp">CE —</span><span style="color:#9dbdd8;font-size:12px;">/</span><span class="g-pe-ltp" id="greeksPeLtp">PE —</span></div>
    </div>
    <div class="g-row"><div class="g-label"><span class="g-name">Δ Delta</span><span class="g-sub">CE / PE</span></div><div class="g-vals" id="greeksDeltaWrap"><div class="delta-bar-wrap"><div class="delta-bar-track"><div class="delta-bar-fill" id="dbarCe" style="background:var(--green);"></div></div><span class="g-ce-val" id="greeksDeltaCe">—</span></div><div class="delta-bar-wrap"><div class="delta-bar-track"><div class="delta-bar-fill" id="dbarPe" style="background:var(--red);"></div></div><span class="g-pe-val" id="greeksDeltaPe">—</span></div></div></div>
    <div class="g-row"><div class="g-label"><span class="g-name">σ IV</span><span class="skew-lbl" id="greeksSkewLbl" style="color:var(--purple);">—</span></div><div class="g-vals"><div style="display:flex;align-items:center;gap:5px;"><span style="font-size:12px;font-weight:700;color:#00d8f5;">CE</span><span class="g-ce-val" id="greeksIvCe">—</span></div><div style="display:flex;align-items:center;gap:5px;"><span style="font-size:12px;font-weight:700;color:#ff9090;">PE</span><span class="g-pe-val" id="greeksIvPe">—</span></div></div></div>
    <div class="g-row"><div class="g-label"><span class="g-name">Θ Theta</span><span class="g-sub">per day</span></div><div class="g-vals"><div style="display:flex;align-items:center;gap:5px;"><span style="font-size:12px;font-weight:700;color:#00d8f5;">CE</span><span class="g-ce-val" id="greeksThetaCe">—</span></div><div style="display:flex;align-items:center;gap:5px;"><span style="font-size:12px;font-weight:700;color:#ff9090;">PE</span><span class="g-pe-val" id="greeksThetaPe">—</span></div></div></div>
    <div class="g-row"><div class="g-label"><span class="g-name">ν Vega</span><span class="g-sub">per 1% IV</span></div><div class="g-vals"><div style="display:flex;align-items:center;gap:5px;"><span style="font-size:12px;font-weight:700;color:#00d8f5;">CE</span><span class="g-ce-val" id="greeksVegaCe">—</span></div><div style="display:flex;align-items:center;gap:5px;"><span style="font-size:12px;font-weight:700;color:#ff9090;">PE</span><span class="g-pe-val" id="greeksVegaPe">—</span></div></div></div>
    <div class="g-row"><div class="g-label"><span class="g-name">Γ Gamma</span><span class="g-sub">delta/point</span></div><div class="g-vals"><div style="display:flex;align-items:center;gap:5px;"><span style="font-size:12px;font-weight:700;color:#00d8f5;">CE</span><span class="g-ce-val" id="greeksGammaCe">—</span></div><div style="display:flex;align-items:center;gap:5px;"><span style="font-size:12px;font-weight:700;color:#ff9090;">PE</span><span class="g-pe-val" id="greeksGammaPe">—</span></div></div></div>
    <div class="iv-gauge-wrap">
      <div class="iv-gauge-row"><span class="iv-gauge-lbl">IV Average</span><span class="iv-gauge-val" id="greeksIvAvg" style="color:var(--green);">—</span></div>
      <div class="iv-gauge-track"><div class="iv-gauge-fill" id="greeksIvBar" style="width:0%;background:var(--green);"></div></div>
      <div class="iv-regime" id="greeksIvRegime" style="color:var(--green);">—</div>
    </div>
  </div>

</div>

<!-- STRATEGY RECOMMENDATIONS -->
<div class="strat-section">
  <div class="sec-hdr">
    <div class="sec-title">🏆 Strategy Recommendations <span class="sec-tag" id="stratCount">0 STRATEGIES</span></div>
    <select class="sel" style="width:180px;" id="sortSel" onchange="renderStrategies()">
      <option value="score">Sort: Best Score</option>
      <option value="pop">Sort: Probability</option>
      <option value="profit">Sort: Max Profit</option>
      <option value="rr">Sort: Risk/Reward</option>
    </select>
  </div>
  <div class="strat-grid" id="stratGrid">
    <div class="empty"><div class="empty-icon">📋</div><p>Select a mode (S/R or Breakeven)<br>enter your levels, then click Analyze</p></div>
  </div>
</div>

<!-- PAYOFF CHART -->
<div class="panel" style="margin-bottom:18px;">
  <div class="panel-hdr">
    <div class="panel-title">📈 Payoff Diagram<span style="font-size:12px;color:#9dbdd8;margin-left:8px;font-family:'JetBrains Mono',monospace;">🟢 Today (BSM) &nbsp;|&nbsp; 🔵 At Expiry &nbsp;|&nbsp; bars = OI</span></div>
    <select class="sel" style="width:200px;" id="payoffSel" onchange="drawPayoff()"><option value="">— Select Strategy —</option></select>
  </div>
  <div id="payoffStats" style="display:none;grid-template-columns:repeat(4,1fr);gap:10px;padding:14px 16px 0;"></div>
  <div class="payoff-wrap" style="height:320px;padding:16px;position:relative;"><canvas id="payoffChart"></canvas></div>
  <div id="payoffTooltip" style="display:none;position:fixed;z-index:9999;background:rgba(8,18,30,0.97);border:1px solid rgba(0,212,255,0.28);border-radius:10px;padding:13px 16px;min-width:250px;max-width:280px;box-shadow:0 8px 36px rgba(0,0,0,0.7);pointer-events:none;font-family:'JetBrains Mono',monospace;backdrop-filter:blur(8px);"></div>
  <div id="payoffFooter" style="display:none;padding:10px 16px 12px;border-top:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="font-size:12px;color:#d8eeff;font-family:'JetBrains Mono',monospace;letter-spacing:1px;">BREAKEVENS</span>
      <div id="beBadges" style="display:flex;gap:6px;"></div>
    </div>
    <div style="font-size:14px;font-family:'JetBrains Mono',monospace;color:#d8eeff;" id="projBadge">Select a strategy to see projected P&L</div>
  </div>
  <div id="projBadgeFallback" style="text-align:center;padding:10px 16px 14px;font-size:14px;font-family:'JetBrains Mono',monospace;border-top:1px solid var(--border);color:#d8eeff;">Select a strategy to see projected P&L</div>
</div>

<div class="footer">NIFTY OPTIONS ANALYZER · NSE INDIA DATA · FOR EDUCATIONAL PURPOSE ONLY · NOT FINANCIAL ADVICE</div>
</div>

<script>
// ── Baked-in data ─────────────────────────────────────────────
const ALL_DATA    = {data_json};
const EXPIRY_LIST = {expiry_list_j};
const LOT_SIZE    = 65;
const RISK_FREE   = 0.065;

// ── State ─────────────────────────────────────────────────────
let currentExpiry  = EXPIRY_LIST[0] || "";
let marketBias     = "neutral";
let strategies     = [];
let payoffChart    = null;
let currentMode    = "sr";   // "sr" or "be"

// ── Init ──────────────────────────────────────────────────────
window.onload = () => {{
  populateExpiries();
  restoreUserState();
  if (currentExpiry) {{
    updateTicker();
    renderChain();
    renderGreeks(null);
  }}
  startCountdown();
  document.getElementById("lotSize").addEventListener("input", saveUserState);
  document.getElementById("maxCap").addEventListener("input",  saveUserState);
  document.querySelectorAll(".sup-inp,.res-inp").forEach(inp => {{
    inp.addEventListener("input", saveUserState);
  }});
}};

// ── Countdown & silent refresh ────────────────────────────────
function startCountdown() {{
  function updateClock() {{
    const now = new Date();
    const ist = new Date(now.getTime() + (5.5 * 60 * 60 * 1000));
    const hh  = String(ist.getUTCHours()).padStart(2,"0");
    const mm  = String(ist.getUTCMinutes()).padStart(2,"0");
    const ss  = String(ist.getUTCSeconds()).padStart(2,"0");
    const el  = document.getElementById("istClock");
    if (el) el.textContent = hh+":"+mm+":"+ss;
  }}
  updateClock();
  setInterval(updateClock, 1000);
  let secs = 30;
  const el = document.getElementById("countdown");
  setInterval(() => {{
    secs--;
    if (secs <= 0) {{ secs = 30; silentRefresh(); }}
    el.textContent = secs+"s";
  }}, 1000);
}}

async function silentRefresh() {{
  try {{
    const resp = await fetch(location.href+"?_="+Date.now(), {{cache:"no-store"}});
    if (!resp.ok) return;
    const html   = await resp.text();
    const parser = new DOMParser();
    const doc    = parser.parseFromString(html,"text/html");
    const scripts = doc.querySelectorAll("script");
    for (const s of scripts) {{
      const m = s.textContent.match(/const ALL_DATA[\\s]*=[\\s]*(\\{{[\\s\\S]*?\\}});/);
      if (m) {{
        try {{
          const freshData = JSON.parse(m[1]);
          Object.assign(ALL_DATA, freshData);
          updateTicker(); renderChain();
          updateGreeksForStrike(parseInt(document.getElementById("greeksStrikeSel").value)||ALL_DATA[currentExpiry]?.atm_strike);
          const genEl = doc.querySelector(".gen-time");
          if (genEl) document.querySelector(".gen-time") && (document.querySelector(".gen-time").textContent = genEl.textContent);
        }} catch(e) {{ console.warn("Silent refresh parse error:",e); }}
        break;
      }}
    }}
  }} catch(e) {{ console.warn("Silent refresh fetch error:",e); }}
}}

// ── Expiry ────────────────────────────────────────────────────
function populateExpiries() {{
  const sel  = document.getElementById("expirySel");
  const sel2 = document.getElementById("expirySel2");
  const opts = EXPIRY_LIST.map((e,i)=>`<option value="${{e}}"${{i===0?" selected":""}}>${{e}}${{i===0?" (Weekly)":""}}</option>`).join("");
  sel.innerHTML  = opts;
  sel2.innerHTML = opts;
  currentExpiry  = EXPIRY_LIST[0] || "";
}}

function onExpiryChange()  {{ currentExpiry = document.getElementById("expirySel").value;  updateTicker(); renderChain(); renderGreeks(null); saveUserState(); }}
function onExpiryChange2() {{ currentExpiry = document.getElementById("expirySel2").value; updateTicker(); renderChain(); renderGreeks(null); saveUserState(); }}

// ── Ticker ────────────────────────────────────────────────────
function updateTicker() {{
  const d = ALL_DATA[currentExpiry];
  if (!d) return;
  document.getElementById("spotVal").textContent = "₹"+d.underlying.toLocaleString("en-IN");
  document.getElementById("atmVal").textContent  = d.atm_strike.toLocaleString("en-IN");
  document.getElementById("spotDte").textContent = "DTE: "+d.dte;
  const atm = d.all_strikes.find(s=>s.is_atm);
  if (atm) {{
    const ivAvg = ((atm.ce_iv||0)+(atm.pe_iv||0))/2;
    document.getElementById("atmIv").textContent = "ATM IV: "+ivAvg.toFixed(1)+"%";
  }}
  let maxCeOi=0,maxCeSt=0,maxPeOi=0,maxPeSt=0;
  d.all_strikes.forEach(s=>{{
    if(s.ce_oi>maxCeOi){{maxCeOi=s.ce_oi;maxCeSt=s.strike;}}
    if(s.pe_oi>maxPeOi){{maxPeOi=s.pe_oi;maxPeSt=s.strike;}}
  }});
  document.getElementById("maxCeOi").textContent = maxCeSt.toLocaleString("en-IN");
  document.getElementById("maxPeOi").textContent = maxPeSt.toLocaleString("en-IN");
}}

// ── Option Chain ──────────────────────────────────────────────
function renderChain() {{
  const d = ALL_DATA[currentExpiry];
  if (!d) return;
  document.getElementById("chainExpLbl").textContent = currentExpiry;
  const chainSpot = document.getElementById("chainSpotLbl");
  const chainDte  = document.getElementById("chainDteLbl");
  if (chainSpot) chainSpot.textContent = "₹"+d.underlying.toLocaleString("en-IN");
  if (chainDte)  chainDte.textContent  = d.dte;

  const supports    = getSupports();
  const resistances = getResistances();
  // Also get BE levels if in BE mode
  const beLo = currentMode==="be" ? (parseFloat(document.getElementById("beLower").value)||null) : null;
  const beHi = currentMode==="be" ? (parseFloat(document.getElementById("beUpper").value)||null) : null;
  const beLo_st = beLo ? nearestStrike(d.all_strikes, beLo) : null;
  const beHi_st = beHi ? nearestStrike(d.all_strikes, beHi) : null;

  const rows   = [...d.all_strikes].sort((a,b)=>b.strike-a.strike);
  const maxOi  = Math.max(...rows.flatMap(r=>[r.ce_oi,r.pe_oi]),1);

  document.getElementById("chainBody").innerHTML = rows.map(r => {{
    const isAtm = r.is_atm;
    const isSup = supports.includes(r.strike);
    const isRes = resistances.includes(r.strike);
    const isBeL = beLo_st === r.strike;
    const isBeH = beHi_st === r.strike;
    const ceOiL = (r.ce_oi/1e5).toFixed(1);
    const peOiL = (r.pe_oi/1e5).toFixed(1);
    const ceHeat = Math.round((r.ce_oi/maxOi)*100);
    const peHeat = Math.round((r.pe_oi/maxOi)*100);
    const ceChgStr = (r.ce_oi_chg>=0?"+":"")+(r.ce_oi_chg/1e5).toFixed(1)+"L";
    const peChgStr = (r.pe_oi_chg>=0?"+":"")+(r.pe_oi_chg/1e5).toFixed(1)+"L";
    const ceChgCls = r.ce_oi_chg>=0?"up":"down";
    const peChgCls = r.pe_oi_chg>=0?"up":"down";
    let rc = isAtm?"atm-row":isSup?"sup-row":isRes?"res-row":isBeL?"be-lo-row":isBeH?"be-hi-row":"";
    const smark = isSup?'<span style="color:var(--green);font-size:9px;font-weight:700;letter-spacing:.5px;">▲ SUP</span>':"";
    const rmark = isRes?'<span style="color:var(--red);font-size:9px;font-weight:700;letter-spacing:.5px;">▼ RES</span>':"";
    const blomark = isBeL?'<span style="color:var(--red);font-size:9px;font-weight:700;letter-spacing:.5px;">BE ▼</span>':"";
    const bhimark = isBeH?'<span style="color:var(--green);font-size:9px;font-weight:700;letter-spacing:.5px;">BE ▲</span>':"";
    return `<div class="chain-row ${{rc}}">
      <div class="ce-side">
        <div class="ce-heat-bg" style="width:${{ceHeat}}%"></div>
        <span class="cv-ltp ce-ltp-v">${{r.ce_ltp.toFixed(2)}}</span>
        <span class="cv-iv">${{r.ce_iv.toFixed(1)}}%</span>
        <span class="cv-oi">${{ceOiL}}L</span>
        <span class="cv-doi ${{ceChgCls}}">${{ceChgStr}}</span>
      </div>
      <div class="stk-cell">
        ${{isAtm?'<span class="atm-tag">ATM</span>':""}}
        <span style="color:${{isAtm?"var(--cyan)":isBeL?"var(--red)":isBeH?"var(--green)":"var(--text)"}}">${{r.strike.toLocaleString("en-IN")}}</span>
        ${{(smark||rmark||blomark||bhimark)?`<span style="display:flex;gap:4px;">${{smark}}${{rmark}}${{blomark}}${{bhimark}}</span>`:""}}
      </div>
      <div class="pe-side">
        <div class="pe-heat-bg" style="width:${{peHeat}}%"></div>
        <span class="cv-doi ${{peChgCls}}">${{peChgStr}}</span>
        <span class="cv-oi">${{peOiL}}L</span>
        <span class="cv-iv">${{r.pe_iv.toFixed(1)}}%</span>
        <span class="cv-ltp pe-ltp-v">${{r.pe_ltp.toFixed(2)}}</span>
      </div>
    </div>`;
  }}).join("");
}}

function nearestStrike(allStrikes, val) {{
  return allStrikes.reduce((a,b)=>Math.abs(b.strike-val)<Math.abs(a.strike-val)?b:a).strike;
}}

// ── Greeks ────────────────────────────────────────────────────
function renderGreeks(strikeOverride) {{
  const d = ALL_DATA[currentExpiry];
  if (!d) return;
  document.getElementById("greeksExpTag").textContent = currentExpiry;
  const sel   = document.getElementById("greeksStrikeSel");
  const atm   = d.atm_strike;
  const rows  = [...d.all_strikes].sort((a,b)=>b.strike-a.strike);
  let ceOpts="",atmOpt="",peOpts="";
  rows.forEach(r=>{{
    const s=r.strike, dist=Math.abs(s-atm)/50;
    if(r.is_atm) atmOpt=`<option value="${{s}}" ${{!strikeOverride?"selected":""}}>\u2605 ATM ₹${{s.toLocaleString("en-IN")}}</option>`;
    else if(s>atm) ceOpts+=`<option value="${{s}}" ${{strikeOverride==s?"selected":""}}>\u25b2 CE+${{dist}} ₹${{s.toLocaleString("en-IN")}}</option>`;
    else peOpts+=`<option value="${{s}}" ${{strikeOverride==s?"selected":""}}>\u25bc PE-${{dist}} ₹${{s.toLocaleString("en-IN")}}</option>`;
  }});
  sel.innerHTML=`<optgroup label="─ OTM CALLS (CE) ─">${{ceOpts}}</optgroup><optgroup label="─ ATM ─">${{atmOpt}}</optgroup><optgroup label="─ OTM PUTS (PE) ─">${{peOpts}}</optgroup>`;
  updateGreeksForStrike(strikeOverride||atm);
}}

function updateGreeksForStrike(strike) {{
  strike=parseInt(strike);
  const d=ALL_DATA[currentExpiry]; if(!d) return;
  const row=d.all_strikes.find(s=>s.strike===strike); if(!row) return;
  const atm=d.atm_strike, dist=Math.abs(strike-atm)/50;
  const typeLabel=row.is_atm?"ATM":strike>atm?`OTM CE +${{dist}}`:`OTM PE -${{dist}}`;
  document.getElementById("greeksTypeLabel").textContent=typeLabel;
  document.getElementById("greeksStrikeVal").textContent="₹"+strike.toLocaleString("en-IN");
  document.getElementById("greeksCeLtp").textContent="CE ₹"+row.ce_ltp.toFixed(1);
  document.getElementById("greeksPeLtp").textContent="PE ₹"+row.pe_ltp.toFixed(1);
  document.getElementById("greeksDeltaCe").textContent=(row.ce_delta>=0?"+":"")+row.ce_delta.toFixed(3);
  document.getElementById("greeksDeltaPe").textContent=(row.pe_delta>=0?"+":"")+row.pe_delta.toFixed(3);
  document.getElementById("greeksIvCe").textContent=row.ce_iv.toFixed(1)+"%";
  document.getElementById("greeksIvPe").textContent=row.pe_iv.toFixed(1)+"%";
  document.getElementById("greeksThetaCe").textContent="₹"+Math.abs(row.ce_theta).toFixed(2);
  document.getElementById("greeksThetaPe").textContent="₹"+Math.abs(row.pe_theta).toFixed(2);
  document.getElementById("greeksVegaCe").textContent=row.ce_vega.toFixed(4);
  document.getElementById("greeksVegaPe").textContent=row.pe_vega.toFixed(4);
  document.getElementById("greeksGammaCe").textContent=row.ce_gamma.toFixed(6);
  document.getElementById("greeksGammaPe").textContent=row.pe_gamma.toFixed(6);
  document.getElementById("dbarCe").style.width=Math.abs(row.ce_delta)*100+"%";
  document.getElementById("dbarPe").style.width=Math.abs(row.pe_delta)*100+"%";
  const ivAvg=(row.ce_iv+row.pe_iv)/2;
  const ivPct=Math.min(100,(ivAvg/60)*100);
  const ivCol=ivAvg>25?"var(--red)":ivAvg>18?"var(--gold)":"var(--green)";
  const ivReg=ivAvg>25?"High IV · Buy Premium":ivAvg>15?"Normal IV · Balanced":"Low IV · Sell Premium";
  document.getElementById("greeksIvAvg").textContent=ivAvg.toFixed(1)+"%";
  document.getElementById("greeksIvAvg").style.color=ivCol;
  document.getElementById("greeksIvBar").style.width=ivPct.toFixed(1)+"%";
  document.getElementById("greeksIvBar").style.background=ivCol;
  document.getElementById("greeksIvBar").style.boxShadow=`0 0 6px ${{ivCol}}88`;
  document.getElementById("greeksIvRegime").textContent=ivReg;
  document.getElementById("greeksIvRegime").style.color=ivCol;
  const skew=row.pe_iv-row.ce_iv;
  const skewCol=skew>1.5?"var(--red)":skew<-1.5?"var(--green)":"var(--purple)";
  document.getElementById("greeksSkewLbl").textContent=skew>0?`PE Skew +${{skew.toFixed(1)}}`:`CE Skew ${{skew.toFixed(1)}}`;
  document.getElementById("greeksSkewLbl").style.color=skewCol;
}}

// ── Bias ──────────────────────────────────────────────────────
function setBias(b) {{
  marketBias=b;
  document.getElementById("btnBull").className="bias-btn"+(b==="bullish"?" bias-bull":"");
  document.getElementById("btnBear").className="bias-btn"+(b==="bearish"?" bias-bear":"");
  document.getElementById("btnNeut").className="bias-btn"+(b==="neutral"?" bias-neut":"");
  saveUserState();
}}
setBias("neutral");

// ── MODE SWITCH ───────────────────────────────────────────────
function setMode(m) {{
  currentMode = m;
  const srC=document.getElementById("srContent");
  const beC=document.getElementById("beContent");
  const srB=document.getElementById("modeSR");
  const beB=document.getElementById("modeBE");

  if (m==="sr") {{
    srC.style.display="block"; srC.classList.remove("mode-anim"); void srC.offsetWidth; srC.classList.add("mode-anim");
    beC.style.display="none";
    srB.className="mode-btn active-sr";
    beB.className="mode-btn";
    beB.querySelector(".mode-dot").style.opacity=".3";
    srB.querySelector(".mode-dot").style.opacity="1";
  }} else {{
    beC.style.display="block"; beC.classList.remove("mode-anim"); void beC.offsetWidth; beC.classList.add("mode-anim");
    srC.style.display="none";
    beB.className="mode-btn active-be";
    srB.className="mode-btn";
    srB.querySelector(".mode-dot").style.opacity=".3";
    beB.querySelector(".mode-dot").style.opacity="1";
    updateBEVisual();
  }}
  // Clear results
  document.getElementById("stratGrid").innerHTML='<div class="empty"><div class="empty-icon">📋</div><p>Set your '+(m==="sr"?"support & resistance":"breakeven levels")+'<br>then click Analyze</p></div>';
  document.getElementById("stratCount").textContent="0 STRATEGIES";
  saveUserState();
}}

// ── BE VISUAL UPDATE ──────────────────────────────────────────
function updateBEVisual() {{
  const d    = ALL_DATA[currentExpiry];
  const spot = d ? d.underlying : 0;
  const lo   = parseFloat(document.getElementById("beLower").value) || null;
  const hi   = parseFloat(document.getElementById("beUpper").value) || null;

  const hasLo = lo !== null && lo > 0;
  const hasHi = hi !== null && hi > 0;

  // Show/hide range bar
  const rbWrap = document.getElementById("beRangeBarWrap");
  rbWrap.style.display = (hasLo || hasHi) ? "block" : "none";

  // Show/hide chips (only when both)
  document.getElementById("beChips").style.display = (hasLo && hasHi) ? "grid" : "none";

  // Hints
  const hintEl = document.getElementById("beModeHint");
  if (!hasLo && !hasHi) {{
    hintEl.style.display = "none";
  }} else {{
    hintEl.style.display = "block";
    if (hasLo && hasHi) {{
      hintEl.style.background = "rgba(255,209,102,.07)";
      hintEl.style.border     = "1px solid rgba(255,209,102,.2)";
      hintEl.style.color      = "rgba(255,209,102,.8)";
      hintEl.innerHTML = `⚖️ Both BEs set → Will suggest <b>Iron Condor, Short Strangle</b> + single-sided spreads`;
    }} else if (hasLo && !hasHi) {{
      hintEl.style.background = "rgba(0,200,150,.07)";
      hintEl.style.border     = "1px solid rgba(0,200,150,.2)";
      hintEl.style.color      = "rgba(0,200,150,.8)";
      hintEl.innerHTML = `🐂 Only Lower BE → Will suggest <b>Bull Put Spread, Bull Call Spread</b> (bullish plays)`;
    }} else {{
      hintEl.style.background = "rgba(255,107,107,.07)";
      hintEl.style.border     = "1px solid rgba(255,107,107,.2)";
      hintEl.style.color      = "rgba(255,150,150,.8)";
      hintEl.innerHTML = `🐻 Only Upper BE → Will suggest <b>Bear Call Spread, Bear Put Spread</b> (bearish plays)`;
    }}
  }}

  if (!hasLo && !hasHi) return;

  // Build visual range
  const visMin = hasLo && hasHi ? lo - (hi-lo)*0.25 : hasLo ? lo-500 : hi-500;
  const visMax = hasLo && hasHi ? hi + (hi-lo)*0.25 : hasLo ? lo+500 : hi+500;
  const visR   = visMax - visMin || 1;

  const spotPct = Math.min(97, Math.max(3, ((spot-visMin)/visR*100)));
  document.getElementById("beRbSpot").style.left  = spotPct+"%";
  document.getElementById("beRbSpotLbl").textContent = "SPOT ₹"+(spot||0).toLocaleString("en-IN");

  // Both
  if (hasLo && hasHi) {{
    const fillL = ((lo-visMin)/visR*100).toFixed(1);
    const fillW = ((hi-lo)/visR*100).toFixed(1);
    document.getElementById("beRbFillBoth").style.display  = "block";
    document.getElementById("beRbFillLower").style.width   = "0%";
    document.getElementById("beRbFillUpper").style.width   = "0%";
    document.getElementById("beRbFillBoth").style.left     = fillL+"%";
    document.getElementById("beRbFillBoth").style.width    = fillW+"%";
    document.getElementById("beRbMarkerLo").style.display  = "block";
    document.getElementById("beRbMarkerHi").style.display  = "block";
    document.getElementById("beRbMarkerLo").style.left     = fillL+"%";
    document.getElementById("beRbMarkerHi").style.left     = (parseFloat(fillL)+parseFloat(fillW))+"%";
    document.getElementById("beRbLoLbl").textContent  = "▼ ₹"+lo.toLocaleString("en-IN");
    document.getElementById("beRbHiLbl").textContent  = "▲ ₹"+hi.toLocaleString("en-IN");
    document.getElementById("beWidthLabel").textContent = "Width: "+(hi-lo).toLocaleString("en-IN")+" pts";
    // chips
    document.getElementById("chipWidth").textContent  = (hi-lo).toLocaleString("en-IN")+" pts";
    const mid = (lo+hi)/2;
    const sd  = spot-mid; const sdSign = sd>=0?"+":"";
    document.getElementById("chipMid").textContent    = sdSign+Math.round(sd).toLocaleString("en-IN")+" pts";
    document.getElementById("chipLoBuf").textContent  = Math.round(spot-lo).toLocaleString("en-IN")+" pts";
    document.getElementById("chipHiBuf").textContent  = Math.round(hi-spot).toLocaleString("en-IN")+" pts";
  }} else if (hasLo) {{
    const loPct = ((lo-visMin)/visR*100).toFixed(1);
    document.getElementById("beRbFillBoth").style.display  = "none";
    document.getElementById("beRbFillLower").style.left    = "0%";
    document.getElementById("beRbFillLower").style.width   = loPct+"%";
    document.getElementById("beRbFillUpper").style.width   = "0%";
    document.getElementById("beRbMarkerLo").style.display  = "block";
    document.getElementById("beRbMarkerHi").style.display  = "none";
    document.getElementById("beRbMarkerLo").style.left     = loPct+"%";
    document.getElementById("beRbLoLbl").textContent  = "▼ BE ₹"+lo.toLocaleString("en-IN");
    document.getElementById("beRbHiLbl").textContent  = "";
    document.getElementById("beWidthLabel").textContent = "Lower BE only";
  }} else {{
    const hiPct = ((hi-visMin)/visR*100).toFixed(1);
    document.getElementById("beRbFillBoth").style.display  = "none";
    document.getElementById("beRbFillLower").style.width   = "0%";
    document.getElementById("beRbFillUpper").style.left    = hiPct+"%";
    document.getElementById("beRbFillUpper").style.width   = (100-parseFloat(hiPct))+"%";
    document.getElementById("beRbMarkerLo").style.display  = "none";
    document.getElementById("beRbMarkerHi").style.display  = "block";
    document.getElementById("beRbMarkerHi").style.left     = hiPct+"%";
    document.getElementById("beRbLoLbl").textContent  = "";
    document.getElementById("beRbHiLbl").textContent  = "▲ BE ₹"+hi.toLocaleString("en-IN");
    document.getElementById("beWidthLabel").textContent = "Upper BE only";
  }}
  // Re-render chain to highlight BE strikes
  renderChain();
}}

// ── State persistence ─────────────────────────────────────────
function saveUserState() {{
  const state = {{
    bias:        marketBias,
    mode:        currentMode,
    expiry:      currentExpiry,
    supports:    getSupports(),
    resistances: getResistances(),
    beLower:     document.getElementById("beLower").value,
    beUpper:     document.getElementById("beUpper").value,
    lotSize:     document.getElementById("lotSize").value,
    maxCap:      document.getElementById("maxCap").value,
  }};
  try {{ sessionStorage.setItem("noa_state", JSON.stringify(state)); }} catch(e) {{}}
}}

function restoreUserState() {{
  try {{
    const raw = sessionStorage.getItem("noa_state");
    if (!raw) return;
    const state = JSON.parse(raw);
    if (state.bias) setBias(state.bias);
    if (state.mode) setMode(state.mode);
    if (state.expiry && EXPIRY_LIST.includes(state.expiry)) {{
      currentExpiry = state.expiry;
      document.getElementById("expirySel").value  = state.expiry;
      document.getElementById("expirySel2").value = state.expiry;
    }}
    if (state.supports && state.supports.length) {{
      const supC=document.getElementById("supContainer"); supC.innerHTML="";
      state.supports.forEach(v=>{{
        const div=document.createElement("div"); div.className="sr-row";
        div.innerHTML=`<input type="number" class="inp sup-inp" value="${{v}}" placeholder="e.g. 21800"/><button class="rm-btn" onclick="rmRow(this)">✕</button>`;
        supC.appendChild(div);
      }});
    }}
    if (state.resistances && state.resistances.length) {{
      const resC=document.getElementById("resContainer"); resC.innerHTML="";
      state.resistances.forEach(v=>{{
        const div=document.createElement("div"); div.className="sr-row";
        div.innerHTML=`<input type="number" class="inp res-inp" value="${{v}}" placeholder="e.g. 22600"/><button class="rm-btn" onclick="rmRow(this)">✕</button>`;
        resC.appendChild(div);
      }});
    }}
    if (state.beLower) document.getElementById("beLower").value = state.beLower;
    if (state.beUpper) document.getElementById("beUpper").value = state.beUpper;
    if (state.lotSize) document.getElementById("lotSize").value = state.lotSize;
    if (state.maxCap)  document.getElementById("maxCap").value  = state.maxCap;
    if (state.mode === "be") updateBEVisual();
  }} catch(e) {{ console.warn("State restore error:",e); }}
}}

// ── S/R helpers ───────────────────────────────────────────────
function addLevel(type) {{
  const c=document.getElementById(type==="sup"?"supContainer":"resContainer");
  const cls=type==="sup"?"sup-inp":"res-inp";
  const ph=type==="sup"?"e.g. 21800":"e.g. 22600";
  const div=document.createElement("div"); div.className="sr-row";
  div.innerHTML=`<input type="number" class="inp ${{cls}}" placeholder="${{ph}}" oninput="saveUserState()"/><button class="rm-btn" onclick="rmRow(this)">✕</button>`;
  c.appendChild(div);
}}
function rmRow(btn) {{
  const row=btn.parentElement;
  if(row.parentElement.children.length>1) row.remove();
  else row.querySelector("input").value="";
  saveUserState();
}}
function getSupports()    {{ return [...document.querySelectorAll(".sup-inp")].map(i=>parseFloat(i.value)).filter(v=>!isNaN(v)&&v>0); }}
function getResistances() {{ return [...document.querySelectorAll(".res-inp")].map(i=>parseFloat(i.value)).filter(v=>!isNaN(v)&&v>0); }}

// ── BSM JS ────────────────────────────────────────────────────
function normCdf(x) {{
  const a1=.254829592,a2=-.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=.3275911;
  const sign=x<0?-1:1,t=1/(1+p*Math.abs(x));
  return 0.5*(1+sign*(1-(((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*Math.exp(-x*x/2)));
}}
function normPdf(x) {{ return Math.exp(-0.5*x*x)/Math.sqrt(2*Math.PI); }}
function bsm(S,K,T,r,sigma,type) {{
  if(T<=0||sigma<=0){{ const intr=type==="CE"?Math.max(0,S-K):Math.max(0,K-S); return {{price:intr,delta:0,pop:0}}; }}
  const d1=(Math.log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*Math.sqrt(T));
  const d2=d1-sigma*Math.sqrt(T);
  const price=type==="CE"?S*normCdf(d1)-K*Math.exp(-r*T)*normCdf(d2):K*Math.exp(-r*T)*normCdf(-d2)-S*normCdf(-d1);
  const delta=type==="CE"?normCdf(d1):normCdf(d1)-1;
  const pop  =type==="CE"?normCdf(d2):normCdf(-d2);
  return {{price:Math.max(0,price),delta,pop}};
}}

// ── Payoff calc ───────────────────────────────────────────────
function payoffsFor(legs, underlying) {{
  const range=[];
  for(let p=underlying-1500;p<=underlying+1500;p+=25) range.push(p);
  return {{
    range,
    vals: range.map(price=>{{
      let pnl=0;
      legs.forEach(l=>{{
        const intr=(l.opt_type||l.type)==="CE"?Math.max(0,price-l.strike):Math.max(0,l.strike-price);
        pnl+=l.action==="buy"?(intr-l.premium):(l.premium-intr);
      }});
      return Math.round(pnl*LOT_SIZE*100)/100;
    }})
  }};
}}

// ── PoP calculation ───────────────────────────────────────────
function calcPoP(legs, underlying, T, bes, sType) {{
  const allSell=legs.every(l=>l.action==="sell");
  const allBuy =legs.every(l=>l.action==="buy");
  const sigma  =(legs.reduce((a,l)=>a+(l.iv||15),0)/legs.length)/100;

  if (bes.length >= 1) {{
    if (sType==="debit_spread") {{
      const be=bes[0]; const isCall=legs.find(l=>l.action==="buy")?.type==="CE";
      return isCall
        ? Math.round((1-bsm(underlying,be,T,RISK_FREE,sigma,"CE").pop)*1000)/10
        : Math.round(bsm(underlying,be,T,RISK_FREE,sigma,"PE").pop*1000)/10;
    }}
    if (allBuy&&(sType==="straddle"||sType==="strangle")) {{
      const lo=bes[0],hi=bes.length>=2?bes[1]:null;
      const pb=bsm(underlying,lo,T,RISK_FREE,sigma,"PE").pop;
      const pa=hi?(1-bsm(underlying,hi,T,RISK_FREE,sigma,"CE").pop):0;
      return Math.round(Math.min((pb+pa)*100,99)*10)/10;
    }}
    if (allSell||sType==="credit_spread"||sType==="iron_condor"||sType==="iron_butterfly") {{
      const lo=bes[0],hi=bes.length>=2?bes[1]:null;
      const pb=bsm(underlying,lo,T,RISK_FREE,sigma,"PE").pop;
      const pa=hi?(1-bsm(underlying,hi,T,RISK_FREE,sigma,"CE").pop):0;
      return Math.round(Math.min((1-pb-Math.max(0,pa))*100,99)*10)/10;
    }}
  }}
  let s=0;
  legs.forEach(l=>{{ const b=bsm(underlying,l.strike,T,RISK_FREE,l.iv/100,l.opt_type||l.type); s+=l.action==="sell"?b.pop:(1-b.pop); }});
  return Math.round(Math.min(s/legs.length*100,99)*10)/10;
}}

// ═══════════════════════════════════════════════════════
// SR MODE ANALYZE — identical logic to original script
// ═══════════════════════════════════════════════════════
function analyze() {{
  const d=ALL_DATA[currentExpiry];
  if(!d){{ alert("No data loaded for this expiry."); return; }}
  const supports=getSupports(), ress=getResistances();
  if(!supports.length||!ress.length){{ alert("Please enter at least one support and one resistance level."); return; }}

  renderChain();

  const underlying=d.underlying, atm=d.atm_strike, dte=d.dte, T=Math.max(dte/365,0.001);
  const strikes=d.all_strikes; const smap={{}};
  strikes.forEach(s=>{{ smap[s.strike]=s; }});
  const allSt=strikes.map(s=>s.strike).sort((a,b)=>a-b);
  const nearest=val=>allSt.reduce((a,b)=>Math.abs(b-val)<Math.abs(a-val)?b:a);
  const get=(st,field,def=0)=>(smap[st]||{{}})[field]||def;

  // ── Proximity-based support: closest level below spot (aggressive), or lowest (conservative)
  const supBelow=supports.filter(v=>v<underlying);
  const bestSup=supBelow.length>0?supBelow.reduce((a,b)=>Math.abs(b-underlying)<Math.abs(a-underlying)?b:a):Math.min(...supports);
  // ── Proximity-based resistance: closest level above spot, or highest as fallback
  const resAbove=ress.filter(v=>v>underlying);
  const bestRes=resAbove.length>0?resAbove.reduce((a,b)=>Math.abs(b-underlying)<Math.abs(a-underlying)?b:a):Math.max(...ress);
  // ── ATR-based wing width (~1.5× daily ATR, min 100, snapped to 50)
  const atmIV=(get(atm,"ce_iv",15)+get(atm,"pe_iv",15))/2;
  const dailyATR=underlying*(atmIV/100)*Math.sqrt(1/365);
  const atrWing=Math.max(Math.round(dailyATR*1.5/50)*50,100);

  const srRange=bestRes-bestSup;
  const s_st=nearest(bestSup),r_st=nearest(bestRes);
  const far_c=nearest(bestRes+atrWing),far_p=nearest(bestSup-atrWing);
  const otm_c=nearest(underlying+srRange*0.3),otm_p=nearest(underlying-srRange*0.3);

  const raw=[];
  const bias=marketBias;

  function makeStrat(name,legs,biasTag,sType) {{
    const netPrem=legs.reduce((a,l)=>a+(l.action==="sell"?l.premium:-l.premium),0);
    const po=payoffsFor(legs,underlying);
    const maxP=Math.max(...po.vals), maxL=Math.min(...po.vals);
    if(maxP<=0) return null;
    const bes=[];
    for(let i=0;i<po.vals.length-1;i++){{
      if((po.vals[i]<0)!==(po.vals[i+1]<0)){{
        const be=po.range[i]+(po.range[i+1]-po.range[i])*Math.abs(po.vals[i])/(Math.abs(po.vals[i])+Math.abs(po.vals[i+1]));
        bes.push(Math.round(be));
      }}
    }}
    const rr=maxL!==0?Math.round(Math.abs(maxP/maxL)*100)/100:0;
    const pop=Math.max(calcPoP(legs,underlying,T,bes,sType),1);
    const rrNorm=Math.min(rr,5.0)/5.0;
    const rrAdj=rrNorm*(pop/100);
    const efficiency=Math.min(Math.abs(maxP)/Math.max(Math.abs(maxL),1),5.0)/5.0;
    const score=Math.round(((pop/100)*0.40*100+rrAdj*0.35*100+efficiency*0.25*100)*10)/10;
    return {{name,biasTag,sType,legs,netPrem:Math.round(netPrem*100)/100,isDebit:netPrem<0,
             maxProfit:Math.round(maxP*100)/100,maxLoss:Math.round(Math.abs(maxL)*100)/100,
             breakevens:bes,rr,pop,score,margin:Math.round(Math.abs(maxL)*100)/100,
             payoffs:po.vals,priceRange:po.range,isBEMode:false}};
  }}

  if(bias!=="bearish"){{ const s=makeStrat("Bull Call Spread",[{{action:"buy",strike:atm,type:"CE",opt_type:"CE",premium:get(atm,"ce_ltp"),iv:get(atm,"ce_iv",15)}},{{action:"sell",strike:r_st,type:"CE",opt_type:"CE",premium:get(r_st,"ce_ltp"),iv:get(r_st,"ce_iv",15)}}],"bullish","debit_spread"); if(s) raw.push(s); }}
  if(bias!=="bullish"){{ const s=makeStrat("Bear Put Spread",[{{action:"buy",strike:atm,type:"PE",opt_type:"PE",premium:get(atm,"pe_ltp"),iv:get(atm,"pe_iv",15)}},{{action:"sell",strike:s_st,type:"PE",opt_type:"PE",premium:get(s_st,"pe_ltp"),iv:get(s_st,"pe_iv",15)}}],"bearish","debit_spread"); if(s) raw.push(s); }}
  if(bias!=="bearish"){{ const s=makeStrat("Bull Put Spread",[{{action:"sell",strike:s_st,type:"PE",opt_type:"PE",premium:get(s_st,"pe_ltp"),iv:get(s_st,"pe_iv",15)}},{{action:"buy",strike:far_p,type:"PE",opt_type:"PE",premium:get(far_p,"pe_ltp"),iv:get(far_p,"pe_iv",15)}}],"bullish","credit_spread"); if(s) raw.push(s); }}
  if(bias!=="bullish"){{ const s=makeStrat("Bear Call Spread",[{{action:"sell",strike:r_st,type:"CE",opt_type:"CE",premium:get(r_st,"ce_ltp"),iv:get(r_st,"ce_iv",15)}},{{action:"buy",strike:far_c,type:"CE",opt_type:"CE",premium:get(far_c,"ce_ltp"),iv:get(far_c,"ce_iv",15)}}],"bearish","credit_spread"); if(s) raw.push(s); }}
  {{ const s=makeStrat("Iron Condor",[{{action:"sell",strike:r_st,type:"CE",opt_type:"CE",premium:get(r_st,"ce_ltp"),iv:get(r_st,"ce_iv",15)}},{{action:"buy",strike:far_c,type:"CE",opt_type:"CE",premium:get(far_c,"ce_ltp"),iv:get(far_c,"ce_iv",15)}},{{action:"sell",strike:s_st,type:"PE",opt_type:"PE",premium:get(s_st,"pe_ltp"),iv:get(s_st,"pe_iv",15)}},{{action:"buy",strike:far_p,type:"PE",opt_type:"PE",premium:get(far_p,"pe_ltp"),iv:get(far_p,"pe_iv",15)}}],"neutral","iron_condor"); if(s) raw.push(s); }}
  {{ const wing=Math.max(Math.round(srRange*0.4/50)*50,100),bc=nearest(atm+wing),bp=nearest(atm-wing);
     const s=makeStrat("Iron Butterfly",[{{action:"sell",strike:atm,type:"CE",opt_type:"CE",premium:get(atm,"ce_ltp"),iv:get(atm,"ce_iv",15)}},{{action:"sell",strike:atm,type:"PE",opt_type:"PE",premium:get(atm,"pe_ltp"),iv:get(atm,"pe_iv",15)}},{{action:"buy",strike:bc,type:"CE",opt_type:"CE",premium:get(bc,"ce_ltp"),iv:get(bc,"ce_iv",15)}},{{action:"buy",strike:bp,type:"PE",opt_type:"PE",premium:get(bp,"pe_ltp"),iv:get(bp,"pe_iv",15)}}],"neutral","iron_butterfly"); if(s) raw.push(s); }}
  {{ const s=makeStrat("Long Straddle",[{{action:"buy",strike:atm,type:"CE",opt_type:"CE",premium:get(atm,"ce_ltp"),iv:get(atm,"ce_iv",15)}},{{action:"buy",strike:atm,type:"PE",opt_type:"PE",premium:get(atm,"pe_ltp"),iv:get(atm,"pe_iv",15)}}],"volatile","straddle"); if(s) raw.push(s); }}
  if(bias==="neutral"){{ const s=makeStrat("Short Straddle",[{{action:"sell",strike:atm,type:"CE",opt_type:"CE",premium:get(atm,"ce_ltp"),iv:get(atm,"ce_iv",15)}},{{action:"sell",strike:atm,type:"PE",opt_type:"PE",premium:get(atm,"pe_ltp"),iv:get(atm,"pe_iv",15)}}],"neutral","straddle"); if(s) raw.push(s); }}
  {{ const s=makeStrat("Long Strangle",[{{action:"buy",strike:otm_c,type:"CE",opt_type:"CE",premium:get(otm_c,"ce_ltp"),iv:get(otm_c,"ce_iv",15)}},{{action:"buy",strike:otm_p,type:"PE",opt_type:"PE",premium:get(otm_p,"pe_ltp"),iv:get(otm_p,"pe_iv",15)}}],"volatile","strangle"); if(s) raw.push(s); }}
  if(bias==="neutral"){{ const s=makeStrat("Short Strangle",[{{action:"sell",strike:otm_c,type:"CE",opt_type:"CE",premium:get(otm_c,"ce_ltp"),iv:get(otm_c,"ce_iv",15)}},{{action:"sell",strike:otm_p,type:"PE",opt_type:"PE",premium:get(otm_p,"pe_ltp"),iv:get(otm_p,"pe_iv",15)}}],"neutral","strangle"); if(s) raw.push(s); }}

  strategies=raw.filter(Boolean);
  renderStrategies();
  populatePayoffSel();
}}

// ═══════════════════════════════════════════════════════
// BE MODE ANALYZE — flexible: lo only / hi only / both
// ═══════════════════════════════════════════════════════
function analyzeBE() {{
  const d = ALL_DATA[currentExpiry];
  if (!d) {{ alert("No data loaded."); return; }}

  const lo  = parseFloat(document.getElementById("beLower").value) || null;
  const hi  = parseFloat(document.getElementById("beUpper").value) || null;
  const hasLo = lo !== null && lo > 0;
  const hasHi = hi !== null && hi > 0;

  if (!hasLo && !hasHi) {{ alert("Enter at least one Breakeven level (Lower or Upper)."); return; }}
  if (hasLo && hasHi && lo >= hi) {{ alert("BE Lower must be less than BE Upper."); return; }}

  const underlying = d.underlying;
  const dte        = d.dte;
  const T          = Math.max(dte / 365, 0.001);
  const strikes    = d.all_strikes;
  const smap       = {{}};
  strikes.forEach(s => {{ smap[s.strike] = s; }});
  const allSt = strikes.map(s => s.strike).sort((a,b) => a-b);
  const nearest = val => allSt.reduce((a,b) => Math.abs(b-val)<Math.abs(a-val)?b:a);
  const get = (st, field, def=0) => (smap[st]||{{}})[field] || def;

  // ── Strike step: inferred from actual option chain data (e.g. 50 for Nifty) ──
  const strikeStep = (allSt.length > 1)
    ? allSt.slice(1).reduce((min,k,i) => Math.min(min, k - allSt[i]), Infinity)
    : 50;

  // ── Mid-price approximation: NSE API provides only LTP (no bid/ask).
  //    We apply a half-tick slippage buffer to simulate realistic fill prices.
  //    This prevents BE calculations from using stale "ghost" print prices.
  const halfTick = 0.05;  // minimum NSE option tick = ₹0.05
  function midPrice(ltp) {{
    if (!ltp || ltp <= 0) return 0;
    return Math.max(ltp - halfTick, halfTick);  // conservative: assume slight adverse fill
  }}

  // ── wingDist: derived from IV-based daily ATR so it scales with volatility.
  //    Formula: 1.5 × daily_ATR, snapped to strike step, minimum = 2 × strikeStep.
  //    For two-sided mode: also constrained to be ≤ 30% of the user's BE range.
  const atmIV_be   = (get(d.atm_strike,"ce_iv",15) + get(d.atm_strike,"pe_iv",15)) / 2;
  const dailyATR_be = underlying * (atmIV_be / 100) * Math.sqrt(1 / 365);
  const atrWing_be  = Math.max(
    Math.round(dailyATR_be * 1.5 / strikeStep) * strikeStep,
    2 * strikeStep
  );
  const wingDist = (hasLo && hasHi)
    ? Math.min(atrWing_be, Math.max(Math.round((hi - lo) * 0.30 / strikeStep) * strikeStep, 2 * strikeStep))
    : atrWing_be;

  // ── Liquidity guard: a strike is liquid if CE or PE OI > 0 ──
  function isLiquid(strike, optType) {{
    const oiField = optType === "CE" ? "ce_oi" : "pe_oi";
    return get(strike, oiField, 0) > 0;
  }}

  // ── Yield-to-Risk filter threshold (max_profit / max_loss) ──
  const MIN_YIELD_TO_RISK = 0.20;  // flag if < 1:5

  // ── SOLVER: find sell strike whose net-credit-implied BE best matches target ──
  // Resolves the circular dependency: wing width is FIXED first (atrWing_be),
  // then we iterate sell strikes and compute impliedBE from both legs' mid-prices.

  function findSellPEStrikeForBE(targetBE) {{
    // Bull Put Spread BE = sell_strike − (mid(pe_sell) − mid(pe_buy))
    let best = null, bestDiff = Infinity;
    allSt.forEach(k => {{
      if (k >= underlying) return;   // sell strike must be below spot for a put spread
      const peSell = midPrice(get(k, "pe_ltp", 0));
      if (peSell <= 0) return;
      const buyK   = nearest(k - wingDist);
      if (buyK >= k) return;
      const peBuy  = midPrice(get(buyK, "pe_ltp", 0));
      const netCredit = peSell - peBuy;
      if (netCredit <= 0) return;
      const impliedBE = k - netCredit;
      const diff = Math.abs(impliedBE - targetBE);
      if (diff < bestDiff) {{ bestDiff = diff; best = k; }}
    }});
    return best || nearest(targetBE);
  }}

  function findSellCEStrikeForBE(targetBE) {{
    // Bear Call Spread BE = sell_strike + (mid(ce_sell) − mid(ce_buy))
    let best = null, bestDiff = Infinity;
    allSt.forEach(k => {{
      if (k <= underlying) return;   // sell strike must be above spot for a call spread
      const ceSell = midPrice(get(k, "ce_ltp", 0));
      if (ceSell <= 0) return;
      const buyK   = nearest(k + wingDist);
      if (buyK <= k) return;
      const ceBuy  = midPrice(get(buyK, "ce_ltp", 0));
      const netCredit = ceSell - ceBuy;
      if (netCredit <= 0) return;
      const impliedBE = k + netCredit;
      const diff = Math.abs(impliedBE - targetBE);
      if (diff < bestDiff) {{ bestDiff = diff; best = k; }}
    }});
    return best || nearest(targetBE);
  }}

  // Derive sell strikes (circular dependency resolved: wing is fixed before iteration)
  const lo_st   = hasLo ? findSellPEStrikeForBE(lo) : null;
  const hi_st   = hasHi ? findSellCEStrikeForBE(hi) : null;
  const lo_wing = hasLo ? nearest(lo_st - wingDist) : null;
  const hi_wing = hasHi ? nearest(hi_st + wingDist) : null;

  // ── "Tight wing": half of atrWing, minimum 1 strike step ──
  const tightWing = Math.max(Math.round(atrWing_be * 0.5 / strikeStep) * strikeStep, strikeStep);
  const atm      = d.atm_strike;

  const raw = [];
  const bias = marketBias;

  // ── Helper — same makeStrat but marks isBEMode=true ──
  function makeStratBE(name, legs, biasTag, sType, beInsight) {{
    const netPrem = legs.reduce((a,l)=>a+(l.action==="sell"?l.premium:-l.premium),0);
    const po      = payoffsFor(legs, underlying);
    const maxP    = Math.max(...po.vals), maxL = Math.min(...po.vals);
    if (maxP <= 0) return null;
    const bes = [];
    for (let i=0;i<po.vals.length-1;i++) {{
      if ((po.vals[i]<0)!==(po.vals[i+1]<0)) {{
        const be = po.range[i]+(po.range[i+1]-po.range[i])*Math.abs(po.vals[i])/(Math.abs(po.vals[i])+Math.abs(po.vals[i+1]));
        bes.push(Math.round(be));
      }}
    }}
    const rr    = maxL!==0?Math.round(Math.abs(maxP/maxL)*100)/100:0;
    const pop   = Math.max(calcPoP(legs, underlying, T, bes, sType), 1);
    const rrNorm2=Math.min(rr,5.0)/5.0;
    const rrAdj2=rrNorm2*(pop/100);
    const eff2=Math.min(Math.abs(maxP)/Math.max(Math.abs(maxL),1),5.0)/5.0;
    const score = Math.round(((pop/100)*0.40*100+rrAdj2*0.35*100+eff2*0.25*100)*10)/10;
    // Fit: compare strategy's computed BEs against user's EXACT input values
    const fit   = calcBEFit(bes, hasLo ? lo : null, hasHi ? hi : null);
    // BE accuracy: show user how far actual BEs are from their targets
    const beAccuracy = [];
    if (hasLo && bes.length > 0) {{
      const diff = bes[0] - lo;
      beAccuracy.push({{ side:"Lower", target:lo, actual:bes[0], diff:Math.round(diff), close: Math.abs(diff) <= 50 }});
    }}
    if (hasHi && bes.length > 1) {{
      const diff = bes[bes.length-1] - hi;
      beAccuracy.push({{ side:"Upper", target:hi, actual:bes[bes.length-1], diff:Math.round(diff), close: Math.abs(diff) <= 50 }});
    }} else if (hasHi && bes.length > 0 && !hasLo) {{
      const diff = bes[0] - hi;
      beAccuracy.push({{ side:"Upper", target:hi, actual:bes[0], diff:Math.round(diff), close: Math.abs(diff) <= 50 }});
    }}
    // ── Yield-to-Risk filter ──
    const yieldToRisk = maxL !== 0 ? Math.abs(maxP) / Math.abs(maxL) : 0;
    const poorValue   = yieldToRisk > 0 && yieldToRisk < MIN_YIELD_TO_RISK;
    // ── Liquidity check: flag if any sell leg has zero OI ──
    const lowLiquidity = legs.some(l => l.action === "sell" && !isLiquid(l.strike, l.opt_type));
    return {{
      name, biasTag, sType, legs,
      netPrem: Math.round(netPrem*100)/100, isDebit: netPrem < 0,
      maxProfit: Math.round(maxP*100)/100, maxLoss: Math.round(Math.abs(maxL)*100)/100,
      breakevens: bes, rr, pop, score, fit, beAccuracy,
      margin: Math.round(Math.abs(maxL)*100)/100,
      payoffs: po.vals, priceRange: po.range,
      isBEMode: true, beInsight,
      poorValue, lowLiquidity, yieldToRisk: Math.round(yieldToRisk*100)/100,
    }};
  }}

  function calcBEFit(stratBes, targetLo, targetHi) {{
    // Compare strategy's actual computed BEs against user's EXACT input values
    const range = (targetHi && targetLo) ? (targetHi - targetLo) : 500;
    let err = 0, n = 0;
    if (targetLo && stratBes.length > 0) {{ err += Math.abs(stratBes[0] - targetLo) / range; n++; }}
    if (targetHi && stratBes.length > 1) {{ err += Math.abs(stratBes[stratBes.length-1] - targetHi) / range; n++; }}
    else if (targetHi && stratBes.length > 0) {{ err += Math.abs(stratBes[0] - targetHi) / range; n++; }}
    return n ? Math.round(Math.max(10, Math.min(99, 100 - err * 60))) : 55;
  }}

  // Use exact user inputs (lo, hi) — NOT lo_st/hi_st — for fit calculation

  // ══ BOTH BE LOWER + BE UPPER ══════════════════════════════════
  if (hasLo && hasHi) {{
    // ── Iron Condor: iterative Seed → Expand → Adjust → Finalize ──────────────
    // Step 1 (Seed): start with sell strikes closest OTM to user's ceiling/floor
    // Step 2 (Expand): compute credit using atrWing_be-based wings
    // Step 3 (Adjust): if actual BE is too wide → move sell inward; too narrow → outward
    // Step 4 (Finalize): apply 25% dynamic wing rule on final sell strikes, compute fit
    (function() {{
      const MAX_ITER = allSt.length;
      const ceAbove  = allSt.filter(k => k > underlying).sort((a,b) => a-b);
      const peBelow  = allSt.filter(k => k < underlying).sort((a,b) => b-a);
      if (!ceAbove.length || !peBelow.length) return;

      // Step 1: seed — closest OTM strikes to user's BE inputs
      let sellCE = nearest(hi), sellPE = nearest(lo);
      let buyCE  = nearest(sellCE + wingDist), buyPE = nearest(sellPE - wingDist);

      for (let iter = 0; iter < MAX_ITER; iter++) {{
        const ceSellP = midPrice(get(sellCE, "ce_ltp", 0));
        const ceBuyP  = midPrice(get(buyCE,  "ce_ltp", 0));
        const peSellP = midPrice(get(sellPE, "pe_ltp", 0));
        const peBuyP  = midPrice(get(buyPE,  "pe_ltp", 0));
        if (ceSellP <= 0 || peSellP <= 0) break;
        const ncCE = ceSellP - ceBuyP, ncPE = peSellP - peBuyP;
        if (ncCE <= 0 || ncPE <= 0) break;
        const actualHiBE = sellCE + ncCE;
        const actualLoBE = sellPE - ncPE;
        // Step 3: adjust — check if actual BEs match user targets
        const hiErr = actualHiBE - hi, loErr = actualLoBE - lo;
        const converged = Math.abs(hiErr) <= strikeStep && Math.abs(loErr) <= strikeStep;
        if (converged) break;
        // Move sell strikes to correct BE direction
        const ceIdx = ceAbove.indexOf(sellCE);
        const peIdx = peBelow.indexOf(sellPE);
        if (hiErr > strikeStep && ceIdx > 0) {{
          sellCE = ceAbove[ceIdx - 1];  // move inward (lower CE sell → lower BE)
        }} else if (hiErr < -strikeStep && ceIdx < ceAbove.length - 1) {{
          sellCE = ceAbove[ceIdx + 1];  // move outward
        }}
        if (loErr < -strikeStep && peIdx > 0) {{
          sellPE = peBelow[peIdx - 1];  // move inward
        }} else if (loErr > strikeStep && peIdx < peBelow.length - 1) {{
          sellPE = peBelow[peIdx + 1];  // move outward
        }}
        // Step 4: recalculate wings at 25% of new sell-to-sell range
        const finalWing = Math.max(
          Math.round((sellCE - sellPE) * 0.25 / strikeStep) * strikeStep,
          2 * strikeStep
        );
        buyCE = nearest(sellCE + finalWing);
        buyPE = nearest(sellPE - finalWing);
      }}

      if (new Set([sellCE, buyCE, sellPE, buyPE]).size === 4) {{
        const s = makeStratBE("Iron Condor", [
          {{action:"sell",strike:sellCE, type:"CE",opt_type:"CE",premium:get(sellCE,"ce_ltp"), iv:get(sellCE,"ce_iv",15), why:`SELL CE ₹${{sellCE.toLocaleString("en-IN")}} — iteratively adjusted to match your upper BE ₹${{hi.toLocaleString("en-IN")}}`}},
          {{action:"buy", strike:buyCE,  type:"CE",opt_type:"CE",premium:get(buyCE,"ce_ltp"),  iv:get(buyCE,"ce_iv",15),  why:`BUY CE wing ₹${{buyCE.toLocaleString("en-IN")}} — caps upside loss (25% dynamic wing)`}},
          {{action:"sell",strike:sellPE, type:"PE",opt_type:"PE",premium:get(sellPE,"pe_ltp"), iv:get(sellPE,"pe_iv",15), why:`SELL PE ₹${{sellPE.toLocaleString("en-IN")}} — iteratively adjusted to match your lower BE ₹${{lo.toLocaleString("en-IN")}}`}},
          {{action:"buy", strike:buyPE,  type:"PE",opt_type:"PE",premium:get(buyPE,"pe_ltp"),  iv:get(buyPE,"pe_iv",15),  why:`BUY PE wing ₹${{buyPE.toLocaleString("en-IN")}} — caps downside loss (25% dynamic wing)`}},
        ], "neutral", "iron_condor",
        `Sell strikes refined via iterative solver to match your BEs. Wings set at 25% of final spread width.`);
        if (s) raw.push(s);
      }}
    }})();

    // Short Strangle — sell at both BEs, no wings
    if (lo_st && hi_st && lo_st !== hi_st) {{
      const s = makeStratBE("Short Strangle", [
        {{action:"sell",strike:hi_st,type:"CE",opt_type:"CE",premium:get(hi_st,"ce_ltp"),iv:get(hi_st,"ce_iv",15),why:`SELL CE at Upper BE ₹${{hi_st.toLocaleString("en-IN")}} — max premium collected`}},
        {{action:"sell",strike:lo_st,type:"PE",opt_type:"PE",premium:get(lo_st,"pe_ltp"),iv:get(lo_st,"pe_iv",15),why:`SELL PE at Lower BE ₹${{lo_st.toLocaleString("en-IN")}} — max premium collected`}},
      ], "neutral", "credit_spread",
      `Sell both BEs outright. Highest credit but unlimited risk if Nifty breaks out of your range.`);
      if (s) raw.push(s);
    }}

    // Iron Butterfly — only if range is tight (< 600 pts)
    if ((hi-lo) <= 600) {{
      const wing = Math.max(Math.round((hi-lo)*0.5/50)*50, 100);
      const buyC = nearest(atm + wing), buyP = nearest(atm - wing);
      if (new Set([atm,buyC,buyP]).size===3) {{
        const s = makeStratBE("Iron Butterfly", [
          {{action:"sell",strike:atm, type:"CE",opt_type:"CE",premium:get(atm,"ce_ltp"), iv:get(atm,"ce_iv",15),  why:`SELL ATM CE ₹${{atm.toLocaleString("en-IN")}} — maximum ATM premium`}},
          {{action:"sell",strike:atm, type:"PE",opt_type:"PE",premium:get(atm,"pe_ltp"), iv:get(atm,"pe_iv",15),  why:`SELL ATM PE ₹${{atm.toLocaleString("en-IN")}} — centered at spot`}},
          {{action:"buy", strike:buyC,type:"CE",opt_type:"CE",premium:get(buyC,"ce_ltp"),iv:get(buyC,"ce_iv",15), why:`BUY upper wing CE ₹${{buyC.toLocaleString("en-IN")}} — caps upside loss`}},
          {{action:"buy", strike:buyP,type:"PE",opt_type:"PE",premium:get(buyP,"pe_ltp"),iv:get(buyP,"pe_iv",15), why:`BUY lower wing PE ₹${{buyP.toLocaleString("en-IN")}} — caps downside loss`}},
        ], "neutral", "iron_butterfly",
        `Tight range play — sell ATM, buy wings near your BEs. Best if Nifty pins near ₹${{atm.toLocaleString("en-IN")}} at expiry.`);
        if (s) raw.push(s);
      }}
    }}

    // Bull Put Spread (lower side only)
    if (lo_st && lo_wing && lo_st !== lo_wing && bias !== "bearish") {{
      const s = makeStratBE("Bull Put Spread", [
        {{action:"sell",strike:lo_st,  type:"PE",opt_type:"PE",premium:get(lo_st,"pe_ltp"),  iv:get(lo_st,"pe_iv",15), why:`SELL PE at Lower BE ₹${{lo_st.toLocaleString("en-IN")}} — profit if Nifty holds above your lower BE`}},
        {{action:"buy", strike:lo_wing,type:"PE",opt_type:"PE",premium:get(lo_wing,"pe_ltp"),iv:get(lo_wing,"pe_iv",15),why:`BUY PE hedge below ₹${{lo_wing.toLocaleString("en-IN")}} — defines max loss if support breaks`}},
      ], "bullish", "credit_spread",
      `Only plays the lower BE. Lower margin than Iron Condor. Use if you're more confident about downside support holding.`);
      if (s) raw.push(s);
    }}

    // Bear Call Spread (upper side only)
    if (hi_st && hi_wing && hi_st !== hi_wing && bias !== "bullish") {{
      const s = makeStratBE("Bear Call Spread", [
        {{action:"sell",strike:hi_st,  type:"CE",opt_type:"CE",premium:get(hi_st,"ce_ltp"),  iv:get(hi_st,"ce_iv",15), why:`SELL CE at Upper BE ₹${{hi_st.toLocaleString("en-IN")}} — profit if Nifty stays below your upper BE`}},
        {{action:"buy", strike:hi_wing,type:"CE",opt_type:"CE",premium:get(hi_wing,"ce_ltp"),iv:get(hi_wing,"ce_iv",15),why:`BUY CE hedge above ₹${{hi_wing.toLocaleString("en-IN")}} — defines max loss if resistance breaks`}},
      ], "bearish", "credit_spread",
      `Only plays the upper BE. Lower margin than Iron Condor. Use if you're more confident about upside resistance holding.`);
      if (s) raw.push(s);
    }}
  }}

  // ══ ONLY LOWER BE ═════════════════════════════════════════════
  if (hasLo && !hasHi) {{
    // Bull Put Spread — primary play
    if (lo_st && lo_wing && lo_st !== lo_wing) {{
      const s = makeStratBE("Bull Put Spread", [
        {{action:"sell",strike:lo_st,  type:"PE",opt_type:"PE",premium:get(lo_st,"pe_ltp"),  iv:get(lo_st,"pe_iv",15), why:`SELL PE at your BE ₹${{lo_st.toLocaleString("en-IN")}} — you believe Nifty stays ABOVE this level at expiry`}},
        {{action:"buy", strike:lo_wing,type:"PE",opt_type:"PE",premium:get(lo_wing,"pe_ltp"),iv:get(lo_wing,"pe_iv",15),why:`BUY PE protection below ₹${{lo_wing.toLocaleString("en-IN")}} — limits your loss if Nifty breaks the floor`}},
      ], "bullish", "credit_spread",
      `Your lower BE ₹${{lo_st.toLocaleString("en-IN")}} is the SELL strike. Keep full credit if Nifty closes above it at expiry.`);
      if (s) raw.push(s);
    }}

    // Bull Call Spread — buy ATM call, sell call at BE-based OTM level
    if (atm !== lo_st) {{
      const sellC = nearest(lo + (underlying - lo) * 0.5); // midpoint target
      if (atm !== sellC) {{
        const s = makeStratBE("Bull Call Spread", [
          {{action:"buy", strike:atm,   type:"CE",opt_type:"CE",premium:get(atm,"ce_ltp"),  iv:get(atm,"ce_iv",15),  why:`BUY ATM CE ₹${{atm.toLocaleString("en-IN")}} — long position, profits as Nifty rallies`}},
          {{action:"sell",strike:sellC, type:"CE",opt_type:"CE",premium:get(sellC,"ce_ltp"),iv:get(sellC,"ce_iv",15), why:`SELL CE at ₹${{sellC.toLocaleString("en-IN")}} — reduces cost, caps max profit`}},
        ], "bullish", "debit_spread",
        `Directional play using your lower BE as conviction that Nifty is supported here. Buy ATM, sell a higher CE.`);
        if (s) raw.push(s);
      }}
    }}

    // Short Put at lower BE (if bias is neutral/bullish and premium is attractive)
    if (lo_st && bias !== "bearish") {{
      const netPremVal = get(lo_st, "pe_ltp");
      if (netPremVal > 0) {{
        const wing2 = nearest(lo_st - tightWing);
        if (wing2 !== lo_st) {{
          const s = makeStratBE("Bull Put Spread (Tight)", [
            {{action:"sell",strike:lo_st,type:"PE",opt_type:"PE",premium:get(lo_st,"pe_ltp"),iv:get(lo_st,"pe_iv",15),why:`SELL PE exactly at your BE floor ₹${{lo_st.toLocaleString("en-IN")}}`}},
            {{action:"buy", strike:wing2,type:"PE",opt_type:"PE",premium:get(wing2,"pe_ltp"),iv:get(wing2,"pe_iv",15),why:`BUY tight wing PE ₹${{wing2.toLocaleString("en-IN")}} — narrow spread, lower margin required`}},
          ], "bullish", "credit_spread",
          `Tight spread right at your floor BE. Lower max profit but also lower capital at risk.`);
          if (s) raw.push(s);
        }}
      }}
    }}
  }}

  // ══ ONLY UPPER BE ═════════════════════════════════════════════
  if (!hasLo && hasHi) {{
    // Bear Call Spread — primary play
    if (hi_st && hi_wing && hi_st !== hi_wing) {{
      const s = makeStratBE("Bear Call Spread", [
        {{action:"sell",strike:hi_st,  type:"CE",opt_type:"CE",premium:get(hi_st,"ce_ltp"),  iv:get(hi_st,"ce_iv",15), why:`SELL CE at your BE ₹${{hi_st.toLocaleString("en-IN")}} — you believe Nifty stays BELOW this level at expiry`}},
        {{action:"buy", strike:hi_wing,type:"CE",opt_type:"CE",premium:get(hi_wing,"ce_ltp"),iv:get(hi_wing,"ce_iv",15),why:`BUY CE protection above ₹${{hi_wing.toLocaleString("en-IN")}} — limits your loss if Nifty breaks the ceiling`}},
      ], "bearish", "credit_spread",
      `Your upper BE ₹${{hi_st.toLocaleString("en-IN")}} is the SELL strike. Keep full credit if Nifty closes below it at expiry.`);
      if (s) raw.push(s);
    }}

    // Bear Put Spread — reverse-engineer buy strike so natural BE ≈ user's exact hi input
    // FIX: Old code used ATM blindly → BE was 1000+ pts from user input.
    // FIX: wingDist was hardcoded 200 → too small when BE is far above spot.
    // NEW: iterate all strikes as buy leg, pair with sell leg wingDist below,
    //      compute impliedBE = buyStrike - netDebit, pick closest to user's hi.
    (function() {{
      let bestBuy = null, bestSell = null, bestDiff = Infinity;
      allSt.forEach(buyK => {{
        const peBuy = get(buyK, "pe_ltp", 0);
        if (peBuy <= 0) return;
        const sellK = nearest(buyK - wingDist);
        if (sellK >= buyK) return;
        const peSell = get(sellK, "pe_ltp", 0);
        if (peSell <= 0) return;
        const netDebit  = peBuy - peSell;
        if (netDebit <= 0) return;
        const impliedBE = buyK - netDebit;  // BE = BuyStrike - NetDebit
        const diff      = Math.abs(impliedBE - hi);
        if (diff < bestDiff) {{ bestDiff = diff; bestBuy = buyK; bestSell = sellK; }}
      }});
      if (bestBuy && bestSell && bestBuy !== bestSell) {{
        const netD     = get(bestBuy,"pe_ltp",0) - get(bestSell,"pe_ltp",0);
        const actualBE = Math.round(bestBuy - netD);
        const diffPts  = actualBE - hi;
        const s = makeStratBE("Bear Put Spread", [
          {{action:"buy", strike:bestBuy,  type:"PE",opt_type:"PE",
            premium:get(bestBuy,"pe_ltp"),  iv:get(bestBuy,"pe_iv",15),
            why:`BUY PE ₹${{bestBuy.toLocaleString("en-IN")}} — chosen so BE ≈ your target ₹${{hi.toLocaleString("en-IN")}} (actual BE ₹${{actualBE.toLocaleString("en-IN")}})`}},
          {{action:"sell",strike:bestSell, type:"PE",opt_type:"PE",
            premium:get(bestSell,"pe_ltp"), iv:get(bestSell,"pe_iv",15),
            why:`SELL PE ₹${{bestSell.toLocaleString("en-IN")}} — reduces net debit, caps max profit below this strike`}},
        ], "bearish", "debit_spread",
        `Strikes chosen to match your Upper BE ₹${{hi.toLocaleString("en-IN")}}. Actual BE ≈ ₹${{actualBE.toLocaleString("en-IN")}} (diff: ${{diffPts>=0?"+":""}}${{diffPts}} pts).`);
        if (s) raw.push(s);
      }}
    }})();

    // Bear Call Spread (Tight)
    if (hi_st && bias !== "bullish") {{
      const wing2 = nearest(hi_st + tightWing);
      if (wing2 !== hi_st) {{
        const s = makeStratBE("Bear Call Spread (Tight)", [
          {{action:"sell",strike:hi_st,type:"CE",opt_type:"CE",premium:get(hi_st,"ce_ltp"),iv:get(hi_st,"ce_iv",15),why:`SELL CE exactly at your BE ceiling ₹${{hi_st.toLocaleString("en-IN")}}`}},
          {{action:"buy", strike:wing2,type:"CE",opt_type:"CE",premium:get(wing2,"ce_ltp"),iv:get(wing2,"ce_iv",15),why:`BUY tight wing CE ₹${{wing2.toLocaleString("en-IN")}} — narrow spread, lower margin required`}},
        ], "bearish", "credit_spread",
        `Tight spread right at your ceiling BE. Lower max profit but also lower capital at risk.`);
        if (s) raw.push(s);
      }}
    }}
  }}

  // Sort: first by fit (BE mode), then by score
  raw.sort((a,b) => ((b.fit||0)-(a.fit||0)) || (b.score-a.score));
  strategies = raw.filter(Boolean);
  renderStrategies();
  populatePayoffSel();
}}



// ── Render Strategy Cards — accordion design ─────────────────
function toggleCard(uid) {{
  const card = document.getElementById("sc-card-"+uid);
  if (!card) return;
  const isOpen = card.classList.contains("sc-open");
  document.querySelectorAll(".strat-card.sc-open").forEach(c=>c.classList.remove("sc-open"));
  if (!isOpen) {{
    card.classList.add("sc-open");
    const nm = card.dataset.stratname;
    if (nm) selectPayoff(nm);
  }}
}}

function renderStrategies() {{
  const sortBy = document.getElementById("sortSel").value;
  const sorted = [...strategies].sort((a,b)=>{{
    if(sortBy==="pop")    return b.pop-a.pop;
    if(sortBy==="profit") return b.maxProfit-a.maxProfit;
    if(sortBy==="rr")     return b.rr-a.rr;
    return b.score-a.score;
  }});

  document.getElementById("stratCount").textContent = sorted.length+" STRATEGIES";
  if(!sorted.length){{
    document.getElementById("stratGrid").innerHTML='<div class="empty"><div class="empty-icon">📋</div><p>No strategies found for current settings.</p></div>';
    return;
  }}

  const colors={{bullish:"var(--green)",bearish:"var(--red)",neutral:"var(--cyan)",volatile:"var(--purple)"}};
  const emojis={{bullish:"🐂",bearish:"🐻",neutral:"⚖️",volatile:"⚡"}};
  const pillCls={{bullish:"sc-pill-bull",bearish:"sc-pill-bear",neutral:"sc-pill-neut",volatile:"sc-pill-volt"}};
  const maxScore=Math.max(...sorted.map(s=>s.score));

  document.getElementById("stratGrid").innerHTML = sorted.map((s,i)=>{{
    const cc     = colors[s.biasTag]||"var(--cyan)";
    const popCol = s.pop>=60?"var(--green)":s.pop>=45?"var(--gold)":"var(--red)";
    const popBg  = s.pop>=60?"#00c89620":s.pop>=45?"#ffd16620":"#ff6b6b20";
    const rrDisp = s.rr===0?"∞":s.rr.toFixed(2)+"x";
    const sw     = Math.round((s.score/maxScore)*100);
    const beStr  = s.breakevens.length ? s.breakevens.map(b=>"₹"+b.toLocaleString("en-IN")).join(" / ") : "—";
    const netDisp= s.isDebit?`<span class="down">-₹${{Math.abs(s.netPrem).toFixed(2)}}</span>`:`<span class="up">+₹${{s.netPrem.toFixed(2)}}</span>`;
    const uid    = (s.name+i).replace(/[^a-zA-Z0-9]/g,"_");

    const legChips = s.legs.map(l=>
      `<span class="sc-leg-chip ${{l.action}}">${{l.action==="sell"?"SELL":"BUY"}} ${{l.opt_type||l.type}} ${{l.strike.toLocaleString("en-IN")}} @${{(+l.premium).toFixed(1)}}</span>`
    ).join("");

    const poorValueBadge = (s.isBEMode && s.poorValue)
      ? `<div style="padding:6px 14px;background:rgba(255,107,107,.15);border-bottom:1px solid rgba(255,107,107,.40);font-size:12px;font-weight:800;color:#ff8080;font-family:'DM Mono',monospace;letter-spacing:.8px;">⚠️ POOR VALUE — Yield-to-Risk ${{s.yieldToRisk.toFixed(2)}}x (below 1:5 threshold). Consider a wider spread.</div>`
      : "";
    const lowLiquidityBadge = (s.isBEMode && s.lowLiquidity)
      ? `<div style="padding:6px 14px;background:rgba(255,209,102,.14);border-bottom:1px solid rgba(255,209,102,.40);font-size:12px;font-weight:800;color:#ffd166;font-family:'DM Mono',monospace;letter-spacing:.8px;">⚠️ LOW LIQUIDITY — One or more sell legs have zero OI. Verify fills before trading.</div>`
      : "";

    const beAccuracyPanel = (s.isBEMode && s.beAccuracy && s.beAccuracy.length) ? `
      <div style="padding:8px 14px;background:rgba(6,10,18,.6);border-bottom:1px solid var(--border);">
        <div style="font-size:13px;font-weight:800;text-transform:uppercase;letter-spacing:1.2px;color:#d8eeff;font-family:'DM Mono',monospace;margin-bottom:6px;">🎯 YOUR INPUT vs ACTUAL BREAKEVENS</div>
        <div style="display:grid;grid-template-columns:${{s.beAccuracy.length>1?"1fr 1fr":"1fr"}};gap:8px;">
          ${{s.beAccuracy.map(a=>{{
            const sign=a.diff>=0?"+":"";
            const col=Math.abs(a.diff)<=30?"var(--green)":Math.abs(a.diff)<=80?"var(--gold)":"var(--red)";
            const icon=Math.abs(a.diff)<=30?"✅":Math.abs(a.diff)<=80?"⚠️":"❌";
            return `<div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:7px;padding:7px 10px;">
              <div style="font-size:13px;color:#b8d4e8;font-family:'DM Mono',monospace;margin-bottom:4px;">${{icon}} ${{a.side.toUpperCase()}} BE</div>
              <div style="display:flex;align-items:center;justify-content:space-between;gap:6px;">
                <div style="text-align:center;"><div style="font-size:13px;font-weight:700;color:#c8dff0;font-family:'DM Mono',monospace;">YOUR INPUT</div><div style="font-size:15px;font-weight:800;font-family:'DM Mono',monospace;color:var(--gold);">₹${{a.target.toLocaleString("en-IN")}}</div></div>
                <div style="color:#d8eeff;font-size:16px;font-weight:700;">→</div>
                <div style="text-align:center;"><div style="font-size:13px;font-weight:700;color:#c8dff0;font-family:'DM Mono',monospace;">ACTUAL BE</div><div style="font-size:15px;font-weight:800;font-family:'DM Mono',monospace;color:${{col}};">₹${{a.actual.toLocaleString("en-IN")}}</div></div>
                <div style="text-align:center;"><div style="font-size:13px;font-weight:700;color:#c8dff0;font-family:'DM Mono',monospace;">DIFF</div><div style="font-size:14px;font-weight:700;font-family:'DM Mono',monospace;color:${{col}};">${{sign}}${{a.diff}}</div></div>
              </div>
            </div>`;
          }}).join("")}}
        </div>
        <div style="margin-top:6px;font-size:13px;color:#b8d4e8;font-family:'DM Mono',monospace;line-height:1.6;">
          ℹ️ Diff = (Sell Strike ± actual premium collected) vs your input. Small diff = sell strikes were well-chosen.
        </div>
      </div>` : "";

    const fitBar = s.isBEMode ? `
      <div class="fit-bar-wrap">
        <div class="fit-bar-hdr"><span>Breakeven Fit Score</span><span style="color:${{s.fit>=80?"var(--green)":s.fit>=60?"var(--gold)":"var(--red)"}};">${{s.fit}}%</span></div>
        <div class="fit-bar-track"><div class="fit-bar-fill" style="width:${{s.fit}}%;background:${{s.fit>=80?"var(--green)":s.fit>=60?"var(--gold)":"var(--red)"}};box-shadow:0 0 6px ${{s.fit>=80?"var(--green)":s.fit>=60?"var(--gold)":"var(--red)"}}55;"></div></div>
      </div>` : "";

    const legsSection = s.isBEMode
      ? `<div class="sc-legs-detail">
          <div class="sc-legs-detail-title">📋 LEGS TO TRADE — HOW & WHY</div>
          ${{s.legs.map(l=>`
          <div class="leg-detail-row ${{l.action.toLowerCase()}}">
            <span class="leg-detail-badge ${{l.action.toLowerCase()}}">${{l.action.toUpperCase()}}</span>
            <div class="leg-detail-body">
              <div class="leg-detail-main">
                <span class="leg-${{(l.opt_type||l.type).toLowerCase()}}">${{l.opt_type||l.type}}</span>
                <span class="leg-stk">₹${{l.strike.toLocaleString("en-IN")}}</span>
                <span class="leg-prem">@ ₹${{l.premium}}</span>
              </div>
              <div class="leg-why">↳ ${{l.why}}</div>
            </div>
          </div>`).join("")}}
          ${{s.beInsight?`<div style="margin-top:8px;padding:7px 10px;background:rgba(255,209,102,.06);border:1px solid rgba(255,209,102,.15);border-radius:7px;font-size:12px;font-family:'DM Mono',monospace;color:rgba(255,209,102,.8);line-height:1.6;">💡 ${{s.beInsight}}</div>`:""}}
        </div>`
      : `<div class="sc-legs">
          ${{s.legs.map(l=>`<span class="leg-tag leg-${{l.action}}">${{l.action.toUpperCase()}} ${{l.strike}} ${{l.opt_type||l.type}} @${{l.premium.toFixed(2)}}</span>`).join("")}}
        </div>`;

    const headerRow = `
      <div class="sc-header" onclick="toggleCard('${{uid}}')">
        <span class="sc-chevron">▶</span>
        <span class="sc-pill-bias ${{pillCls[s.biasTag]||"sc-pill-neut"}}">${{(s.biasTag||"NEUTRAL").toUpperCase()}}</span>
        <span class="sc-name">${{emojis[s.biasTag]||"📊"}} ${{s.name}}</span>
        <div class="sc-header-right">
          ${{s.isBEMode?`<span class="sc-fit-badge">FIT ${{s.fit||"—"}}%</span>`:""}}
          <div class="sc-legs-mini">${{legChips}}</div>
          <div class="pop-pill" style="background:${{popBg}};color:${{popCol}};border:1px solid ${{popCol}}33;">${{s.pop}}%<br><span style="font-size:11px;font-weight:700;">PoP</span></div>
        </div>
      </div>`;

    const bodyContent = `
      <div class="sc-body">
        ${{poorValueBadge}}${{lowLiquidityBadge}}
        <div class="sc-top">
          <div>
            <div class="sc-name" style="font-size:16px;">${{emojis[s.biasTag]||"📊"}} ${{s.name}}</div>
            <div class="sc-sub">${{s.biasTag.toUpperCase()}} · ${{s.isDebit?"DEBIT":"CREDIT"}} SPREAD · DTE:${{ALL_DATA[currentExpiry]?.dte||"—"}}</div>
          </div>
          <div class="pop-pill" style="background:${{popBg}};color:${{popCol}};border:1px solid ${{popCol}}33;">${{s.pop}}%<br><span style="font-size:12px;font-weight:700;">PoP</span></div>
        </div>
        ${{fitBar}}
        ${{beAccuracyPanel}}
        <div class="sc-fields">
          <div class="sc-field"><span class="sc-field-lbl">Strike Price</span><span class="sc-field-val" style="color:var(--cyan);">ATM ₹${{ALL_DATA[currentExpiry]?.atm_strike?.toLocaleString("en-IN")||"—"}}</span></div>
          <div class="sc-field"><span class="sc-field-lbl">Max Profit</span><span class="sc-field-val up">₹${{s.maxProfit.toLocaleString("en-IN")}}</span></div>
          <div class="sc-field"><span class="sc-field-lbl">Max Loss</span><span class="sc-field-val down">₹${{s.maxLoss.toLocaleString("en-IN")}}</span></div>
          <div class="sc-field"><span class="sc-field-lbl">Max RR Ratio</span><span class="sc-field-val" style="color:var(--gold);">1:${{rrDisp}}</span></div>
          <div class="sc-field"><span class="sc-field-lbl">Breakevens</span><span class="sc-field-val" style="font-size:14px;font-weight:700;color:#d8eeff;">${{beStr}}</span></div>
          <div class="sc-field"><span class="sc-field-lbl">Net Credit/Debit</span><span class="sc-field-val">${{netDisp}}</span></div>
          <div class="sc-field" style="grid-column:1/-1"><span class="sc-field-lbl">Est. Margin / Premium</span><span class="sc-field-val" style="color:var(--purple);">₹${{s.margin.toLocaleString("en-IN")}}</span></div>
        </div>
        ${{legsSection}}
        <div class="sc-score">
          <span class="score-lbl">SCORE</span>
          <div class="score-bar-track"><div class="score-bar-fill" style="width:${{sw}}%"></div></div>
          <span class="score-num">${{s.score}}</span>
        </div>
        <div style="padding:8px 12px;border-top:1px solid rgba(255,255,255,.05);" onclick="event.stopPropagation()">
          <button data-simuid="${{uid}}" onclick="toggleSim('${{uid}}',this)"
            style="width:100%;background:rgba(245,197,24,.07);border:1px solid rgba(245,197,24,.2);border-radius:8px;
                   padding:7px 12px;cursor:pointer;display:flex;align-items:center;justify-content:space-between;
                   font-family:'DM Mono',monospace;font-size:12px;font-weight:700;color:rgba(255,209,102,.8);
                   letter-spacing:.8px;text-transform:uppercase;transition:all .2s;">
            <span style="display:flex;align-items:center;gap:7px;"><span style="font-size:15px;">📊</span> Intraday P&L Simulator</span>
            <span id="sim-arrow-${{uid}}" style="font-size:14px;transition:transform .25s;">▼</span>
          </button>
        </div>
        <div id="sim-wrap-${{uid}}" style="display:none;overflow:hidden;">${{buildIntradaySim(s, uid)}}</div>
      </div>`;

    return `<div class="strat-card" id="sc-card-${{uid}}" style="--cc:${{cc}};animation-delay:${{i*0.05}}s" data-stratname="${{s.name}}">
      ${{headerRow}}
      ${{bodyContent}}
    </div>`;
  }}).join("");

  // Cards start collapsed — user clicks to open
}}

// ── Intraday Simulator ────────────────────────────────────────
function buildIntradaySim(s, uid) {{
  const d=ALL_DATA[currentExpiry]; if(!d) return "";
  const underlying=d.underlying, dte=d.dte;
  let netDelta=0,netTheta=0,netVega=0;
  s.legs.forEach(l=>{{
    const st=d.all_strikes.find(x=>x.strike===l.strike); if(!st) return;
    const isCE=(l.opt_type||l.type)==="CE"; const mult=l.action==="buy"?1:-1;
    netDelta+=mult*(isCE?(st.ce_delta||0):Math.abs(st.pe_delta||0));
    netTheta+=mult*(isCE?(st.ce_theta||0):(st.pe_theta||0));
    netVega +=mult*(isCE?(st.ce_vega||0) :(st.pe_vega||0));
  }});
  const thetaDay=Math.round(netTheta*LOT_SIZE), deltaPerPt=netDelta*LOT_SIZE, vegaPerIV=netVega*LOT_SIZE;
  const maxP=s.maxProfit, maxL=s.maxLoss;
  const moves=[{{label:"+300 pts",pts:300,cls:"bull"}},{{label:"+200 pts",pts:200,cls:"bull"}},{{label:"+150 pts",pts:150,cls:"bull"}},{{label:"+100 pts",pts:100,cls:"bull"}},{{label:"+50 pts",pts:50,cls:"bull"}},{{label:"Flat",pts:0,cls:"flat"}},{{label:"−50 pts",pts:-50,cls:"bear"}},{{label:"−100 pts",pts:-100,cls:"bear"}},{{label:"−150 pts",pts:-150,cls:"bear"}},{{label:"−200 pts",pts:-200,cls:"bear"}},{{label:"−300 pts",pts:-300,cls:"bear"}}];
  const scenarioRows=moves.map(m=>{{
    const dPnl=Math.round(m.pts*deltaPerPt);
    const total=Math.max(-maxL,Math.min(maxP*0.92,dPnl+thetaDay));
    const pct=maxP>0?((total/maxP)*100).toFixed(0):"—";
    const spot=Math.round(underlying+m.pts);
    const pCol=total>0?"#00c896":total<0?"#ff6b6b":"#6480ff";
    const mBg=m.cls==="bull"?"rgba(0,200,150,.12)":m.cls==="bear"?"rgba(255,107,107,.12)":"rgba(245,197,24,.1)";
    const mTxt=m.cls==="bull"?"#00c896":m.cls==="bear"?"#ff6b6b":"#ffd166";
    const isFlat=m.pts===0?'class="sim-flat"':"";
    const legCols=s.legs.map(l=>{{
      const st=d.all_strikes.find(x=>x.strike===l.strike);
      const isCE=(l.opt_type||l.type)==="CE";
      const ltp=isCE?(st?.ce_ltp||l.premium):(st?.pe_ltp||l.premium);
      const dlt=isCE?(st?.ce_delta||0):Math.abs(st?.pe_delta||0);
      const tht=isCE?(st?.ce_theta||0):(st?.pe_theta||0);
      const est=Math.max(0,ltp+dlt*m.pts+tht).toFixed(0);
      const col=l.action==="buy"?"#00c8e0":"#ff9090";
      return `<td style="color:${{col}};font-size:12px;">₹${{est}}</td>`;
    }}).join("");
    return `<tr ${{isFlat}}><td><span class="sim-move-lbl" style="background:${{mBg}};color:${{mTxt}};">${{m.label}}</span></td><td style="color:#d8eeff;font-size:14px;font-weight:600;">₹${{spot.toLocaleString("en-IN")}}</td>${{legCols}}<td><span class="sim-pnl-val" style="color:${{pCol}};">${{total>=0?"+":""}}₹${{Math.abs(Math.round(total)).toLocaleString("en-IN")}}</span><span style="font-size:12px;color:#c8dff0;font-weight:600;margin-left:3px;">${{total>=0?"+":""}}${{pct}}%</span></td></tr>`;
  }}).join("");
  const legHeaders=s.legs.map(l=>`<th style="color:${{l.action==="buy"?"rgba(0,200,220,.7)":"rgba(255,144,144,.7)"}};font-size:7.5px;">${{l.action.toUpperCase()}} ${{l.strike}}</th>`).join("");
  const absMax=Math.max(Math.abs(thetaDay),Math.abs(Math.round(deltaPerPt*100)),Math.abs(Math.round(vegaPerIV)),1);
  const dBar=Math.round(Math.abs(thetaDay)/absMax*100);
  const thetaStr=thetaDay>=0?`+₹${{thetaDay}}`:`−₹${{Math.abs(thetaDay)}}`;
  const vegaStr=vegaPerIV>=0?`~+₹${{Math.round(Math.abs(vegaPerIV))}}`:`~−₹${{Math.round(Math.abs(vegaPerIV))}}`;
  const sliderMin=Math.round(underlying-400),sliderMax=Math.round(underlying+400);
  return `<div class="intraday-sim" onclick="event.stopPropagation()">
    <div class="sim-tabs">
      <button class="sim-tab active" onclick="simTab('${{uid}}','sc',this)">📊 Scenarios</button>
      <button class="sim-tab"        onclick="simTab('${{uid}}','gr',this)">🔬 Greeks</button>
      <button class="sim-tab"        onclick="simTab('${{uid}}','sl',this)">🎚 Slider</button>
    </div>
    <div id="sim-sc-${{uid}}">
      <div class="sim-hdr"><div style="display:flex;align-items:center;gap:7px;"><div class="sim-icon">📅</div><div><div class="sim-title">TODAY'S P&L SCENARIOS</div><div class="sim-subtitle">Exit before market close — Delta + Theta estimate</div></div></div><div style="font-family:'DM Mono',monospace;font-size:14px;font-weight:700;color:#d8eeff;">DTE: ${{dte}}</div></div>
      <div style="overflow-x:auto;padding:0 10px 10px;"><table class="sim-tbl"><thead><tr><th>Nifty Move</th><th>Spot</th>${{legHeaders}}<th style="color:#d8eeff;">Today P&L</th></tr></thead><tbody>${{scenarioRows}}</tbody></table></div>
      <div class="sim-note"><span style="flex-shrink:0;">⏱</span><span>Formula: <strong style="color:#ffd166;">Delta × move + Theta/day</strong>. Actual P&L may vary with IV. Max profit of ₹${{maxP.toLocaleString("en-IN")}} is only achievable <strong>at expiry</strong>.</span></div>
    </div>
    <div id="sim-gr-${{uid}}" style="display:none;">
      <div class="sim-hdr"><div style="display:flex;align-items:center;gap:7px;"><div class="sim-icon">🔬</div><div><div class="sim-title">GREEKS BREAKDOWN</div><div class="sim-subtitle">P&L contribution from each Greek</div></div></div></div>
      <div class="sim-live-pnl">
        <div class="slpb" style="background:rgba(0,200,150,.07);border:1px solid rgba(0,200,150,.18);"><div class="slpb-lbl">Δ Delta P&L</div><div class="slpb-num" style="color:#00c896;" id="sim-dp-${{uid}}">+₹0</div><div class="slpb-sub">flat = ₹0</div></div>
        <div class="slpb" style="background:rgba(255,107,107,.07);border:1px solid rgba(255,107,107,.18);"><div class="slpb-lbl">Θ Theta Cost</div><div class="slpb-num" style="color:#ff9090;">${{thetaStr}}/day</div><div class="slpb-sub">time decay</div></div>
        <div class="slpb" style="background:rgba(138,160,255,.07);border:1px solid rgba(138,160,255,.18);"><div class="slpb-lbl">ν Vega ±1%</div><div class="slpb-num" style="color:#8aa0ff;">${{vegaStr}}</div><div class="slpb-sub">per 1% IV</div></div>
        <div class="slpb" style="background:rgba(245,197,24,.07);border:1px solid rgba(245,197,24,.18);"><div class="slpb-lbl">Net (Flat)</div><div class="slpb-num" style="color:#ffd166;">${{thetaStr}}</div><div class="slpb-sub">theta drag</div></div>
      </div>
      <div style="padding:10px 10px 0;"><div style="font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#b8d4e8;margin-bottom:8px;">Greek contribution bars</div>
        <div class="cbar-row"><div class="cbar-lbl" style="color:#00c896;">Δ Delta</div><div class="cbar-track"><div class="cbar-fill" style="width:${{dBar}}%;background:#00c896;" id="sim-db-${{uid}}"></div></div><div class="cbar-val" style="color:#00c896;" id="sim-dv-${{uid}}">₹0 (flat)</div></div>
        <div class="cbar-row"><div class="cbar-lbl" style="color:#ff9090;">Θ Theta</div><div class="cbar-track"><div class="cbar-fill" style="width:100%;background:#ff6b6b;"></div></div><div class="cbar-val" style="color:#ff9090;">${{thetaStr}}/day</div></div>
        <div class="cbar-row"><div class="cbar-lbl" style="color:#8aa0ff;">ν Vega</div><div class="cbar-track"><div class="cbar-fill" style="width:${{Math.round(Math.abs(Math.round(vegaPerIV))/absMax*100)}}%;background:#8aa0ff;"></div></div><div class="cbar-val" style="color:#8aa0ff;">${{vegaStr}}</div></div>
      </div>
      <div style="margin:10px 10px 12px;padding:9px 11px;background:rgba(0,0,0,.18);border-radius:8px;font-size:12px;color:#c8dff0;line-height:1.8;"><strong style="color:rgba(255,255,255,.55);">Net Delta:</strong> ${{(netDelta>=0?"+":"")+netDelta.toFixed(3)}} per point → <strong style="color:#00c896;">₹${{deltaPerPt.toFixed(1)}} per Nifty point</strong><br><strong style="color:rgba(255,255,255,.55);">Net Theta:</strong> ${{thetaStr}} per trading day</div>
    </div>
    <div id="sim-sl-${{uid}}" style="display:none;">
      <div class="sim-hdr"><div style="display:flex;align-items:center;gap:7px;"><div class="sim-icon">🎚</div><div><div class="sim-title">LIVE SCENARIO SLIDER</div><div class="sim-subtitle">Drag to see today's estimated exit P&L</div></div></div></div>
      <div class="sim-slide-wrap">
        <div class="sim-slide-labels"><span class="sim-slide-edge">₹${{sliderMin.toLocaleString("en-IN")}}</span><span class="sim-slide-cur" id="sim-sc-lbl-${{uid}}">Spot: ₹${{underlying.toLocaleString("en-IN")}}</span><span class="sim-slide-edge">₹${{sliderMax.toLocaleString("en-IN")}}</span></div>
        <input class="sim-range" type="range" id="sim-range-${{uid}}" min="${{sliderMin}}" max="${{sliderMax}}" value="${{underlying}}" step="25" style="--pct:50%" oninput="simSlide('${{uid}}',${{underlying}},${{deltaPerPt}},${{thetaDay}},${{maxP}},${{maxL}},this.value)">
      </div>
      <div style="padding:0 10px 14px;"><div style="background:rgba(0,0,0,.25);border-radius:10px;padding:14px;text-align:center;border:1px solid rgba(255,255,255,.06);">
        <div style="font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#b8d4e8;margin-bottom:6px;">ESTIMATED EXIT P&L TODAY</div>
        <div style="font-family:'DM Mono',monospace;font-size:30px;font-weight:700;" id="sim-bigpnl-${{uid}}">${{thetaStr}}</div>
        <div style="font-size:12px;color:#b8d4e8;margin-top:4px;" id="sim-note-${{uid}}">Theta drag only (flat market)</div>
        <div style="display:flex;gap:12px;justify-content:center;margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,.05);">
          <div style="text-align:center;"><div style="font-size:12px;font-weight:700;color:#d8eeff;letter-spacing:1px;text-transform:uppercase;margin-bottom:2px;">Delta P&L</div><div style="font-family:'DM Mono',monospace;font-size:15px;font-weight:700;color:#00c896;" id="sim-slide-d-${{uid}}">₹0</div></div>
          <div style="text-align:center;"><div style="font-size:7.5px;color:#b8d4e8;letter-spacing:1px;text-transform:uppercase;margin-bottom:2px;">Theta Cost</div><div style="font-family:'DM Mono',monospace;font-size:15px;font-weight:700;color:#ff9090;">${{thetaStr}}</div></div>
          <div style="text-align:center;"><div style="font-size:12px;font-weight:700;color:#d8eeff;letter-spacing:1px;text-transform:uppercase;margin-bottom:2px;">% of Max</div><div style="font-family:'DM Mono',monospace;font-size:15px;font-weight:700;color:#ffd166;" id="sim-slide-pct-${{uid}}">—</div></div>
        </div>
      </div></div>
    </div>
  </div>`;
}}

function toggleSim(uid,btn) {{
  const wrap=document.getElementById("sim-wrap-"+uid);
  const arrow=document.getElementById("sim-arrow-"+uid);
  if(!wrap) return;
  const isOpen=wrap.style.display!=="none";
  document.querySelectorAll('[data-simuid]').forEach(ob=>{{
    const oid=ob.getAttribute("data-simuid"); if(oid===uid) return;
    const ow=document.getElementById("sim-wrap-"+oid),oa=document.getElementById("sim-arrow-"+oid);
    if(ow) ow.style.display="none"; if(oa) oa.style.transform="rotate(0deg)";
    ob.style.background="rgba(245,197,24,.07)"; ob.style.borderColor="rgba(245,197,24,.2)";
  }});
  wrap.style.display=isOpen?"none":"block";
  if(arrow) arrow.style.transform=isOpen?"rotate(0deg)":"rotate(180deg)";
  btn.style.background=isOpen?"rgba(245,197,24,.07)":"rgba(245,197,24,.13)";
  btn.style.borderColor=isOpen?"rgba(245,197,24,.2)":"rgba(245,197,24,.4)";
}}

function simTab(uid,tab,btn) {{
  ["sc","gr","sl"].forEach(t=>{{ const el=document.getElementById("sim-"+t+"-"+uid); if(el) el.style.display=t===tab?"block":"none"; }});
  btn.closest(".intraday-sim").querySelectorAll(".sim-tab").forEach(b=>b.classList.remove("active"));
  btn.classList.add("active");
}}

function simSlide(uid,spot,dPerPt,thetaDay,maxP,maxL,val) {{
  const cur=parseInt(val),move=cur-spot,dPnl=Math.round(move*dPerPt),total=Math.max(-maxL,Math.min(maxP*0.92,dPnl+thetaDay));
  const pct=maxP>0?((total/maxP)*100).toFixed(1):"—";
  const range=document.getElementById("sim-range-"+uid);
  const pctFill=((cur-parseInt(range.min))/(parseInt(range.max)-parseInt(range.min))*100).toFixed(1);
  range.style.setProperty("--pct",pctFill+"%");
  const bigEl=document.getElementById("sim-bigpnl-"+uid);
  const noteEl=document.getElementById("sim-note-"+uid);
  const dEl=document.getElementById("sim-slide-d-"+uid);
  const pctEl=document.getElementById("sim-slide-pct-"+uid);
  const lblEl=document.getElementById("sim-sc-lbl-"+uid);
  const col=total>100?"#00c896":total>0?"#4de8b8":total>-200?"#ffd166":"#ff6b6b";
  if(bigEl){{ bigEl.style.color=col; bigEl.textContent=(total>=0?"+":"")+"₹"+Math.abs(Math.round(total)).toLocaleString("en-IN"); }}
  if(noteEl) noteEl.textContent=move>0?`Nifty up ${{move}} pts`:move<0?`Nifty down ${{Math.abs(move)}} pts`:"Flat — theta drag only";
  if(dEl){{ dEl.style.color=dPnl>=0?"#00c896":"#ff6b6b"; dEl.textContent=(dPnl>=0?"+":"")+"₹"+Math.abs(dPnl).toLocaleString("en-IN"); }}
  if(pctEl){{ pctEl.style.color=total>=0?"#00c896":"#ff9090"; pctEl.textContent=(total>=0?"+":"")+pct+"%"; }}
  if(lblEl) lblEl.textContent="Spot: ₹"+cur.toLocaleString("en-IN");
  const dbEl=document.getElementById("sim-db-"+uid),dvEl=document.getElementById("sim-dv-"+uid),dpEl=document.getElementById("sim-dp-"+uid);
  const absMax=Math.max(Math.abs(thetaDay),Math.abs(dPnl),1);
  if(dbEl) dbEl.style.width=Math.round(Math.abs(dPnl)/absMax*100)+"%";
  if(dvEl){{ dvEl.style.color=dPnl>=0?"#00c896":"#ff6b6b"; dvEl.textContent=(dPnl>=0?"+₹":"-₹")+Math.abs(dPnl).toLocaleString("en-IN"); }}
  if(dpEl){{ dpEl.style.color=dPnl>=0?"#00c896":"#ff9090"; dpEl.textContent=(dPnl>=0?"+₹":"−₹")+Math.abs(dPnl).toLocaleString("en-IN"); }}
}}

function populatePayoffSel() {{
  document.getElementById("payoffSel").innerHTML='<option value="">— Select Strategy —</option>'+strategies.map(s=>`<option value="${{s.name}}">${{s.name}}</option>`).join("");
}}

function selectPayoff(name) {{
  document.getElementById("payoffSel").value=name;
  drawPayoff();
  document.getElementById("payoffSel").scrollIntoView({{behavior:"smooth"}});
}}

// ── Payoff Chart (unchanged from original) ────────────────────
function bsmPrice(S,K,T,r,sigma,type) {{
  if(T<=0||sigma<=0) return type==="CE"?Math.max(0,S-K):Math.max(0,K-S);
  const d1=(Math.log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*Math.sqrt(T));
  const d2=d1-sigma*Math.sqrt(T);
  if(type==="CE") return Math.max(0,S*normCdf(d1)-K*Math.exp(-r*T)*normCdf(d2));
  return Math.max(0,K*Math.exp(-r*T)*normCdf(-d2)-S*normCdf(-d1));
}}

function stratPnlAtSpot(legs,spotPrice,T) {{
  let pnl=0;
  legs.forEach(l=>{{
    const sigma=(l.iv||15)/100, optType=l.opt_type||l.type||"CE";
    const theoVal=bsmPrice(spotPrice,l.strike,T,RISK_FREE,sigma,optType);
    const legPnl=l.action==="buy"?(theoVal-l.premium):(l.premium-theoVal);
    pnl+=legPnl;
  }});
  return Math.round(pnl*LOT_SIZE*100)/100;
}}

function drawPayoff() {{
  const name=document.getElementById("payoffSel").value;
  const s=strategies.find(x=>x.name===name); if(!s) return;
  const d=ALL_DATA[currentExpiry], underlying=d?.underlying||0, dte=d?.dte||7;
  const T_today=Math.max(dte/365,0.5/365);
  const priceRange=[];
  for(let p=underlying-1500;p<=underlying+1500;p+=25) priceRange.push(p);
  const todayPnl=priceRange.map(p=>stratPnlAtSpot(s.legs,p,T_today));
  const expiryPnl=priceRange.map(p=>stratPnlAtSpot(s.legs,p,0));
  const allStrikes=d?.all_strikes||[];
  const ceOiData=priceRange.map(p=>{{ const row=allStrikes.find(r=>r.strike===p); return row?Math.round(row.ce_oi/1e3):null; }});
  const peOiData=priceRange.map(p=>{{ const row=allStrikes.find(r=>r.strike===p); return row?Math.round(row.pe_oi/1e3):null; }});
  const netCost=Math.abs(s.netPrem)*LOT_SIZE||1;
  const projPnl=stratPnlAtSpot(s.legs,underlying,T_today);
  const projPct=((projPnl/netCost)*100).toFixed(1);
  const projCol=projPnl>=0?"var(--green)":"var(--red)";
  const projSign=projPnl>=0?"+":"";
  const statsEl=document.getElementById("payoffStats");
  const bes=s.breakevens||[];
  statsEl.style.display="grid";
  statsEl.innerHTML=`
    <div style="background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--green);border-radius:10px;padding:10px 14px;"><div style="font-size:12px;color:#d8eeff;text-transform:uppercase;letter-spacing:.8px;">Max Profit</div><div style="font-size:17px;font-weight:800;font-family:'JetBrains Mono',monospace;color:var(--green);margin-top:3px;">${{s.maxProfit>=999999?"Unlimited":"₹"+s.maxProfit.toLocaleString("en-IN")}}</div></div>
    <div style="background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--red);border-radius:10px;padding:10px 14px;"><div style="font-size:12px;color:#d8eeff;text-transform:uppercase;letter-spacing:.8px;">Max Loss</div><div style="font-size:17px;font-weight:800;font-family:'JetBrains Mono',monospace;color:var(--red);margin-top:3px;">₹${{s.maxLoss.toLocaleString("en-IN")}}</div></div>
    <div style="background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--gold);border-radius:10px;padding:10px 14px;"><div style="font-size:12px;color:#d8eeff;text-transform:uppercase;letter-spacing:.8px;">Lower BE</div><div style="font-size:17px;font-weight:800;font-family:'JetBrains Mono',monospace;color:var(--gold);margin-top:3px;">${{bes[0]?"₹"+bes[0].toLocaleString("en-IN"):"—"}}</div></div>
    <div style="background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--gold);border-radius:10px;padding:10px 14px;"><div style="font-size:12px;color:#d8eeff;text-transform:uppercase;letter-spacing:.8px;">Upper BE</div><div style="font-size:17px;font-weight:800;font-family:'JetBrains Mono',monospace;color:var(--gold);margin-top:3px;">${{bes[1]?"₹"+bes[1].toLocaleString("en-IN"):bes[0]?"₹"+bes[0].toLocaleString("en-IN"):"—"}}</div></div>`;
  const footerEl=document.getElementById("payoffFooter"), fallbackEl=document.getElementById("projBadgeFallback");
  if(footerEl) footerEl.style.display="flex"; if(fallbackEl) fallbackEl.style.display="none";
  const beBadgesEl=document.getElementById("beBadges");
  if(beBadgesEl) beBadgesEl.innerHTML=bes.map((be,i)=>`<span style="background:rgba(255,209,102,0.1);border:1px solid rgba(255,209,102,0.35);border-radius:6px;padding:3px 10px;font-size:13px;font-weight:700;color:var(--gold);font-family:'JetBrains Mono',monospace;">${{i===0?"▼":"▲"}} ₹${{be.toLocaleString("en-IN")}}</span>`).join("");
  const projBadgeEl=document.getElementById("projBadge");
  if(projBadgeEl) projBadgeEl.innerHTML=`Projected P&L now: <span style="color:${{projCol}};font-weight:800;">${{projSign}}₹${{Math.round(projPnl).toLocaleString("en-IN")}} (${{projSign}}${{projPct}}%)</span>`;
  const ctx=document.getElementById("payoffChart").getContext("2d");
  if(payoffChart) payoffChart.destroy();
  let crosshairX=null;
  payoffChart=new Chart(ctx,{{
    type:"bar",
    data:{{
      labels:priceRange,
      datasets:[
        {{label:"CE OI",type:"bar",data:ceOiData,backgroundColor:"rgba(0,200,150,0.18)",borderColor:"rgba(0,200,150,0.35)",borderWidth:1,yAxisID:"yOI",order:3,barPercentage:0.6}},
        {{label:"PE OI",type:"bar",data:peOiData,backgroundColor:"rgba(255,107,107,0.18)",borderColor:"rgba(255,107,107,0.35)",borderWidth:1,yAxisID:"yOI",order:3,barPercentage:0.6}},
        {{label:"Today (DTE:"+dte+")",type:"line",data:todayPnl,borderColor:"#00c896",borderWidth:2.5,pointRadius:0,fill:{{target:{{value:0}},above:"rgba(0,200,150,0.12)",below:"rgba(255,107,107,0.10)"}},tension:0.3,yAxisID:"yPnl",order:1}},
        {{label:"At Expiry",type:"line",data:expiryPnl,borderColor:"#5ba3ff",borderWidth:2,pointRadius:0,borderDash:[5,4],fill:false,tension:0.1,yAxisID:"yPnl",order:2}},
        {{label:"Zero",type:"line",data:priceRange.map(()=>0),borderColor:"rgba(255,255,255,0.10)",borderWidth:1,pointRadius:0,fill:false,yAxisID:"yPnl",order:4}},
      ]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      interaction:{{mode:"index",intersect:false,axis:"x"}},
      plugins:{{
        legend:{{display:true,position:"top",labels:{{color:"#b0cce0",font:{{family:"DM Mono",size:10}},boxWidth:14,filter:item=>item.text!=="Zero"}}}},
        tooltip:{{enabled:false,mode:"index",intersect:false}},
      }},
      scales:{{
        x:{{ticks:{{color:"#6a90b8",font:{{family:"DM Mono",size:9}},maxTicksLimit:12,callback:(val,idx)=>{{const price=priceRange[idx]??val;return price>=1000?Math.round(price).toLocaleString("en-IN"):price;}}}},grid:{{color:"#0b1520"}},border:{{color:"#1a2535"}}}},
        yPnl:{{position:"left",ticks:{{color:"#6a90b8",font:{{family:"DM Mono",size:9}},callback:v=>"₹"+(Math.abs(v)>=1000?(v/1000).toFixed(0)+"K":v)}},grid:{{color:"#0b1520"}},border:{{color:"#1a2535"}},title:{{display:true,text:"Profit / Loss",color:"#7aA8c8",font:{{size:9,family:"DM Mono"}}}}}},
        yOI:{{position:"right",ticks:{{color:"#6a90b8",font:{{family:"DM Mono",size:9}},callback:v=>v>=1000?(v/1000).toFixed(0)+"L":v}},grid:{{drawOnChartArea:false}},border:{{color:"#1a2535"}},title:{{display:true,text:"Open Interest",color:"#7aA8c8",font:{{size:9,family:"DM Mono"}}}}}},
      }},
    }},
    plugins:[{{
      id:"crosshairSpot",
      afterDatasetsDraw(chart) {{
        const xScale=chart.scales.x,yScale=chart.scales.yPnl,ctx2=chart.ctx;
        const spotIdx=priceRange.findIndex(p=>p>=underlying);
        if(spotIdx>=0) {{
          const xPx=xScale.getPixelForValue(spotIdx);
          ctx2.save(); ctx2.setLineDash([6,4]); ctx2.strokeStyle="rgba(0,200,150,0.65)"; ctx2.lineWidth=1.5;
          ctx2.beginPath(); ctx2.moveTo(xPx,yScale.top); ctx2.lineTo(xPx,yScale.bottom); ctx2.stroke();
          ctx2.setLineDash([]); ctx2.fillStyle="rgba(0,200,150,0.9)"; ctx2.font="bold 17pxpxpx DM Mono,monospace"; ctx2.textAlign="center";
          ctx2.fillText("▼ "+underlying.toLocaleString("en-IN"),xPx,yScale.top-4); ctx2.restore();
        }}
        const bePoints=[];
        for(let i=0;i<expiryPnl.length-1;i++) {{
          if((expiryPnl[i]<0)!==(expiryPnl[i+1]<0)) {{
            const frac=Math.abs(expiryPnl[i])/(Math.abs(expiryPnl[i])+Math.abs(expiryPnl[i+1]));
            const bePx=xScale.getPixelForValue(i)+frac*(xScale.getPixelForValue(i+1)-xScale.getPixelForValue(i));
            const bePrice=priceRange[i]+frac*(priceRange[i+1]-priceRange[i]);
            bePoints.push({{bePx,bePrice}});
          }}
        }}
        bePoints.forEach((be)=>{{
          ctx2.save(); ctx2.setLineDash([4,3]); ctx2.strokeStyle="rgba(255,209,102,0.7)"; ctx2.lineWidth=1.5;
          ctx2.beginPath(); ctx2.moveTo(be.bePx,yScale.top); ctx2.lineTo(be.bePx,yScale.bottom); ctx2.stroke(); ctx2.setLineDash([]);
          const label="BE "+Math.round(be.bePrice).toLocaleString("en-IN");
          ctx2.font="bold 17pxpxpx DM Mono,monospace"; const tw=ctx2.measureText(label).width+10;
          const tx=be.bePx-tw/2,ty=yScale.top+6;
          ctx2.fillStyle="rgba(255,209,102,0.15)"; ctx2.strokeStyle="rgba(255,209,102,0.6)"; ctx2.lineWidth=1;
          ctx2.beginPath(); ctx2.roundRect(tx,ty,tw,14,3); ctx2.fill(); ctx2.stroke();
          ctx2.fillStyle="rgba(255,209,102,1)"; ctx2.textAlign="center"; ctx2.fillText(label,be.bePx,ty+10);
          const zeroPx=yScale.getPixelForValue(0);
          ctx2.fillStyle="#ffd166"; ctx2.strokeStyle="#060910"; ctx2.lineWidth=2;
          ctx2.beginPath(); ctx2.arc(be.bePx,zeroPx,4,0,Math.PI*2); ctx2.fill(); ctx2.stroke();
          ctx2.restore();
        }});
        if(crosshairX===null) return;
        const nearXPx=xScale.getPixelForValue(crosshairX); if(isNaN(nearXPx)) return;
        ctx2.save(); ctx2.setLineDash([3,3]); ctx2.strokeStyle="rgba(255,255,255,0.18)"; ctx2.lineWidth=1;
        ctx2.beginPath(); ctx2.moveTo(nearXPx,yScale.top); ctx2.lineTo(nearXPx,yScale.bottom); ctx2.stroke(); ctx2.setLineDash([]);
        ctx2.fillStyle="#00c896"; ctx2.strokeStyle="#060910"; ctx2.lineWidth=2.5;
        ctx2.beginPath(); ctx2.arc(nearXPx,yScale.getPixelForValue(todayPnl[crosshairX]),5,0,Math.PI*2); ctx2.fill(); ctx2.stroke();
        ctx2.fillStyle="#5ba3ff"; ctx2.strokeStyle="#060910"; ctx2.lineWidth=2;
        ctx2.beginPath(); ctx2.arc(nearXPx,yScale.getPixelForValue(expiryPnl[crosshairX]),5,0,Math.PI*2); ctx2.fill(); ctx2.stroke();
        ctx2.restore();
      }},
    }}],
  }});
  const canvas=document.getElementById("payoffChart"), tt=document.getElementById("payoffTooltip");
  function showTooltip(clientX,clientY) {{
    const rect=canvas.getBoundingClientRect(), xScale=payoffChart.scales.x, yScale=payoffChart.scales.yPnl;
    const canvasX=clientX-rect.left;
    if(canvasX<xScale.left||canvasX>xScale.right){{ crosshairX=null; tt.style.display="none"; payoffChart.draw(); return; }}
    const ratio=(canvasX-xScale.left)/(xScale.right-xScale.left);
    const bestIdx=Math.max(0,Math.min(priceRange.length-1,Math.round(ratio*(priceRange.length-1))));
    if(crosshairX!==bestIdx){{ crosshairX=bestIdx; payoffChart.draw(); }}
    const price=priceRange[bestIdx], pctChg=(((price-underlying)/underlying)*100).toFixed(1);
    const sign=parseFloat(pctChg)>=0?"+":"", pCol=parseFloat(pctChg)>=0?"#00c896":"#ff6b6b";
    const todayVal=todayPnl[bestIdx], expVal=expiryPnl[bestIdx];
    const tSign=todayVal>=0?"+":"", eSign=expVal>=0?"+":"";
    const tCol=todayVal>=0?"#00c896":"#ff6b6b", eCol=expVal>=0?"#00c896":"#ff6b6b";
    const tPct=((todayVal/netCost)*100).toFixed(1), ePct=((expVal/netCost)*100).toFixed(1);
    tt.innerHTML=`<div style="font-size:12px;color:#b0cce0;margin-bottom:5px;letter-spacing:1px;">WHEN PRICE IS AT</div>
      <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:10px;"><span style="font-size:18px;font-weight:800;color:#ddeeff;font-family:'DM Mono',monospace;">&#8377;${{price.toLocaleString("en-IN")}}</span><span style="font-size:14px;font-weight:700;color:${{pCol}};font-family:'DM Mono',monospace;">${{sign}}${{pctChg}}%&nbsp;(${{sign}}${{(price-underlying).toLocaleString("en-IN")}})</span></div>
      <div style="height:1px;background:rgba(255,255,255,0.07);margin-bottom:10px;"></div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;"><span style="font-size:12px;color:#b0cce0;display:flex;align-items:center;gap:6px;"><span style="width:9px;height:9px;border-radius:50%;background:#00c896;display:inline-block;"></span>Today <span style="color:#7aA8c8;margin-left:2px;">DTE:${{dte}}</span></span><span style="font-size:16px;font-weight:800;color:${{tCol}};font-family:'DM Mono',monospace;">${{tSign}}&#8377;${{Math.round(todayVal).toLocaleString("en-IN")}}<span style="font-size:12px;opacity:.8;"> (${{tSign}}${{tPct}}%)</span></span></div>
      <div style="display:flex;justify-content:space-between;align-items:center;"><span style="font-size:12px;color:#b0cce0;display:flex;align-items:center;gap:6px;"><span style="width:9px;height:9px;border-radius:50%;background:#5ba3ff;display:inline-block;"></span>At Expiry <span style="color:#7aA8c8;margin-left:2px;">T=0</span></span><span style="font-size:16px;font-weight:800;color:${{eCol}};font-family:'DM Mono',monospace;">${{eSign}}&#8377;${{Math.round(expVal).toLocaleString("en-IN")}}<span style="font-size:12px;opacity:.8;"> (${{eSign}}${{ePct}}%)</span></span></div>
      <div style="margin-top:9px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.06);font-size:12px;color:#7aA8c8;">Theta loss by expiry: <span style="color:${{(todayVal-expVal)>0?"#ff6b6b":"#00c896"}};font-weight:700;">${{(todayVal-expVal)>=0?"-":"+"}}&#8377;${{Math.abs(Math.round(todayVal-expVal)).toLocaleString("en-IN")}}</span></div>`;
    const ttW=268,ttH=140;
    let fixedL=clientX+18,fixedT=clientY-ttH/2;
    if(fixedL+ttW>window.innerWidth-10) fixedL=clientX-ttW-18;
    if(fixedT<8) fixedT=8; if(fixedT+ttH>window.innerHeight-10) fixedT=window.innerHeight-ttH-10;
    tt.style.left=fixedL+"px"; tt.style.top=fixedT+"px"; tt.style.display="block";
  }}
  function hideTooltip(){{ crosshairX=null; tt.style.display="none"; payoffChart.draw(); }}
  canvas.addEventListener("mousemove",e=>showTooltip(e.clientX,e.clientY));
  canvas.addEventListener("mouseleave",hideTooltip);
  canvas.addEventListener("touchmove",e=>{{e.preventDefault();const t=e.touches[0];showTooltip(t.clientX,t.clientY);}},{{passive:false}});
  canvas.addEventListener("touchend",hideTooltip);
}}
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
</body>
</html>"""
    return html


# =================================================================
#  SECTION 6 -- MAIN RUNNER
# =================================================================

def main():
    print("=" * 60)
    print("  Nifty Options Analyzer — GitHub Actions Runner")
    print(f"  Started: {now_ist_str()}")
    print("=" * 60)

    nse = NSEOptionChain()

    print("\n[1/3] Warming up NSE session & fetching expiries...")
    result, session, headers = nse.fetch()

    print("\n[2/3] Fetching all expiries...")
    all_expiry_data, expiry_list = nse.fetch_multiple_expiries(session, headers, n=7)

    if not all_expiry_data:
        print("ERROR: No data fetched. Exiting.")
        return

    print(f"\n[3/3] Generating index.html with {len(all_expiry_data)} expiries...")
    html = build_html(all_expiry_data, expiry_list, now_ist_str())

    out_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  index.html written → {out_path}")
    print(f"  Expiries baked in: {expiry_list}")
    print(f"  Done: {now_ist_str()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
