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
                lower_bound = underlying - 600
                upper_bound = underlying + 600
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
    avg_support    = sum(support_levels)    / len(support_levels)    if support_levels    else underlying - 300
    avg_resistance = sum(resistance_levels) / len(resistance_levels) if resistance_levels else underlying + 300
    sr_range       = avg_resistance - avg_support
    atm  = atm_strike
    s_st = nearest(avg_support)
    r_st = nearest(avg_resistance)
    otm_c = nearest(underlying + sr_range * 0.3)
    otm_p = nearest(underlying - sr_range * 0.3)
    far_c = nearest(avg_resistance + 150)
    far_p = nearest(avg_support    - 150)
    atm_iv = (get(atm, "ce_iv", 15) + get(atm, "pe_iv", 15)) / 2
    exp_move = round(underlying * (atm_iv / 100) * math.sqrt(T), 0)
    raw = []
    def make_strategy(name, legs, bias_tag, strategy_type):
        net_premium  = sum((-l["premium"] if l["action"] == "buy" else l["premium"]) for l in legs)
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
        pop = _strategy_pop(legs, underlying, T, breakevens=breakevens, strategy_type=strategy_type)
        score = round(pop * 0.40 + min(rr_ratio * 35, 35) + min(max_profit / 5000 * 25, 25), 1)
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
                {"action": "buy",  "strike": buy_st,  "opt_type": "CE", "premium": get(buy_st,  "ce_ltp"), "iv": get(buy_st,  "ce_iv", 15)},
                {"action": "sell", "strike": sell_st, "opt_type": "CE", "premium": get(sell_st, "ce_ltp"), "iv": get(sell_st, "ce_iv", 15)},
            ], "bullish", "debit_spread")
            if s: raw.append(s)
    if bias != "bullish":
        buy_st, sell_st = atm, s_st
        if buy_st != sell_st:
            s = make_strategy("Bear Put Spread", [
                {"action": "buy",  "strike": buy_st,  "opt_type": "PE", "premium": get(buy_st,  "pe_ltp"), "iv": get(buy_st,  "pe_iv", 15)},
                {"action": "sell", "strike": sell_st, "opt_type": "PE", "premium": get(sell_st, "pe_ltp"), "iv": get(sell_st, "pe_iv", 15)},
            ], "bearish", "debit_spread")
            if s: raw.append(s)
    if bias != "bearish":
        sell_st = s_st
        buy_st  = nearest(avg_support - 150)
        if sell_st != buy_st:
            s = make_strategy("Bull Put Spread", [
                {"action": "sell", "strike": sell_st, "opt_type": "PE", "premium": get(sell_st, "pe_ltp"), "iv": get(sell_st, "pe_iv", 15)},
                {"action": "buy",  "strike": buy_st,  "opt_type": "PE", "premium": get(buy_st,  "pe_ltp"), "iv": get(buy_st,  "pe_iv", 15)},
            ], "bullish", "credit_spread")
            if s: raw.append(s)
    if bias != "bullish":
        sell_st = r_st
        buy_st  = nearest(avg_resistance + 150)
        if sell_st != buy_st:
            s = make_strategy("Bear Call Spread", [
                {"action": "sell", "strike": sell_st, "opt_type": "CE", "premium": get(sell_st, "ce_ltp"), "iv": get(sell_st, "ce_iv", 15)},
                {"action": "buy",  "strike": buy_st,  "opt_type": "CE", "premium": get(buy_st,  "ce_ltp"), "iv": get(buy_st,  "ce_iv", 15)},
            ], "bearish", "credit_spread")
            if s: raw.append(s)
    sell_ce, buy_ce = r_st, far_c
    sell_pe, buy_pe = s_st, far_p
    if len({sell_ce, buy_ce, sell_pe, buy_pe}) == 4:
        s = make_strategy("Iron Condor", [
            {"action": "sell", "strike": sell_ce, "opt_type": "CE", "premium": get(sell_ce, "ce_ltp"), "iv": get(sell_ce, "ce_iv", 15)},
            {"action": "buy",  "strike": buy_ce,  "opt_type": "CE", "premium": get(buy_ce,  "ce_ltp"), "iv": get(buy_ce,  "ce_iv", 15)},
            {"action": "sell", "strike": sell_pe, "opt_type": "PE", "premium": get(sell_pe, "pe_ltp"), "iv": get(sell_pe, "pe_iv", 15)},
            {"action": "buy",  "strike": buy_pe,  "opt_type": "PE", "premium": get(buy_pe,  "pe_ltp"), "iv": get(buy_pe,  "pe_iv", 15)},
        ], "neutral", "iron_condor")
        if s: raw.append(s)
    wing = int(sr_range * 0.4 / 50) * 50
    wing = max(wing, 100)
    buy_c_ibf = nearest(atm + wing)
    buy_p_ibf = nearest(atm - wing)
    if len({atm, buy_c_ibf, buy_p_ibf}) == 3:
        s = make_strategy("Iron Butterfly", [
            {"action": "sell", "strike": atm,       "opt_type": "CE", "premium": get(atm,       "ce_ltp"), "iv": get(atm,       "ce_iv", 15)},
            {"action": "sell", "strike": atm,       "opt_type": "PE", "premium": get(atm,       "pe_ltp"), "iv": get(atm,       "pe_iv", 15)},
            {"action": "buy",  "strike": buy_c_ibf, "opt_type": "CE", "premium": get(buy_c_ibf, "ce_ltp"), "iv": get(buy_c_ibf, "ce_iv", 15)},
            {"action": "buy",  "strike": buy_p_ibf, "opt_type": "PE", "premium": get(buy_p_ibf, "pe_ltp"), "iv": get(buy_p_ibf, "pe_iv", 15)},
        ], "neutral", "iron_butterfly")
        if s: raw.append(s)
    s = make_strategy("Long Straddle", [
        {"action": "buy", "strike": atm, "opt_type": "CE", "premium": get(atm, "ce_ltp"), "iv": get(atm, "ce_iv", 15)},
        {"action": "buy", "strike": atm, "opt_type": "PE", "premium": get(atm, "pe_ltp"), "iv": get(atm, "pe_iv", 15)},
    ], "volatile", "straddle")
    if s: raw.append(s)
    if bias == "neutral":
        s = make_strategy("Short Straddle", [
            {"action": "sell", "strike": atm, "opt_type": "CE", "premium": get(atm, "ce_ltp"), "iv": get(atm, "ce_iv", 15)},
            {"action": "sell", "strike": atm, "opt_type": "PE", "premium": get(atm, "pe_ltp"), "iv": get(atm, "pe_iv", 15)},
        ], "neutral", "straddle")
        if s: raw.append(s)
    s = make_strategy("Long Strangle", [
        {"action": "buy", "strike": otm_c, "opt_type": "CE", "premium": get(otm_c, "ce_ltp"), "iv": get(otm_c, "ce_iv", 15)},
        {"action": "buy", "strike": otm_p, "opt_type": "PE", "premium": get(otm_p, "pe_ltp"), "iv": get(otm_p, "pe_iv", 15)},
    ], "volatile", "strangle")
    if s: raw.append(s)
    if bias == "neutral":
        s = make_strategy("Short Strangle", [
            {"action": "sell", "strike": otm_c, "opt_type": "CE", "premium": get(otm_c, "ce_ltp"), "iv": get(otm_c, "ce_iv", 15)},
            {"action": "sell", "strike": otm_p, "opt_type": "PE", "premium": get(otm_p, "pe_ltp"), "iv": get(otm_p, "pe_iv", 15)},
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
  --bg3:        #111826;
  --border:     #1a2535;
  --border2:    #0e1e30;
  --cyan:       #00d4ff;
  --green:      #00c896;
  --red:        #ff6b6b;
  --gold:       #ffd166;
  --purple:     #8aa0ff;
  --text:       #ddeeff;
  --text2:      #6a8aaa;
  --text3:      #2d4560;
  --grid:       #0b1520;
}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh;overflow-x:hidden;}}
body::before{{content:'';position:fixed;inset:0;background-image:linear-gradient(var(--grid) 1px,transparent 1px),linear-gradient(90deg,var(--grid) 1px,transparent 1px);background-size:44px 44px;pointer-events:none;z-index:0;}}
.wrap{{position:relative;z-index:1;max-width:1500px;margin:0 auto;padding:18px;}}

/* ── HEADER ── */
.hdr{{display:flex;align-items:center;justify-content:space-between;padding:16px 22px;background:linear-gradient(135deg,#0d111799,#11182699);border:1px solid var(--border);border-top:2px solid var(--cyan);border-radius:12px;margin-bottom:18px;backdrop-filter:blur(12px);}}
.hdr-left{{display:flex;align-items:center;gap:14px;}}
.logo{{width:44px;height:44px;background:linear-gradient(135deg,var(--cyan),var(--purple));border-radius:10px;display:flex;align-items:center;justify-content:center;font-family:'DM Mono',monospace;font-weight:500;font-size:17px;color:#000;box-shadow:0 0 18px #00d4ff33;}}
.hdr h1{{font-size:20px;font-weight:800;letter-spacing:-.5px;}}
.hdr h1 span{{color:var(--cyan);}}
.hdr-sub{{font-size:11px;color:var(--text2);font-family:'DM Mono',monospace;margin-top:2px;}}
.live-pill{{display:flex;align-items:center;gap:7px;background:#00c89611;border:1px solid #00c89633;border-radius:20px;padding:5px 13px;font-size:11px;font-weight:700;color:var(--green);font-family:'DM Mono',monospace;}}
.live-dot{{width:7px;height:7px;background:var(--green);border-radius:50%;animation:pulse 1.5s infinite;}}

/* ── TICKER ── */
.ticker{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px;}}
.tick-card{{background:var(--bg3);border:1px solid var(--border);border-top:2px solid var(--tc,var(--cyan));border-radius:10px;padding:14px 18px;transition:border-color .3s;}}
.tick-card:hover{{border-color:var(--tc,var(--cyan));}}
.tick-lbl{{font-size:10px;color:var(--text2);font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:1px;}}
.tick-val{{font-size:22px;font-weight:800;font-family:'DM Mono',monospace;margin-top:3px;}}
.tick-sub{{font-size:10px;color:var(--text3);font-family:'DM Mono',monospace;margin-top:1px;}}

/* ── MAIN LAYOUT ── */
.main{{display:grid;grid-template-columns:340px 1fr 210px;gap:16px;margin-bottom:18px;}}

/* ── PANEL ── */
.panel{{background:var(--bg3);border:1px solid var(--border);border-radius:12px;overflow:hidden;}}
.panel-hdr{{padding:13px 18px;background:linear-gradient(90deg,#0d1a26,#0d1117);border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;}}
.panel-title{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:var(--text2);display:flex;align-items:center;gap:7px;}}
.panel-body{{padding:18px;}}

/* ── PARAM PANEL TABS ── */
.ptab-row{{display:flex;border-bottom:1px solid var(--border);margin:-18px -18px 16px;}}
.ptab{{flex:1;padding:11px 6px;font-family:'DM Mono',monospace;font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;cursor:pointer;text-align:center;color:var(--text2);border:none;background:transparent;transition:all .2s;border-bottom:2px solid transparent;margin-bottom:-1px;}}
.ptab.active{{color:var(--cyan);border-bottom-color:var(--cyan);background:rgba(0,212,255,.05);}}
.ptab.ptab-be.active{{color:var(--gold);border-bottom-color:var(--gold);background:rgba(255,209,102,.05);}}
.ptab:hover:not(.active){{color:var(--text);background:rgba(255,255,255,.03);}}

/* ── TARGET BE SECTION ── */
.be-find-btn{{width:100%;padding:13px;background:linear-gradient(135deg,var(--gold),#ff9500);border:none;border-radius:9px;color:#000;font-family:'Syne',sans-serif;font-size:14px;font-weight:800;text-transform:uppercase;letter-spacing:2px;cursor:pointer;transition:all .3s;margin-top:6px;}}
.be-find-btn:hover{{transform:translateY(-2px);box-shadow:0 8px 28px rgba(255,209,102,.33);}}
.be-find-btn:active{{transform:translateY(0);}}
.be-info-box{{background:rgba(255,209,102,.05);border:1px solid rgba(255,209,102,.18);border-radius:8px;padding:10px 12px;margin-bottom:14px;font-size:9.5px;color:rgba(255,209,102,.75);line-height:1.7;font-family:'DM Mono',monospace;}}
.be-results-wrap{{margin-top:14px;}}
.be-match-card{{background:var(--bg2);border:1px solid var(--border);border-radius:9px;padding:12px 14px;margin-bottom:9px;cursor:pointer;transition:all .25s;position:relative;overflow:hidden;}}
.be-match-card:hover{{border-color:var(--cyan);transform:translateY(-1px);box-shadow:0 6px 20px rgba(0,0,0,.3);}}
.be-match-card.be-best{{border-color:var(--gold);border-top:2px solid var(--gold);background:rgba(255,209,102,.03);}}
.be-match-rank{{position:absolute;top:8px;right:10px;font-size:8px;font-weight:700;font-family:'DM Mono',monospace;padding:2px 7px;border-radius:10px;}}
.be-match-name{{font-size:12px;font-weight:800;margin-bottom:6px;}}
.be-match-legs{{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;}}
.be-leg-tag{{border-radius:5px;padding:3px 7px;font-size:9px;font-weight:700;font-family:'DM Mono',monospace;}}
.be-leg-buy{{border:1px solid var(--green);color:var(--green);background:#00c89608;}}
.be-leg-sell{{border:1px solid var(--red);color:var(--red);background:#ff6b6b08;}}
.be-accuracy-row{{display:grid;grid-template-columns:1fr 1fr;gap:6px;}}
.be-accuracy-box{{background:var(--bg3);border-radius:6px;padding:7px 9px;}}
.be-acc-lbl{{font-size:8px;color:var(--text3);text-transform:uppercase;letter-spacing:.7px;font-family:'DM Mono',monospace;}}
.be-acc-target{{font-size:9px;color:var(--text2);font-family:'DM Mono',monospace;margin-top:1px;}}
.be-acc-actual{{font-size:12px;font-weight:700;font-family:'DM Mono',monospace;margin-top:2px;}}
.be-acc-diff{{font-size:9px;font-family:'DM Mono',monospace;margin-top:1px;}}
.be-diff-good{{color:var(--green);}}
.be-diff-ok{{color:var(--gold);}}
.be-diff-bad{{color:var(--red);}}
.be-net-prem{{font-size:10px;font-family:'DM Mono',monospace;color:var(--text2);margin-top:6px;padding-top:6px;border-top:1px solid var(--border2);display:flex;justify-content:space-between;align-items:center;}}
.be-empty{{text-align:center;padding:24px 12px;color:var(--text3);font-family:'DM Mono',monospace;font-size:10px;line-height:1.8;}}
.be-score-bar{{height:2px;background:var(--border);border-radius:2px;overflow:hidden;margin-top:6px;}}
.be-score-fill{{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--gold),#ff9500);transition:width .6s;}}

/* ── INPUTS ── */
.form-grp{{margin-bottom:14px;}}
.form-lbl{{display:block;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--text2);margin-bottom:6px;font-family:'DM Mono',monospace;}}
.inp{{width:100%;background:var(--bg2);border:1px solid var(--border);border-radius:7px;padding:9px 12px;color:var(--text);font-family:'DM Mono',monospace;font-size:13px;outline:none;transition:all .2s;}}
.inp:focus{{border-color:var(--cyan);box-shadow:0 0 0 3px #00d4ff12;}}
.inp-gold:focus{{border-color:var(--gold)!important;box-shadow:0 0 0 3px rgba(255,209,102,.12)!important;}}
.sel{{width:100%;background:var(--bg2);border:1px solid var(--border);border-radius:7px;padding:9px 12px;color:var(--text);font-family:'DM Mono',monospace;font-size:12px;outline:none;cursor:pointer;appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'%3E%3Cpath fill='%2300d4ff' d='M5 7L0 2h10z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 12px center;transition:all .2s;}}
.sel:focus{{border-color:var(--cyan);box-shadow:0 0 0 3px #00d4ff12;}}
.sel option{{background:#0d1117;}}
.bias-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;}}
.bias-btn{{padding:9px 6px;border:1px solid var(--border);border-radius:7px;background:var(--bg2);color:var(--text2);font-family:'DM Mono',monospace;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;cursor:pointer;transition:all .2s;text-align:center;}}
.bias-btn:hover{{border-color:var(--cyan);color:var(--cyan);}}
.bias-bull{{border-color:var(--green)!important;background:#00c89611!important;color:var(--green)!important;}}
.bias-bear{{border-color:var(--red)!important;background:#ff6b6b11!important;color:var(--red)!important;}}
.bias-neut{{border-color:var(--gold)!important;background:#ffd16611!important;color:var(--gold)!important;}}
.sr-row{{display:flex;gap:7px;margin-bottom:7px;align-items:center;}}
.add-btn{{background:transparent;border:1px dashed var(--border);border-radius:7px;color:var(--text3);padding:7px 11px;font-size:16px;cursor:pointer;transition:all .2s;line-height:1;}}
.add-btn:hover{{border-color:var(--cyan);color:var(--cyan);}}
.rm-btn{{background:transparent;border:none;color:var(--text3);font-size:15px;cursor:pointer;padding:3px 7px;transition:color .2s;line-height:1;}}
.rm-btn:hover{{color:var(--red);}}
.divider{{height:1px;background:var(--border);margin:12px 0;}}
.analyze-btn{{width:100%;padding:13px;background:linear-gradient(135deg,var(--cyan),var(--purple));border:none;border-radius:9px;color:#000;font-family:'Syne',sans-serif;font-size:14px;font-weight:800;text-transform:uppercase;letter-spacing:2px;cursor:pointer;transition:all .3s;margin-top:6px;}}
.analyze-btn:hover{{transform:translateY(-2px);box-shadow:0 8px 28px #00d4ff33;}}
.analyze-btn:active{{transform:translateY(0);}}

/* ── OPTION CHAIN ── */
.chain-wrap{{overflow:auto;max-height:540px;}}
.chain-side-hdr{{display:grid;grid-template-columns:1fr 100px 1fr;padding:7px 0;border-bottom:1px solid var(--border);background:linear-gradient(90deg,#0d1a26,#0d1117);position:sticky;top:0;z-index:3;}}
.chain-side-hdr .ce-hdr{{text-align:right;padding-right:12px;font-size:9px;font-weight:800;color:var(--green);letter-spacing:1.5px;text-transform:uppercase;}}
.chain-side-hdr .st-hdr{{text-align:center;font-size:9px;font-weight:800;color:var(--text2);letter-spacing:1px;text-transform:uppercase;}}
.chain-side-hdr .pe-hdr{{text-align:left;padding-left:12px;font-size:9px;font-weight:800;color:var(--red);letter-spacing:1.5px;text-transform:uppercase;}}
.chain-col-hdr{{display:grid;grid-template-columns:1fr 100px 1fr;padding:5px 0 4px;border-bottom:1px solid var(--border2);background:#0a1218;position:sticky;top:32px;z-index:2;}}
.chain-col-hdr .ce-cols{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;padding:0 10px 0 6px;}}
.chain-col-hdr .pe-cols{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;padding:0 6px 0 10px;}}
.chain-col-hdr span{{font-size:8px;color:var(--text3);text-transform:uppercase;letter-spacing:.8px;font-family:'JetBrains Mono',monospace;text-align:right;}}
.chain-col-hdr .pe-cols span{{text-align:left;}}
.chain-row{{display:grid;grid-template-columns:1fr 100px 1fr;border-bottom:1px solid var(--border2);transition:background .12s;position:relative;}}
.chain-row:hover{{background:#ffffff04;}}
.chain-row.atm-row{{background:#00d4ff07;border-left:2px solid var(--cyan);}}
.chain-row.sup-row .stk-cell{{border-left:2px solid var(--green)!important;}}
.chain-row.res-row .stk-cell{{border-left:2px solid var(--red)!important;}}
.ce-side{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;align-items:center;padding:8px 10px 8px 6px;gap:2px;position:relative;overflow:hidden;}}
.ce-heat-bg{{position:absolute;top:0;right:0;bottom:0;background:var(--green);opacity:0.09;pointer-events:none;border-radius:2px 0 0 2px;}}
.pe-side{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;align-items:center;padding:8px 6px 8px 10px;gap:2px;position:relative;overflow:hidden;}}
.pe-heat-bg{{position:absolute;top:0;left:0;bottom:0;background:var(--red);opacity:0.09;pointer-events:none;border-radius:0 2px 2px 0;}}
.stk-cell{{display:flex;align-items:center;justify-content:center;font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;border-left:1px solid var(--border);border-right:1px solid var(--border);background:#0a1520;min-height:36px;position:relative;flex-direction:column;gap:2px;}}
.atm-tag{{background:var(--cyan);color:#000;font-size:7px;font-weight:800;padding:1px 6px;border-radius:0 0 4px 4px;position:absolute;top:0;letter-spacing:.5px;}}
.cv-ltp{{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;text-align:right;}}
.cv-iv{{font-size:9px;color:var(--text2);font-family:'JetBrains Mono',monospace;text-align:right;}}
.cv-oi{{font-size:9px;font-family:'JetBrains Mono',monospace;color:var(--text2);text-align:right;}}
.cv-doi{{font-size:9px;font-family:'JetBrains Mono',monospace;text-align:right;}}
.pe-side .cv-ltp,.pe-side .cv-iv,.pe-side .cv-oi,.pe-side .cv-doi{{text-align:left;}}
.ce-ltp-v{{color:var(--green);}} .pe-ltp-v{{color:var(--red);}}
.up{{color:var(--green)!important;}} .down{{color:var(--red)!important;}}

/* ── GREEKS SIDEBAR ── */
.greeks-panel{{padding:0;}}
.greeks-title{{padding:13px 14px 10px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:var(--text2);display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border);}}
.g-exp-tag{{font-size:9px;color:var(--cyan);background:#00d4ff12;border:1px solid #00d4ff22;padding:2px 8px;border-radius:10px;font-family:'DM Mono',monospace;}}
.g-sel-wrap{{padding:10px 12px 8px;border-bottom:1px solid var(--border);}}
.g-sel{{width:100%;background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:7px 10px;color:var(--text);font-family:'DM Mono',monospace;font-size:10px;outline:none;cursor:pointer;appearance:none;}}
.g-sel option{{background:#0d1117;}}
.g-atm-badge{{margin:8px 12px;padding:8px 10px;background:linear-gradient(135deg,#00d4ff0a,#8aa0ff0a);border:1px solid #00d4ff22;border-radius:8px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:4px;}}
.g-strike-type{{font-size:8px;font-weight:700;color:rgba(138,160,255,.9);font-family:'DM Mono',monospace;}}
.g-strike-val{{font-size:16px;font-weight:800;color:var(--cyan);font-family:'DM Mono',monospace;}}
.g-ltp-row{{display:flex;gap:6px;margin-top:2px;width:100%;}}
.g-ce-ltp{{font-size:9px;color:#00c8e0;font-family:'DM Mono',monospace;}}
.g-pe-ltp{{font-size:9px;color:#ff9090;font-family:'DM Mono',monospace;}}
.g-row{{display:flex;align-items:center;justify-content:space-between;padding:8px 12px;border-bottom:1px solid var(--border2);}}
.g-label{{display:flex;flex-direction:column;}}
.g-name{{font-size:11px;font-weight:700;color:var(--text);}}
.g-sub{{font-size:9px;color:var(--text3);margin-top:1px;font-family:'DM Mono',monospace;}}
.g-vals{{display:flex;flex-direction:column;gap:3px;align-items:flex-end;}}
.g-ce-val{{font-family:'DM Mono',monospace;font-size:11px;font-weight:700;color:#00c8e0;}}
.g-pe-val{{font-family:'DM Mono',monospace;font-size:11px;font-weight:700;color:#ff9090;}}
.delta-bar-wrap{{display:flex;align-items:center;gap:4px;}}
.delta-bar-track{{width:32px;height:3px;background:rgba(255,255,255,.08);border-radius:2px;overflow:hidden;}}
.delta-bar-fill{{height:100%;border-radius:2px;}}
.iv-gauge-wrap{{padding:8px 12px 6px;border-bottom:1px solid var(--border2);}}
.iv-gauge-row{{display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;}}
.iv-gauge-lbl{{font-size:9px;color:var(--text2);font-family:'DM Mono',monospace;}}
.iv-gauge-val{{font-size:11px;font-weight:700;font-family:'DM Mono',monospace;}}
.iv-gauge-track{{height:4px;background:var(--border);border-radius:2px;overflow:hidden;margin-bottom:4px;}}
.iv-gauge-fill{{height:100%;border-radius:2px;transition:width .6s;}}
.iv-regime{{font-size:8.5px;text-align:center;font-weight:700;letter-spacing:.5px;padding:4px 0 8px;font-family:'DM Mono',monospace;}}
.skew-lbl{{font-size:9px;font-weight:700;font-family:'DM Mono',monospace;}}

/* ── STRATEGIES ── */
.strat-section{{margin-bottom:18px;}}
.sec-hdr{{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;}}
.sec-title{{font-size:15px;font-weight:800;display:flex;align-items:center;gap:9px;}}
.sec-tag{{font-size:9px;font-weight:700;background:#00d4ff12;border:1px solid #00d4ff28;color:var(--cyan);padding:2px 9px;border-radius:14px;font-family:'DM Mono',monospace;letter-spacing:1px;}}
.strat-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:14px;}}
.strat-card{{background:var(--bg3);border:1px solid var(--border);border-radius:11px;overflow:hidden;transition:all .3s;cursor:pointer;}}
.strat-card:hover{{transform:translateY(-3px);box-shadow:0 10px 32px rgba(0,0,0,.4);border-color:var(--cc,var(--cyan));}}
.sc-top{{padding:13px 15px 10px;border-bottom:1px solid var(--border);display:flex;align-items:flex-start;justify-content:space-between;}}
.sc-name{{font-size:13px;font-weight:800;}}
.sc-sub{{font-size:10px;color:var(--text2);margin-top:2px;font-family:'DM Mono',monospace;}}
.pop-pill{{padding:4px 10px;border-radius:14px;font-size:12px;font-weight:800;font-family:'DM Mono',monospace;white-space:nowrap;}}
.sc-fields{{padding:10px 15px;display:grid;grid-template-columns:1fr 1fr;gap:6px 10px;border-bottom:1px solid var(--border);}}
.sc-field{{display:flex;flex-direction:column;}}
.sc-field-lbl{{font-size:8.5px;color:var(--text3);text-transform:uppercase;letter-spacing:.7px;font-family:'DM Mono',monospace;}}
.sc-field-val{{font-size:12px;font-weight:700;font-family:'DM Mono',monospace;margin-top:2px;}}
.sc-legs{{padding:9px 15px;background:#0d111766;display:flex;flex-wrap:wrap;gap:5px;border-bottom:1px solid var(--border);}}
.leg-tag{{border-radius:5px;padding:3px 8px;font-size:9.5px;font-weight:700;font-family:'DM Mono',monospace;}}
.leg-buy{{border:1px solid var(--green);color:var(--green);background:#00c89608;}}
.leg-sell{{border:1px solid var(--red);color:var(--red);background:#ff6b6b08;}}
.sc-score{{padding:8px 15px;display:flex;align-items:center;gap:8px;}}
.score-bar-track{{flex:1;height:3px;background:var(--border);border-radius:2px;overflow:hidden;}}
.score-bar-fill{{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--cyan),var(--purple));transition:width 1s ease;}}
.score-lbl{{font-size:9px;color:var(--text3);font-family:'DM Mono',monospace;}}
.score-num{{font-size:11px;font-weight:700;color:var(--cyan);font-family:'DM Mono',monospace;}}

/* ── INTRADAY P&L SIMULATOR ── */
.intraday-sim{{border-top:2px solid rgba(255,209,102,.22);background:linear-gradient(135deg,rgba(245,197,24,.03),rgba(200,155,10,.015));}}
.sim-tabs{{display:flex;border-bottom:1px solid rgba(255,255,255,.06);}}
.sim-tab{{flex:1;padding:8px 4px;font-family:'DM Mono',monospace;font-size:8.5px;font-weight:700;letter-spacing:1px;text-transform:uppercase;cursor:pointer;text-align:center;color:rgba(255,255,255,.28);border:none;background:transparent;transition:all .2s;border-bottom:2px solid transparent;margin-bottom:-1px;}}
.sim-tab.active{{color:#ffd166;border-bottom-color:#ffd166;background:rgba(245,197,24,.06);}}
.sim-tab:hover:not(.active){{color:rgba(255,255,255,.55);background:rgba(255,255,255,.03);}}
.sim-hdr{{display:flex;align-items:center;justify-content:space-between;padding:8px 10px 6px;}}
.sim-icon{{width:20px;height:20px;border-radius:5px;background:rgba(245,197,24,.15);border:1px solid rgba(245,197,24,.3);display:flex;align-items:center;justify-content:center;font-size:10px;flex-shrink:0;}}
.sim-title{{font-size:9px;font-weight:700;color:rgba(255,209,102,.9);letter-spacing:.8px;text-transform:uppercase;}}
.sim-subtitle{{font-size:8px;color:rgba(255,255,255,.28);margin-top:1px;}}
.sim-tbl{{width:100%;border-collapse:collapse;}}
.sim-tbl thead tr{{background:rgba(255,255,255,.03);}}
.sim-tbl th{{padding:5px 8px;font-size:7.5px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:rgba(255,255,255,.28);text-align:center;border-bottom:1px solid rgba(255,255,255,.06);}}
.sim-tbl th:first-child{{text-align:left;}}
.sim-tbl td{{padding:6px 8px;font-family:'DM Mono',monospace;font-size:10px;text-align:center;border-bottom:1px solid rgba(255,255,255,.03);transition:background .12s;}}
.sim-tbl td:first-child{{text-align:left;}}
.sim-tbl tr:last-child td{{border-bottom:none;}}
.sim-tbl tr:hover td{{background:rgba(255,255,255,.025);}}
.sim-tbl tr.sim-flat td{{background:rgba(245,197,24,.06);border-left:2px solid rgba(245,197,24,.35);}}
.sim-move-lbl{{font-size:9px;font-weight:700;padding:2px 6px;border-radius:5px;display:inline-block;}}
.sim-pnl-val{{font-weight:700;font-size:11px;}}
.sim-live-pnl{{display:flex;align-items:center;justify-content:center;gap:8px;padding:10px;flex-wrap:wrap;border-bottom:1px solid rgba(255,255,255,.05);}}
.slpb{{display:flex;flex-direction:column;align-items:center;gap:2px;padding:8px 12px;border-radius:9px;min-width:90px;}}
.slpb-lbl{{font-size:7.5px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:rgba(255,255,255,.28);}}
.slpb-num{{font-family:'DM Mono',monospace;font-size:16px;font-weight:700;line-height:1;}}
.slpb-sub{{font-size:8px;color:rgba(255,255,255,.25);}}
.cbar-row{{display:flex;align-items:center;gap:6px;margin-bottom:5px;}}
.cbar-lbl{{font-family:'DM Mono',monospace;font-size:9px;font-weight:700;width:44px;flex-shrink:0;}}
.cbar-track{{flex:1;height:4px;background:rgba(255,255,255,.07);border-radius:2px;overflow:hidden;}}
.cbar-fill{{height:100%;border-radius:2px;transition:width .4s;}}
.cbar-val{{font-family:'DM Mono',monospace;font-size:9px;font-weight:700;min-width:58px;text-align:right;}}
.sim-note{{margin:0 10px 10px;padding:7px 10px;background:rgba(255,107,107,.05);border:1px solid rgba(255,107,107,.13);border-radius:7px;font-size:9px;color:rgba(255,150,150,.65);display:flex;align-items:flex-start;gap:6px;line-height:1.5;}}
.sim-slide-wrap{{padding:0 10px 10px;}}
.sim-slide-labels{{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;}}
.sim-slide-cur{{font-family:'DM Mono',monospace;font-size:10px;font-weight:700;color:#ffd166;background:rgba(245,197,24,.1);border:1px solid rgba(245,197,24,.28);border-radius:5px;padding:2px 8px;}}
.sim-slide-edge{{font-size:8px;color:rgba(255,255,255,.25);}}
input.sim-range{{width:100%;height:4px;border-radius:2px;outline:none;border:none;-webkit-appearance:none;cursor:pointer;background:linear-gradient(90deg,#ffd166 var(--pct,50%),rgba(255,255,255,.09) var(--pct,50%));}}
input.sim-range::-webkit-slider-thumb{{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:#ffd166;border:2px solid var(--bg);box-shadow:0 0 6px rgba(245,197,24,.45);cursor:pointer;}}

/* ── PAYOFF ── */
.payoff-wrap{{position:relative;height:320px;padding:16px;background:var(--bg2);border-radius:0 0 11px 11px;}}
canvas#payoffChart{{width:100%!important;height:288px!important;}}

/* ── EMPTY STATE ── */
.empty{{text-align:center;padding:50px 20px;color:var(--text3);grid-column:1/-1;}}
.empty-icon{{font-size:42px;margin-bottom:12px;opacity:.3;}}
.empty p{{font-size:11px;font-family:'DM Mono',monospace;line-height:1.8;}}

/* ── FOOTER ── */
.footer{{text-align:center;padding:16px;font-size:10px;color:var(--text3);font-family:'DM Mono',monospace;border-top:1px solid var(--border);margin-top:6px;}}

/* ── ANIMATIONS ── */
@keyframes pulse{{0%,100%{{opacity:1;box-shadow:0 0 0 0 #00c89655;}}50%{{opacity:.6;box-shadow:0 0 0 5px transparent;}}}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(14px);}}to{{opacity:1;transform:translateY(0);}}}}
.strat-card{{animation:fadeUp .35s ease both;}}
.strat-card:nth-child(2){{animation-delay:.05s;}}
.strat-card:nth-child(3){{animation-delay:.10s;}}
.strat-card:nth-child(4){{animation-delay:.15s;}}
.strat-card:nth-child(5){{animation-delay:.20s;}}
.strat-card:nth-child(6){{animation-delay:.25s;}}
::-webkit-scrollbar{{width:5px;height:5px;}}
::-webkit-scrollbar-track{{background:var(--bg);}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:3px;}}

/* ── RESPONSIVE ── */
@media(max-width:1100px){{
  .main{{grid-template-columns:300px 1fr!important;}}
  .greeks-panel{{display:none;}}
  .ticker{{grid-template-columns:repeat(2,1fr);}}
}}
@media(max-width:800px){{
  .main{{grid-template-columns:1fr!important;}}
  .ticker{{grid-template-columns:repeat(2,1fr);}}
  .strat-grid{{grid-template-columns:1fr!important;}}
  .hdr{{flex-direction:column;gap:10px;align-items:flex-start;}}
  .hdr>div:last-child{{align-self:flex-end;}}
  .wrap{{padding:10px;}}
  .sec-hdr{{flex-direction:column;gap:8px;align-items:flex-start;}}
  .panel-body{{padding:12px;}}
}}
@media(max-width:500px){{
  .ticker{{grid-template-columns:1fr 1fr;gap:8px;}}
  .tick-val{{font-size:17px;}}
  .hdr h1{{font-size:16px;}}
  .bias-row{{grid-template-columns:repeat(3,1fr);}}
  .sc-fields{{grid-template-columns:1fr 1fr;}}
}}
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
      <div style="font-size:9px;color:var(--text3);font-family:'DM Mono',monospace;">GENERATED</div>
      <div class="gen-time" style="font-size:11px;color:var(--text2);font-family:'DM Mono',monospace;">{generated_at}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:9px;color:var(--text3);font-family:'DM Mono',monospace;">IST TIME</div>
      <div style="font-size:14px;font-weight:700;color:var(--cyan);font-family:'DM Mono',monospace;letter-spacing:1px;" id="istClock">--:--:--</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:9px;color:var(--text3);font-family:'DM Mono',monospace;">NEXT REFRESH</div>
      <div style="font-size:11px;color:var(--cyan);font-family:'DM Mono',monospace;" id="countdown">30s</div>
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

  <!-- INPUT PANEL -->
  <div class="panel">
    <div class="panel-hdr">
      <div class="panel-title">⚙ Strategy Parameters</div>
    </div>
    <div class="panel-body">

      <!-- ═══ PARAM PANEL TABS ═══ -->
      <div class="ptab-row">
        <button class="ptab active" id="ptab-sr" onclick="switchParamTab('sr',this)">⚡ S/R Strategy</button>
        <button class="ptab ptab-be" id="ptab-be" onclick="switchParamTab('be',this)">🎯 Target BE</button>
      </div>

      <!-- ═══ TAB 1: S/R CONTENT (existing) ═══ -->
      <div id="param-sr-content">

        <div class="form-grp">
          <label class="form-lbl">Market Bias</label>
          <div class="bias-row">
            <button class="bias-btn" id="btnBull" onclick="setBias('bullish')">🐂 Bullish</button>
            <button class="bias-btn" id="btnBear" onclick="setBias('bearish')">🐻 Bearish</button>
            <button class="bias-btn" id="btnNeut" onclick="setBias('neutral')">⚖️ Neutral</button>
          </div>
        </div>

        <div class="divider"></div>

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
          <div class="form-grp">
            <label class="form-lbl">Lot Size</label>
            <input type="number" class="inp" id="lotSize" value="65"/>
          </div>
          <div class="form-grp">
            <label class="form-lbl">Max Capital ₹</label>
            <input type="number" class="inp" id="maxCap" placeholder="500000"/>
          </div>
        </div>

        <button class="analyze-btn" onclick="analyze()">⚡ Analyze Strategies</button>

      </div><!-- end param-sr-content -->

      <!-- ═══ TAB 2: TARGET BE CONTENT (NEW) ═══ -->
      <div id="param-be-content" style="display:none;">

        <div class="be-info-box">
          🎯 Enter your <strong>desired break-even prices</strong>.<br>
          The engine will scan every live NSE strike and find the best matching strategy + exact legs to get you closest to those BEs.
        </div>

        <div class="form-grp">
          <label class="form-lbl" style="color:var(--red);">▼ Lower Break-Even</label>
          <input type="number" class="inp inp-gold" id="beLower" placeholder="e.g. 24100"
                 style="border-color:rgba(255,107,107,.4);" oninput="saveUserState()"/>
          <div style="font-size:9px;color:var(--text3);margin-top:4px;font-family:'DM Mono',monospace;">
            The lowest price where you want to break even
          </div>
        </div>

        <div class="form-grp">
          <label class="form-lbl" style="color:var(--green);">▲ Upper Break-Even</label>
          <input type="number" class="inp inp-gold" id="beUpper" placeholder="e.g. 25380"
                 style="border-color:rgba(0,200,150,.4);" oninput="saveUserState()"/>
          <div style="font-size:9px;color:var(--text3);margin-top:4px;font-family:'DM Mono',monospace;">
            The highest price where you want to break even
          </div>
        </div>

        <div style="background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin-bottom:12px;" id="beCalcPreview">
          <div style="font-size:9px;color:var(--text3);font-family:'DM Mono',monospace;text-align:center;">
            Enter both BEs above to see required premium
          </div>
        </div>

        <div class="form-grp">
          <label class="form-lbl">Expiry</label>
          <select class="sel" id="expirySelBe" onchange="saveUserState()"></select>
        </div>

        <button class="be-find-btn" onclick="findBEStrategies()">🎯 Find Matching Strategies</button>

        <!-- Results appear here -->
        <div class="be-results-wrap" id="beResultsWrap">
          <div class="be-empty">Enter your target break-even levels<br>and click Find to see matching strategies</div>
        </div>

      </div><!-- end param-be-content -->

    </div>
  </div>

  <!-- OPTION CHAIN -->
  <div class="panel">
    <div class="panel-hdr">
      <div class="panel-title">📊 Live Option Chain</div>
      <div style="display:flex;align-items:center;gap:14px;">
        <span style="font-size:10px;color:var(--text2);font-family:'JetBrains Mono',monospace;">
          SPOT <b style="color:var(--cyan);" id="chainSpotLbl">—</b>
        </span>
        <span style="font-size:10px;color:var(--text2);font-family:'JetBrains Mono',monospace;">
          DTE <b style="color:var(--gold);" id="chainDteLbl">—</b>
        </span>
        <span style="font-size:10px;color:var(--text2);font-family:'JetBrains Mono',monospace;" id="chainExpLbl"></span>
      </div>
    </div>
    <div class="chain-wrap">
      <div class="chain-side-hdr">
        <div class="ce-hdr">── CALLS (CE) ──</div>
        <div class="st-hdr">STRIKE</div>
        <div class="pe-hdr">── PUTS (PE) ──</div>
      </div>
      <div class="chain-col-hdr">
        <div class="ce-cols">
          <span>LTP</span><span>IV%</span><span>OI(L)</span><span>ΔOI</span>
        </div>
        <div></div>
        <div class="pe-cols">
          <span>ΔOI</span><span>OI(L)</span><span>IV%</span><span>LTP</span>
        </div>
      </div>
      <div id="chainBody"><div style="text-align:center;padding:50px;color:var(--text3);font-family:'JetBrains Mono',monospace;font-size:11px;">Loading…</div></div>
    </div>
  </div>

  <!-- GREEKS SIDEBAR -->
  <div class="panel greeks-panel" id="greeksPanel">
    <div class="greeks-title">▲ GREEKS <span class="g-exp-tag" id="greeksExpTag">—</span></div>
    <div class="g-sel-wrap">
      <select class="g-sel" id="greeksStrikeSel" onchange="updateGreeksForStrike(this.value)"></select>
    </div>
    <div class="g-atm-badge" id="greeksAtmBadge">
      <div>
        <div class="g-strike-type" id="greeksTypeLabel">ATM</div>
        <div class="g-strike-val" id="greeksStrikeVal">—</div>
      </div>
      <div class="g-ltp-row">
        <span class="g-ce-ltp" id="greeksCeLtp">CE —</span>
        <span style="color:var(--text3);font-size:9px;">/</span>
        <span class="g-pe-ltp" id="greeksPeLtp">PE —</span>
      </div>
    </div>
    <div class="g-row">
      <div class="g-label"><span class="g-name">Δ Delta</span><span class="g-sub">CE / PE</span></div>
      <div class="g-vals" id="greeksDeltaWrap">
        <div class="delta-bar-wrap"><div class="delta-bar-track"><div class="delta-bar-fill" id="dbarCe" style="background:var(--green);"></div></div><span class="g-ce-val" id="greeksDeltaCe">—</span></div>
        <div class="delta-bar-wrap"><div class="delta-bar-track"><div class="delta-bar-fill" id="dbarPe" style="background:var(--red);"></div></div><span class="g-pe-val" id="greeksDeltaPe">—</span></div>
      </div>
    </div>
    <div class="g-row">
      <div class="g-label"><span class="g-name">σ IV</span><span class="skew-lbl" id="greeksSkewLbl" style="color:var(--purple);">—</span></div>
      <div class="g-vals">
        <div style="display:flex;align-items:center;gap:5px;"><span style="font-size:8px;color:rgba(0,200,220,.8);">CE</span><span class="g-ce-val" id="greeksIvCe">—</span></div>
        <div style="display:flex;align-items:center;gap:5px;"><span style="font-size:8px;color:rgba(255,144,144,.8);">PE</span><span class="g-pe-val" id="greeksIvPe">—</span></div>
      </div>
    </div>
    <div class="g-row">
      <div class="g-label"><span class="g-name">Θ Theta</span><span class="g-sub">per day</span></div>
      <div class="g-vals">
        <div style="display:flex;align-items:center;gap:5px;"><span style="font-size:8px;color:rgba(0,200,220,.8);">CE</span><span class="g-ce-val" id="greeksThetaCe">—</span></div>
        <div style="display:flex;align-items:center;gap:5px;"><span style="font-size:8px;color:rgba(255,144,144,.8);">PE</span><span class="g-pe-val" id="greeksThetaPe">—</span></div>
      </div>
    </div>
    <div class="g-row">
      <div class="g-label"><span class="g-name">ν Vega</span><span class="g-sub">per 1% IV</span></div>
      <div class="g-vals">
        <div style="display:flex;align-items:center;gap:5px;"><span style="font-size:8px;color:rgba(0,200,220,.8);">CE</span><span class="g-ce-val" id="greeksVegaCe">—</span></div>
        <div style="display:flex;align-items:center;gap:5px;"><span style="font-size:8px;color:rgba(255,144,144,.8);">PE</span><span class="g-pe-val" id="greeksVegaPe">—</span></div>
      </div>
    </div>
    <div class="g-row">
      <div class="g-label"><span class="g-name">Γ Gamma</span><span class="g-sub">delta/point</span></div>
      <div class="g-vals">
        <div style="display:flex;align-items:center;gap:5px;"><span style="font-size:8px;color:rgba(0,200,220,.8);">CE</span><span class="g-ce-val" id="greeksGammaCe">—</span></div>
        <div style="display:flex;align-items:center;gap:5px;"><span style="font-size:8px;color:rgba(255,144,144,.8);">PE</span><span class="g-pe-val" id="greeksGammaPe">—</span></div>
      </div>
    </div>
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
    <div class="empty"><div class="empty-icon">📋</div><p>Enter support &amp; resistance levels<br>then click ⚡ Analyze Strategies<br><br>— or —<br><br>Switch to 🎯 Target BE tab<br>to find strategies by break-even</p></div>
  </div>
</div>

<!-- PAYOFF CHART -->
<div class="panel" style="margin-bottom:18px;">
  <div class="panel-hdr">
    <div class="panel-title">📈 Payoff Diagram
      <span style="font-size:9px;color:var(--text3);margin-left:8px;font-family:'JetBrains Mono',monospace;">🟢 Today (BSM) &nbsp;|&nbsp; 🔵 At Expiry &nbsp;|&nbsp; bars = OI</span>
    </div>
    <select class="sel" style="width:200px;" id="payoffSel" onchange="drawPayoff()">
      <option value="">— Select Strategy —</option>
    </select>
  </div>
  <div id="payoffStats" style="display:none;grid-template-columns:repeat(4,1fr);gap:10px;padding:14px 16px 0;"></div>
  <div class="payoff-wrap" style="height:320px;padding:16px;position:relative;">
    <canvas id="payoffChart"></canvas>
  </div>
  <div id="payoffTooltip" style="
    display:none;position:fixed;z-index:9999;
    background:rgba(8,18,30,0.97);border:1px solid rgba(0,212,255,0.28);
    border-radius:10px;padding:13px 16px;min-width:250px;max-width:280px;
    box-shadow:0 8px 36px rgba(0,0,0,0.7);pointer-events:none;
    font-family:'JetBrains Mono',monospace;backdrop-filter:blur(8px);
  "></div>
  <div id="payoffFooter" style="display:none;padding:10px 16px 12px;border-top:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="font-size:9px;color:var(--text2);font-family:'JetBrains Mono',monospace;letter-spacing:1px;">BREAKEVENS</span>
      <div id="beBadges" style="display:flex;gap:6px;"></div>
    </div>
    <div style="font-size:11px;font-family:'JetBrains Mono',monospace;color:var(--text2);" id="projBadge">
      Select a strategy to see projected P&L
    </div>
  </div>
  <div id="projBadgeFallback" style="text-align:center;padding:10px 16px 14px;font-size:11px;font-family:'JetBrains Mono',monospace;border-top:1px solid var(--border);color:var(--text2);">
    Select a strategy to see projected P&L
  </div>
</div>

<div class="footer">NIFTY OPTIONS ANALYZER · NSE INDIA DATA · FOR EDUCATIONAL PURPOSE ONLY · NOT FINANCIAL ADVICE</div>
</div>

<!-- ═══════════════════════════════════════════════════════════
     JAVASCRIPT
═══════════════════════════════════════════════════════════ -->
<script>
// ── Baked-in data from Python ─────────────────────────────────
const ALL_DATA    = {data_json};
const EXPIRY_LIST = {expiry_list_j};
const LOT_SIZE    = 65;
const RISK_FREE   = 0.065;

// ── State ─────────────────────────────────────────────────────
let currentExpiry  = EXPIRY_LIST[0] || "";
let marketBias     = "neutral";
let strategies     = [];
let payoffChart    = null;
let activeParamTab = "sr";

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
  // BE preview updates
  document.getElementById("beLower").addEventListener("input", updateBePreview);
  document.getElementById("beUpper").addEventListener("input", updateBePreview);
}};

// ── Param tab switcher ────────────────────────────────────────
function switchParamTab(tab, btn) {{
  activeParamTab = tab;
  document.getElementById("param-sr-content").style.display = tab === "sr" ? "block" : "none";
  document.getElementById("param-be-content").style.display = tab === "be" ? "block" : "none";
  document.querySelectorAll(".ptab").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  saveUserState();
}}

// ── BE Preview calculation ────────────────────────────────────
function updateBePreview() {{
  const lower = parseFloat(document.getElementById("beLower").value);
  const upper = parseFloat(document.getElementById("beUpper").value);
  const prev  = document.getElementById("beCalcPreview");
  if (isNaN(lower) || isNaN(upper) || lower <= 0 || upper <= 0) {{
    prev.innerHTML = '<div style="font-size:9px;color:var(--text3);font-family:\'DM Mono\',monospace;text-align:center;">Enter both BEs above to see required premium</div>';
    return;
  }}
  if (upper <= lower) {{
    prev.innerHTML = '<div style="font-size:9px;color:var(--red);font-family:\'DM Mono\',monospace;text-align:center;">⚠ Upper BE must be greater than Lower BE</div>';
    return;
  }}
  const range      = upper - lower;
  const center     = (upper + lower) / 2;
  const stradPrem  = (range / 2).toFixed(2);
  const d          = ALL_DATA[currentExpiry || EXPIRY_LIST[0]];
  const spot       = d ? d.underlying : 0;
  prev.innerHTML = `
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
      <div>
        <div style="font-size:8px;color:var(--text3);font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:.7px;">BE Range</div>
        <div style="font-size:13px;font-weight:700;color:var(--gold);font-family:'DM Mono',monospace;">${{range.toFixed(0)}} pts</div>
      </div>
      <div>
        <div style="font-size:8px;color:var(--text3);font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:.7px;">Center Strike</div>
        <div style="font-size:13px;font-weight:700;color:var(--cyan);font-family:'DM Mono',monospace;">~₹${{center.toFixed(0)}}</div>
      </div>
      <div>
        <div style="font-size:8px;color:var(--text3);font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:.7px;">Min Premium Needed</div>
        <div style="font-size:13px;font-weight:700;color:var(--green);font-family:'DM Mono',monospace;">₹${{stradPrem}}</div>
      </div>
      <div>
        <div style="font-size:8px;color:var(--text3);font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:.7px;">Spot vs Center</div>
        <div style="font-size:11px;font-weight:700;font-family:'DM Mono',monospace;color:var(--text2);">${{spot ? (spot > center ? "▲ +" : "▼ ") + Math.abs(spot - center).toFixed(0) + " pts" : "—"}}</div>
      </div>
    </div>`;
}}

// ── Silent background refresh ─────────────────────────────────
function startCountdown() {{
  function updateClock() {{
    const now = new Date();
    const ist = new Date(now.getTime() + (5.5 * 60 * 60 * 1000));
    const hh  = String(ist.getUTCHours()).padStart(2, "0");
    const mm  = String(ist.getUTCMinutes()).padStart(2, "0");
    const ss  = String(ist.getUTCSeconds()).padStart(2, "0");
    const el  = document.getElementById("istClock");
    if (el) el.textContent = hh + ":" + mm + ":" + ss;
  }}
  updateClock();
  setInterval(updateClock, 1000);
  let secs = 30;
  const el = document.getElementById("countdown");
  setInterval(() => {{
    secs--;
    if (secs <= 0) {{ secs = 30; silentRefresh(); }}
    el.textContent = secs + "s";
  }}, 1000);
}}

async function silentRefresh() {{
  try {{
    const resp = await fetch(location.href + "?_=" + Date.now(), {{cache:"no-store"}});
    if (!resp.ok) return;
    const html   = await resp.text();
    const parser = new DOMParser();
    const doc    = parser.parseFromString(html, "text/html");
    const scripts = doc.querySelectorAll("script");
    for (const s of scripts) {{
      const m = s.textContent.match(/const ALL_DATA\s*=\s*(\{{[\s\S]*?\}});/);
      if (m) {{
        try {{
          const freshData = JSON.parse(m[1]);
          Object.assign(ALL_DATA, freshData);
          updateTicker();
          renderChain();
          updateGreeksForStrike(parseInt(document.getElementById("greeksStrikeSel").value) || ALL_DATA[currentExpiry]?.atm_strike);
          const genEl = doc.querySelector(".gen-time");
          if (genEl) document.querySelector(".gen-time") && (document.querySelector(".gen-time").textContent = genEl.textContent);
        }} catch(e) {{ console.warn("Silent refresh parse error:", e); }}
        break;
      }}
    }}
  }} catch(e) {{ console.warn("Silent refresh fetch error:", e); }}
}}

// ── Expiry ────────────────────────────────────────────────────
function populateExpiries() {{
  const sel   = document.getElementById("expirySel");
  const selBe = document.getElementById("expirySelBe");
  const opts  = EXPIRY_LIST.map((e, i) =>
    `<option value="${{e}}"${{i===0?' selected':''}}>${{e}}${{i===0?' (Weekly)':''}}</option>`
  ).join("");
  sel.innerHTML   = opts;
  selBe.innerHTML = opts;
  currentExpiry = EXPIRY_LIST[0] || "";
}}

function onExpiryChange() {{
  currentExpiry = document.getElementById("expirySel").value;
  document.getElementById("expirySelBe").value = currentExpiry;
  updateTicker();
  renderChain();
  renderGreeks(null);
  updateBePreview();
  saveUserState();
}}

// ── Ticker ────────────────────────────────────────────────────
function updateTicker() {{
  const d = ALL_DATA[currentExpiry];
  if (!d) return;
  document.getElementById("spotVal").textContent = "₹" + d.underlying.toLocaleString("en-IN");
  document.getElementById("atmVal").textContent  = d.atm_strike.toLocaleString("en-IN");
  document.getElementById("spotDte").textContent = "DTE: " + d.dte;
  const atm = d.all_strikes.find(s => s.is_atm);
  if (atm) {{
    const ivAvg = ((atm.ce_iv || 0) + (atm.pe_iv || 0)) / 2;
    document.getElementById("atmIv").textContent = "ATM IV: " + ivAvg.toFixed(1) + "%";
  }}
  let maxCeOi = 0, maxCeSt = 0, maxPeOi = 0, maxPeSt = 0;
  d.all_strikes.forEach(s => {{
    if (s.ce_oi > maxCeOi) {{ maxCeOi = s.ce_oi; maxCeSt = s.strike; }}
    if (s.pe_oi > maxPeOi) {{ maxPeOi = s.pe_oi; maxPeSt = s.strike; }}
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
  if (chainSpot) chainSpot.textContent = "₹" + d.underlying.toLocaleString("en-IN");
  if (chainDte)  chainDte.textContent  = d.dte;
  const supports    = getSupports();
  const resistances = getResistances();
  const rows        = [...d.all_strikes].sort((a, b) => b.strike - a.strike);
  const maxOi       = Math.max(...rows.flatMap(r => [r.ce_oi, r.pe_oi]), 1);
  const chainBody = document.getElementById("chainBody");
  chainBody.innerHTML = rows.map(r => {{
    const isAtm = r.is_atm;
    const isSup = supports.includes(r.strike);
    const isRes = resistances.includes(r.strike);
    const ceOiL = (r.ce_oi / 1e5).toFixed(1);
    const peOiL = (r.pe_oi / 1e5).toFixed(1);
    const ceHeat = Math.round((r.ce_oi / maxOi) * 100);
    const peHeat = Math.round((r.pe_oi / maxOi) * 100);
    const ceChgStr = (r.ce_oi_chg >= 0 ? "+" : "") + (r.ce_oi_chg/1e5).toFixed(1) + "L";
    const peChgStr = (r.pe_oi_chg >= 0 ? "+" : "") + (r.pe_oi_chg/1e5).toFixed(1) + "L";
    const ceChgCls = r.ce_oi_chg >= 0 ? "up" : "down";
    const peChgCls = r.pe_oi_chg >= 0 ? "up" : "down";
    const rc       = isAtm ? "atm-row" : isSup ? "sup-row" : isRes ? "res-row" : "";
    const smark    = isSup ? '<span style="color:var(--green);font-size:7px;font-weight:700;letter-spacing:.5px;">▲ SUP</span>' : "";
    const rmark    = isRes ? '<span style="color:var(--red);font-size:7px;font-weight:700;letter-spacing:.5px;">▼ RES</span>' : "";
    return `<div class="chain-row ${{rc}}">
      <div class="ce-side">
        <div class="ce-heat-bg" style="width:${{ceHeat}}%"></div>
        <span class="cv-ltp ce-ltp-v">${{r.ce_ltp.toFixed(2)}}</span>
        <span class="cv-iv">${{r.ce_iv.toFixed(1)}}%</span>
        <span class="cv-oi">${{ceOiL}}L</span>
        <span class="cv-doi ${{ceChgCls}}">${{ceChgStr}}</span>
      </div>
      <div class="stk-cell">
        ${{isAtm ? '<span class="atm-tag">ATM</span>' : ""}}
        <span style="color:${{isAtm ? "var(--cyan)" : "var(--text)"}}">${{r.strike.toLocaleString("en-IN")}}</span>
        ${{smark || rmark ? `<span style="display:flex;gap:4px;">${{smark}}${{rmark}}</span>` : ""}}
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

// ── Greeks ────────────────────────────────────────────────────
function renderGreeks(strikeOverride) {{
  const d = ALL_DATA[currentExpiry];
  if (!d) return;
  document.getElementById("greeksExpTag").textContent = currentExpiry;
  const sel   = document.getElementById("greeksStrikeSel");
  const atm   = d.atm_strike;
  const rows  = [...d.all_strikes].sort((a, b) => b.strike - a.strike);
  let ceOpts = "", atmOpt = "", peOpts = "";
  rows.forEach(r => {{
    const s    = r.strike;
    const dist = Math.abs(s - atm) / 50;
    if (r.is_atm) {{
      atmOpt = `<option value="${{s}}" ${{!strikeOverride?'selected':''}}>\u2605 ATM ₹${{s.toLocaleString("en-IN")}}</option>`;
    }} else if (s > atm) {{
      ceOpts += `<option value="${{s}}" ${{strikeOverride==s?'selected':''}}>\u25b2 CE+${{dist}} ₹${{s.toLocaleString("en-IN")}}</option>`;
    }} else {{
      peOpts += `<option value="${{s}}" ${{strikeOverride==s?'selected':''}}>\u25bc PE-${{dist}} ₹${{s.toLocaleString("en-IN")}}</option>`;
    }}
  }});
  sel.innerHTML =
    `<optgroup label="─ OTM CALLS (CE) ─">${{ceOpts}}</optgroup>` +
    `<optgroup label="─ ATM ─">${{atmOpt}}</optgroup>` +
    `<optgroup label="─ OTM PUTS (PE) ─">${{peOpts}}</optgroup>`;
  const targetStrike = strikeOverride || atm;
  updateGreeksForStrike(targetStrike);
}}

function updateGreeksForStrike(strike) {{
  strike = parseInt(strike);
  const d = ALL_DATA[currentExpiry];
  if (!d) return;
  const row = d.all_strikes.find(s => s.strike === strike);
  if (!row) return;
  const atm      = d.atm_strike;
  const dist     = Math.abs(strike - atm) / 50;
  const typeLabel = row.is_atm ? "ATM" : strike > atm ? `OTM CE +${{dist}}` : `OTM PE -${{dist}}`;
  document.getElementById("greeksTypeLabel").textContent  = typeLabel;
  document.getElementById("greeksStrikeVal").textContent  = "₹" + strike.toLocaleString("en-IN");
  document.getElementById("greeksCeLtp").textContent      = "CE ₹" + row.ce_ltp.toFixed(1);
  document.getElementById("greeksPeLtp").textContent      = "PE ₹" + row.pe_ltp.toFixed(1);
  document.getElementById("greeksDeltaCe").textContent    = (row.ce_delta >= 0 ? "+" : "") + row.ce_delta.toFixed(3);
  document.getElementById("greeksDeltaPe").textContent    = (row.pe_delta >= 0 ? "+" : "") + row.pe_delta.toFixed(3);
  document.getElementById("greeksIvCe").textContent       = row.ce_iv.toFixed(1) + "%";
  document.getElementById("greeksIvPe").textContent       = row.pe_iv.toFixed(1) + "%";
  document.getElementById("greeksThetaCe").textContent    = "₹" + Math.abs(row.ce_theta).toFixed(2);
  document.getElementById("greeksThetaPe").textContent    = "₹" + Math.abs(row.pe_theta).toFixed(2);
  document.getElementById("greeksVegaCe").textContent     = row.ce_vega.toFixed(4);
  document.getElementById("greeksVegaPe").textContent     = row.pe_vega.toFixed(4);
  document.getElementById("greeksGammaCe").textContent    = row.ce_gamma.toFixed(6);
  document.getElementById("greeksGammaPe").textContent    = row.pe_gamma.toFixed(6);
  document.getElementById("dbarCe").style.width = Math.abs(row.ce_delta) * 100 + "%";
  document.getElementById("dbarPe").style.width = Math.abs(row.pe_delta) * 100 + "%";
  const ivAvg   = (row.ce_iv + row.pe_iv) / 2;
  const ivPct   = Math.min(100, (ivAvg / 60) * 100);
  const ivCol   = ivAvg > 25 ? "var(--red)" : ivAvg > 18 ? "var(--gold)" : "var(--green)";
  const ivReg   = ivAvg > 25 ? "High IV · Buy Premium" : ivAvg > 15 ? "Normal IV · Balanced" : "Low IV · Sell Premium";
  document.getElementById("greeksIvAvg").textContent     = ivAvg.toFixed(1) + "%";
  document.getElementById("greeksIvAvg").style.color     = ivCol;
  document.getElementById("greeksIvBar").style.width     = ivPct.toFixed(1) + "%";
  document.getElementById("greeksIvBar").style.background = ivCol;
  document.getElementById("greeksIvBar").style.boxShadow = `0 0 6px ${{ivCol}}88`;
  document.getElementById("greeksIvRegime").textContent  = ivReg;
  document.getElementById("greeksIvRegime").style.color  = ivCol;
  const skew    = row.pe_iv - row.ce_iv;
  const skewCol = skew > 1.5 ? "var(--red)" : skew < -1.5 ? "var(--green)" : "var(--purple)";
  const skewTxt = skew > 0 ? `PE Skew +${{skew.toFixed(1)}}` : `CE Skew ${{skew.toFixed(1)}}`;
  document.getElementById("greeksSkewLbl").textContent = skewTxt;
  document.getElementById("greeksSkewLbl").style.color = skewCol;
}}

// ── Bias ──────────────────────────────────────────────────────
function setBias(b) {{
  marketBias = b;
  document.getElementById("btnBull").className = "bias-btn" + (b==="bullish"?" bias-bull":"");
  document.getElementById("btnBear").className = "bias-btn" + (b==="bearish"?" bias-bear":"");
  document.getElementById("btnNeut").className = "bias-btn" + (b==="neutral"?" bias-neut":"");
  saveUserState();
}}
setBias("neutral");

// ── Persist user state ────────────────────────────────────────
function saveUserState() {{
  const state = {{
    bias:        marketBias,
    expiry:      currentExpiry,
    supports:    getSupports(),
    resistances: getResistances(),
    lotSize:     document.getElementById("lotSize").value,
    maxCap:      document.getElementById("maxCap").value,
    activeTab:   activeParamTab,
    beLower:     document.getElementById("beLower").value,
    beUpper:     document.getElementById("beUpper").value,
  }};
  try {{ sessionStorage.setItem("noa_state", JSON.stringify(state)); }} catch(e) {{}}
}}

function restoreUserState() {{
  try {{
    const raw = sessionStorage.getItem("noa_state");
    if (!raw) return;
    const state = JSON.parse(raw);
    if (state.bias) setBias(state.bias);
    if (state.expiry && EXPIRY_LIST.includes(state.expiry)) {{
      currentExpiry = state.expiry;
      document.getElementById("expirySel").value    = state.expiry;
      document.getElementById("expirySelBe").value  = state.expiry;
    }}
    if (state.supports && state.supports.length) {{
      const supC = document.getElementById("supContainer");
      supC.innerHTML = "";
      state.supports.forEach(v => {{
        const div = document.createElement("div");
        div.className = "sr-row";
        div.innerHTML = `<input type="number" class="inp sup-inp" value="${{v}}" placeholder="e.g. 21800"/><button class="rm-btn" onclick="rmRow(this)">✕</button>`;
        supC.appendChild(div);
      }});
    }}
    if (state.resistances && state.resistances.length) {{
      const resC = document.getElementById("resContainer");
      resC.innerHTML = "";
      state.resistances.forEach(v => {{
        const div = document.createElement("div");
        div.className = "sr-row";
        div.innerHTML = `<input type="number" class="inp res-inp" value="${{v}}" placeholder="e.g. 22600"/><button class="rm-btn" onclick="rmRow(this)">✕</button>`;
        resC.appendChild(div);
      }});
    }}
    if (state.lotSize) document.getElementById("lotSize").value = state.lotSize;
    if (state.maxCap)  document.getElementById("maxCap").value  = state.maxCap;
    if (state.beLower) document.getElementById("beLower").value = state.beLower;
    if (state.beUpper) document.getElementById("beUpper").value = state.beUpper;
    if (state.activeTab && state.activeTab === "be") {{
      const btn = document.getElementById("ptab-be");
      if (btn) switchParamTab("be", btn);
    }}
  }} catch(e) {{ console.warn("State restore error:", e); }}
}}

// ── S/R Input ─────────────────────────────────────────────────
function addLevel(type) {{
  const c   = document.getElementById(type==="sup"?"supContainer":"resContainer");
  const cls = type==="sup"?"sup-inp":"res-inp";
  const ph  = type==="sup"?"e.g. 21800":"e.g. 22600";
  const div = document.createElement("div");
  div.className = "sr-row";
  div.innerHTML = `<input type="number" class="inp ${{cls}}" placeholder="${{ph}}" oninput="saveUserState()"/><button class="rm-btn" onclick="rmRow(this)">✕</button>`;
  c.appendChild(div);
}}
function rmRow(btn) {{
  const row = btn.parentElement;
  if (row.parentElement.children.length > 1) row.remove();
  else row.querySelector("input").value = "";
  saveUserState();
}}
function getSupports()    {{ return [...document.querySelectorAll(".sup-inp")].map(i=>parseFloat(i.value)).filter(v=>!isNaN(v)&&v>0); }}
function getResistances() {{ return [...document.querySelectorAll(".res-inp")].map(i=>parseFloat(i.value)).filter(v=>!isNaN(v)&&v>0); }}

// ── BSM (JS) ─────────────────────────────────────────────────
function normCdf(x) {{
  const a1=.254829592,a2=-.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=.3275911;
  const sign = x<0?-1:1;
  const t = 1/(1+p*Math.abs(x));
  const y = 1-(((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*Math.exp(-x*x/2);
  return 0.5*(1+sign*y);
}}
function normPdf(x) {{ return Math.exp(-0.5*x*x)/Math.sqrt(2*Math.PI); }}

function bsm(S,K,T,r,sigma,type) {{
  if(T<=0||sigma<=0) {{ const intr=type==="CE"?Math.max(0,S-K):Math.max(0,K-S); return {{price:intr,delta:0,pop:0}}; }}
  const d1=(Math.log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*Math.sqrt(T));
  const d2=d1-sigma*Math.sqrt(T);
  const price=type==="CE"?S*normCdf(d1)-K*Math.exp(-r*T)*normCdf(d2):K*Math.exp(-r*T)*normCdf(-d2)-S*normCdf(-d1);
  const delta=type==="CE"?normCdf(d1):normCdf(d1)-1;
  const pop  =type==="CE"?normCdf(d2):normCdf(-d2);
  return {{price:Math.max(0,price),delta,pop}};
}}

// =================================================================
// 🎯 TARGET BE ENGINE — Find best strategy matching your break-evens
// =================================================================

function findBEStrategies() {{
  const expKey    = document.getElementById("expirySelBe").value || currentExpiry;
  const d         = ALL_DATA[expKey];
  if (!d) {{ alert("No data loaded for this expiry."); return; }}

  const lowerBE = parseFloat(document.getElementById("beLower").value);
  const upperBE = parseFloat(document.getElementById("beUpper").value);

  if (isNaN(lowerBE) || isNaN(upperBE) || lowerBE <= 0 || upperBE <= 0) {{
    alert("Please enter both Lower and Upper Break-Even values.");
    return;
  }}
  if (upperBE <= lowerBE) {{
    alert("Upper BE must be greater than Lower BE.");
    return;
  }}

  const underlying = d.underlying;
  const dte        = d.dte;
  const T          = Math.max(dte / 365, 0.001);
  const strikes    = d.all_strikes;
  const smap       = {{}};
  strikes.forEach(s => {{ smap[s.strike] = s; }});
  const allSt      = strikes.map(s => s.strike).sort((a,b) => a - b);
  const nearest    = val => allSt.reduce((a,b) => Math.abs(b-val) < Math.abs(a-val) ? b : a);
  const get        = (st, f, def=0) => (smap[st]||{{}})[f] || def;

  const targetCenter = (upperBE + lowerBE) / 2;
  const targetRange  = upperBE - lowerBE;
  const candidates   = [];

  // ─────────────────────────────────────────────────────────────
  // Helper: compute actual BEs from a leg array via payoff curve
  // ─────────────────────────────────────────────────────────────
  function computeActualBEs(legs) {{
    const range = [];
    for (let p = underlying - 2000; p <= underlying + 2000; p += 10) range.push(p);
    const payoffs = range.map(price => {{
      let pnl = 0;
      legs.forEach(l => {{
        const intr = l.type === "CE" ? Math.max(0, price - l.strike) : Math.max(0, l.strike - price);
        pnl += l.action === "buy" ? (intr - l.premium) : (l.premium - intr);
      }});
      return pnl;
    }});
    const bes = [];
    for (let i = 0; i < payoffs.length - 1; i++) {{
      if ((payoffs[i] < 0) !== (payoffs[i+1] < 0)) {{
        const frac = Math.abs(payoffs[i]) / (Math.abs(payoffs[i]) + Math.abs(payoffs[i+1]));
        bes.push(Math.round(range[i] + frac * (range[i+1] - range[i])));
      }}
    }}
    return bes;
  }}

  function makeCandidate(name, legs, icon) {{
    const netPrem = legs.reduce((a,l) => a + (l.action==="sell" ? l.premium : -l.premium), 0);
    if (netPrem <= 0) return null; // only credit strategies for BE targeting
    const bes = computeActualBEs(legs);
    if (bes.length < 1) return null;
    const actualLower = bes[0];
    const actualUpper = bes.length >= 2 ? bes[1] : bes[0];
    const lowerDiff   = Math.abs(actualLower - lowerBE);
    const upperDiff   = bes.length >= 2 ? Math.abs(actualUpper - upperBE) : 9999;
    const totalDiff   = lowerDiff + upperDiff;
    // accuracy score: 0 = perfect match, higher = worse
    const accuracy    = Math.max(0, 100 - (totalDiff / targetRange) * 100);
    const maxPayoff   = [];
    for (let p = underlying-2000; p<=underlying+2000; p+=25) {{
      let pnl=0;
      legs.forEach(l=>{{
        const intr=l.type==="CE"?Math.max(0,p-l.strike):Math.max(0,l.strike-p);
        pnl+=l.action==="buy"?(intr-l.premium):(l.premium-intr);
      }});
      maxPayoff.push(pnl*LOT_SIZE);
    }}
    const maxProfit = Math.max(...maxPayoff);
    const maxLoss   = Math.abs(Math.min(...maxPayoff));

    // PoP
    let pop = 50;
    if (bes.length >= 2) {{
      const sigma = ((legs.find(l=>l.type==="CE")?.iv||15) + (legs.find(l=>l.type==="PE")?.iv||15)) / 2 / 100;
      const probBelow = bsm(underlying, actualLower, T, RISK_FREE, sigma, "PE").pop;
      const probAbove = 1 - bsm(underlying, actualUpper, T, RISK_FREE, sigma, "CE").pop;
      pop = Math.round(Math.min((1 - probBelow - probAbove) * 100, 99) * 10) / 10;
    }}

    return {{
      name, legs, icon, netPrem: Math.round(netPrem*100)/100,
      actualLower, actualUpper,
      lowerDiff: Math.round(actualLower - lowerBE),
      upperDiff: bes.length >= 2 ? Math.round(actualUpper - upperBE) : null,
      totalDiff, accuracy: Math.round(accuracy),
      maxProfit: Math.round(maxProfit), maxLoss: Math.round(maxLoss), pop,
      bes,
    }};
  }}

  // ─────────────────────────────────────────────────────────────
  // STRATEGY 1: Short Straddle
  // BE_upper = strike + premium, BE_lower = strike - premium
  // Best strike = center of BEs, need premium ≈ range/2
  // ─────────────────────────────────────────────────────────────
  const stCenter = nearest(targetCenter);
  const stRow    = smap[stCenter];
  if (stRow) {{
    const c = makeCandidate("Short Straddle", [
      {{action:"sell", strike:stCenter, type:"CE", premium:stRow.ce_ltp||0.01, iv:stRow.ce_iv||15}},
      {{action:"sell", strike:stCenter, type:"PE", premium:stRow.pe_ltp||0.01, iv:stRow.pe_iv||15}},
    ], "⚖️");
    if (c) candidates.push(c);
  }}

  // Also try ±1 strike either side of center
  [nearest(targetCenter - 50), nearest(targetCenter + 50)].forEach(st => {{
    if (st === stCenter) return;
    const row = smap[st];
    if (!row) return;
    const c = makeCandidate("Short Straddle", [
      {{action:"sell", strike:st, type:"CE", premium:row.ce_ltp||0.01, iv:row.ce_iv||15}},
      {{action:"sell", strike:st, type:"PE", premium:row.pe_ltp||0.01, iv:row.pe_iv||15}},
    ], "⚖️");
    if (c) candidates.push(c);
  }});

  // ─────────────────────────────────────────────────────────────
  // STRATEGY 2: Short Strangle
  // Sell CE above upper BE area, sell PE below lower BE area
  // BE_upper = sell_CE + total_premium, BE_lower = sell_PE - total_premium
  // Scan: CE strikes from ATM to upperBE, PE strikes from lowerBE to ATM
  // ─────────────────────────────────────────────────────────────
  const ceRange = allSt.filter(s => s >= underlying && s <= upperBE + 300);
  const peRange = allSt.filter(s => s >= lowerBE - 300 && s <= underlying);

  // Try a focused set of CE/PE combinations (most promising strikes)
  const ceCands = ceRange.slice(0, 8);
  const peCands = peRange.slice(-8);

  ceCands.forEach(ces => {{
    peCands.forEach(pes => {{
      if (ces <= pes) return;
      const ceRow = smap[ces], peRow = smap[pes];
      if (!ceRow || !peRow) return;
      const c = makeCandidate("Short Strangle", [
        {{action:"sell", strike:ces, type:"CE", premium:ceRow.ce_ltp||0.01, iv:ceRow.ce_iv||15}},
        {{action:"sell", strike:pes, type:"PE", premium:peRow.pe_ltp||0.01, iv:peRow.pe_iv||15}},
      ], "🔀");
      if (c && c.totalDiff < targetRange * 1.5) candidates.push(c);
    }});
  }});

  // ─────────────────────────────────────────────────────────────
  // STRATEGY 3: Iron Condor
  // Sell CE/PE inside the BEs, buy CE/PE outside for protection
  // BE_upper ≈ sell_CE + net_credit, BE_lower ≈ sell_PE - net_credit
  // ─────────────────────────────────────────────────────────────
  // Best sell CE: scan strikes just below upperBE
  const icCeSells = allSt.filter(s => s >= underlying && s <= upperBE).slice(-4);
  const icPeSells = allSt.filter(s => s >= lowerBE && s <= underlying).slice(0, 4);

  icCeSells.forEach(sceStr => {{
    icPeSells.forEach(speStr => {{
      if (sceStr <= speStr) return;
      const buyC = nearest(sceStr + 150);
      const buyP = nearest(speStr - 150);
      if (buyC === sceStr || buyP === speStr) return;
      const sceRow = smap[sceStr], speRow = smap[speStr];
      const bcRow  = smap[buyC],   bpRow  = smap[buyP];
      if (!sceRow || !speRow || !bcRow || !bpRow) return;
      const c = makeCandidate("Iron Condor", [
        {{action:"sell", strike:sceStr, type:"CE", premium:sceRow.ce_ltp||0.01, iv:sceRow.ce_iv||15}},
        {{action:"buy",  strike:buyC,   type:"CE", premium:bcRow.ce_ltp||0.01,  iv:bcRow.ce_iv||15}},
        {{action:"sell", strike:speStr, type:"PE", premium:speRow.pe_ltp||0.01, iv:speRow.pe_iv||15}},
        {{action:"buy",  strike:buyP,   type:"PE", premium:bpRow.pe_ltp||0.01,  iv:bpRow.pe_iv||15}},
      ], "🦅");
      if (c && c.totalDiff < targetRange * 1.5) candidates.push(c);
    }});
  }});

  // ─────────────────────────────────────────────────────────────
  // STRATEGY 4: Bull Put Spread (lower BE focus)
  // BE = sell_strike - net_credit
  // Scan sell strikes; find where actual BE lands near lowerBE
  // ─────────────────────────────────────────────────────────────
  allSt.filter(s => s >= lowerBE - 100 && s <= underlying + 100).forEach(sellSt => {{
    const buySt = nearest(sellSt - 150);
    if (buySt === sellSt) return;
    const sRow = smap[sellSt], bRow = smap[buySt];
    if (!sRow || !bRow) return;
    const c = makeCandidate("Bull Put Spread", [
      {{action:"sell", strike:sellSt, type:"PE", premium:sRow.pe_ltp||0.01, iv:sRow.pe_iv||15}},
      {{action:"buy",  strike:buySt,  type:"PE", premium:bRow.pe_ltp||0.01, iv:bRow.pe_iv||15}},
    ], "🐂");
    if (c && c.bes.length >= 1 && Math.abs(c.bes[0] - lowerBE) < 300) candidates.push(c);
  }});

  // ─────────────────────────────────────────────────────────────
  // STRATEGY 5: Bear Call Spread (upper BE focus)
  // BE = sell_strike + net_credit
  // ─────────────────────────────────────────────────────────────
  allSt.filter(s => s >= underlying - 100 && s <= upperBE + 100).forEach(sellSt => {{
    const buySt = nearest(sellSt + 150);
    if (buySt === sellSt) return;
    const sRow = smap[sellSt], bRow = smap[buySt];
    if (!sRow || !bRow) return;
    const c = makeCandidate("Bear Call Spread", [
      {{action:"sell", strike:sellSt, type:"CE", premium:sRow.ce_ltp||0.01, iv:sRow.ce_iv||15}},
      {{action:"buy",  strike:buySt,  type:"CE", premium:bRow.ce_ltp||0.01, iv:bRow.ce_iv||15}},
    ], "🐻");
    if (c && c.bes.length >= 1 && Math.abs(c.bes[0] - upperBE) < 300) candidates.push(c);
  }});

  // ─────────────────────────────────────────────────────────────
  // STRATEGY 6: Iron Butterfly
  // ─────────────────────────────────────────────────────────────
  const ibfCenter = nearest(targetCenter);
  const ibfWing   = Math.max(Math.round(targetRange * 0.45 / 50) * 50, 100);
  const ibfBuyC   = nearest(ibfCenter + ibfWing);
  const ibfBuyP   = nearest(ibfCenter - ibfWing);
  if (ibfBuyC !== ibfCenter && ibfBuyP !== ibfCenter) {{
    const cRow = smap[ibfCenter], bcRow = smap[ibfBuyC], bpRow = smap[ibfBuyP];
    if (cRow && bcRow && bpRow) {{
      const c = makeCandidate("Iron Butterfly", [
        {{action:"sell", strike:ibfCenter, type:"CE", premium:cRow.ce_ltp||0.01,  iv:cRow.ce_iv||15}},
        {{action:"sell", strike:ibfCenter, type:"PE", premium:cRow.pe_ltp||0.01,  iv:cRow.pe_iv||15}},
        {{action:"buy",  strike:ibfBuyC,   type:"CE", premium:bcRow.ce_ltp||0.01, iv:bcRow.ce_iv||15}},
        {{action:"buy",  strike:ibfBuyP,   type:"PE", premium:bpRow.pe_ltp||0.01, iv:bpRow.pe_iv||15}},
      ], "🦋");
      if (c) candidates.push(c);
    }}
  }}

  // ─────────────────────────────────────────────────────────────
  // Deduplicate by name+strikes, sort by totalDiff ascending
  // ─────────────────────────────────────────────────────────────
  const seen = new Set();
  const unique = candidates.filter(c => {{
    if (!c) return false;
    const key = c.name + "|" + c.legs.map(l=>l.action[0]+l.strike+l.type).join(",");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  }});
  unique.sort((a,b) => a.totalDiff - b.totalDiff);

  const top = unique.slice(0, 6);

  // Push to main strategies array and render in strategy grid too
  strategies = top.map((c, idx) => {{
    // Convert to strategy grid format
    const range = [];
    for (let p = underlying - 1500; p <= underlying + 1500; p += 25) range.push(p);
    const payoffs = range.map(price => {{
      let pnl = 0;
      c.legs.forEach(l => {{
        const intr = l.type==="CE" ? Math.max(0,price-l.strike) : Math.max(0,l.strike-price);
        pnl += l.action==="buy" ? (intr-l.premium) : (l.premium-intr);
      }});
      return Math.round(pnl*LOT_SIZE*100)/100;
    }});
    return {{
      name:       c.name + " (BE Match #" + (idx+1) + ")",
      type:       "be_match",
      bias:       "neutral",
      legs:       c.legs.map(l => ({{...l, opt_type: l.type, premium: l.premium}})),
      netPrem:    c.netPrem,
      isDebit:    false,
      maxProfit:  c.maxProfit,
      maxLoss:    c.maxLoss,
      breakevens: c.bes,
      rr:         c.maxLoss > 0 ? Math.round(Math.abs(c.maxProfit/c.maxLoss)*100)/100 : 0,
      pop:        c.pop,
      score:      c.accuracy,
      margin:     c.maxLoss,
      payoffs, priceRange: range,
      _beMatch: c,
    }};
  }});

  renderBEResults(top, lowerBE, upperBE);
  renderStrategies();
  populatePayoffSel();

  // Auto-select best match in payoff chart
  if (top.length > 0) {{
    setTimeout(() => {{
      document.getElementById("payoffSel").value = strategies[0].name;
      drawPayoff();
    }}, 200);
  }}
}}

// ── Render BE results in left panel ──────────────────────────
function renderBEResults(results, lowerBE, upperBE) {{
  const wrap = document.getElementById("beResultsWrap");
  if (!results.length) {{
    wrap.innerHTML = '<div class="be-empty">No matching strategies found.<br>Try adjusting your BE range.</div>';
    return;
  }}

  const maxAcc = Math.max(...results.map(r => r.accuracy), 1);

  wrap.innerHTML = `
    <div style="font-size:9px;font-weight:700;color:var(--text2);font-family:'DM Mono',monospace;
                text-transform:uppercase;letter-spacing:1px;margin:12px 0 8px;
                display:flex;align-items:center;justify-content:space-between;">
      <span>🎯 ${{results.length}} Matches Found</span>
      <span style="color:var(--text3);">Click to view payoff</span>
    </div>` +
    results.map((r, i) => {{
      const isBest    = i === 0;
      const accCol    = r.accuracy >= 75 ? "var(--green)" : r.accuracy >= 45 ? "var(--gold)" : "var(--red)";
      const accLabel  = r.accuracy >= 75 ? "Excellent" : r.accuracy >= 45 ? "Good" : "Approximate";
      const lDiffCol  = Math.abs(r.lowerDiff) <= 25 ? "be-diff-good" : Math.abs(r.lowerDiff) <= 75 ? "be-diff-ok" : "be-diff-bad";
      const uDiffCol  = r.upperDiff !== null ? (Math.abs(r.upperDiff) <= 25 ? "be-diff-good" : Math.abs(r.upperDiff) <= 75 ? "be-diff-ok" : "be-diff-bad") : "";
      const lDiffStr  = r.lowerDiff >= 0 ? "+" + r.lowerDiff : "" + r.lowerDiff;
      const uDiffStr  = r.upperDiff !== null ? (r.upperDiff >= 0 ? "+" + r.upperDiff : "" + r.upperDiff) : "N/A";
      const legTags   = r.legs.map(l =>
        `<span class="be-leg-tag ${{l.action==="buy"?"be-leg-buy":"be-leg-sell"}}">${{l.action.toUpperCase()}} ${{l.strike}} ${{l.type}} @₹${{l.premium.toFixed(1)}}</span>`
      ).join("");
      const swPct     = Math.round((r.accuracy / maxAcc) * 100);
      const stratName = strategies[i]?.name || r.name;

      return `
      <div class="be-match-card ${{isBest?"be-best":""}}" onclick="selectBEPayoff('${{stratName.replace(/'/g,"\\'")}}')">
        ${{isBest ? '<span class="be-match-rank" style="background:rgba(255,209,102,.15);color:var(--gold);border:1px solid rgba(255,209,102,.3);">🏆 BEST</span>' :
                    `<span class="be-match-rank" style="background:var(--bg3);color:var(--text3);border:1px solid var(--border);">#${{i+1}}</span>`}}
        <div class="be-match-name">${{r.icon}} ${{r.name}}</div>
        <div class="be-match-legs">${{legTags}}</div>
        <div class="be-accuracy-row">
          <div class="be-accuracy-box">
            <div class="be-acc-lbl">▼ Lower BE</div>
            <div class="be-acc-target" style="color:var(--text3);">Target: ₹${{lowerBE.toLocaleString("en-IN")}}</div>
            <div class="be-acc-actual" style="color:var(--red);">₹${{r.actualLower.toLocaleString("en-IN")}}</div>
            <div class="be-acc-diff ${{lDiffCol}}">${{lDiffStr}} pts off</div>
          </div>
          <div class="be-accuracy-box">
            <div class="be-acc-lbl">▲ Upper BE</div>
            <div class="be-acc-target" style="color:var(--text3);">Target: ₹${{upperBE.toLocaleString("en-IN")}}</div>
            <div class="be-acc-actual" style="color:var(--green);">${{r.upperDiff !== null ? "₹" + r.actualUpper.toLocaleString("en-IN") : "—"}}</div>
            <div class="be-acc-diff ${{uDiffCol}}">${{uDiffStr}} pts off</div>
          </div>
        </div>
        <div class="be-net-prem">
          <span>Net Credit: <b style="color:var(--green);">₹${{r.netPrem.toFixed(2)}}</b></span>
          <span>PoP: <b style="color:${{r.pop>=60?"var(--green)":r.pop>=45?"var(--gold)":"var(--red)"}};">${{r.pop}}%</b></span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;margin-top:6px;">
          <span style="font-size:8px;color:var(--text3);font-family:'DM Mono',monospace;">MATCH</span>
          <div class="be-score-bar" style="flex:1;"><div class="be-score-fill" style="width:${{swPct}}%;"></div></div>
          <span style="font-size:10px;font-weight:700;font-family:'DM Mono',monospace;color:${{accCol}};">${{r.accuracy}}% ${{accLabel}}</span>
        </div>
      </div>`;
    }}).join("");
}}

function selectBEPayoff(name) {{
  document.getElementById("payoffSel").value = name;
  drawPayoff();
  document.querySelector(".payoff-wrap")?.scrollIntoView({{behavior:"smooth", block:"start"}});
}}

// ── Strategy Engine (JS) ──────────────────────────────────────
function analyze() {{
  const d = ALL_DATA[currentExpiry];
  if (!d) {{ alert("No data loaded for this expiry."); return; }}
  const supports    = getSupports();
  const resistances = getResistances();
  if (!supports.length || !resistances.length) {{
    alert("Please enter at least one support and one resistance level.");
    return;
  }}
  renderChain();
  const underlying  = d.underlying;
  const atm         = d.atm_strike;
  const dte         = d.dte;
  const T           = Math.max(dte / 365, 0.001);
  const strikesArr  = d.all_strikes;
  const smap        = {{}};
  strikesArr.forEach(s => {{ smap[s.strike] = s; }});
  const allSt       = strikesArr.map(s => s.strike).sort((a,b)=>a-b);
  const nearest = val => allSt.reduce((a,b)=>Math.abs(b-val)<Math.abs(a-val)?b:a);
  const get     = (st, field, def=0) => (smap[st]||{{}})[field] || def;
  const avgSup  = supports.reduce((a,b)=>a+b,0)/supports.length;
  const avgRes  = resistances.reduce((a,b)=>a+b,0)/resistances.length;
  const srRange = avgRes - avgSup;
  const s_st  = nearest(avgSup);
  const r_st  = nearest(avgRes);
  const otm_c = nearest(underlying + srRange*0.3);
  const otm_p = nearest(underlying - srRange*0.3);
  const far_c = nearest(avgRes + 150);
  const far_p = nearest(avgSup - 150);

  function payoffsFor(legs) {{
    const range=[];
    for(let p=underlying-1500;p<=underlying+1500;p+=25) range.push(p);
    return {{
      range,
      vals: range.map(price => {{
        let pnl=0;
        legs.forEach(l=>{{
          const intr=(l.opt_type||l.type)==="CE"?Math.max(0,price-l.strike):Math.max(0,l.strike-price);
          pnl+= l.action==="buy"?(intr-l.premium):(l.premium-intr);
        }});
        return Math.round(pnl*LOT_SIZE*100)/100;
      }})
    }};
  }}

  function makeStrat(name,legs,biasTag,sType) {{
    const netPrem = legs.reduce((a,l)=>a+(l.action==="sell"?l.premium:-l.premium),0);
    const po      = payoffsFor(legs);
    const maxP    = Math.max(...po.vals);
    const maxL    = Math.min(...po.vals);
    if(maxP<=0) return null;
    const bes=[];
    for(let i=0;i<po.vals.length-1;i++){{
      if((po.vals[i]<0)!==(po.vals[i+1]<0)){{
        const be=po.range[i]+(po.range[i+1]-po.range[i])*Math.abs(po.vals[i])/(Math.abs(po.vals[i])+Math.abs(po.vals[i+1]));
        bes.push(Math.round(be));
      }}
    }}
    let pop;
    const allSell  = legs.every(l => l.action === "sell");
    const allBuy   = legs.every(l => l.action === "buy");
    const avgIv    = legs.reduce((a,l) => a + (l.iv||15), 0) / legs.length;
    const sigma    = avgIv / 100;
    if (bes.length >= 1) {{
      if (sType === "debit_spread") {{
        const be = bes[0];
        const isCall = legs.find(l => l.action==="buy")?.type === "CE";
        if (isCall) {{
          pop = Math.round((1 - bsm(underlying, be, T, RISK_FREE, sigma, "CE").pop) * 1000) / 10;
        }} else {{
          pop = Math.round(bsm(underlying, be, T, RISK_FREE, sigma, "PE").pop * 1000) / 10;
        }}
      }} else if (allBuy && (sType === "straddle" || sType === "strangle")) {{
        const lowerBE2 = bes[0];
        const upperBE2 = bes.length >= 2 ? bes[1] : null;
        const probBelow = bsm(underlying, lowerBE2, T, RISK_FREE, sigma, "PE").pop;
        const probAbove = upperBE2 ? (1 - bsm(underlying, upperBE2, T, RISK_FREE, sigma, "CE").pop) : 0;
        pop = Math.round(Math.min((probBelow + probAbove) * 100, 99) * 10) / 10;
      }} else if (allSell || sType === "credit_spread" || sType === "iron_condor" || sType === "iron_butterfly") {{
        const lowerBE2 = bes[0];
        const upperBE2 = bes.length >= 2 ? bes[1] : null;
        const probBelow = bsm(underlying, lowerBE2, T, RISK_FREE, sigma, "PE").pop;
        const probAbove = upperBE2 ? (1 - bsm(underlying, upperBE2, T, RISK_FREE, sigma, "CE").pop) : 0;
        pop = Math.round(Math.min((1 - probBelow - probAbove) * 100, 99) * 10) / 10;
      }} else {{
        let s=0;
        legs.forEach(l=>{{ const b=bsm(underlying,l.strike,T,RISK_FREE,l.iv/100,l.opt_type||l.type); s+=l.action==="sell"?b.pop:(1-b.pop); }});
        pop = Math.round(Math.min(s/legs.length*100,99)*10)/10;
      }}
    }} else {{
      let s=0;
      legs.forEach(l=>{{ const b=bsm(underlying,l.strike,T,RISK_FREE,l.iv/100,l.opt_type||l.type); s+=l.action==="sell"?b.pop:(1-b.pop); }});
      pop = Math.round(Math.min(s/legs.length*100,99)*10)/10;
    }}
    pop = Math.max(pop, 1);
    const rr   = maxL!==0?Math.round(Math.abs(maxP/maxL)*100)/100:0;
    const score= Math.round(pop*0.40+Math.min(rr*35,35)+Math.min(maxP/5000*25,25)*10)/10;
    return {{
      name, biasTag, sType, legs,
      netPrem:  Math.round(netPrem*100)/100,
      isDebit:  netPrem<0,
      maxProfit: Math.round(maxP*100)/100,
      maxLoss:   Math.round(Math.abs(maxL)*100)/100,
      breakevens: bes,
      rr, pop, score,
      margin: Math.round(Math.abs(maxL)*100)/100,
      payoffs: po.vals,
      priceRange: po.range,
    }};
  }}

  const raw=[];
  const bias=marketBias;
  if(bias!=="bearish"){{
    const s=makeStrat("Bull Call Spread",[
      {{action:"buy", strike:atm,  type:"CE",premium:get(atm, "ce_ltp"),iv:get(atm, "ce_iv",15)}},
      {{action:"sell",strike:r_st, type:"CE",premium:get(r_st,"ce_ltp"),iv:get(r_st,"ce_iv",15)}},
    ],"bullish","debit_spread");
    if(s) raw.push(s);
  }}
  if(bias!=="bullish"){{
    const s=makeStrat("Bear Put Spread",[
      {{action:"buy", strike:atm,  type:"PE",premium:get(atm, "pe_ltp"),iv:get(atm, "pe_iv",15)}},
      {{action:"sell",strike:s_st, type:"PE",premium:get(s_st,"pe_ltp"),iv:get(s_st,"pe_iv",15)}},
    ],"bearish","debit_spread");
    if(s) raw.push(s);
  }}
  if(bias!=="bearish"){{
    const s=makeStrat("Bull Put Spread",[
      {{action:"sell",strike:s_st, type:"PE",premium:get(s_st,"pe_ltp"),iv:get(s_st,"pe_iv",15)}},
      {{action:"buy", strike:far_p,type:"PE",premium:get(far_p,"pe_ltp"),iv:get(far_p,"pe_iv",15)}},
    ],"bullish","credit_spread");
    if(s) raw.push(s);
  }}
  if(bias!=="bullish"){{
    const s=makeStrat("Bear Call Spread",[
      {{action:"sell",strike:r_st, type:"CE",premium:get(r_st,"ce_ltp"),iv:get(r_st,"ce_iv",15)}},
      {{action:"buy", strike:far_c,type:"CE",premium:get(far_c,"ce_ltp"),iv:get(far_c,"ce_iv",15)}},
    ],"bearish","credit_spread");
    if(s) raw.push(s);
  }}
  {{
    const s=makeStrat("Iron Condor",[
      {{action:"sell",strike:r_st, type:"CE",premium:get(r_st, "ce_ltp"),iv:get(r_st, "ce_iv",15)}},
      {{action:"buy", strike:far_c,type:"CE",premium:get(far_c,"ce_ltp"),iv:get(far_c,"ce_iv",15)}},
      {{action:"sell",strike:s_st, type:"PE",premium:get(s_st, "pe_ltp"),iv:get(s_st, "pe_iv",15)}},
      {{action:"buy", strike:far_p,type:"PE",premium:get(far_p,"pe_ltp"),iv:get(far_p,"pe_iv",15)}},
    ],"neutral","iron_condor");
    if(s) raw.push(s);
  }}
  {{
    const wing=Math.max(Math.round(srRange*0.4/50)*50,100);
    const bc=nearest(atm+wing), bp=nearest(atm-wing);
    const s=makeStrat("Iron Butterfly",[
      {{action:"sell",strike:atm,type:"CE",premium:get(atm,"ce_ltp"),iv:get(atm,"ce_iv",15)}},
      {{action:"sell",strike:atm,type:"PE",premium:get(atm,"pe_ltp"),iv:get(atm,"pe_iv",15)}},
      {{action:"buy", strike:bc, type:"CE",premium:get(bc, "ce_ltp"),iv:get(bc, "ce_iv",15)}},
      {{action:"buy", strike:bp, type:"PE",premium:get(bp, "pe_ltp"),iv:get(bp, "pe_iv",15)}},
    ],"neutral","iron_butterfly");
    if(s) raw.push(s);
  }}
  {{
    const s=makeStrat("Long Straddle",[
      {{action:"buy",strike:atm,type:"CE",premium:get(atm,"ce_ltp"),iv:get(atm,"ce_iv",15)}},
      {{action:"buy",strike:atm,type:"PE",premium:get(atm,"pe_ltp"),iv:get(atm,"pe_iv",15)}},
    ],"volatile","straddle");
    if(s) raw.push(s);
  }}
  if(bias==="neutral"){{
    const s=makeStrat("Short Straddle",[
      {{action:"sell",strike:atm,type:"CE",premium:get(atm,"ce_ltp"),iv:get(atm,"ce_iv",15)}},
      {{action:"sell",strike:atm,type:"PE",premium:get(atm,"pe_ltp"),iv:get(atm,"pe_iv",15)}},
    ],"neutral","straddle");
    if(s) raw.push(s);
  }}
  {{
    const s=makeStrat("Long Strangle",[
      {{action:"buy",strike:otm_c,type:"CE",premium:get(otm_c,"ce_ltp"),iv:get(otm_c,"ce_iv",15)}},
      {{action:"buy",strike:otm_p,type:"PE",premium:get(otm_p,"pe_ltp"),iv:get(otm_p,"pe_iv",15)}},
    ],"volatile","strangle");
    if(s) raw.push(s);
  }}
  if(bias==="neutral"){{
    const s=makeStrat("Short Strangle",[
      {{action:"sell",strike:otm_c,type:"CE",premium:get(otm_c,"ce_ltp"),iv:get(otm_c,"ce_iv",15)}},
      {{action:"sell",strike:otm_p,type:"PE",premium:get(otm_p,"pe_ltp"),iv:get(otm_p,"pe_iv",15)}},
    ],"neutral","strangle");
    if(s) raw.push(s);
  }}
  strategies=raw.filter(s=>s!==null);
  renderStrategies();
  populatePayoffSel();
}}

// ── Render Strategy Cards ─────────────────────────────────────
function renderStrategies() {{
  const sortBy = document.getElementById("sortSel").value;
  const sorted = [...strategies].sort((a,b)=>{{
    if(sortBy==="pop")    return b.pop-a.pop;
    if(sortBy==="profit") return b.maxProfit-a.maxProfit;
    if(sortBy==="rr")     return b.rr-a.rr;
    return b.score-a.score;
  }});
  document.getElementById("stratCount").textContent = sorted.length+" STRATEGIES";
  if(!sorted.length) {{
    document.getElementById("stratGrid").innerHTML='<div class="empty"><div class="empty-icon">📋</div><p>No strategies found for current settings.</p></div>';
    return;
  }}
  const colors={{bullish:"var(--green)",bearish:"var(--red)",neutral:"var(--cyan)",volatile:"var(--purple)"  }};
  const emojis={{bullish:"🐂",bearish:"🐻",neutral:"⚖️",volatile:"⚡"}};
  const maxScore=Math.max(...sorted.map(s=>s.score));
  document.getElementById("stratGrid").innerHTML = sorted.map((s,i)=>{{
    // detect BE match
    const isBEMatch = s.type === "be_match";
    const cc      = isBEMatch ? "var(--gold)" : (colors[s.biasTag]||"var(--cyan)");
    const popCol  = s.pop>=60?"var(--green)":s.pop>=45?"var(--gold)":"var(--red)";
    const popBg   = s.pop>=60?"#00c89620":s.pop>=45?"#ffd16620":"#ff6b6b20";
    const rrDisp  = s.rr===0?"∞":s.rr.toFixed(2)+"x";
    const sw      = Math.round((s.score/maxScore)*100);
    const beStr   = s.breakevens?.length ? s.breakevens.map(b=>"₹"+b.toLocaleString("en-IN")).join(" / ") : "—";
    const netPremField = s.netPrem !== undefined ? s.netPrem : 0;
    const netDisp = (netPremField < 0)
      ? `<span class="down">-₹${{Math.abs(netPremField).toFixed(2)}}</span>`
      : `<span class="up">+₹${{netPremField.toFixed(2)}}</span>`;
    const uid = s.name.replace(/[^a-zA-Z0-9]/g,"_");
    const legTags = (s.legs||[]).map(l=>`<span class="leg-tag leg-${{l.action}}">${{l.action.toUpperCase()}} ${{l.strike}} ${{l.opt_type||l.type}} @${{(l.premium||0).toFixed(2)}}</span>`).join("");
    const beMatchBadge = isBEMatch ? `<div style="background:rgba(255,209,102,.1);border:1px solid rgba(255,209,102,.3);border-radius:6px;padding:3px 8px;font-size:8px;font-weight:700;color:var(--gold);font-family:'DM Mono',monospace;margin-bottom:6px;display:inline-block;">🎯 TARGET BE MATCH · ${{s.score}}% accuracy</div>` : "";
    return `<div class="strat-card" style="--cc:${{cc}};animation-delay:${{i*0.05}}s${{isBEMatch?";border-top:2px solid var(--gold)":""}}" onclick="selectPayoff('${{s.name.replace(/'/g,"\\'")}}')">
      <div class="sc-top">
        <div>
          ${{beMatchBadge}}
          <div class="sc-name">${{isBEMatch?"🎯":emojis[s.biasTag]||"📊"}} ${{s.name}}</div>
          <div class="sc-sub">${{isBEMatch?"BE MATCH":"${{(s.biasTag||"").toUpperCase()}}"}} · ${{(s.isDebit?"DEBIT":"CREDIT")}} · DTE:${{ALL_DATA[currentExpiry]?.dte||"—"}}</div>
        </div>
        <div class="pop-pill" style="background:${{popBg}};color:${{popCol}};border:1px solid ${{popCol}}33;">${{s.pop}}%<br><span style="font-size:8px;font-weight:400;">PoP</span></div>
      </div>
      <div class="sc-fields">
        <div class="sc-field"><span class="sc-field-lbl">Max Profit</span><span class="sc-field-val up">₹${{(s.maxProfit||0).toLocaleString("en-IN")}}</span></div>
        <div class="sc-field"><span class="sc-field-lbl">Max Loss</span><span class="sc-field-val down">₹${{(s.maxLoss||0).toLocaleString("en-IN")}}</span></div>
        <div class="sc-field"><span class="sc-field-lbl">RR Ratio</span><span class="sc-field-val" style="color:var(--gold);">1:${{rrDisp}}</span></div>
        <div class="sc-field"><span class="sc-field-lbl">Net Credit</span><span class="sc-field-val">${{netDisp}}</span></div>
        <div class="sc-field" style="grid-column:1/-1"><span class="sc-field-lbl">Break-Evens</span><span class="sc-field-val" style="font-size:10px;color:${{isBEMatch?"var(--gold)":"var(--text2)"}};">${{beStr}}</span></div>
      </div>
      <div class="sc-legs">${{legTags}}</div>
      <div class="sc-score">
        <span class="score-lbl">${{isBEMatch?"ACCURACY":"SCORE"}}</span>
        <div class="score-bar-track"><div class="score-bar-fill" style="width:${{sw}}%;${{isBEMatch?"background:linear-gradient(90deg,var(--gold),#ff9500);":""}}"></div></div>
        <span class="score-num" style="${{isBEMatch?"color:var(--gold);":''}}">${{s.score}}</span>
      </div>
    </div>`;
  }}).join("");
}}

function populatePayoffSel() {{
  document.getElementById("payoffSel").innerHTML =
    '<option value="">— Select Strategy —</option>' +
    strategies.map(s=>`<option value="${{s.name}}">${{s.name}}</option>`).join("");
}}

function selectPayoff(name) {{
  document.getElementById("payoffSel").value = name;
  drawPayoff();
  document.getElementById("payoffSel").scrollIntoView({{behavior:"smooth"}});
}}

// ═══════════════════════════════════════════════════════════
// RICH PAYOFF CHART
// ═══════════════════════════════════════════════════════════

function bsmPrice(S, K, T, r, sigma, type) {{
  if (T <= 0 || sigma <= 0) {{
    return type === "CE" ? Math.max(0, S - K) : Math.max(0, K - S);
  }}
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  if (type === "CE") return Math.max(0, S * normCdf(d1) - K * Math.exp(-r * T) * normCdf(d2));
  return Math.max(0, K * Math.exp(-r * T) * normCdf(-d2) - S * normCdf(-d1));
}}

function stratPnlAtSpot(legs, spotPrice, T) {{
  let pnl = 0;
  legs.forEach(l => {{
    const sigma   = (l.iv || 15) / 100;
    const optType = l.opt_type || l.type || "CE";
    const theoVal = bsmPrice(spotPrice, l.strike, T, RISK_FREE, sigma, optType);
    const legPnl  = l.action === "buy" ? (theoVal - l.premium) : (l.premium - theoVal);
    pnl += legPnl;
  }});
  return Math.round(pnl * LOT_SIZE * 100) / 100;
}}

function drawPayoff() {{
  const name = document.getElementById("payoffSel").value;
  const s    = strategies.find(x => x.name === name);
  if (!s) return;
  const d          = ALL_DATA[currentExpiry];
  const underlying = d?.underlying || 0;
  const dte        = d?.dte || 7;
  const T_expiry = 0;
  const T_today  = Math.max(dte / 365, 0.5 / 365);
  const priceRange = [];
  for (let p = underlying - 1500; p <= underlying + 1500; p += 25) priceRange.push(p);
  const todayPnl  = priceRange.map(p => stratPnlAtSpot(s.legs, p, T_today));
  const expiryPnl = priceRange.map(p => stratPnlAtSpot(s.legs, p, T_expiry));
  const allStrikes = d?.all_strikes || [];
  const ceOiData   = priceRange.map(p => {{ const row = allStrikes.find(r => r.strike === p); return row ? Math.round(row.ce_oi / 1e3) : null; }});
  const peOiData   = priceRange.map(p => {{ const row = allStrikes.find(r => r.strike === p); return row ? Math.round(row.pe_oi / 1e3) : null; }});
  const netCost = Math.abs((s.netPrem||0)) * LOT_SIZE || 1;
  const projPnl  = stratPnlAtSpot(s.legs, underlying, T_today);
  const projPct  = ((projPnl / netCost) * 100).toFixed(1);
  const projCol  = projPnl >= 0 ? "var(--green)" : "var(--red)";
  const projSign = projPnl >= 0 ? "+" : "";
  const isBEMatch = s.type === "be_match";

  const statsEl = document.getElementById("payoffStats");
  const maxP    = s.maxProfit;
  const maxL    = s.maxLoss;
  const bes     = s.breakevens || [];
  statsEl.style.display = "grid";
  statsEl.innerHTML = `
    <div style="background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--green);border-radius:10px;padding:10px 14px;">
      <div style="font-size:9px;color:var(--text2);text-transform:uppercase;letter-spacing:.8px;">Max Profit</div>
      <div style="font-size:15px;font-weight:800;font-family:'JetBrains Mono',monospace;color:var(--green);margin-top:3px;">${{maxP >= 999999 ? "Unlimited" : "₹" + maxP.toLocaleString("en-IN")}}</div>
    </div>
    <div style="background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--red);border-radius:10px;padding:10px 14px;">
      <div style="font-size:9px;color:var(--text2);text-transform:uppercase;letter-spacing:.8px;">Max Loss</div>
      <div style="font-size:15px;font-weight:800;font-family:'JetBrains Mono',monospace;color:var(--red);margin-top:3px;">₹${{maxL.toLocaleString("en-IN")}}</div>
    </div>
    <div style="background:var(--bg2);border:1px solid var(--border);border-top:2px solid ${{isBEMatch?"var(--gold)":"var(--gold)"}};border-radius:10px;padding:10px 14px;">
      <div style="font-size:9px;color:var(--text2);text-transform:uppercase;letter-spacing:.8px;">Lower BE</div>
      <div style="font-size:15px;font-weight:800;font-family:'JetBrains Mono',monospace;color:var(--gold);margin-top:3px;">${{bes[0] ? "₹" + bes[0].toLocaleString("en-IN") : "—"}}</div>
    </div>
    <div style="background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--gold);border-radius:10px;padding:10px 14px;">
      <div style="font-size:9px;color:var(--text2);text-transform:uppercase;letter-spacing:.8px;">Upper BE</div>
      <div style="font-size:15px;font-weight:800;font-family:'JetBrains Mono',monospace;color:var(--gold);margin-top:3px;">${{bes[1] ? "₹" + bes[1].toLocaleString("en-IN") : bes[0] ? "₹" + bes[0].toLocaleString("en-IN") : "—"}}</div>
    </div>`;

  const footerEl   = document.getElementById("payoffFooter");
  const fallbackEl = document.getElementById("projBadgeFallback");
  if (footerEl)   footerEl.style.display   = "flex";
  if (fallbackEl) fallbackEl.style.display = "none";
  const beBadgesEl = document.getElementById("beBadges");
  if (beBadgesEl) {{
    beBadgesEl.innerHTML = bes.map((be, i) =>
      `<span style="background:rgba(255,209,102,0.1);border:1px solid rgba(255,209,102,0.35);border-radius:6px;padding:3px 10px;font-size:10px;font-weight:700;color:var(--gold);font-family:'JetBrains Mono',monospace;">
        ${{i === 0 ? "▼" : "▲"}} ₹${{be.toLocaleString("en-IN")}}
      </span>`
    ).join("");
  }}
  const projBadgeEl = document.getElementById("projBadge");
  if (projBadgeEl) projBadgeEl.innerHTML =
    `Projected P&L now: <span style="color:${{projCol}};font-weight:800;">${{projSign}}₹${{Math.round(projPnl).toLocaleString("en-IN")}} (${{projSign}}${{projPct}}%)</span>`;

  const ctx = document.getElementById("payoffChart").getContext("2d");
  if (payoffChart) payoffChart.destroy();
  let crosshairX = null;

  payoffChart = new Chart(ctx, {{
    type: "bar",
    data: {{
      labels: priceRange,
      datasets: [
        {{
          label: "CE OI", type: "bar", data: ceOiData,
          backgroundColor: "rgba(0,200,150,0.18)", borderColor: "rgba(0,200,150,0.35)",
          borderWidth: 1, yAxisID: "yOI", order: 3, barPercentage: 0.6,
        }},
        {{
          label: "PE OI", type: "bar", data: peOiData,
          backgroundColor: "rgba(255,107,107,0.18)", borderColor: "rgba(255,107,107,0.35)",
          borderWidth: 1, yAxisID: "yOI", order: 3, barPercentage: 0.6,
        }},
        {{
          label: "Today (DTE:" + dte + ")", type: "line", data: todayPnl,
          borderColor: "#00c896", borderWidth: 2.5, pointRadius: 0,
          fill: {{target: {{value: 0}}, above: "rgba(0,200,150,0.12)", below: "rgba(255,107,107,0.10)"}},
          tension: 0.3, yAxisID: "yPnl", order: 1,
        }},
        {{
          label: "At Expiry", type: "line", data: expiryPnl,
          borderColor: "#5ba3ff", borderWidth: 2, pointRadius: 0,
          borderDash: [5, 4], fill: false, tension: 0.1, yAxisID: "yPnl", order: 2,
        }},
        {{
          label: "Zero", type: "line", data: priceRange.map(() => 0),
          borderColor: "rgba(255,255,255,0.10)", borderWidth: 1,
          pointRadius: 0, fill: false, yAxisID: "yPnl", order: 4,
        }},
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{mode: "index", intersect: false, axis: "x"}},
      plugins: {{
        legend: {{
          display: true, position: "top",
          labels: {{color:"#6a8aaa",font:{{family:"DM Mono",size:10}},boxWidth:14,filter:item=>item.text!=="Zero"}}
        }},
        tooltip: {{enabled: false, mode: "index", intersect: false}},
      }},
      scales: {{
        x: {{
          ticks: {{color:"#2d4560",font:{{family:"DM Mono",size:9}},maxTicksLimit:12,
                   callback:(val,idx)=>{{const price=priceRange[idx]??val;return price>=1000?Math.round(price).toLocaleString("en-IN"):price;}}}},
          grid:{{color:"#0b1520"}},border:{{color:"#1a2535"}},
        }},
        yPnl: {{
          position:"left",
          ticks:{{color:"#2d4560",font:{{family:"DM Mono",size:9}},callback:v=>"₹"+(Math.abs(v)>=1000?(v/1000).toFixed(0)+"K":v)}},
          grid:{{color:"#0b1520"}},border:{{color:"#1a2535"}},
          title:{{display:true,text:"Profit / Loss",color:"#3d5a73",font:{{size:9,family:"DM Mono"}}}},
        }},
        yOI: {{
          position:"right",
          ticks:{{color:"#2d4560",font:{{family:"DM Mono",size:9}},callback:v=>v>=1000?(v/1000).toFixed(0)+"L":v}},
          grid:{{drawOnChartArea:false}},border:{{color:"#1a2535"}},
          title:{{display:true,text:"Open Interest",color:"#3d5a73",font:{{size:9,family:"DM Mono"}}}},
        }},
      }},
    }},
    plugins: [{{
      id: "crosshairSpot",
      afterDatasetsDraw(chart) {{
        const xScale = chart.scales.x;
        const yScale = chart.scales.yPnl;
        const ctx2   = chart.ctx;
        const spotIdx = priceRange.findIndex(p => p >= underlying);
        if (spotIdx >= 0) {{
          const xPx = xScale.getPixelForValue(spotIdx);
          ctx2.save();
          ctx2.setLineDash([6,4]);
          ctx2.strokeStyle = "rgba(0,200,150,0.65)";
          ctx2.lineWidth   = 1.5;
          ctx2.beginPath();
          ctx2.moveTo(xPx, yScale.top);
          ctx2.lineTo(xPx, yScale.bottom);
          ctx2.stroke();
          ctx2.setLineDash([]);
          ctx2.fillStyle  = "rgba(0,200,150,0.9)";
          ctx2.font       = "bold 9px DM Mono,monospace";
          ctx2.textAlign  = "center";
          ctx2.fillText("▼ " + underlying.toLocaleString("en-IN"), xPx, yScale.top - 4);
          ctx2.restore();
        }}
        const bePoints = [];
        for (let i = 0; i < expiryPnl.length - 1; i++) {{
          if ((expiryPnl[i] < 0) !== (expiryPnl[i+1] < 0)) {{
            const frac  = Math.abs(expiryPnl[i]) / (Math.abs(expiryPnl[i]) + Math.abs(expiryPnl[i+1]));
            const bePx  = xScale.getPixelForValue(i) + frac * (xScale.getPixelForValue(i+1) - xScale.getPixelForValue(i));
            const bePrice = priceRange[i] + frac * (priceRange[i+1] - priceRange[i]);
            bePoints.push({{ bePx, bePrice }});
          }}
        }}
        bePoints.forEach((be) => {{
          ctx2.save();
          ctx2.setLineDash([4, 3]);
          ctx2.strokeStyle = "rgba(255,209,102,0.7)";
          ctx2.lineWidth   = 1.5;
          ctx2.beginPath();
          ctx2.moveTo(be.bePx, yScale.top);
          ctx2.lineTo(be.bePx, yScale.bottom);
          ctx2.stroke();
          ctx2.setLineDash([]);
          const label    = "BE " + Math.round(be.bePrice).toLocaleString("en-IN");
          ctx2.font      = "bold 8px DM Mono,monospace";
          const tw       = ctx2.measureText(label).width + 10;
          const tx       = be.bePx - tw / 2;
          const ty       = yScale.top + 6;
          ctx2.fillStyle = "rgba(255,209,102,0.15)";
          ctx2.strokeStyle = "rgba(255,209,102,0.6)";
          ctx2.lineWidth = 1;
          ctx2.beginPath();
          ctx2.roundRect(tx, ty, tw, 14, 3);
          ctx2.fill();
          ctx2.stroke();
          ctx2.fillStyle = "rgba(255,209,102,1)";
          ctx2.textAlign = "center";
          ctx2.fillText(label, be.bePx, ty + 10);
          const zeroPx = yScale.getPixelForValue(0);
          ctx2.fillStyle   = "#ffd166";
          ctx2.strokeStyle = "#060910";
          ctx2.lineWidth   = 2;
          ctx2.beginPath();
          ctx2.arc(be.bePx, zeroPx, 4, 0, Math.PI*2);
          ctx2.fill(); ctx2.stroke();
          ctx2.restore();
        }});
        if (bePoints.length >= 2) {{
          const midLossX = (bePoints[0].bePx + bePoints[bePoints.length-1].bePx) / 2;
          const zeroPx   = yScale.getPixelForValue(0);
          ctx2.save();
          ctx2.font      = "bold 9px DM Mono,monospace";
          ctx2.textAlign = "center";
          ctx2.fillStyle = "rgba(255,107,107,0.5)";
          ctx2.fillText("▼ LOSS ZONE", midLossX, zeroPx + 14);
          ctx2.restore();
        }}
        if (bePoints.length >= 1) {{
          const leftX  = (xScale.left + bePoints[0].bePx) / 2;
          const rightX = (bePoints[bePoints.length-1].bePx + xScale.right) / 2;
          const topY   = yScale.top + 28;
          ctx2.save();
          ctx2.font      = "bold 9px DM Mono,monospace";
          ctx2.textAlign = "center";
          ctx2.fillStyle = "rgba(0,200,150,0.5)";
          ctx2.fillText("▲ PROFIT", leftX, topY);
          ctx2.fillText("▲ PROFIT", rightX, topY);
          ctx2.restore();
        }}
        if (crosshairX === null) return;
        const nearXPx  = xScale.getPixelForValue(crosshairX);
        if (isNaN(nearXPx)) return;
        ctx2.save();
        ctx2.setLineDash([3,3]);
        ctx2.strokeStyle = "rgba(255,255,255,0.18)";
        ctx2.lineWidth   = 1;
        ctx2.beginPath();
        ctx2.moveTo(nearXPx, yScale.top);
        ctx2.lineTo(nearXPx, yScale.bottom);
        ctx2.stroke();
        ctx2.setLineDash([]);
        ctx2.fillStyle   = "#00c896";
        ctx2.strokeStyle = "#060910";
        ctx2.lineWidth   = 2.5;
        ctx2.beginPath();
        ctx2.arc(nearXPx, yScale.getPixelForValue(todayPnl[crosshairX]), 5, 0, Math.PI*2);
        ctx2.fill(); ctx2.stroke();
        ctx2.fillStyle   = "#5ba3ff";
        ctx2.strokeStyle = "#060910";
        ctx2.lineWidth   = 2;
        ctx2.beginPath();
        ctx2.arc(nearXPx, yScale.getPixelForValue(expiryPnl[crosshairX]), 5, 0, Math.PI*2);
        ctx2.fill(); ctx2.stroke();
        ctx2.restore();
      }},
    }}],
  }});

  const canvas = document.getElementById("payoffChart");
  const tt     = document.getElementById("payoffTooltip");

  function showTooltip(clientX, clientY) {{
    const rect   = canvas.getBoundingClientRect();
    const xScale = payoffChart.scales.x;
    const yScale = payoffChart.scales.yPnl;
    const canvasX = clientX - rect.left;
    if (canvasX < xScale.left || canvasX > xScale.right) {{
      crosshairX = null; tt.style.display = "none"; payoffChart.draw(); return;
    }}
    const chartLeft  = xScale.left, chartRight = xScale.right, chartW = chartRight - chartLeft;
    const ratio      = (canvasX - chartLeft) / chartW;
    const rawIdx     = ratio * (priceRange.length - 1);
    const bestIdx    = Math.max(0, Math.min(priceRange.length - 1, Math.round(rawIdx)));
    if (crosshairX !== bestIdx) {{ crosshairX = bestIdx; payoffChart.draw(); }}
    const price    = priceRange[bestIdx];
    const pctChg   = (((price - underlying) / underlying) * 100).toFixed(1);
    const sign     = parseFloat(pctChg) >= 0 ? "+" : "";
    const pCol     = parseFloat(pctChg) >= 0 ? "#00c896" : "#ff6b6b";
    const todayVal = todayPnl[bestIdx];
    const expVal   = expiryPnl[bestIdx];
    const tSign    = todayVal >= 0 ? "+" : "";
    const eSign    = expVal   >= 0 ? "+" : "";
    const tCol     = todayVal >= 0 ? "#00c896" : "#ff6b6b";
    const eCol     = expVal   >= 0 ? "#00c896" : "#ff6b6b";
    const tPct     = ((todayVal / netCost) * 100).toFixed(1);
    const ePct     = ((expVal   / netCost) * 100).toFixed(1);
    tt.innerHTML = `
      <div style="font-size:9px;color:#6a8aaa;margin-bottom:5px;letter-spacing:1px;">WHEN PRICE IS AT</div>
      <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:10px;">
        <span style="font-size:16px;font-weight:800;color:#ddeeff;font-family:'DM Mono',monospace;">&#8377;${{price.toLocaleString("en-IN")}}</span>
        <span style="font-size:11px;font-weight:700;color:${{pCol}};font-family:'DM Mono',monospace;">${{sign}}${{pctChg}}%&nbsp;(${{sign}}${{(price-underlying).toLocaleString("en-IN")}})</span>
      </div>
      <div style="height:1px;background:rgba(255,255,255,0.07);margin-bottom:10px;"></div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;">
        <span style="font-size:9px;color:#6a8aaa;display:flex;align-items:center;gap:6px;">
          <span style="width:9px;height:9px;border-radius:50%;background:#00c896;display:inline-block;"></span>
          Today <span style="color:#3d5a73;margin-left:2px;">DTE:${{dte}}</span>
        </span>
        <span style="font-size:13px;font-weight:800;color:${{tCol}};font-family:'DM Mono',monospace;">
          ${{tSign}}&#8377;${{Math.round(todayVal).toLocaleString("en-IN")}}
          <span style="font-size:9px;opacity:.8;"> (${{tSign}}${{tPct}}%)</span>
        </span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:9px;color:#6a8aaa;display:flex;align-items:center;gap:6px;">
          <span style="width:9px;height:9px;border-radius:50%;background:#5ba3ff;display:inline-block;"></span>
          At Expiry <span style="color:#3d5a73;margin-left:2px;">T=0</span>
        </span>
        <span style="font-size:13px;font-weight:800;color:${{eCol}};font-family:'DM Mono',monospace;">
          ${{eSign}}&#8377;${{Math.round(expVal).toLocaleString("en-IN")}}
          <span style="font-size:9px;opacity:.8;"> (${{eSign}}${{ePct}}%)</span>
        </span>
      </div>
      <div style="margin-top:9px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.06);font-size:9px;color:#3d5a73;">
        Theta loss by expiry:
        <span style="color:${{(todayVal-expVal)>0?"#ff6b6b":"#00c896"}};font-weight:700;">
          ${{(todayVal-expVal)>=0?"-":"+"}}&#8377;${{Math.abs(Math.round(todayVal-expVal)).toLocaleString("en-IN")}}
        </span>
      </div>`;
    const ttW=268,ttH=140;
    let fixedL=clientX+18, fixedT=clientY-ttH/2;
    if(fixedL+ttW>window.innerWidth-10) fixedL=clientX-ttW-18;
    if(fixedT<8) fixedT=8;
    if(fixedT+ttH>window.innerHeight-10) fixedT=window.innerHeight-ttH-10;
    tt.style.left=fixedL+"px"; tt.style.top=fixedT+"px"; tt.style.display="block";
  }}
  function hideTooltip() {{ crosshairX=null; tt.style.display="none"; payoffChart.draw(); }}
  canvas.addEventListener("mousemove",  e => showTooltip(e.clientX, e.clientY));
  canvas.addEventListener("mouseleave", hideTooltip);
  canvas.addEventListener("touchmove", e => {{ e.preventDefault(); const t=e.touches[0]; showTooltip(t.clientX,t.clientY); }}, {{passive:false}});
  canvas.addEventListener("touchend", hideTooltip);
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
