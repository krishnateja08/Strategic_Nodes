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

# ── Import holidays ───────────────────────────────────────────────
from nse_holidays import NSE_HOLIDAYS

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
    """Return True if the given date is an NSE trading holiday or weekend."""
    if dt.weekday() >= 5:
        return True
    return dt in _HOLIDAY_DATES


def get_prev_trading_day(dt):
    """Return the nearest previous trading day."""
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
        """Find current/next Tuesday with holiday adjustment → previous trading day."""
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
        """Fallback: fetch real expiry list from NSE and pick nearest upcoming."""
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
        """Fetch full option chain for a specific expiry date."""
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
        """Fetch next n expiries directly from NSE API."""
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

        # Fallback: generate Tuesdays
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
        """Main entry point — returns (result, session, headers, expiry_list)."""
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

        # Cache full expiry list
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
    """
    Black-Scholes option pricing.
    S     = spot price
    K     = strike price
    T     = time to expiry in years
    r     = risk-free rate (e.g. 0.065)
    sigma = implied volatility (e.g. 0.15 for 15%)
    """
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
    """
    Compute/enrich Greeks for entire option chain.
    Uses NSE Greeks if available, falls back to BSM.
    Returns enriched DataFrame + atm_greeks dict + all_strikes list.
    """
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

        # ── IV: use NSE value, fallback to BSM approximation ──
        ce_iv = row.get("CE_IV", 0) or 0
        pe_iv = row.get("PE_IV", 0) or 0
        if ce_iv <= 0:
            ce_iv = 15.0
        if pe_iv <= 0:
            pe_iv = 15.0

        sigma_ce = ce_iv / 100
        sigma_pe = pe_iv / 100

        # ── Greeks: use NSE if non-zero, else BSM ─────────────
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

        # ── Probability of profit via BSM d2 ──────────────────
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


def _strategy_pop(legs, underlying, T, r=RISK_FREE_R):
    """Estimate probability of profit using BSM for each leg."""
    total_pop = 0
    for leg in legs:
        sigma = leg["iv"] / 100
        bs    = black_scholes(underlying, leg["strike"], T, r, sigma, leg["opt_type"])
        pop   = bs.get("prob_profit", 0.5)
        total_pop += (pop if leg["action"] == "sell" else 1 - pop)
    return round((total_pop / len(legs)) * 100, 1)


def _payoff_at_expiry(legs, price_range):
    """Calculate net P&L across a range of expiry prices."""
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
    """
    Build and score all option spread strategies based on:
    - Live NSE option chain data
    - Support / Resistance levels
    - Market bias
    - BSM probability of profit
    Returns sorted list of strategy dicts.
    """
    if not oc_analysis:
        return []

    all_strikes = oc_analysis["all_strikes"]
    underlying  = oc_analysis["underlying"]
    atm_strike  = oc_analysis["atm_strike"]
    expiry_str  = oc_analysis["expiry"]
    dte         = oc_analysis["dte"]
    T           = max(dte / 365.0, 0.001)

    # Build quick lookup
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

    # Expected move
    atm_iv = (get(atm, "ce_iv", 15) + get(atm, "pe_iv", 15)) / 2
    exp_move = round(underlying * (atm_iv / 100) * math.sqrt(T), 0)

    raw = []

    # ── Helper to build a strategy dict ───────────────────────────
    def make_strategy(name, legs, bias_tag, strategy_type):
        net_premium  = sum(
            (-l["premium"] if l["action"] == "buy" else l["premium"]) for l in legs
        )
        is_debit     = net_premium < 0
        net_premium  = round(net_premium, 2)

        # Max profit / loss
        price_range  = list(range(int(underlying - 1500), int(underlying + 1500), 25))
        payoffs      = _payoff_at_expiry(legs, price_range)
        max_profit   = max(payoffs)
        max_loss     = min(payoffs)

        if max_profit <= 0:
            return None

        # Breakevens
        breakevens = []
        for i in range(len(payoffs) - 1):
            if (payoffs[i] < 0) != (payoffs[i + 1] < 0):
                be = price_range[i] + (price_range[i + 1] - price_range[i]) * abs(payoffs[i]) / (abs(payoffs[i]) + abs(payoffs[i + 1]))
                breakevens.append(round(be, 0))

        # RR ratio
        rr_ratio = round(abs(max_profit / max_loss), 2) if max_loss != 0 else 0

        # PoP via BSM
        pop = _strategy_pop(legs, underlying, T)

        # Score: 40% PoP + 35% RR + 25% max profit normalised
        score = round(pop * 0.40 + min(rr_ratio * 35, 35) + min(max_profit / 5000 * 25, 25), 1)

        # Margin estimate (simplified: max loss for debit, wing width for credit)
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

    # ── 1. Bull Call Spread ────────────────────────────────────────
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

    # ── 2. Bear Put Spread ─────────────────────────────────────────
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

    # ── 3. Bull Put Spread (Credit) ────────────────────────────────
    if bias != "bearish":
        sell_st = s_st
        buy_st  = nearest(avg_support - 150)
        if sell_st != buy_st:
            s = make_strategy("Bull Put Spread", [
                {"action": "sell", "strike": sell_st, "opt_type": "PE",
                 "premium": get(sell_st, "pe_ltp"), "iv": get(sell_st, "pe_iv", 15)},
                {"action": "buy",  "strike": buy_st,  "opt_type": "PE",
                 "premium": get(buy_st,  "pe_ltp"), "iv": get(buy_st,  "pe_iv", 15)},
            ], "bullish", "credit_spread")
            if s: raw.append(s)

    # ── 4. Bear Call Spread (Credit) ───────────────────────────────
    if bias != "bullish":
        sell_st = r_st
        buy_st  = nearest(avg_resistance + 150)
        if sell_st != buy_st:
            s = make_strategy("Bear Call Spread", [
                {"action": "sell", "strike": sell_st, "opt_type": "CE",
                 "premium": get(sell_st, "ce_ltp"), "iv": get(sell_st, "ce_iv", 15)},
                {"action": "buy",  "strike": buy_st,  "opt_type": "CE",
                 "premium": get(buy_st,  "ce_ltp"), "iv": get(buy_st,  "ce_iv", 15)},
            ], "bearish", "credit_spread")
            if s: raw.append(s)

    # ── 5. Iron Condor ─────────────────────────────────────────────
    {
        "sell_ce": r_st, "buy_ce": far_c,
        "sell_pe": s_st, "buy_pe": far_p,
    }
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

    # ── 6. Iron Butterfly ──────────────────────────────────────────
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

    # ── 7. Long Straddle ───────────────────────────────────────────
    s = make_strategy("Long Straddle", [
        {"action": "buy", "strike": atm, "opt_type": "CE",
         "premium": get(atm, "ce_ltp"), "iv": get(atm, "ce_iv", 15)},
        {"action": "buy", "strike": atm, "opt_type": "PE",
         "premium": get(atm, "pe_ltp"), "iv": get(atm, "pe_iv", 15)},
    ], "volatile", "straddle")
    if s: raw.append(s)

    # ── 8. Short Straddle ──────────────────────────────────────────
    if bias == "neutral":
        s = make_strategy("Short Straddle", [
            {"action": "sell", "strike": atm, "opt_type": "CE",
             "premium": get(atm, "ce_ltp"), "iv": get(atm, "ce_iv", 15)},
            {"action": "sell", "strike": atm, "opt_type": "PE",
             "premium": get(atm, "pe_ltp"), "iv": get(atm, "pe_iv", 15)},
        ], "neutral", "straddle")
        if s: raw.append(s)

    # ── 9. Long Strangle ───────────────────────────────────────────
    s = make_strategy("Long Strangle", [
        {"action": "buy", "strike": otm_c, "opt_type": "CE",
         "premium": get(otm_c, "ce_ltp"), "iv": get(otm_c, "ce_iv", 15)},
        {"action": "buy", "strike": otm_p, "opt_type": "PE",
         "premium": get(otm_p, "pe_ltp"), "iv": get(otm_p, "pe_iv", 15)},
    ], "volatile", "strangle")
    if s: raw.append(s)

    # ── 10. Short Strangle ─────────────────────────────────────────
    if bias == "neutral":
        s = make_strategy("Short Strangle", [
            {"action": "sell", "strike": otm_c, "opt_type": "CE",
             "premium": get(otm_c, "ce_ltp"), "iv": get(otm_c, "ce_iv", 15)},
            {"action": "sell", "strike": otm_p, "opt_type": "PE",
             "premium": get(otm_p, "pe_ltp"), "iv": get(otm_p, "pe_iv", 15)},
        ], "neutral", "strangle")
        if s: raw.append(s)

    # Sort by score descending
    raw.sort(key=lambda x: x["score"], reverse=True)
    return raw


# =================================================================
#  SECTION 5 -- HTML GENERATOR
# =================================================================

def build_html(all_expiry_data, expiry_list, generated_at):
    """
    Build the complete index.html with all data baked in as JSON.
    All strategy calculation + Greeks rendering happens in JavaScript.
    """

    # Serialize data per expiry
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
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
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

/* ── INPUTS ── */
.form-grp{{margin-bottom:14px;}}
.form-lbl{{display:block;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--text2);margin-bottom:6px;font-family:'DM Mono',monospace;}}
.inp{{width:100%;background:var(--bg2);border:1px solid var(--border);border-radius:7px;padding:9px 12px;color:var(--text);font-family:'DM Mono',monospace;font-size:13px;outline:none;transition:all .2s;}}
.inp:focus{{border-color:var(--cyan);box-shadow:0 0 0 3px #00d4ff12;}}
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
.chain-wrap{{overflow:auto;max-height:520px;}}
.chain-tbl{{width:100%;border-collapse:collapse;font-family:'DM Mono',monospace;font-size:11px;}}
.chain-tbl thead tr{{background:linear-gradient(90deg,#0d1a26,#0d1117);position:sticky;top:0;z-index:2;}}
.chain-tbl th{{padding:9px 10px;text-align:right;font-size:9px;text-transform:uppercase;letter-spacing:1px;color:var(--text3);border-bottom:1px solid var(--border);font-weight:700;}}
.chain-tbl th.ce{{color:var(--green);}} .chain-tbl th.pe{{color:var(--red);}} .chain-tbl th.ctr{{text-align:center;}}
.chain-tbl td{{padding:7px 10px;text-align:right;border-bottom:1px solid var(--border2);transition:background .15s;}}
.chain-tbl tr:hover td{{background:#ffffff04;}}
.atm-row td{{background:#00d4ff07!important;}}
.atm-row .stk-col{{color:var(--cyan)!important;font-weight:700;}}
.stk-col{{text-align:center;font-weight:700;color:var(--text);background:#0d1a2677;border-left:1px solid var(--border);border-right:1px solid var(--border);min-width:85px;}}
.sup-row .stk-col{{border-left:2px solid var(--green)!important;}}
.res-row .stk-col{{border-left:2px solid var(--red)!important;}}
.atm-badge{{display:inline-block;background:var(--cyan);color:#000;font-size:8px;font-weight:700;padding:1px 5px;border-radius:3px;margin-left:4px;letter-spacing:.5px;}}
.oi-bar-w{{display:flex;align-items:center;gap:4px;justify-content:flex-end;}}
.oi-bar{{height:3px;border-radius:2px;background:var(--green);max-width:50px;min-width:1px;}}
.oi-bar.pe{{background:var(--red);}}
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
  .chain-tbl th,.chain-tbl td{{padding:5px 6px;font-size:10px;}}
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
    </div>
  </div>

  <!-- OPTION CHAIN -->
  <div class="panel">
    <div class="panel-hdr">
      <div class="panel-title">📊 Live Option Chain</div>
      <span style="font-size:10px;color:var(--text2);font-family:'DM Mono',monospace;" id="chainExpLbl"></span>
    </div>
    <div class="chain-wrap">
      <table class="chain-tbl">
        <thead><tr>
          <th class="ce">LTP</th><th class="ce">IV%</th><th class="ce">OI(L)</th><th class="ce">ΔOI</th>
          <th class="ctr">STRIKE</th>
          <th class="pe">ΔOI</th><th class="pe">OI(L)</th><th class="pe">IV%</th><th class="pe">LTP</th>
        </tr></thead>
        <tbody id="chainBody"><tr><td colspan="9" style="text-align:center;padding:50px;color:var(--text3);font-family:'DM Mono',monospace;font-size:11px;">Loading…</td></tr></tbody>
      </table>
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
    <div class="empty"><div class="empty-icon">📋</div><p>Enter support &amp; resistance levels<br>then click ⚡ Analyze Strategies</p></div>
  </div>
</div>

<!-- PAYOFF CHART -->
<div class="panel" style="margin-bottom:18px;">
  <div class="panel-hdr">
    <div class="panel-title">📈 Payoff Diagram
      <span style="font-size:9px;color:var(--text3);margin-left:8px;font-family:'DM Mono',monospace;">🟢 Today (BSM+Greeks) &nbsp;|&nbsp; 🔵 At Expiry &nbsp;|&nbsp; bars = OI</span>
    </div>
    <select class="sel" style="width:200px;" id="payoffSel" onchange="drawPayoff()">
      <option value="">— Select Strategy —</option>
    </select>
  </div>
  <div class="payoff-wrap" style="height:320px;padding:16px;position:relative;">
    <canvas id="payoffChart"></canvas>
    <div id="payoffTooltip" style="
      display:none;position:absolute;z-index:99;
      background:rgba(10,20,34,0.97);
      border:1px solid rgba(0,212,255,0.25);
      border-radius:10px;padding:12px 14px;
      min-width:240px;max-width:270px;
      box-shadow:0 8px 32px rgba(0,0,0,0.6);
      pointer-events:none;
      font-family:'DM Mono',monospace;
    "></div>
  </div>
  <div style="text-align:center;padding:10px 16px 14px;font-size:11px;font-family:'DM Mono',monospace;border-top:1px solid var(--border);" id="projBadge">
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

// ── Init ──────────────────────────────────────────────────────
window.onload = () => {{
  populateExpiries();
  restoreUserState();   // ← restore before first render
  if (currentExpiry) {{
    updateTicker();
    renderChain();
    renderGreeks(null);
  }}
  startCountdown();
  // Save on lot/capital change
  document.getElementById("lotSize").addEventListener("input", saveUserState);
  document.getElementById("maxCap").addEventListener("input",  saveUserState);
  // Save existing SR inputs
  document.querySelectorAll(".sup-inp,.res-inp").forEach(inp => {{
    inp.addEventListener("input", saveUserState);
  }});
}};

// ── Silent background refresh (no flicker, no reload) ────────
function startCountdown() {{
  let secs = 30;
  const el = document.getElementById("countdown");
  setInterval(() => {{
    secs--;
    if (secs <= 0) {{
      secs = 30;
      silentRefresh();
    }}
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
    // Extract fresh JSON data from new page
    const scripts = doc.querySelectorAll("script");
    for (const s of scripts) {{
      const m = s.textContent.match(/const ALL_DATA\s*=\s*(\{{[\s\S]*?\}});/);
      if (m) {{
        try {{
          const freshData = JSON.parse(m[1]);
          // Merge fresh data into ALL_DATA
          Object.assign(ALL_DATA, freshData);
          // Re-render data panels silently (preserve user inputs)
          updateTicker();
          renderChain();
          updateGreeksForStrike(parseInt(document.getElementById("greeksStrikeSel").value) || ALL_DATA[currentExpiry]?.atm_strike);
          // Update generated time
          const genEl = doc.querySelector(".gen-time");
          if (genEl) document.querySelector(".gen-time") && (document.querySelector(".gen-time").textContent = genEl.textContent);
          console.log("Silent refresh done:", new Date().toLocaleTimeString());
        }} catch(e) {{ console.warn("Silent refresh parse error:", e); }}
        break;
      }}
    }}
  }} catch(e) {{
    console.warn("Silent refresh fetch error:", e);
  }}
}}

// ── Expiry ────────────────────────────────────────────────────
function populateExpiries() {{
  const sel = document.getElementById("expirySel");
  sel.innerHTML = EXPIRY_LIST.map((e, i) =>
    `<option value="${{e}}"${{i===0?' selected':''}}>${{e}}${{i===0?' (Weekly)':''}}</option>`
  ).join("");
  currentExpiry = EXPIRY_LIST[0] || "";
}}

function onExpiryChange() {{
  currentExpiry = document.getElementById("expirySel").value;
  updateTicker();
  renderChain();
  renderGreeks(null);
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
  // Max OI
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
  const supports    = getSupports();
  const resistances = getResistances();
  const underlying  = d.underlying;
  const rows        = [...d.all_strikes].sort((a, b) => b.strike - a.strike);
  const maxOi       = Math.max(...rows.flatMap(r => [r.ce_oi, r.pe_oi]));

  const tbody = document.getElementById("chainBody");
  tbody.innerHTML = rows.map(r => {{
    const isAtm = r.is_atm;
    const isSup = supports.includes(r.strike);
    const isRes = resistances.includes(r.strike);
    const ceOiL = (r.ce_oi / 1e5).toFixed(1);
    const peOiL = (r.pe_oi / 1e5).toFixed(1);
    const ceBar = Math.round((r.ce_oi / maxOi) * 50);
    const peBar = Math.round((r.pe_oi / maxOi) * 50);
    const ceChg = r.ce_oi_chg >= 0
      ? `<span class="up">+${{(r.ce_oi_chg/1e5).toFixed(1)}}L</span>`
      : `<span class="down">${{(r.ce_oi_chg/1e5).toFixed(1)}}L</span>`;
    const peChg = r.pe_oi_chg >= 0
      ? `<span class="up">+${{(r.pe_oi_chg/1e5).toFixed(1)}}L</span>`
      : `<span class="down">${{(r.pe_oi_chg/1e5).toFixed(1)}}L</span>`;
    const rc    = isAtm ? "atm-row" : isSup ? "sup-row" : isRes ? "res-row" : "";
    const badge = isAtm ? '<span class="atm-badge">ATM</span>' : "";
    const smark = isSup ? '<span style="color:var(--green);font-size:8px;margin-left:3px;">▲S</span>' : "";
    const rmark = isRes ? '<span style="color:var(--red);font-size:8px;margin-left:3px;">▼R</span>' : "";
    return `<tr class="${{rc}}">
      <td style="color:var(--green)">${{r.ce_ltp.toFixed(2)}}</td>
      <td>${{r.ce_iv.toFixed(1)}}%</td>
      <td><div class="oi-bar-w">${{ceOiL}}L<div class="oi-bar" style="width:${{ceBar}}px"></div></div></td>
      <td>${{ceChg}}</td>
      <td class="stk-col">${{r.strike.toLocaleString("en-IN")}}${{badge}}${{smark}}${{rmark}}</td>
      <td>${{peChg}}</td>
      <td><div class="oi-bar-w"><div class="oi-bar pe" style="width:${{peBar}}px"></div>${{peOiL}}L</div></td>
      <td>${{r.pe_iv.toFixed(1)}}%</td>
      <td style="color:var(--red)">${{r.pe_ltp.toFixed(2)}}</td>
    </tr>`;
  }}).join("");
}}

// ── Greeks ────────────────────────────────────────────────────
function renderGreeks(strikeOverride) {{
  const d = ALL_DATA[currentExpiry];
  if (!d) return;
  document.getElementById("greeksExpTag").textContent = currentExpiry;

  // Populate strike dropdown
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

  // Delta bars
  document.getElementById("dbarCe").style.width = Math.abs(row.ce_delta) * 100 + "%";
  document.getElementById("dbarPe").style.width = Math.abs(row.pe_delta) * 100 + "%";

  // IV gauge
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

  // IV Skew
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

// ── Persist user state across silent refreshes ────────────────
function saveUserState() {{
  const state = {{
    bias:        marketBias,
    expiry:      currentExpiry,
    supports:    getSupports(),
    resistances: getResistances(),
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

    // Restore bias
    if (state.bias) setBias(state.bias);

    // Restore expiry
    if (state.expiry && EXPIRY_LIST.includes(state.expiry)) {{
      currentExpiry = state.expiry;
      document.getElementById("expirySel").value = state.expiry;
    }}

    // Restore support levels
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

    // Restore resistance levels
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

    // Restore lot size & capital
    if (state.lotSize) document.getElementById("lotSize").value = state.lotSize;
    if (state.maxCap)  document.getElementById("maxCap").value  = state.maxCap;

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

  renderChain(); // re-render chain to mark S/R rows

  const underlying  = d.underlying;
  const atm         = d.atm_strike;
  const dte         = d.dte;
  const T           = Math.max(dte / 365, 0.001);
  const strikes     = d.all_strikes;
  const smap        = {{}};
  strikes.forEach(s => {{ smap[s.strike] = s; }});
  const allSt       = strikes.map(s => s.strike).sort((a,b)=>a-b);

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
          const intr=l.type==="CE"?Math.max(0,price-l.strike):Math.max(0,l.strike-price);
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

    // Breakevens
    const bes=[];
    for(let i=0;i<po.vals.length-1;i++){{
      if((po.vals[i]<0)!==(po.vals[i+1]<0)){{
        const be=po.range[i]+(po.range[i+1]-po.range[i])*Math.abs(po.vals[i])/(Math.abs(po.vals[i])+Math.abs(po.vals[i+1]));
        bes.push(Math.round(be));
      }}
    }}

    // PoP via BSM average
    let popSum=0;
    legs.forEach(l=>{{
      const b=bsm(underlying,l.strike,T,RISK_FREE,l.iv/100,l.type);
      popSum+= l.action==="sell"?b.pop:(1-b.pop);
    }});
    const pop=Math.round(popSum/legs.length*100*10)/10;

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

  // Bull Call Spread
  if(bias!=="bearish"){{
    const s=makeStrat("Bull Call Spread",[
      {{action:"buy", strike:atm,  type:"CE",premium:get(atm, "ce_ltp"),iv:get(atm, "ce_iv",15)}},
      {{action:"sell",strike:r_st, type:"CE",premium:get(r_st,"ce_ltp"),iv:get(r_st,"ce_iv",15)}},
    ],"bullish","debit_spread");
    if(s) raw.push(s);
  }}
  // Bear Put Spread
  if(bias!=="bullish"){{
    const s=makeStrat("Bear Put Spread",[
      {{action:"buy", strike:atm,  type:"PE",premium:get(atm, "pe_ltp"),iv:get(atm, "pe_iv",15)}},
      {{action:"sell",strike:s_st, type:"PE",premium:get(s_st,"pe_ltp"),iv:get(s_st,"pe_iv",15)}},
    ],"bearish","debit_spread");
    if(s) raw.push(s);
  }}
  // Bull Put Spread
  if(bias!=="bearish"){{
    const s=makeStrat("Bull Put Spread",[
      {{action:"sell",strike:s_st, type:"PE",premium:get(s_st,"pe_ltp"),iv:get(s_st,"pe_iv",15)}},
      {{action:"buy", strike:far_p,type:"PE",premium:get(far_p,"pe_ltp"),iv:get(far_p,"pe_iv",15)}},
    ],"bullish","credit_spread");
    if(s) raw.push(s);
  }}
  // Bear Call Spread
  if(bias!=="bullish"){{
    const s=makeStrat("Bear Call Spread",[
      {{action:"sell",strike:r_st, type:"CE",premium:get(r_st,"ce_ltp"),iv:get(r_st,"ce_iv",15)}},
      {{action:"buy", strike:far_c,type:"CE",premium:get(far_c,"ce_ltp"),iv:get(far_c,"ce_iv",15)}},
    ],"bearish","credit_spread");
    if(s) raw.push(s);
  }}
  // Iron Condor
  {{
    const s=makeStrat("Iron Condor",[
      {{action:"sell",strike:r_st, type:"CE",premium:get(r_st, "ce_ltp"),iv:get(r_st, "ce_iv",15)}},
      {{action:"buy", strike:far_c,type:"CE",premium:get(far_c,"ce_ltp"),iv:get(far_c,"ce_iv",15)}},
      {{action:"sell",strike:s_st, type:"PE",premium:get(s_st, "pe_ltp"),iv:get(s_st, "pe_iv",15)}},
      {{action:"buy", strike:far_p,type:"PE",premium:get(far_p,"pe_ltp"),iv:get(far_p,"pe_iv",15)}},
    ],"neutral","iron_condor");
    if(s) raw.push(s);
  }}
  // Iron Butterfly
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
  // Long Straddle
  {{
    const s=makeStrat("Long Straddle",[
      {{action:"buy",strike:atm,type:"CE",premium:get(atm,"ce_ltp"),iv:get(atm,"ce_iv",15)}},
      {{action:"buy",strike:atm,type:"PE",premium:get(atm,"pe_ltp"),iv:get(atm,"pe_iv",15)}},
    ],"volatile","straddle");
    if(s) raw.push(s);
  }}
  // Short Straddle
  if(bias==="neutral"){{
    const s=makeStrat("Short Straddle",[
      {{action:"sell",strike:atm,type:"CE",premium:get(atm,"ce_ltp"),iv:get(atm,"ce_iv",15)}},
      {{action:"sell",strike:atm,type:"PE",premium:get(atm,"pe_ltp"),iv:get(atm,"pe_iv",15)}},
    ],"neutral","straddle");
    if(s) raw.push(s);
  }}
  // Long Strangle
  {{
    const s=makeStrat("Long Strangle",[
      {{action:"buy",strike:otm_c,type:"CE",premium:get(otm_c,"ce_ltp"),iv:get(otm_c,"ce_iv",15)}},
      {{action:"buy",strike:otm_p,type:"PE",premium:get(otm_p,"pe_ltp"),iv:get(otm_p,"pe_iv",15)}},
    ],"volatile","strangle");
    if(s) raw.push(s);
  }}
  // Short Strangle
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

  const colors={{bullish:"var(--green)",bearish:"var(--red)",neutral:"var(--cyan)",volatile:"var(--purple)"}};
  const emojis={{bullish:"🐂",bearish:"🐻",neutral:"⚖️",volatile:"⚡"}};
  const maxScore=Math.max(...sorted.map(s=>s.score));

  document.getElementById("stratGrid").innerHTML = sorted.map((s,i)=>{{
    const cc      = colors[s.biasTag]||"var(--cyan)";
    const popCol  = s.pop>=60?"var(--green)":s.pop>=45?"var(--gold)":"var(--red)";
    const popBg   = s.pop>=60?"#00c89620":s.pop>=45?"#ffd16620":"#ff6b6b20";
    const rrDisp  = s.rr===0?"∞":s.rr.toFixed(2)+"x";
    const sw      = Math.round((s.score/maxScore)*100);
    const beStr   = s.breakevens.length ? s.breakevens.map(b=>"₹"+b.toLocaleString("en-IN")).join(" / ") : "—";
    const netDisp = s.isDebit ? `<span class="down">-₹${{Math.abs(s.netPrem).toFixed(2)}}</span>` : `<span class="up">+₹${{s.netPrem.toFixed(2)}}</span>`;
    const legTags = s.legs.map(l=>`<span class="leg-tag leg-${{l.action}}">${{l.action.toUpperCase()}} ${{l.strike}} ${{l.type}} @${{l.premium.toFixed(2)}}</span>`).join("");

    return `<div class="strat-card" style="--cc:${{cc}};animation-delay:${{i*0.05}}s" onclick="selectPayoff('${{s.name}}')">
      <div class="sc-top">
        <div>
          <div class="sc-name">${{emojis[s.biasTag]||"📊"}} ${{s.name}}</div>
          <div class="sc-sub">${{s.biasTag.toUpperCase()}} · ${{s.isDebit?"DEBIT":"CREDIT"}} SPREAD · DTE:${{ALL_DATA[currentExpiry]?.dte||"—"}}</div>
        </div>
        <div class="pop-pill" style="background:${{popBg}};color:${{popCol}};border:1px solid ${{popCol}}33;">${{s.pop}}%<br><span style="font-size:8px;font-weight:400;">PoP</span></div>
      </div>
      <div class="sc-fields">
        <div class="sc-field"><span class="sc-field-lbl">Strike Price</span><span class="sc-field-val" style="color:var(--cyan);">ATM ₹${{ALL_DATA[currentExpiry]?.atm_strike?.toLocaleString("en-IN")||"—"}}</span></div>
        <div class="sc-field"><span class="sc-field-lbl">Max Profit</span><span class="sc-field-val up">₹${{s.maxProfit.toLocaleString("en-IN")}}</span></div>
        <div class="sc-field"><span class="sc-field-lbl">Max Loss</span><span class="sc-field-val down">₹${{s.maxLoss.toLocaleString("en-IN")}}</span></div>
        <div class="sc-field"><span class="sc-field-lbl">Max RR Ratio</span><span class="sc-field-val" style="color:var(--gold);">1:${{rrDisp}}</span></div>
        <div class="sc-field"><span class="sc-field-lbl">Breakevens</span><span class="sc-field-val" style="font-size:10px;color:var(--text2);">${{beStr}}</span></div>
        <div class="sc-field"><span class="sc-field-lbl">Net Credit/Debit</span><span class="sc-field-val">${{netDisp}}</span></div>
        <div class="sc-field" style="grid-column:1/-1"><span class="sc-field-lbl">Est. Margin / Premium</span><span class="sc-field-val" style="color:var(--purple);">₹${{s.margin.toLocaleString("en-IN")}}</span></div>
      </div>
      <div class="sc-legs">${{legTags}}</div>
      <div class="sc-score">
        <span class="score-lbl">SCORE</span>
        <div class="score-bar-track"><div class="score-bar-fill" style="width:${{sw}}%"></div></div>
        <span class="score-num">${{s.score}}</span>
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
// RICH PAYOFF CHART — BSM Today line + Expiry line + OI bars
// ═══════════════════════════════════════════════════════════

// BSM option price at a given spot & time
function bsmPrice(S, K, T, r, sigma, type) {{
  if (T <= 0 || sigma <= 0) {{
    return type === "CE" ? Math.max(0, S - K) : Math.max(0, K - S);
  }}
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  if (type === "CE") return Math.max(0, S * normCdf(d1) - K * Math.exp(-r * T) * normCdf(d2));
  return Math.max(0, K * Math.exp(-r * T) * normCdf(-d2) - S * normCdf(-d1));
}}

// Compute P&L for a strategy at a given spot price and time T
function stratPnlAtSpot(legs, spotPrice, T) {{
  let pnl = 0;
  legs.forEach(l => {{
    const sigma  = (l.iv || 15) / 100;
    const theoVal = bsmPrice(spotPrice, l.strike, T, RISK_FREE, sigma, l.type);
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
  const dte        = d?.dte || 1;
  const T_today    = Math.max(dte / 365, 0.0001);
  const T_expiry   = 0;

  // Price range: spot ± 1500 in steps of 25
  const priceRange = [];
  for (let p = underlying - 1500; p <= underlying + 1500; p += 25) priceRange.push(p);

  // Today P&L (BSM with T_today — includes Theta, Vega, Delta effects)
  const todayPnl   = priceRange.map(p => stratPnlAtSpot(s.legs, p, T_today));
  // Expiry P&L (intrinsic only, T=0)
  const expiryPnl  = priceRange.map(p => stratPnlAtSpot(s.legs, p, T_expiry));

  // OI data for background bars (align to priceRange)
  const allStrikes = d?.all_strikes || [];
  const strikeSet  = new Set(priceRange);
  const oiLabels   = priceRange;
  const ceOiData   = priceRange.map(p => {{
    const row = allStrikes.find(r => r.strike === p);
    return row ? Math.round(row.ce_oi / 1e3) : null;  // in thousands
  }});
  const peOiData   = priceRange.map(p => {{
    const row = allStrikes.find(r => r.strike === p);
    return row ? Math.round(row.pe_oi / 1e3) : null;
  }});

  // Net cost of strategy (for % calculation)
  const netCost = Math.abs(s.netPrem) * LOT_SIZE || 1;

  // Current projected P&L at spot
  const projPnl     = stratPnlAtSpot(s.legs, underlying, T_today);
  const projPct     = ((projPnl / netCost) * 100).toFixed(1);
  const projCol     = projPnl >= 0 ? "#00c896" : "#ff6b6b";
  const projSign    = projPnl >= 0 ? "+" : "";
  document.getElementById("projBadge").innerHTML =
    `Projected P&L: <span style="color:${{projCol}};font-weight:800;">${{projSign}}₹${{Math.round(projPnl).toLocaleString("en-IN")}} (${{projSign}}${{projPct}}%)</span>`;

  const ctx = document.getElementById("payoffChart").getContext("2d");
  if (payoffChart) payoffChart.destroy();

  // Crosshair state
  let crosshairX = null;

  payoffChart = new Chart(ctx, {{
    type: "bar",
    data: {{
      labels: priceRange,
      datasets: [
        // ── CE OI bars (green, background, secondary Y) ──
        {{
          label:           "CE OI",
          type:            "bar",
          data:            ceOiData,
          backgroundColor: "rgba(0,200,150,0.18)",
          borderColor:     "rgba(0,200,150,0.35)",
          borderWidth:     1,
          yAxisID:         "yOI",
          order:           3,
          barPercentage:   0.6,
        }},
        // ── PE OI bars (red, background, secondary Y) ──
        {{
          label:           "PE OI",
          type:            "bar",
          data:            peOiData,
          backgroundColor: "rgba(255,107,107,0.18)",
          borderColor:     "rgba(255,107,107,0.35)",
          borderWidth:     1,
          yAxisID:         "yOI",
          order:           3,
          barPercentage:   0.6,
        }},
        // ── Today P&L line (green, BSM with Greeks) ──
        {{
          label:       "Today P&L (BSM)",
          type:        "line",
          data:        todayPnl,
          borderColor: "#00c896",
          borderWidth: 2.5,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: "#00c896",
          fill: {{target: {{value: 0}}, above: "rgba(0,200,150,0.12)", below: "rgba(255,107,107,0.10)"}},
          tension:  0.3,
          yAxisID:  "yPnl",
          order:    1,
        }},
        // ── Expiry P&L line (blue, intrinsic) ──
        {{
          label:       "Expiry P&L",
          type:        "line",
          data:        expiryPnl,
          borderColor: "#5ba3ff",
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: "#5ba3ff",
          borderDash:  [4, 3],
          fill:        false,
          tension:     0.1,
          yAxisID:     "yPnl",
          order:       2,
        }},
        // ── Zero line ──
        {{
          label:       "Zero",
          type:        "line",
          data:        priceRange.map(() => 0),
          borderColor: "rgba(255,255,255,0.10)",
          borderWidth: 1,
          pointRadius: 0,
          fill:        false,
          yAxisID:     "yPnl",
          order:       4,
        }},
      ]
    }},
    options: {{
      responsive:          true,
      maintainAspectRatio: false,
      interaction: {{mode: "index", intersect: false, axis: "x"}},
      plugins: {{
        legend: {{
          display: true,
          position: "top",
          labels: {{
            color:     "#6a8aaa",
            font:      {{family: "DM Mono", size: 10}},
            boxWidth:  14,
            filter:    item => item.text !== "Zero",
          }}
        }},
        tooltip: {{
          enabled:         false,   // use custom tooltip
          mode:            "index",
          intersect:       false,
        }},
        // Spot price vertical annotation drawn via afterDraw
      }},
      scales: {{
        x: {{
          ticks: {{
            color:         "#2d4560",
            font:          {{family: "DM Mono", size: 9}},
            maxTicksLimit: 12,
            callback:      v => v.toLocaleString("en-IN"),
          }},
          grid:   {{color: "#0b1520"}},
          border: {{color: "#1a2535"}},
        }},
        yPnl: {{
          position: "left",
          ticks: {{
            color:    "#2d4560",
            font:     {{family: "DM Mono", size: 9}},
            callback: v => "₹" + (Math.abs(v) >= 1000 ? (v/1000).toFixed(0)+"K" : v),
          }},
          grid:   {{color: "#0b1520"}},
          border: {{color: "#1a2535"}},
          title:  {{display:true, text:"Profit / Loss", color:"#3d5a73", font:{{size:9, family:"DM Mono"}}}},
        }},
        yOI: {{
          position: "right",
          ticks: {{
            color:    "#2d4560",
            font:     {{family: "DM Mono", size: 9}},
            callback: v => v >= 1000 ? (v/1000).toFixed(0)+"L" : v,
          }},
          grid: {{drawOnChartArea: false}},
          border: {{color: "#1a2535"}},
          title: {{display:true, text:"Open Interest", color:"#3d5a73", font:{{size:9, family:"DM Mono"}}}},
        }},
      }},
    }},
    plugins: [{{
      // ── Spot line + crosshair drawn on canvas ──────────────────
      id: "crosshairSpot",
      afterDatasetsDraw(chart) {{
        const xScale = chart.scales.x;
        const yScale = chart.scales.yPnl;
        const ctx2   = chart.ctx;

        // ── Spot vertical line ──
        const spotIdx = priceRange.findIndex(p => p >= underlying);
        if (spotIdx >= 0) {{
          const xPx = xScale.getPixelForValue(priceRange[spotIdx]);
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

        // ── Crosshair line + dots ──
        if (crosshairX === null) return;
        const nearXPx  = xScale.getPixelForValue(priceRange[crosshairX]);
        const todayVal = todayPnl[crosshairX];
        const expVal   = expiryPnl[crosshairX];

        ctx2.save();
        ctx2.setLineDash([3,3]);
        ctx2.strokeStyle = "rgba(255,255,255,0.18)";
        ctx2.lineWidth   = 1;
        ctx2.beginPath();
        ctx2.moveTo(nearXPx, yScale.top);
        ctx2.lineTo(nearXPx, yScale.bottom);
        ctx2.stroke();
        ctx2.setLineDash([]);

        // Today dot
        ctx2.fillStyle   = "#00c896";
        ctx2.strokeStyle = "#060910";
        ctx2.lineWidth   = 2.5;
        ctx2.beginPath();
        ctx2.arc(nearXPx, yScale.getPixelForValue(todayVal), 5, 0, Math.PI*2);
        ctx2.fill(); ctx2.stroke();

        // Expiry dot
        ctx2.fillStyle   = "#5ba3ff";
        ctx2.strokeStyle = "#060910";
        ctx2.beginPath();
        ctx2.arc(nearXPx, yScale.getPixelForValue(expVal), 5, 0, Math.PI*2);
        ctx2.fill(); ctx2.stroke();
        ctx2.restore();
      }},

      // ── Mouse/Touch → update HTML tooltip overlay ──────────────
      afterEvent(chart, args) {{
        const event  = args.event;
        const xScale = chart.scales.x;
        const tt     = document.getElementById("payoffTooltip");

        if (["mouseleave","touchend","mouseout"].includes(event.type)) {{
          crosshairX = null;
          tt.style.display = "none";
          chart.draw();
          return;
        }}
        if (!["mousemove","touchmove","touchstart"].includes(event.type)) return;

        const evtX = event.x;
        if (evtX < xScale.left || evtX > xScale.right) {{
          crosshairX = null;
          tt.style.display = "none";
          chart.draw();
          return;
        }}

        // Nearest price index
        let minDist = Infinity, bestIdx = 0;
        priceRange.forEach((p, i) => {{
          const dist = Math.abs(xScale.getPixelForValue(p) - evtX);
          if (dist < minDist) {{ minDist = dist; bestIdx = i; }}
        }});

        if (crosshairX !== bestIdx) {{
          crosshairX = bestIdx;
          chart.draw();
        }}

        // ── Build HTML tooltip ──
        const price    = priceRange[bestIdx];
        const pctChg   = (((price - underlying) / underlying) * 100).toFixed(1);
        const sign     = pctChg >= 0 ? "+" : "";
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
          <div style="font-size:9px;color:#6a8aaa;margin-bottom:4px;letter-spacing:.5px;">WHEN PRICE IS AT</div>
          <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:8px;">
            <span style="font-size:16px;font-weight:800;color:#ddeeff;font-family:'DM Mono',monospace;">₹${{price.toLocaleString("en-IN")}}</span>
            <span style="font-size:11px;font-weight:700;color:${{pCol}};font-family:'DM Mono',monospace;">${{sign}}${{pctChg}}% (${{sign}}${{(price-underlying).toLocaleString("en-IN")}})</span>
          </div>
          <div style="height:1px;background:rgba(255,255,255,0.07);margin-bottom:8px;"></div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
            <span style="font-size:9px;color:#6a8aaa;display:flex;align-items:center;gap:5px;"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#00c896;"></span>Today (BSM)</span>
            <span style="font-size:12px;font-weight:800;color:${{tCol}};font-family:'DM Mono',monospace;">${{tSign}}₹${{Math.round(todayVal).toLocaleString("en-IN")}}&nbsp;<span style="font-size:9px;">(${{tSign}}${{tPct}}%)</span></span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:9px;color:#6a8aaa;display:flex;align-items:center;gap:5px;"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#5ba3ff;"></span>At Expiry</span>
            <span style="font-size:12px;font-weight:800;color:${{eCol}};font-family:'DM Mono',monospace;">${{eSign}}₹${{Math.round(expVal).toLocaleString("en-IN")}}&nbsp;<span style="font-size:9px;">(${{eSign}}${{ePct}}%)</span></span>
          </div>`;

        // Position tooltip — flip left if near right edge
        const canvasRect = chart.canvas.getBoundingClientRect();
        const xPxAbs     = xScale.getPixelForValue(price);
        const ttW        = 260;
        let   leftPx     = xPxAbs + 14;
        if (leftPx + ttW > chart.width - 10) leftPx = xPxAbs - ttW - 14;
        tt.style.left    = leftPx + "px";
        tt.style.top     = (chart.scales.yPnl.top + 10) + "px";
        tt.style.display = "block";
      }},
    }}],
  }});
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
