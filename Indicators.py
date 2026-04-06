import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from datetime import datetime, timedelta
import requests
import time
import warnings
from utils.API import FRED_API_KEY
from io import BytesIO

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler

warnings.filterwarnings('ignore')

CORRELATION_DROP_THRESHOLD = 0.9

# ========== 1. CONFIGURATION ==========

# FRED series IDs (TED removed - obsolete after 2023)
FRED_SERIES = {
    "HY_OAS": "BAMLH0A0HYM2",      # High-yield credit spread
    "IG_OAS": "BAMLC0A0CM",        # Investment-grade spread
    "T10Y2Y": "T10Y2Y",            # 10Y-2Y yield spread
    "T10Y3M": "T10Y3M",            # 10Y-3M yield spread
    "TIPS": "T10YIE",              # TIPS breakeven inflation
}

# Yahoo Finance symbols
YAHOO_SYMBOLS = {
    "VIX": "^VIX",
    "VVIX": "^VVIX",
    "SKEW": "^SKEW",
    "DXY": "DX-Y.NYB",
    "GOLD": "GC=F",
    "OVX": "^OVX",
}

# VIX term structure
VIX_TERM_SYMBOLS = {
    "VIX9D": "^VIX9D",
    "VIX3M": "^VIX3M",
    "VIX6M": "^VIX6M",
}

# ========== 2. FETCH FUNCTIONS ==========

def fetch_fred_data(fred, series_dict, start_date, end_date):
    """Fetch all FRED indicators"""
    dfs = []
    print("\nFetching from FRED...")
    
    for name, series_id in series_dict.items():
        try:
            data = fred.get_series(series_id, start_date, end_date)
            df = pd.DataFrame(data, columns=[name])
            # Ensure index is just date (no time)
            df.index = pd.to_datetime(df.index)
            dfs.append(df)
            print(f"  ✓ {name}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    return dfs

def fetch_yahoo_data(symbols_dict, start_date, end_date):
    """Fetch all Yahoo Finance indicators and return merged DataFrame"""
    print("\nFetching from Yahoo Finance...")
    
    all_data = {}
    for name, symbol in symbols_dict.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            if not hist.empty:
                # Store as Series with date index
                all_data[name] = hist["Close"]
                print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    if not all_data:
        return None
    
    # Combine all into one DataFrame
    df = pd.DataFrame(all_data)
    # Ensure index is date-only
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


# ========== GPR FETCH ==========

def fetch_gpr(start_date, end_date, xls_path=None):
    """
    Fetch GPR from local XLS file or URL fallback.
    Columns: GPRD (headline), GPRD_THREAT (forward-looking), GPRD_ACT (realised)
    """
    print("\nFetching GPR (Geopolitical Risk Index)...")

    try:
        if xls_path is not None:
            df = pd.read_excel(xls_path, engine="xlrd")
        else:
            # URL fallback
            url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            df = pd.read_excel(BytesIO(response.content), engine="xlrd")

        # Use the 'date' column as index
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index)

        # Keep only signal columns, rename to clean names
        keep = {
          #  "GPRD":        "GPR",
            "GPRD_THREAT": "GPRT",
           # "GPRD_ACT":    "GPRA",
        }
        df = df[[c for c in keep if c in df.columns]].rename(columns=keep)

        # Filter to requested range
        df = df[(df.index >= pd.to_datetime(start_date)) &
                (df.index <= pd.to_datetime(end_date))]

        print(f"  ✓ GPR ({len(df)} days, {df.index.min().date()} → {df.index.max().date()})")
        return df

    except Exception as e:
        print(f"  ✗ GPR: {e}")
        return None


# ========== FED FUNDS FUTURES FETCH ==========

def fetch_fed_funds_futures(fred, start_date, end_date):
    """
    Fed Funds Futures implied rate + surprise signal.

    ZQ=F (Yahoo)  →  implied_rate = 100 - Close
    DFF   (FRED)  →  daily effective Fed Funds rate
    surprise      =  implied_rate - DFF  (market pricing relative to current rate)
                     positive  = market expects hike
                     negative  = market expects cut
    """
    print("\nFetching Fed Funds Futures...")

    try:
        # --- Front-month futures price from Yahoo ---
        ticker = yf.Ticker("ZQ=F")
        hist = ticker.history(start=start_date, end=end_date)
        if hist.empty:
            print("  ✗ ZQ=F: no data returned from Yahoo")
            return None

        futures_df = pd.DataFrame()
        futures_df.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
        futures_df = pd.DataFrame(
        {"FF_futures_price": hist["Close"].values},
        index=pd.to_datetime(hist.index).tz_localize(None).normalize()
        )
        futures_df["FF_implied_rate"] = 100 - futures_df["FF_futures_price"]
        print("  ✓ ZQ=F (front-month futures)")

        # --- Daily effective rate from FRED ---
        dff = fred.get_series("DFF", start_date, end_date)
        dff_df = pd.DataFrame(dff, columns=["FF_effective_rate"])
        dff_df.index = pd.to_datetime(dff_df.index)
        print("  ✓ DFF (daily effective rate)")

        # --- Merge and compute surprise ---
        merged = futures_df.join(dff_df, how="left")
        merged["FF_effective_rate"] = merged["FF_effective_rate"].ffill()
        merged["FF_surprise"] = merged["FF_implied_rate"] - merged["FF_effective_rate"]

        print("  ✓ FF_surprise (implied − effective)")
        return merged[["FF_implied_rate", "FF_effective_rate", "FF_surprise"]]

    except Exception as e:
        print(f"  ✗ Fed Funds Futures: {e}")
        return None

def fetch_crypto_fear_greed(start_date, end_date):
    """Fetch Crypto Fear & Greed Index from alternative.me"""
    print("\nFetching Crypto Fear & Greed Index...")
    
    try:
        # Calculate number of days needed
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days_needed = (end - start).days + 100  # Add buffer
        
        url = "https://api.alternative.me/fng/"
        params = {"limit": min(days_needed, 2000), "format": "json"}
        response = requests.get(url, params=params)
        data = response.json()
        
        records = []
        for item in data["data"]:
            date_str = datetime.fromtimestamp(int(item["timestamp"])).strftime("%Y-%m-%d")
            records.append({
                "date": datetime.strptime(date_str, "%Y-%m-%d").date(),
                "Fear_Greed": int(item["value"])
            })
        
        df = pd.DataFrame(records)
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Filter to date range
        df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        
        print(f"  ✓ Fear & Greed Index ({len(df)} days)")
        return df
    except Exception as e:
        print(f"  ✗ Fear & Greed: {e}")
        return None

def calculate_derived_features(df):
    """Calculate derived features from raw indicators"""
    print("\nCalculating derived features...")
    
    # Fill missing values
    df = df.ffill().bfill().fillna(0)
    
    # HY - IG spread (risk appetite delta)
    if "HY_OAS" in df.columns and "IG_OAS" in df.columns:
        df["HY_IG_spread"] = df["HY_OAS"] - df["IG_OAS"]
        print("  ✓ HY_IG_spread")
    
    # VIX term structure slope (VIX3M - VIX)
    if "VIX3M" in df.columns and "VIX" in df.columns:
        df["VIX_term_slope"] = df["VIX3M"] - df["VIX"]
        print("  ✓ VIX_term_slope")
    
    # 1-day and 5-day differences for slow-moving indicators
    for col in ["HY_OAS", "IG_OAS", "DXY"]:
        if col in df.columns:
            df[f"{col}_diff1"] = df[col].diff(1).fillna(0)
            df[f"{col}_diff5"] = df[col].diff(5).fillna(0)
            print(f"  ✓ {col}_diff1, {col}_diff5")
    
   
    # Yield curve binary regime signal
    if "T10Y2Y" in df.columns:
        df["yield_curve_inverted"] = (df["T10Y2Y"] < 0).astype(int)
        print("  ✓ yield_curve_inverted")
    
    return df

def align_and_lag_all_data(data_dict, lag_days=1):
    """Combine all data sources with proper lag to avoid lookahead bias"""
    print("\nMerging all data sources...")
    
    # Combine all DataFrames
    combined = None
    for name, df in data_dict.items():
        if df is not None and not df.empty:
            print(f"  Merging {name}: {df.shape[0]} rows, {df.shape[1]} cols")
            if combined is None:
                combined = df.copy()
            else:
                combined = combined.join(df, how="outer")
    
    if combined is None:
        print("Error: No data to combine")
        return None
    
    # Sort by date
    combined = combined.sort_index()

    combined = combined[~combined.index.duplicated(keep='last')]

    print("Calculating RVOL on raw data (before ffill)...")
    for col in ["DXY", "VIX", "GOLD"]:
        if col in combined.columns:
            combined[f"RVOL_{col}"] = (
                combined[col]
                .pct_change()
                .rolling(20)
                .std()
            )
            print(f"  ✓ RVOL_{col}")
    
    # Fill missing values before lag
    combined = combined.ffill().bfill().fillna(0)
    
    # Apply lag (t-1 to avoid lookahead)
    print(f"Applying {lag_days}-day lag to all features...")
    combined_lagged = combined.shift(lag_days)
    
    # Fill any new NaNs from lag (first row)
    combined_lagged = combined_lagged.ffill().bfill().fillna(0)
    
    

    combined_lagged = calculate_derived_features(combined_lagged)
    
    return combined_lagged

# ========== 3. MAIN FUNCTION ==========

def fetch_all_features(start_date="2012-01-01", end_date="2026-04-04", fred_api_key=None):
    """
    Main function to fetch all ML features for PPO portfolio optimization
    """
    
    if fred_api_key is None:
        from utils.API import FRED_API_KEY
        print("WARNING: No FRED API key provided.")
        fred = Fred(api_key=FRED_API_KEY)
    else:
        fred = Fred(api_key=fred_api_key)
    
    print("=" * 60)
    print("FETCHING ML FEATURES FOR CRYPTO FORECASTING + PPO")
    print("=" * 60)
    print(f"Date range: {start_date} to {end_date}")
    
    data_dict = {}
    
    # 1. FRED data (credit + macro)
    if fred is not None:
        fred_dfs = fetch_fred_data(fred, FRED_SERIES, start_date, end_date)
        if fred_dfs:
            data_dict["fred"] = pd.concat(fred_dfs, axis=1)
    
    # 2. Yahoo Finance (volatility + cross-asset)
    yahoo_df = fetch_yahoo_data(YAHOO_SYMBOLS, start_date, end_date)
    if yahoo_df is not None and not yahoo_df.empty:
        data_dict["yahoo"] = yahoo_df


    # 3. VIX term structure
    vix_term_df = fetch_yahoo_data(VIX_TERM_SYMBOLS, start_date, end_date)
    if vix_term_df is not None and not vix_term_df.empty:
        data_dict["vix_term"] = vix_term_df
    
    # 4. Crypto Fear & Greed
    fg_df = fetch_crypto_fear_greed(start_date, end_date)
    if fg_df is not None:
        data_dict["fear_greed"] = fg_df
    

    gpr_df = fetch_gpr(start_date, end_date, xls_path="data/gpr_daily.xls")
    if gpr_df is not None:
        data_dict["gpr"] = gpr_df

    # 6. Fed Funds Futures
    ff_df = fetch_fed_funds_futures(fred, start_date, end_date)
    if ff_df is not None:
        data_dict["fed_funds"] = ff_df

    # 5. BTC Dominance (SKIPPED - too slow, optional)
    print("\nSkipping BTC Dominance (API too slow - optional feature)")
    
    # Combine all with proper lag
    final_df = align_and_lag_all_data(data_dict, lag_days=1)
    
    return final_df

def correlation_drop(df, threshold=CORRELATION_DROP_THRESHOLD) -> pd.DataFrame:
    """
    Drop features with pairwise correlation above threshold.
    
    Protected columns are never dropped regardless of correlation.
    For each correlated pair, the column with lower variance is dropped
    (proxy for information content when no target is available).
    """

    # These are NEVER dropped — each carries unique information
    # or was explicitly chosen for a specific reason
    PROTECTED = {
        "VIX",               # core volatility anchor
        "DXY",               # FX risk-off signal
        "T10Y2Y",            # yield curve level
        "Fear_Greed",        # crypto-native sentiment (different methodology from VIX)
        "RVOL_VIX",          # realised vol (complements implied VIX)
        "VIX_term_slope",    # term structure shape — derived, not raw
        "HY_IG_spread",      # credit risk appetite delta — derived
        "GOLD",              # safe-haven cross-asset
        "yield_curve_inverted",  # binary regime signal
        "GPRT",              # geopolitical threat — forward-looking
        "FF_surprise",       # rate surprise signal — derived
        "HY_OAS_diff1",      # credit momentum (different timescale from diff5)
        "HY_OAS_diff5",      # credit momentum (different timescale from diff1)
        "DXY_diff1",         
        "DXY_diff5",         
    }

    cols = df.columns.tolist()
    corr_matrix = df.corr().abs()

    dropped = set()
    drop_log = []

    for i, col_a in enumerate(cols):
        if col_a in dropped:
            continue
        for col_b in cols[i + 1:]:
            if col_b in dropped:
                continue
            if corr_matrix.loc[col_a, col_b] >= threshold:
                # Decide which to drop: protected > variance
                a_protected = col_a in PROTECTED
                b_protected = col_b in PROTECTED

                if a_protected and b_protected:
                    # Both protected — keep both, just log
                    drop_log.append(
                        f"  ⚠ BOTH PROTECTED — keeping [{col_a}, {col_b}] "
                        f"(corr={corr_matrix.loc[col_a, col_b]:.3f})"
                    )
                    continue
                elif b_protected:
                    to_drop = col_a
                elif a_protected:
                    to_drop = col_b
                else:
                    # Neither protected — drop the lower-variance one
                    to_drop = col_a if df[col_a].var() < df[col_b].var() else col_b

                dropped.add(to_drop)
                kept = col_b if to_drop == col_a else col_a
                drop_log.append(
                    f"  ✗ DROP [{to_drop}] — corr({col_a}, {col_b})"
                    f"={corr_matrix.loc[col_a, col_b]:.3f}, kept [{kept}]"
                )

    print("\n" + "=" * 60)
    print(f"CORRELATION DROP (threshold={threshold})")
    print("=" * 60)
    if drop_log:
        for line in drop_log:
            print(line)
    else:
        print("  No pairs above threshold — nothing dropped.")
    
    remaining = [c for c in cols if c not in dropped]
    print(f"\n  Before: {len(cols)} features -> After: {len(remaining)} features")
    if dropped:
        print(f"  Dropped: {sorted(dropped)}")

    return df[remaining]


# ========== 4. SAVE AND VALIDATE ==========
def save_and_validate(df, filename="market_indicators.csv"):
    """Save dataset and print summary statistics"""
    
    if df is None or df.empty:
        print("\n❌ No data to save")
        return False
    
    # df.rename(columns = {'0: Unnamed':'Date'},inplace=True)

    df = correlation_drop(df)
    
    # Save to CSV
    df.to_csv(filename)
    print(f"\n✓ Saved to {filename}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Features: {list(df.columns)}")
    
    # Check for lookahead bias
    print("\n" + "=" * 60)
    print("LOOKAHEAD BIAS CHECK")
    print("=" * 60)
    print(f"Latest date in dataset: {df.index.max()}")
    print(f"End date requested: 2026-04-01")
    
    # Sample preview
    print("\n" + "=" * 60)
    print("LAST 5 DAYS PREVIEW (ALREADY LAGGED)")
    print("=" * 60)
    print(df.tail(5).round(2))
    
    # Show which features are available
    print("\n" + "=" * 60)
    print("AVAILABLE FEATURES")
    print("=" * 60)
    available = [c for c in ["VIX", "DXY", "GOLD", "OVX", "T10Y2Y", "Fear_Greed", "HY_OAS_diff1"] if c in df.columns]
    print(f"Key features present: {available}")
    
    return True

# ========== 5. EXECUTION ==========

if __name__ == "__main__":
    
    # Fetch all features from 2021-01-01 to 2026-04-01
    features = fetch_all_features(
        start_date="2012-01-01",
        end_date="2026-04-01",
        fred_api_key=FRED_API_KEY
    )
    
    # Save and 
    

    if features is not None:
        keep_cols = [
            "VIX",
            "VIX9D", 
            "VIX3M",
            "HY_OAS_diff1", "HY_OAS_diff5",
            "DXY", "DXY_diff1", "DXY_diff5",
            "T10Y2Y",
            "Fear_Greed",
            "RVOL_VIX",
            "VIX_term_slope",
            "HY_IG_spread",
            "T10Y3M",
            "GOLD",
            "yield_curve_inverted",
            # --- new ---
            "GPR",            # headline geopolitical risk
            "GPRT",           # threat sub-index (forward-looking)
            "GPRA",           # act sub-index (events that materialised)
            "FF_implied_rate", # market's priced-in rate
            "FF_surprise",    # implied − effective (hike/cut pressure)
        ]
        features = features[[c for c in keep_cols if c in features.columns]]
        save_and_validate(features, "data/market_indicators.csv")
        
        # Ready to use for PPO state space
        print("\n" + "=" * 60)
        print("READY FOR PPO STATE SPACE")
        print("=" * 60)
        print(f"State dimension: {len(features.columns)} features")


        # correlation matrix print

        print(features.corr().round(2).to_string())