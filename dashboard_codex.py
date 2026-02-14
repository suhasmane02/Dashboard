from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import multiprocessing
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple
import importlib.util

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}

PARTICIPANTS = ["FII", "DII", "Pro", "Client"]
COLOR_DISCRETE_MAP = {
    "Client": "#1f77b4",
    "DII": "#ff9da7",
    "FII": "#d62728",
    "Pro": "#60BD68",
}
INSTRUMENT_COLUMN_MAP = {
    "Index Futures": ("Future Index Long", "Future Index Short"),
    "Index Call Options": ("Option Index Call Long", "Option IndexCall Short"),
    "Index Put Options": ("Option Index Put Long", "Option Index Put Short"),
    "Stock Futures": ("Future Stock Long", "Future Stock Short"),
    "Stock Call Options": ("Option Stock Call Long", "Option StockCall Short"),
    "Stock Put Options": ("Option Stock Put Long", "Option Stock Put Short"),
}

INSTRUMENT_DIFF_COLUMNS = {
    "Index Futures Long": "Future Index Long",
    "Index Futures Short": "Future Index Short",
    "Index Call Long": "Option Index Call Long",
    "Index Call Short": "Option IndexCall Short",
    "Index Put Long": "Option Index Put Long",
    "Index Put Short": "Option Index Put Short",
    "Stock Futures Long": "Future Stock Long",
    "Stock Futures Short": "Future Stock Short",
    "Stock Call Long": "Option Stock Call Long",
    "Stock Call Short": "Option StockCall Short",
    "Stock Put Long": "Option Stock Put Long",
    "Stock Put Short": "Option Stock Put Short",
}

TICK_DTYPE = np.dtype([
    ("symbol", "U64"),
    ("last_traded_time", np.int64),
    ("ltp", np.float64),
    ("vol_traded_today", np.int64),
    ("ask_size", np.int64),
    ("bid_size", np.int64),
    ("ask_price", np.float64),
    ("bid_price", np.float64),
    ("last_traded_qty", np.int64),
    ("tot_buy_qty", np.int64),
    ("tot_sell_qty", np.int64),
    ("avg_trade_price", np.float64),
])
TICK_NUM_ROWS = 5000
COUNTER_SHAPE = (1, 3)
COUNTER_DTYPE = np.int64


@dataclass
class SentimentSnapshot:
    pcr_oi: float
    pcr_change_oi: float
    total_ce_oi: float
    total_pe_oi: float
    total_ce_change_oi: float
    total_pe_change_oi: float
    label: str
    score: float
    rationale: List[str]


class CandleBuilder:
    """Lightweight candle builder adapted from TickMe for incremental volume tracking."""

    def __init__(self, candle_duration_seconds: int = 180):
        self.candle_duration = candle_duration_seconds
        self.current_candle: Optional[dict] = None
        self.candles: List[dict] = []
        self.prev_price: Optional[float] = None

    def process(self, timestamp: int, price: float, incremental_volume: float) -> None:
        if self.current_candle is None or timestamp >= self.current_candle["timestamp"] + self.candle_duration:
            if self.current_candle is not None:
                self.candles.append(self._finalize())
            self._start(timestamp, price)

        side = self._trade_side(price)
        c = self.current_candle
        c["high"] = max(c["high"], price)
        c["low"] = min(c["low"], price)
        c["close"] = price
        c["volume"] += max(incremental_volume, 0)
        if side == "buy":
            c["buy_volume"] += max(incremental_volume, 0)
        elif side == "sell":
            c["sell_volume"] += max(incremental_volume, 0)
        self.prev_price = price

    def _trade_side(self, price: float) -> str:
        if self.prev_price is None:
            return "neutral"
        if price > self.prev_price:
            return "buy"
        if price < self.prev_price:
            return "sell"
        return "neutral"

    def _start(self, timestamp: int, price: float) -> None:
        bucket = (timestamp // self.candle_duration) * self.candle_duration
        self.current_candle = {
            "timestamp": bucket,
            "datetime": datetime.fromtimestamp(bucket),
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 0.0,
            "buy_volume": 0.0,
            "sell_volume": 0.0,
        }

    def _finalize(self) -> dict:
        c = self.current_candle.copy()
        c["delta"] = c["buy_volume"] - c["sell_volume"]
        return c


def nse_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    session.get("https://www.nseindia.com", timeout=15)
    return session


def _clean_participant_df(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Client Type": "Client Type"})
    df = df[df["Client Type"].isin(PARTICIPANTS)].copy()
    numeric_cols = [c for c in df.columns if c != "Client Type"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["Date"] = pd.to_datetime(as_of)
    return df


def fetch_participant_file(kind: str, as_of: date, session: requests.Session) -> pd.DataFrame:
    file_date = as_of.strftime("%d%m%Y")
    url = f"https://nsearchives.nseindia.com/content/nsccl/fao_participant_{kind}_{file_date}.csv"
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    raw = pd.read_csv(pd.io.common.StringIO(resp.text), skiprows=1)
    return _clean_participant_df(raw, as_of)


def fetch_option_chain(symbol: str, session: requests.Session) -> dict:
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()


def compute_pcr(option_chain: dict) -> Dict[str, float]:
    ce_oi = pe_oi = ce_change = pe_change = 0.0
    for row in option_chain.get("records", {}).get("data", []):
        ce = row.get("CE")
        pe = row.get("PE")
        if ce:
            ce_oi += ce.get("openInterest", 0) or 0
            ce_change += ce.get("changeinOpenInterest", 0) or 0
        if pe:
            pe_oi += pe.get("openInterest", 0) or 0
            pe_change += pe.get("changeinOpenInterest", 0) or 0

    return {
        "pcr_oi": (pe_oi / ce_oi) if ce_oi else np.nan,
        "pcr_change_oi": (pe_change / ce_change) if ce_change else np.nan,
        "total_ce_oi": ce_oi,
        "total_pe_oi": pe_oi,
        "total_ce_change_oi": ce_change,
        "total_pe_change_oi": pe_change,
    }


def participant_strength(
    oi_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    previous_oi_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    prev_map: Dict[str, pd.Series] = {}
    if previous_oi_df is not None and not previous_oi_df.empty:
        prev_map = {r["Client Type"]: r for _, r in previous_oi_df.iterrows()}

    metrics = []
    for p in PARTICIPANTS:
        o = oi_df[oi_df["Client Type"] == p]
        v = vol_df[vol_df["Client Type"] == p]
        if o.empty or v.empty:
            continue
        o = o.iloc[0]
        v = v.iloc[0]
        oi_net = float(o["Total Long Contracts"] - o["Total Short Contracts"])
        prev = prev_map.get(p)
        prev_oi_net = (
            float(prev["Total Long Contracts"] - prev["Total Short Contracts"])
            if prev is not None
            else 0.0
        )
        oi_net_diff = oi_net - prev_oi_net
        oi_idx_options_bias = float(
            (o["Option Index Call Long"] + o["Option Index Put Short"])
            - (o["Option IndexCall Short"] + o["Option Index Put Long"])
        ) if "Option IndexCall Short" in o.index else oi_net
        vol_net = float(v["Total Long Contracts"] - v["Total Short Contracts"])
        direction_signal = (oi_net_diff * 0.7) + (vol_net * 0.3)
        row_metrics = {
            "Client Type": p,
            "Net OI": oi_net,
            "Previous Net OI": prev_oi_net,
            "OI Day Difference": oi_net_diff,
            "Net Volume": vol_net,
            "Directional Score": direction_signal,
            "Index Option Bias": oi_idx_options_bias,
        }

        for metric_name, col in INSTRUMENT_DIFF_COLUMNS.items():
            if col not in o.index:
                continue
            current_val = float(o[col])
            prev_val = float(prev[col]) if prev is not None else 0.0
            row_metrics[f"{metric_name} Day Difference"] = current_val - prev_val

        metrics.append(row_metrics)
    out = pd.DataFrame(metrics)
    if out.empty:
        return out
    scale = out["Directional Score"].abs().max() or 1
    out["Directional Score Z"] = out["Directional Score"] / scale
    return out.sort_values("Directional Score", ascending=False)



def build_instrument_diff_history(hist_oi: pd.DataFrame) -> pd.DataFrame:
    hist_oi = hist_oi.sort_values(["Date", "Client Type"]).copy()
    diff_records = []

    for instrument_label, col in INSTRUMENT_DIFF_COLUMNS.items():
        if col not in hist_oi.columns:
            continue
        tmp = hist_oi[["Date", "Client Type", col]].copy()
        tmp["Instrument"] = instrument_label
        tmp["Day Difference"] = tmp.groupby("Client Type")[col].diff().fillna(0)
        diff_records.append(tmp[["Date", "Client Type", "Instrument", "Day Difference"]])

    if not diff_records:
        return pd.DataFrame(columns=["Date", "Client Type", "Instrument", "Day Difference"])
    return pd.concat(diff_records, ignore_index=True)



def build_sentiment_snapshot(
    pcr_metrics: Dict[str, float], participant_df: pd.DataFrame
) -> SentimentSnapshot:
    score = 0.0
    rationale = []

    pcr_oi = pcr_metrics["pcr_oi"]
    pcr_change_oi = pcr_metrics["pcr_change_oi"]

    if pd.notna(pcr_oi):
        if pcr_oi > 1.2:
            score += 1.2
            rationale.append(f"PCR OI elevated at {pcr_oi:.2f} (bullish risk-on put support).")
        elif pcr_oi < 0.8:
            score -= 1.2
            rationale.append(f"PCR OI weak at {pcr_oi:.2f} (bearish call-heavy positioning).")

    if pd.notna(pcr_change_oi):
        if pcr_change_oi > 1.1:
            score += 0.8
            rationale.append("Change-in-OI PCR rising -> fresh put OI growth dominates.")
        elif pcr_change_oi < 0.9:
            score -= 0.8
            rationale.append("Change-in-OI PCR falling -> fresh call OI growth dominates.")

    if not participant_df.empty:
        fii = participant_df[participant_df["Client Type"] == "FII"]
        pro = participant_df[participant_df["Client Type"] == "Pro"]
        client = participant_df[participant_df["Client Type"] == "Client"]
        if not fii.empty and not pro.empty:
            alignment = np.sign(fii["Directional Score"].iloc[0]) == np.sign(pro["Directional Score"].iloc[0])
            if alignment:
                score += 1.4 * np.sign(fii["Directional Score"].iloc[0])
                rationale.append("FII and Pro are aligned directionally.")
        if not client.empty and not pro.empty:
            if np.sign(client["Directional Score"].iloc[0]) != np.sign(pro["Directional Score"].iloc[0]):
                score += 0.6 * np.sign(pro["Directional Score"].iloc[0])
                rationale.append("Smart money vs retail divergence supports contrarian continuation.")

    if score > 1.2:
        label = "Bullish"
    elif score < -1.2:
        label = "Bearish"
    else:
        label = "Neutral"

    return SentimentSnapshot(
        pcr_oi=pcr_oi,
        pcr_change_oi=pcr_change_oi,
        total_ce_oi=pcr_metrics["total_ce_oi"],
        total_pe_oi=pcr_metrics["total_pe_oi"],
        total_ce_change_oi=pcr_metrics["total_ce_change_oi"],
        total_pe_change_oi=pcr_metrics["total_pe_change_oi"],
        label=label,
        score=score,
        rationale=rationale,
    )


def next_day_direction(participant_df: pd.DataFrame) -> Tuple[str, str]:
    if participant_df.empty:
        return "Insufficient", "Participant data unavailable"
    total = participant_df["OI Day Difference"].sum()
    fii_pro = participant_df[participant_df["Client Type"].isin(["FII", "Pro"])]["OI Day Difference"].sum()
    if fii_pro > 0 and total > 0:
        return "Up Bias", "Post-close positioning favors upside continuation"
    if fii_pro < 0 and total < 0:
        return "Down Bias", "Post-close positioning favors downside continuation"
    return "Range / Reversal Risk", "Mixed participant behavior"


def select_itm_strikes(option_chain: dict, count: int = 5) -> Tuple[List[dict], List[dict], float]:
    underlying = option_chain.get("records", {}).get("underlyingValue") or 0
    rows = option_chain.get("records", {}).get("data", [])
    ce_rows = [r for r in rows if r.get("CE") and r.get("strikePrice", 0) <= underlying]
    pe_rows = [r for r in rows if r.get("PE") and r.get("strikePrice", 0) >= underlying]

    ce_rows = sorted(ce_rows, key=lambda x: underlying - x["strikePrice"])[:count]
    pe_rows = sorted(pe_rows, key=lambda x: x["strikePrice"] - underlying)[:count]
    return ce_rows, pe_rows, underlying


def update_candle_state(symbol: str, strikes: List[dict], side: str) -> pd.DataFrame:
    if "candle_state" not in st.session_state:
        st.session_state["candle_state"] = {}
    state = st.session_state["candle_state"]

    now_ts = int(datetime.now().timestamp())
    out = []
    for row in strikes:
        strike = int(row["strikePrice"])
        leg = row.get(side, {})
        key = f"{symbol}_{side}_{strike}"
        tracker = state.get(key)
        if tracker is None:
            tracker = {"builder": CandleBuilder(180), "prev_volume": leg.get("totalTradedVolume", 0)}
            state[key] = tracker

        total_volume = leg.get("totalTradedVolume", 0) or 0
        incremental = total_volume - tracker["prev_volume"]
        tracker["prev_volume"] = total_volume

        tracker["builder"].process(
            timestamp=now_ts,
            price=float(leg.get("lastPrice", 0) or 0),
            incremental_volume=float(max(incremental, 0)),
        )

        candles = tracker["builder"].candles
        if tracker["builder"].current_candle:
            candles = candles + [tracker["builder"]._finalize()]
        for c in candles[-30:]:
            out.append(
                {
                    "datetime": c["datetime"],
                    "strike": strike,
                    "buy_volume": c["buy_volume"],
                    "sell_volume": c["sell_volume"],
                    "delta": c["delta"],
                    "option_type": side,
                }
            )

    return pd.DataFrame(out)


def render_candle_volume_chart(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        st.info(f"No candle build data yet for {title}.")
        return
    fig = px.bar(
        df,
        x="datetime",
        y=["buy_volume", "sell_volume"],
        color_discrete_map={"buy_volume": "#2ca02c", "sell_volume": "#d62728"},
        facet_row="strike",
        title=title,
        barmode="group",
        height=800,
    )
    st.plotly_chart(fig, use_container_width=True)


def create_participant_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
    chart_type: str,
) -> go.Figure:
    if chart_type == "Line":
        fig = px.line(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            color_discrete_map=COLOR_DISCRETE_MAP,
            markers=True,
        )
    else:
        fig = px.bar(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            color_discrete_map=COLOR_DISCRETE_MAP,
            barmode="group",
        )
    fig.update_layout(height=450, hovermode="x unified")
    return fig


def _instrument_frame(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    long_col, short_col = INSTRUMENT_COLUMN_MAP[instrument]
    if long_col not in df.columns or short_col not in df.columns:
        return pd.DataFrame()
    inst_df = df[["Date", "Client Type", long_col, short_col]].copy()
    return inst_df.melt(id_vars=["Date", "Client Type"], var_name="Position Type", value_name="Contracts")


def _instrument_diff_frame(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    inst_df = _instrument_frame(df, instrument)
    if inst_df.empty:
        return inst_df
    inst_df = inst_df.sort_values(["Client Type", "Position Type", "Date"])
    inst_df["Contracts"] = inst_df.groupby(["Client Type", "Position Type"])["Contracts"].diff().fillna(0)
    return inst_df


def render_participant_oi_volume_tab(oi_frames: List[pd.DataFrame], vol_frames: List[pd.DataFrame]) -> None:
    hist_oi = pd.concat(oi_frames, ignore_index=True)
    hist_vol = pd.concat(vol_frames, ignore_index=True)
    hist_oi = hist_oi.sort_values(["Date", "Client Type"])
    hist_vol = hist_vol.sort_values(["Date", "Client Type"])

    selected_clients = st.multiselect("Client Types", PARTICIPANTS, default=PARTICIPANTS)
    chart_type = st.radio("Chart Type", ["Bar", "Line"], horizontal=True)

    filtered_oi = hist_oi[hist_oi["Client Type"].isin(selected_clients)]
    filtered_vol = hist_vol[hist_vol["Client Type"].isin(selected_clients)]

    st.subheader("Overview Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Days", len(filtered_oi["Date"].unique()))
    col2.metric("Start Date", pd.to_datetime(filtered_oi["Date"]).min().strftime("%b %d, %Y"))
    col3.metric("End Date", pd.to_datetime(filtered_oi["Date"]).max().strftime("%b %d, %Y"))
    col4.metric("Total Records", len(filtered_oi))

    oi_tab, vol_tab = st.tabs(["Open Interest", "Volume"])
    instruments = list(INSTRUMENT_COLUMN_MAP.keys())

    with oi_tab:
        instrument_tabs = st.tabs(instruments)
        for idx, instrument in enumerate(instruments):
            with instrument_tabs[idx]:
                inst_df = _instrument_diff_frame(filtered_oi, instrument)
                if inst_df.empty:
                    st.warning(f"No OI data available for {instrument}")
                    continue
                for position in ["Long", "Short"]:
                    pos_df = inst_df[inst_df["Position Type"].str.contains(position)]
                    if pos_df.empty:
                        continue
                    c1, c2 = st.columns([7, 3])
                    with c1:
                        st.plotly_chart(
                            create_participant_chart(
                                pos_df,
                                x="Date",
                                y="Contracts",
                                color="Client Type",
                                title=f"{instrument} - {position} (Day Difference)",
                                chart_type=chart_type,
                            ),
                            use_container_width=True,
                        )
                    with c2:
                        agg = pos_df.groupby(["Client Type", "Position Type"], as_index=False)["Contracts"].sum()
                        st.plotly_chart(
                            create_participant_chart(
                                agg,
                                x="Client Type",
                                y="Contracts",
                                color="Client Type",
                                title=f"{instrument} - {position} (Aggregate Day Difference)",
                                chart_type="Bar",
                            ),
                            use_container_width=True,
                        )

    with vol_tab:
        long_short_vol = filtered_vol[["Date", "Client Type", "Total Long Contracts", "Total Short Contracts"]].melt(
            id_vars=["Date", "Client Type"],
            var_name="Position Type",
            value_name="Contracts",
        )
        for position in ["Long", "Short"]:
            pos_df = long_short_vol[long_short_vol["Position Type"].str.contains(position)]
            c1, c2 = st.columns([7, 3])
            with c1:
                st.plotly_chart(
                    create_participant_chart(
                        pos_df,
                        x="Date",
                        y="Contracts",
                        color="Client Type",
                        title=f"Participant Volume - {position}",
                        chart_type=chart_type,
                    ),
                    use_container_width=True,
                )
            with c2:
                agg = pos_df.groupby(["Client Type", "Position Type"], as_index=False)["Contracts"].sum()
                st.plotly_chart(
                    create_participant_chart(
                        agg,
                        x="Client Type",
                        y="Contracts",
                        color="Client Type",
                        title=f"Participant Volume - {position} (Aggregate)",
                        chart_type="Bar",
                    ),
                    use_container_width=True,
                )


def _fyers_sdk_available() -> bool:
    return importlib.util.find_spec("fyers_apiv3") is not None


def get_fyers_access_token(api_key: str, api_secret: str, redirect_uri: str, auth_code: str) -> str:
    from fyers_apiv3 import fyersModel

    session = fyersModel.SessionModel(
        client_id=api_key,
        secret_key=api_secret,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code",
    )
    session.set_token(auth_code)
    token_response = session.generate_token()
    return token_response.get("access_token", "")


def run_process_symbol_data(access_token_ws: str, symbols: List[str], shm_name: str, shm_c_name: str, queue: multiprocessing.Queue) -> None:
    from tickme.TickMe import myonclose, myonerror, myonopen, on_message_nifty
    from fyers_apiv3.FyersWebsocket import data_ws

    data_type = "SymbolUpdate"
    while True:
        fyers_data = data_ws.FyersDataSocket(
            shm_name=shm_name,
            shm_c_name=shm_c_name,
            num_rows=TICK_NUM_ROWS,
            dtype=TICK_DTYPE,
            counter_data=np.zeros(COUNTER_SHAPE, dtype=COUNTER_DTYPE),
            access_token=access_token_ws,
            log_path="",
            litemode=False,
            write_to_file=False,
            reconnect=True,
            on_connect=myonopen,
            on_close=myonclose,
            on_error=myonerror,
            on_message=on_message_nifty,
            reconnect_retry=50,
        )
        fyers_data.connect()
        fyers_data.subscribe(symbols=symbols, data_type=data_type)
        fyers_data.keep_running()

        next_symbol = queue.get()
        symbols = [next_symbol]


def process_buffer_nifty(shm_name: str, shm_c_name: str, access_token: str, symbols: List[str], queue: multiprocessing.Queue) -> None:
    from tickme.TickMe import CandleBuilder

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    ticks = np.ndarray(shape=(TICK_NUM_ROWS,), dtype=TICK_DTYPE, buffer=existing_shm.buf)

    existing_c_shm = shared_memory.SharedMemory(name=shm_c_name)
    counters = np.ndarray(shape=COUNTER_SHAPE, dtype=COUNTER_DTYPE, buffer=existing_c_shm.buf)

    candle_builders = {s: CandleBuilder(candle_duration_seconds=180, symbol=s) for s in symbols}

    while True:
        read_index = int(counters[0][0])
        write_index = int(counters[0][1])

        while read_index != write_index:
            tick = ticks[read_index]
            symbol = str(tick["symbol"])
            if symbol in candle_builders:
                candle_builders[symbol].process_tick(tick)
            read_index = (read_index + 1) % TICK_NUM_ROWS
        counters[0][0] = read_index


def initialize_fyers_live_pipeline(api_key: str, access_token: str, symbols: List[str]) -> List[multiprocessing.Process]:
    access_token_ws = f"{api_key}:{access_token}"
    shm_size = TICK_NUM_ROWS * TICK_DTYPE.itemsize
    counter_data = np.zeros(COUNTER_SHAPE, dtype=COUNTER_DTYPE)

    data_shm = shared_memory.SharedMemory(create=True, size=shm_size)
    counter_shm = shared_memory.SharedMemory(create=True, size=counter_data.nbytes)

    ticks = np.ndarray(shape=(TICK_NUM_ROWS,), dtype=TICK_DTYPE, buffer=data_shm.buf)
    ticks["symbol"] = ""
    counters = np.ndarray(shape=COUNTER_SHAPE, dtype=COUNTER_DTYPE, buffer=counter_shm.buf)
    counters[:] = counter_data[:]

    queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=1)
    return [
        multiprocessing.Process(target=run_process_symbol_data, args=(access_token_ws, symbols, data_shm.name, counter_shm.name, queue), daemon=True),
        multiprocessing.Process(target=process_buffer_nifty, args=(data_shm.name, counter_shm.name, access_token, symbols, queue), daemon=True),
    ]


def main() -> None:
    st.set_page_config(layout="wide", page_title="Advanced NSE Sentiment Dashboard")
    st.title("Advanced NSE Participant + PCR + OI Sentiment")

    with st.sidebar:
        symbol = st.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"], index=0)
        broker = st.selectbox("Broker", ["NSE", "Fyers"], index=0)
        lookback_days = st.slider("Participant lookback days", 3, 30, 7)
        refresh_sec = st.slider("Auto-refresh seconds", 10, 300, 30)
        fyers_token = ""
        if broker == "Fyers":
            st.caption("Provide credentials to generate Fyers access token.")
            fyers_api_key = st.text_input("Fyers API Key")
            fyers_api_secret = st.text_input("Fyers API Secret", type="password")
            fyers_redirect_uri = st.text_input(
                "Fyers Redirect URI",
                value="https://trade.fyers.in/api-login/redirect-uri/index.html",
            )
            fyers_auth_code = st.text_input("Fyers Auth Code", type="password")

            if st.button("Generate Fyers Token"):
                if not _fyers_sdk_available():
                    st.error("fyers_apiv3 package is not installed in this environment.")
                elif not all([fyers_api_key, fyers_api_secret, fyers_auth_code]):
                    st.warning("API key, secret and auth code are required.")
                else:
                    fyers_token = get_fyers_access_token(
                        fyers_api_key,
                        fyers_api_secret,
                        fyers_redirect_uri,
                        fyers_auth_code,
                    )
                    if fyers_token:
                        st.session_state["fyers_token"] = fyers_token
                        st.success("Fyers token generated and cached in session state.")
                    else:
                        st.error("Failed to generate Fyers token.")

            if st.checkbox("Enable Fyers shared-memory live pipeline"):
                if "fyers_token" not in st.session_state:
                    st.warning("Generate token first.")
                elif st.button("Start Fyers Pipeline"):
                    symbols = [f"NSE:{symbol}-INDEX"]
                    procs = initialize_fyers_live_pipeline(fyers_api_key, st.session_state["fyers_token"], symbols)
                    for proc in procs:
                        proc.start()
                    st.session_state["fyers_procs"] = procs
                    st.success("Started Fyers multiprocessing pipeline.")

    if hasattr(st, "autorefresh"):
        st.autorefresh(interval=refresh_sec * 1000, key="live_refresh")
    else:
        st.caption("Auto-refresh not supported in installed Streamlit version.")

    as_of = datetime.today().date()
    dates = [as_of - timedelta(days=i) for i in range(lookback_days)]

    try:
        session = nse_session()
        option_chain = fetch_option_chain(symbol, session)

        oi_frames, vol_frames = [], []
        for d in dates:
            try:
                oi_frames.append(fetch_participant_file("oi", d, session))
                vol_frames.append(fetch_participant_file("vol", d, session))
            except Exception:
                continue

        if not oi_frames or not vol_frames:
            st.error("Unable to fetch participant OI/volume files from NSE archives for selected range.")
            return

        latest_oi = oi_frames[0]
        latest_vol = vol_frames[0]
        pcr = compute_pcr(option_chain)
        previous_oi = oi_frames[1] if len(oi_frames) > 1 else None
        participant_df = participant_strength(latest_oi, latest_vol, previous_oi)
        snapshot = build_sentiment_snapshot(pcr, participant_df)
        nd_label, nd_comment = next_day_direction(participant_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PCR (OI)", f"{snapshot.pcr_oi:.2f}" if pd.notna(snapshot.pcr_oi) else "NA")
        c2.metric("PCR (Change OI)", f"{snapshot.pcr_change_oi:.2f}" if pd.notna(snapshot.pcr_change_oi) else "NA")
        c3.metric("Sentiment", snapshot.label)
        c4.metric("Next Day Bias (EOD)", nd_label)

        st.caption(nd_comment)
        for r in snapshot.rationale:
            st.write(f"- {r}")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Participant OI + Volume",
            "PCR / OI Monitor",
            "5 ITM Strike Candle Delta",
            "Historical Files",
        ])

        with tab1:
            render_participant_oi_volume_tab(oi_frames, vol_frames)
            st.subheader("Directional strength table")
            st.dataframe(participant_df, use_container_width=True)

        with tab2:
            oi_fig = go.Figure()
            oi_fig.add_bar(name="CE OI", x=["Current"], y=[snapshot.total_ce_oi])
            oi_fig.add_bar(name="PE OI", x=["Current"], y=[snapshot.total_pe_oi])
            oi_fig.add_bar(name="CE Change OI", x=["Current"], y=[snapshot.total_ce_change_oi])
            oi_fig.add_bar(name="PE Change OI", x=["Current"], y=[snapshot.total_pe_change_oi])
            oi_fig.update_layout(barmode="group", title="Real-time OI / Change in OI (NSE option-chain)")
            st.plotly_chart(oi_fig, use_container_width=True)

        with tab3:
            ce_rows, pe_rows, underlying = select_itm_strikes(option_chain, 5)
            st.write(f"Underlying value: **{underlying:.2f}**")
            ce_candle_df = update_candle_state(symbol, ce_rows, "CE")
            pe_candle_df = update_candle_state(symbol, pe_rows, "PE")
            render_candle_volume_chart(ce_candle_df, "CE ITM strikes - per candle buy/sell volume")
            render_candle_volume_chart(pe_candle_df, "PE ITM strikes - per candle buy/sell volume")
            st.info("Buy/sell split is inferred from incremental price direction, adapted from TickMe CandleBuilder logic.")

        with tab4:
            hist_oi = pd.concat(oi_frames, ignore_index=True)
            hist_vol = pd.concat(vol_frames, ignore_index=True)
            hist_oi = hist_oi.sort_values(["Date", "Client Type"])
            hist_oi["Net OI"] = hist_oi["Total Long Contracts"] - hist_oi["Total Short Contracts"]
            hist_oi["Net OI Day Difference"] = hist_oi.groupby("Client Type")["Net OI"].diff().fillna(0)

            oi_fig = px.line(
                hist_oi,
                x="Date",
                y="Net OI Day Difference",
                color="Client Type",
                markers=True,
                title="Participant-wise OI Day Difference (current day - previous day)",
            )
            st.plotly_chart(oi_fig, use_container_width=True)

            instrument_diff_df = build_instrument_diff_history(hist_oi)
            instrument_diff_fig = px.line(
                instrument_diff_df,
                x="Date",
                y="Day Difference",
                color="Client Type",
                markers=True,
                facet_row="Instrument",
                title="Participant-wise Instrument Day Difference (current day - previous day)",
                height=2200,
            )
            instrument_diff_fig.update_yaxes(matches=None)
            st.plotly_chart(instrument_diff_fig, use_container_width=True)

            vol_fig = px.line(
                hist_vol.sort_values(["Date", "Client Type"]),
                x="Date",
                y="Total Long Contracts",
                color="Client Type",
                markers=True,
                title="Participant-wise Volume (Long Contracts)",
            )
            st.plotly_chart(vol_fig, use_container_width=True)

    except Exception as exc:
        st.exception(exc)


if __name__ == "__main__":
    main()
