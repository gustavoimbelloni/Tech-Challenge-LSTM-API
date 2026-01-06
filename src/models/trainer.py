import os
from typing import Dict, Any, Optional
import io
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
# garantir config do yfinance antes do import
os.environ.setdefault("YF_NO_IMPERSONATE", "1")
os.environ.setdefault("YF_USE_CURL_CFFI", "0")
import yfinance as yf
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import json


def build_lstm_model(input_shape: tuple) -> Sequential:
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def _fetch_data_resilient(
    symbol: str,
    period: str,
    interval: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Coleta dados de forma resiliente (Yahoo Chart JSON -> Yahoo CSV -> yfinance download/history).
    Retorna DataFrame com índice datetime e coluna Close.
    """
    df: Optional[pd.DataFrame] = None
    # 1) Yahoo Chart JSON (query2)
    try:
        if start:
            dt_start = int(datetime.fromisoformat(start).timestamp())
            dt_end = int(datetime.fromisoformat(end).timestamp()) if end else int(datetime.now().timestamp())
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?period1={dt_start}&period2={dt_end}&interval={interval}"
        else:
            rng = period if period.endswith(("d", "y", "mo")) else "1y"
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?range={rng}&interval={interval}"
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("application/json"):
            payload = r.json()
            result = payload.get("chart", {}).get("result", [])
            if result:
                r0 = result[0]
                closes = r0.get("indicators", {}).get("quote", [{}])[0].get("close", [])
                ts = r0.get("timestamp", [])
                if closes and ts and len(closes) == len(ts):
                    dates = pd.to_datetime(ts, unit="s", utc=True).tz_convert("UTC").tz_localize(None)
                    df = pd.DataFrame({"Close": closes}, index=dates)
    except Exception:
        df = None

    # 2) Yahoo CSV (query1)
    if df is None or df.empty:
        try:
            headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv"}
            if start:
                dt_start = int(datetime.fromisoformat(start).timestamp())
                dt_end = int(datetime.fromisoformat(end).timestamp()) if end else int(datetime.now().timestamp())
            else:
                now = datetime.now()
                if period.endswith("d"):
                    days = int(period[:-1])
                elif period.endswith("y"):
                    years = int(period[:-1])
                    days = 365 * years
                else:
                    days = 365
                dt_start = int((now - timedelta(days=days)).timestamp())
                dt_end = int(now.timestamp())
            url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={dt_start}&period2={dt_end}&interval={interval}&events=history&includeAdjustedClose=true"
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200 and "Date,Open,High,Low,Close,Adj Close,Volume" in r.text:
                df = pd.read_csv(io.StringIO(r.text))
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    df = df.dropna(subset=["Date"]).set_index("Date")
        except Exception:
            df = None

    # 3) yfinance download
    if df is None or df.empty:
        try:
            if start:
                df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, threads=False, progress=False)
            else:
                df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, threads=False, progress=False)
            if (df is None or df.empty) and not start:
                df = yf.download(symbol, period="1y", interval=interval, auto_adjust=False, threads=False, progress=False)
        except Exception:
            df = None

    # 4) yfinance history
    if df is None or df.empty:
        try:
            t = yf.Ticker(symbol)
            if start:
                df = t.history(start=start, end=end, interval=interval, auto_adjust=False)
            else:
                df = t.history(period=period, interval=interval, auto_adjust=False)
            if (df is None or df.empty) and not start:
                df = t.history(period="1y", interval=interval, auto_adjust=False)
        except Exception:
            df = None

    if df is None or df.empty:
        raise ValueError(f"Nenhum dado retornado para {symbol}. Ajuste 'period' ou informe 'start'/'end'.")

    # Normalização de coluna Close (MultiIndex)
    if "Close" not in df.columns and isinstance(df.columns, pd.MultiIndex):
        try:
            if "Close" in df.columns.get_level_values(0):
                s = df["Close"].iloc[:, 0]
            else:
                s = df.xs("Close", axis=1, level=1)
            df = pd.DataFrame({"Close": s})
        except Exception:
            pass
    if "Close" not in df.columns:
        raise ValueError("Coluna 'Close' não encontrada nos dados coletados.")
    df = df.dropna(subset=["Close"])
    return df


def train_and_save_for_symbol(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    sequence_length: int = 60,
    epochs: int = 10,
    batch_size: int = 32,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Treina um modelo LSTM para o ticker informado e salva artefatos em model/{SYMBOL}/.
    Retorna metadados do treino.
    """
    # 1) Coleta (resiliente)
    df = _fetch_data_resilient(symbol=symbol, period=period, interval=interval, start=start, end=end)
    data = df[["Close"]].copy()
    dataset = data.values

    # 2) Escalonar
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # 3) Dividir (80/20)
    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[0:training_data_len, :]
    x_train, y_train = [], []
    for i in range(sequence_length, len(train_data)):
        x_train.append(train_data[i - sequence_length:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len - sequence_length:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(sequence_length, len(test_data)):
        x_test.append(test_data[i - sequence_length:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # 4) Modelo
    model = build_lstm_model(input_shape=(x_train.shape[1], 1))
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

    # 5) Salvar artefatos
    out_dir = os.path.join("model", symbol)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "lstm_model.h5")
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    meta_path = os.path.join(out_dir, "meta.json")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "sequence_length": sequence_length,
                "epochs": epochs,
                "batch_size": batch_size,
                "train_samples": int(x_train.shape[0]),
                "total_samples": int(len(df)),
                "start": start,
                "end": end,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "symbol": symbol,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "meta_path": meta_path,
        "sequence_length": sequence_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_samples": int(x_train.shape[0]),
        "total_samples": int(len(df)),
        "start": start,
        "end": end,
    }


