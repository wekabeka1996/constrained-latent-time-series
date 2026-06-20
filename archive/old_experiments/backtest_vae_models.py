import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

from src.io_utils import require_file

# 1. Завантаження фінансового ряду (BTCUSDT / S&P 500)
def load_financial_series(symbol="BTCUSDT", start="2017-08-17", end=None):
    # Binance API через yfinance не підтримується, тому використаємо binance-spot-api через python-binance
    try:
        from binance.client import Client
    except ImportError as e:
        raise ImportError(
            "Необхідна бібліотека 'python-binance' для завантаження ринкових даних з Binance. "
            "Встановіть її через pip або надайте локальний файл."
        ) from e
    
    import os
    api_key = os.environ.get('BINANCE_API_KEY', '')
    api_secret = os.environ.get('BINANCE_API_SECRET', '')
    try:
        client = Client(api_key, api_secret)
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, start)
        if not klines:
            raise ValueError(f"Отримано порожній результат від Binance API для {symbol}.")
    except Exception as e:
        raise RuntimeError(
            f"Не вдалося завантажити ринкові дані для {symbol} з Binance: {e}. "
            "Перевірте мережеве з'єднання та налаштування API."
        ) from e
        
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time',
        'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base', 'Taker_buy_quote', 'Ignore'])
    df['Close'] = df['Close'].astype(float)
    df['logret'] = np.log(df['Close']).diff()
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('date')
    return df.dropna()

# 2. Вибірка моделей
# a) ARMA-GARCH: згенеровані моделі (orders_generated.npy)
def load_arma_garch_models(n=15, file="orders_generated.npy"):
    file_path = require_file(file, "Generated models needed for backtesting. Provide the real artifact or generate it through the documented canonical pipeline. Do not use dummy fallback arrays.")
    arr = np.load(file_path, allow_pickle=True)
    # arr: shape (N,4) - [p,q,r,s]
    df = pd.DataFrame(arr, columns=["p","q","r","s"])
    df_unique = df.drop_duplicates()
    return df_unique.sample(min(n, len(df_unique)), random_state=42).to_dict("records")

# b) ARMA з "перекладин" (interpolation_detailed_3.csv)
def load_arma_manifold_models(n=10, file="interpolation_detailed_3.csv"):
    file_path = require_file(file, "Interpolated manifold models needed for backtesting. Provide the real artifact or generate it through the documented canonical pipeline. Do not use dummy fallback arrays.")
    df = pd.read_csv(file_path)
    # Вибрати рівномірно по lambda
    lambdas = np.linspace(0, 0.9, n)
    models = []
    for l in lambdas:
        row = df.iloc[(df["lambda"]-l).abs().argsort()[:1]]
        models.append(row.iloc[0].to_dict())
    return models

# 3. Бектестинг
# a) ARMA-GARCH: прогноз волатильності, long volatility
# b) ARMA: прогноз напрямку, long/short

def backtest_arma_garch(model, returns):
    p, q, r, s = int(model["p"]), int(model["q"]), int(model["r"]), int(model["s"])
    if r==0 and s==0:
        return None
    try:
        am = arch_model(returns, mean="ARX", lags=p, vol="GARCH", p=r, q=s, dist="normal")
        res = am.fit(disp="off")
        forecasts = res.forecast(horizon=1, reindex=False).variance["h.1"]
        mean_vol = forecasts.mean()
        signal = (forecasts > mean_vol).astype(int)  # 1 if vol > mean, else 0
        strat_ret = signal.shift(1).fillna(0) * returns.abs()  # long volatility
        return strat_ret
    except Exception as e:
        print(f"⚠️ ARMA-GARCH({p},{q},{r},{s}) fit failed: {e}")
        return None

def backtest_arma(model, returns):
    p, q = int(model["p"]), int(model["q"])
    try:
        arima = ARIMA(returns, order=(p,0,q))
        res = arima.fit()
        preds = res.predict()
        signal = np.sign(preds)
        strat_ret = signal.shift(1).fillna(0) * returns  # long/short
        return strat_ret
    except Exception as e:
        print(f"⚠️ ARIMA({p},0,{q}) fit failed: {e}")
        return None

def compute_metrics(strat_ret):
    if strat_ret is None or strat_ret.isnull().all():
        return {"Total Return": np.nan, "Sharpe": np.nan, "Max Drawdown": np.nan}
    total = strat_ret.sum()
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else np.nan
    cum = strat_ret.cumsum()
    drawdown = (cum.cummax() - cum).max()
    return {"Total Return": total, "Sharpe": sharpe, "Max Drawdown": drawdown}

if __name__ == "__main__":
    df = load_financial_series()
    returns = df["logret"]
    # ARMA-GARCH
    arma_garch_models = load_arma_garch_models()
    arma_garch_results = []
    for m in arma_garch_models:
        strat_ret = backtest_arma_garch(m, returns)
        metrics = compute_metrics(strat_ret)
        metrics.update({"Type": "ARMA-GARCH", "Order": f"({m['p']},{m['q']},{m['r']},{m['s']})"})
        arma_garch_results.append(metrics)
    # ARMA-manifold
    arma_manifold_models = load_arma_manifold_models()
    arma_manifold_results = []
    for m in arma_manifold_models:
        strat_ret = backtest_arma(m, returns)
        metrics = compute_metrics(strat_ret)
        metrics.update({"Type": "ARMA-manifold", "Order": f"({m['p']},{m['q']})", "lambda": m["lambda"]})
        arma_manifold_results.append(metrics)
    # Зведена таблиця
    all_results = pd.DataFrame(arma_garch_results + arma_manifold_results)
    all_results.to_csv("results/generation_2/robustness/vae_backtest_summary.csv", index=False)
    print(all_results)
    print("\nЗбережено у results/generation_2/robustness/vae_backtest_summary.csv")
