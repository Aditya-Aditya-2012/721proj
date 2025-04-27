#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# ─── CONFIG ────────────────────────────────────────────────────────
TRAIN_CSV  = 'train_daily.csv'    # your 2018–2021 daily-mean data
VAL_CSV    = 'val_daily.csv'      # your 2022 daily-mean data
MODEL_PATH = 'best_lstm.pth'      # saved LSTM weights
DATE_COL   = 'datetime'
LOOKBACK   = 30                   # days
HORIZON    = 30                   # days
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
POLLUTANTS = ['PM2.5', 'PM10', 'NO2']

# ─── METRIC ROUTINE (Sec 4.3) ─────────────────────────────────────
def compute_metrics(y_true, y_pred, train_vals):
    err   = y_pred - y_true
    mae   = np.mean(np.abs(err))
    rmse  = np.sqrt(np.mean(err**2))
    smape = 100 * np.mean(2 * np.abs(err) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))
    denom = np.mean(np.abs(train_vals[1:] - train_vals[:-1]))
    mase  = mae / (denom + 1e-6)
    return mae, rmse, smape, mase

# ─── DATA LOADING & PREPROCESS ───────────────────────────────────
def load_daily(path):
    df = (pd.read_csv(path, parse_dates=[DATE_COL])
            .set_index(DATE_COL))
    # assume it's already daily-aggregated, just fill any gaps
    return df.ffill().bfill()

df_train = load_daily(TRAIN_CSV)
df_val   = load_daily(VAL_CSV)
scaler   = MinMaxScaler().fit(df_train.values)
df_all   = pd.concat([df_train, df_val])  # so windows can straddle Dec21–Jan22

# ─── MODEL DEFINITION & LOAD ─────────────────────────────────────
class LSTMForecast(nn.Module):
    def __init__(self, num_series, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(num_series, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_series * HORIZON)
        self.N    = num_series

    def forward(self, x):
        out, _ = self.lstm(x)
        hS     = out[:, -1, :]
        y      = self.fc(hS)
        return y.view(-1, HORIZON, self.N)

model = LSTMForecast(df_train.shape[1]).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ─── EVALUATION LOOP ───────────────────────────────────────────────
for pollutant in POLLUTANTS:
    # mask to select only the pollutant columns
    mask       = [c.endswith(f"_{pollutant}") for c in df_train.columns]
    train_raw  = df_train.values[:, mask]
    df_subset  = df_all[df_train.columns[mask]]

    results = []
    for m in range(1, 13):
        start     = pd.Timestamp(2022, m, 1)
        hist_beg  = start - pd.Timedelta(days=LOOKBACK)
        hist_end  = start - pd.Timedelta(days=1)
        fut_end   = start + pd.Timedelta(days=HORIZON-1)

        # 1) Build & scale the lookback window [30 days × all series]
        X_hist = df_all.loc[hist_beg:hist_end].values
        X_scaled = scaler.transform(X_hist)[None, ...]     # shape [1, LOOKBACK, N]
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)

        # 2) Forecast next 30 days
        with torch.no_grad():
            y_scaled_full = model(X_tensor).cpu().numpy()[0]  # [HORIZON, N]
        y_pred_full = scaler.inverse_transform(y_scaled_full)  # back to real units

        # 3) Slice out only this pollutant’s columns → [HORIZON, P]
        y_pred = y_pred_full[:, mask]
        y_true = df_subset.loc[start:fut_end].values         # [HORIZON, P]

        # 4) Compute metrics
        mae, rmse, smape, mase = compute_metrics(y_true, y_pred, train_raw)
        results.append({
            'Month': start.strftime('%b'),
            'MAE':    round(mae,   2),
            'RMSE':   round(rmse,  2),
            'SMAPE%': round(smape, 2),
            'MASE':   round(mase,  2),
        })

    # save to CSV
    out_df = pd.DataFrame(results)
    fname  = f'metrics_{pollutant.replace(".", "")}_daily.csv'
    out_df.to_csv(fname, index=False)
    print(f"→ Saved {fname}")
