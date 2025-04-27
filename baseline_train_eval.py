import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── CONFIGURATION ─────────────────────────────────────────────
train_csv  = 'train_daily.csv'   # 2018–2021
val_csv    = 'val_daily.csv'     # 2022
date_col   = 'datetime'
lookback   = 30
horizon    = 30
batch_size = 64
epochs     = 50
lr         = 1e-3
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── 1) LOAD & CONCAT ───────────────────────────────────────────
def load_and_preprocess(path):
    df = pd.read_csv(path, parse_dates=[date_col])
    df.set_index(date_col, inplace=True)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

df_train = load_and_preprocess(train_csv)
df_val   = load_and_preprocess(val_csv)

# Fit scaler on TRAIN, transform both
scaler      = MinMaxScaler()
scaler.fit(df_train.values)
train_scaled = scaler.transform(df_train.values)
val_scaled   = scaler.transform(df_val.values)

feature_cols = df_train.columns.tolist()
num_series   = len(feature_cols)

# ─── 2) DATASET & DATALOADERS ───────────────────────────────────
def create_sequences(data, lookback, horizon):
    X, Y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback])
        Y.append(data[i + lookback : i + lookback + horizon])
    return np.array(X), np.array(Y)

class TSdataset(Dataset):
    def __init__(self, data):
        self.X, self.Y = create_sequences(data, lookback, horizon)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx]).float(),
                torch.from_numpy(self.Y[idx]).float())

train_ds = TSdataset(train_scaled)
val_ds   = TSdataset(val_scaled)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

# ─── 3) MODEL ──────────────────────────────────────────────────
class LSTMForecast(nn.Module):
    def __init__(self, num_series, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(num_series, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_series * horizon)
        self.num_series = num_series

    def forward(self, x):
        out, _ = self.lstm(x)                    # [B, lookback, hidden]
        hS     = out[:, -1, :]                  # [B, hidden]
        y      = self.fc(hS)                    # [B, N*horizon]
        return y.view(-1, horizon, self.num_series)

model     = LSTMForecast(num_series).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ─── 4) TRAIN + EARLY STOPPING ─────────────────────────────────
best_val_loss = float('inf')
patience      = 5
wait          = 0

for ep in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_ds)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model(xb), yb).item() * xb.size(0)
    val_loss /= len(val_ds)

    print(f"Epoch {ep:02d} — Train MSE: {train_loss:.4f}  |  Val MSE: {val_loss:.4f}")

    # early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        torch.save(model.state_dict(), 'best_lstm.pth')
    else:
        wait += 1
        if wait >= patience:
            print(f"⏱ Early stopping at epoch {ep}")
            break

# load best model
model.load_state_dict(torch.load('best_lstm.pth'))

# ─── 5) EVALUATE ON VAL SET (VECTORISED) ────────────────────────
model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        preds.append(model(xb).cpu().numpy())
        trues.append(yb.numpy())

preds = np.vstack(preds)  # [samples, horizon, N]
trues = np.vstack(trues)

# flatten for metrics
diff    = preds - trues
abs_err = np.abs(diff)
squared = diff**2

MAE   = abs_err.mean()
RMSE  = np.sqrt(squared.mean())
SMAPE = 100 * np.mean(2 * abs_err / (np.abs(preds) + np.abs(trues) + 1e-6))

# MASE denominator from TRAIN series
naive_diff = np.abs(train_scaled[1:] - train_scaled[:-1])
denom      = naive_diff.mean()
MASE       = MAE / (denom + 1e-6)

print("\nLSTM on Val set:")
print(f"MAE:   {MAE:.4f}")
print(f"RMSE:  {RMSE:.4f}")
print(f"SMAPE: {SMAPE:.2f}%")
print(f"MASE:  {MASE:.4f}")
