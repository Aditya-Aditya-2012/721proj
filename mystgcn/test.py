#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz
from sklearn.preprocessing import StandardScaler

from script import dataloader, utility
from model.models import STGCNChebGraphConv

# ─── 1) METRICS (Sec 4.3) ───────────────────────────────────────────
def compute_metrics(y_true, y_pred, train_vals):
    err   = y_pred - y_true
    mae   = np.mean(np.abs(err))
    rmse  = np.sqrt(np.mean(err**2))
    smape = 100 * np.mean(2 * np.abs(err) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))
    denom = np.mean(np.abs(train_vals[1:] - train_vals[:-1]))
    mase  = mae / (denom + 1e-6)
    return mae, rmse, smape, mase

# ─── 2) ARGPARSE ───────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',      default='delhiaq',      help='folder under data/')
parser.add_argument('--model_prefix', default='STGCN_delhiaq',help='prefix of your .pt files')
parser.add_argument('--n_his',  type=int, default=30, help='lookback days')
parser.add_argument('--n_pred', type=int, default=30, help='forecast horizon days')
parser.add_argument('--device',  default='cpu', help='cpu or cuda')
args = parser.parse_args()

DEVICE   = torch.device(args.device)
DATA_DIR = os.path.join('data', args.dataset)

# ─── 3) LOAD & BUILD GSO ──────────────────────────────────────────
adj, _ = dataloader.load_adj(args.dataset)
gso     = utility.calc_gso(adj, 'sym_norm_lap')
gso     = utility.calc_chebynet_gso(gso)
gso_mat = torch.from_numpy(gso.toarray().astype(np.float32)).to(DEVICE)

# ─── 4) LOAD DAILY DATA ───────────────────────────────────────────
df_tr = (pd.read_csv(os.path.join(DATA_DIR, 'train_daily.csv'), parse_dates=['datetime'])
           .set_index('datetime'))
df_va = (pd.read_csv(os.path.join(DATA_DIR, 'val_daily.csv'),   parse_dates=['datetime'])
           .set_index('datetime'))
df_all = pd.concat([df_tr, df_va])

# ─── 5) BLOCK‐CONFIG (as in training) ─────────────────────────────
Kt = 3
stblock_num = 2
Ko = args.n_his - (Kt - 1) * 2 * stblock_num
blocks = [[1]] + [[64,16,64]]*stblock_num + ([ [128] ] if Ko==0 else [[128,128]]) + [[1]]

# ─── 6) INFERENCE LOOP ───────────────────────────────────────────
pol_map = {'PM25':'PM2.5','PM10':'PM10','NO2':'NO2'}
for pol in ['PM25','PM10','NO2']:
    suffix     = pol_map[pol]
    cols       = [c for c in df_tr.columns if c.endswith(f"_{suffix}")]
    train_vals = df_tr[cols].values                    # for MASE denom
    df_pol     = df_all[cols]

    # fit scaler on TRAIN only
    scaler = StandardScaler().fit(train_vals)

    # build & load model
    N = len(cols)
    model = STGCNChebGraphConv(
        argparse.Namespace(
            n_his=args.n_his, n_pred=args.n_pred,
            graph_conv_type='cheb_graph_conv',
            gso_type='sym_norm_lap',
            droprate=0.5,
            enable_bias=True,
            Ks=3, Kt=Kt, stblock_num=stblock_num, act_func='glu',
            gso=gso_mat
        ),
        blocks, N
    ).to(DEVICE)

    ckpt = f"{args.model_prefix}{pol}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    results = []
    for month in range(1,13):
        start    = pd.Timestamp(2022, month, 1)
        hist_beg = start - pd.Timedelta(days=args.n_his)
        hist_end = start - pd.Timedelta(days=1)
        fut_end  = start + pd.Timedelta(days=args.n_pred - 1)

        # 1) History → scale (shape [n_his, N])
        X_hist = df_pol.loc[hist_beg:hist_end].values
        X_s    = scaler.transform(X_hist)[None, None, ...]  # [1,1,n_his,N]
        X_t    = torch.tensor(X_s, dtype=torch.float32, device=DEVICE)

        # 2) Forecast (returns [1, n_pred, N])
        with torch.no_grad():
            y_norm = model(X_t).cpu().numpy()[0]          # [n_pred, N]

        # 3) Inverse‐scale
        y_norm = y_norm.squeeze(0)
        y_pred = scaler.inverse_transform(y_norm)        # [n_pred, N]

        # 4) True values
        y_true = df_pol.loc[start:fut_end].values        # [n_pred, N]

        # 5) Compute metrics
        mae, rmse, smape, mase = compute_metrics(y_true, y_pred, train_vals)
        results.append({
            'Month':  start.strftime('%b'),
            'MAE':    round(mae,   2),
            'RMSE':   round(rmse,  2),
            'SMAPE%': round(smape, 2),
            'MASE':   round(mase,  2),
        })

    # ─── 7) Dump CSV ───────────────────────────────────────────────
    out_df = pd.DataFrame(results)
    fname  = f"metrics_{pol}.csv"
    out_df.to_csv(fname, index=False)
    print(f"→ Saved {fname}")
