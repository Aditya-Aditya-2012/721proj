#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from script import dataloader, utility
from model.models import STGCNChebGraphConv
from model.cstgcn import STGCN_CNN_Finetune
import timm

# ─── 1) METRICS (Sec.4.3) ───────────────────────────────────────────
def compute_metrics(y_true, y_pred, train_vals):
    err   = y_pred - y_true
    mae   = np.mean(np.abs(err))
    rmse  = np.sqrt(np.mean(err**2))
    smape = 100 * np.mean(2 * np.abs(err) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))
    denom = np.mean(np.abs(train_vals[1:] - train_vals[:-1]))
    mase  = mae / (denom + 1e-6)
    return mae, rmse, smape, mase

# ─── 2) ARGS ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',      default='delhiaq',
                    help='folder under data/')
parser.add_argument('--model_prefix', default='C-STGCN_delhiaq',
                    help='prefix of your .pt files')
parser.add_argument('--n_his',  type=int, default=30,
                    help='lookback days')
parser.add_argument('--n_pred', type=int, default=30,
                    help='forecast horizon days')
parser.add_argument('--device',  default='cpu', help='cpu or cuda')
parser.add_argument('--cnn_model', default='efficientnet_b0',
                    help='timm model for station patches')
args = parser.parse_args()

DEVICE   = torch.device(args.device)
DATA_DIR = os.path.join('data', args.dataset)

# ─── 3) LOAD & BUILD GSO ──────────────────────────────────────────
adj, _ = dataloader.load_adj(args.dataset)
gso     = utility.calc_gso(adj, 'sym_norm_lap')
gso     = utility.calc_chebynet_gso(gso)
gso_mat = torch.from_numpy(gso.toarray().astype(np.float32)).to(DEVICE)

# ─── 4) LOAD DAILY DATA ───────────────────────────────────────────
df_tr = (pd.read_csv(os.path.join(DATA_DIR, 'train_daily.csv'),
                     parse_dates=['datetime'])
           .set_index('datetime'))
df_va = (pd.read_csv(os.path.join(DATA_DIR, 'val_daily.csv'),
                     parse_dates=['datetime'])
           .set_index('datetime'))
df_all = pd.concat([df_tr, df_va])

# ─── 5) BLOCK‐CONFIG (must match training) ─────────────────────────
Kt = 3
stblock_num = 2
Ko = args.n_his - (Kt - 1) * 2 * stblock_num
blocks = [[1]] + [[64,16,64]]*stblock_num + (
    [[128]] if Ko==0 else [[128,128]]
) + [[1]]

# ─── 6) MAIN LOOP ─────────────────────────────────────────────────
pollutants = ['PM25','PM10','NO2']
pol_map    = {'PM25':'PM2.5','PM10':'PM10','NO2':'NO2'}
patch_dir  = os.path.join(DATA_DIR, 's2_patches')

# Prefill a tiny CNN for computing static‐stats
cnn0 = timm.create_model(args.cnn_model, pretrained=True)
cnn0.global_pool = torch.nn.Identity()
cnn0.classifier  = torch.nn.Identity()
cnn0 = cnn0.to(DEVICE).eval()

for pol in pollutants:
    suffix     = pol_map[pol]
    cols       = [c for c in df_tr.columns if c.endswith(f'_{suffix}')]
    train_vals = df_tr[cols].values    # for MASE denom
    df_pol     = df_all[cols]

    # fit scaler on TRAIN only
    scaler = StandardScaler().fit(train_vals)
    train_s = scaler.transform(train_vals)
    train_ts_std = float(np.std(train_s))

    # ── load and preprocess **2022** patches for these stations ───────
    stations = [c.rsplit('_',1)[0] for c in cols]
    img_list = []
    for st in stations:
        arr = np.load(os.path.join(patch_dir, f'{st}.npy'))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        img_list.append(torch.tensor(arr.transpose(2,0,1),
                                     dtype=torch.float32))
    # stack [N,3,64,64] → to GPU
    X_img = torch.stack(img_list, dim=0).to(DEVICE)  # [N,3,H,W]
    N     = len(stations)

    # ── compute static μ,σ & init_alpha exactly as in training ────────
    with torch.no_grad():
        x = X_img                                # [N,3,64,64]
        x = F.interpolate(x, size=(128,128),
                         mode='bilinear', align_corners=False)
        mean = x.new_tensor([0.485,0.456,0.406]).view(1,3,1,1)
        std  = x.new_tensor([0.229,0.224,0.225]).view(1,3,1,1)
        x = (x - mean)/std                       # [N,3,128,128]
        fmap = cnn0.forward_features(x)          # [N, C_f, h, w]
        vec  = fmap.mean(dim=[2,3])             # [N, C_f]
        node0= vec.mean(dim=1)                  # [N]
        mu    = float(node0.mean().item())
        sigma = float(node0.std().item() or 1e-6)
    init_alpha = train_ts_std / sigma

    # ── build & load your STGCN+CNN‐RefineEmb model ─────────────────
    # we need to package these stats into `args` so the model picks them up
    args.static_mu, args.static_sigma, args.init_alpha = mu, sigma, init_alpha
    n_his=args.n_his
    n_pred=args.n_pred
    args.graph_conv_type='cheb_graph_conv'
    args.gso_type='sym_norm_lap'
    args.droprate=0.5
    args.enable_bias=True
    args.Ks=3
    args.Kt=Kt
    args.stblock_num=stblock_num
    args.act_func='glu'
    args.gso=gso_mat

    model = STGCN_CNN_Finetune(args, blocks, N,
                               cnn_model=args.cnn_model).to(DEVICE)

    ckpt = f"{args.model_prefix}{pol}.pt"
    print(f"Loading weights from {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    # for each month in 2022, forecast & compute metrics
    results = []
    for month in range(1,13):
        start    = pd.Timestamp(2022, month, 1)
        hist_beg = start - pd.Timedelta(days=args.n_his)
        hist_end = start - pd.Timedelta(days=1)
        fut_end  = start + pd.Timedelta(days=args.n_pred-1)

        # history slice & scale → [1,1,n_his,N]
        X_hist = df_pol.loc[hist_beg:hist_end].values
        X_s    = scaler.transform(X_hist)[None,None,...]
        X_t    = torch.tensor(X_s, dtype=torch.float32,
                              device=DEVICE)

        # forecast (model fuses CNN+TS internally)
        with torch.no_grad():
            y_norm = model(X_t, X_img.unsqueeze(0))\
                     .cpu().numpy()[0]       # [n_pred, N]

        # inverse‐scale back to real units
        y_norm = y_norm.squeeze(0)
        y_pred = scaler.inverse_transform(y_norm)

        # true 2022 values
        y_true = df_pol.loc[start:fut_end].values

        # metrics
        mae, rmse, smape, mase = compute_metrics(
            y_true, y_pred, train_vals
        )
        results.append({
            'Month':  start.strftime('%b'),
            'MAE':    round(mae,   2),
            'RMSE':   round(rmse,  2),
            'SMAPE%': round(smape, 2),
            'MASE':   round(mase,  2),
        })

    # dump CSV
    out_df = pd.DataFrame(results)
    fname  = f"metrics_{pol}.csv"
    out_df.to_csv(fname, index=False)
    print(f"→ Saved {fname}")
