#!/usr/bin/env python3
import logging, os, gc, argparse, math, random, warnings
import tqdm
import timm
import numpy as np, pandas as pd
from sklearn import preprocessing
import torch
import torch.nn as nn, torch.optim as optim, torch.utils as utils

from script import dataloader, utility, earlystopping, opt
from model import models
from model.cstgcn import STGCN_CNN_Finetune

def set_env(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN(+CNN) Training')
    # ── core STGCN args ─────────────────────────────────────────────
    parser.add_argument('--enable_cuda',    type=bool, default=True)
    parser.add_argument('--seed',           type=int,  default=42)
    parser.add_argument('--dataset',        type=str,  default='delhiaq')
    parser.add_argument('--pol',            type=str,  default='PM10', choices=['PM25','PM10','NO2'])
    parser.add_argument('--n_his',          type=int,  default=30)
    parser.add_argument('--n_pred',         type=int,  default=30)
    parser.add_argument('--time_intvl',     type=int,  default=1)
    parser.add_argument('--Kt',             type=int,  default=3)
    parser.add_argument('--stblock_num',    type=int,  default=2)
    parser.add_argument('--act_func',       type=str,  default='glu', choices=['glu','gtu'])
    parser.add_argument('--Ks',             type=int,  default=3, choices=[2,3])
    parser.add_argument('--graph_conv_type',type=str,  default='cheb_graph_conv')
    parser.add_argument('--gso_type',       type=str,  default='sym_norm_lap')
    parser.add_argument('--enable_bias',    type=bool, default=True)
    parser.add_argument('--droprate',       type=float,default=0.5)
    parser.add_argument('--lr',             type=float,default=0.001)
    parser.add_argument('--weight_decay_rate', type=float, default=0.001)
    parser.add_argument('--batch_size',     type=int,  default=64)
    parser.add_argument('--epochs',         type=int,  default=50)
    parser.add_argument('--opt',            type=str,  default='adamw', choices=['adamw','nadamw','lion'])
    parser.add_argument('--step_size',      type=int,  default=10)
    parser.add_argument('--gamma',          type=float,default=0.95)
    parser.add_argument('--patience',       type=int,  default=10)
    # ── CNN+finetune args ──────────────────────────────────────────
    parser.add_argument('--use_cnn',        action='store_true', help='Enable STGCN+CNN')
    parser.add_argument('--cnn_model',      type=str,  default='efficientnet_b0')
    args = parser.parse_args()
    print('Configs:', args)

    set_env(args.seed)

    if args.enable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        gc.collect()

    # build ST blocks per Yu et al.
    Ko = args.n_his - (args.Kt - 1)*2*args.stblock_num
    blocks = [[1]] + [[64,16,64]]*args.stblock_num \
           + ([ [128] ] if Ko==0 else [[128,128]]) + [[1]]

    return args, device, blocks

def data_preparate(args, device):
    # ── 1) Graph & GSO ─────────────────────────────────────────────
    adj, n_vertex = dataloader.load_adj(args.dataset)
    gso = utility.calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv':
        gso = utility.calc_chebynet_gso(gso)
    gso = torch.from_numpy(gso.toarray().astype(np.float32)).to(device)
    args.gso = gso

    # ── 2) Read your vel_{pol}.csv as a pure numeric table ─────────
    dp     = os.path.join('data', args.dataset)
    vel_df = pd.read_csv(os.path.join(dp, f'vel_{args.pol}.csv'))
    data   = vel_df.values                # shape [T, N]
    station_cols = vel_df.columns.tolist()# list of length N

    # ── 3) Split into train/val/test ───────────────────────────────
    T       = data.shape[0]
    rate    = 0.15
    n_val   = int(math.floor(T * rate))
    n_test  = n_val
    n_train = T - n_val - n_test

    train = data[:n_train]
    val   = data[n_train : n_train + n_val]
    test  = data[n_train + n_val : n_train + 2*n_val]

    # ── 4) Standard scale the series ────────────────────────────────
    scaler = preprocessing.StandardScaler().fit(train)
    train_s = scaler.transform(train)
    val_s   = scaler.transform(val)
    test_s  = scaler.transform(test)

    # ── 4a) compute global std of the TRAIN time‐series (all features)
    # this is our train_ts_std
    train_ts_std = float(np.std(train_s))  
    args.train_ts_std = train_ts_std

    # ── 5) Create (x,y) sequences ──────────────────────────────────
    x_tr, y_tr = dataloader.data_transform(train_s, args.n_his, args.n_pred, device)
    x_va, y_va = dataloader.data_transform(val_s,   args.n_his, args.n_pred, device)
    x_te, y_te = dataloader.data_transform(test_s,  args.n_his, args.n_pred, device)

    # ── 6) If using CNN, load & tile station patches ───────────────
    if args.use_cnn:
        # load your .npy patches in exact column order
        patch_dir = os.path.join(dp, 's2_patches')
        imgs = []
        for col in station_cols:
            station = col.rsplit('_',1)[0]  # drop the "_PM10" (or "_PM2.5", "_NO2")
            arr = np.load(os.path.join(patch_dir, f'{station}.npy'))
            img = (arr * 255).astype(np.uint8)
            imgs.append(img.transpose(2,0,1))  # [3,H,W]
        imgs_np = np.stack(imgs, axis=0).astype(np.float32)        # [N,3,H,W]
        imgs_t = torch.tensor(imgs_np, dtype=torch.float32).to(device)

        # ─── 6a) Compute μ,σ of static node‐features on the raw patches ───
        # use a throwaway CNN encoder to get per‐node scalar features
        encoder0 = timm.create_model(args.cnn_model, pretrained=True)
        encoder0.global_pool = nn.Identity()
        encoder0.classifier  = nn.Identity()
        encoder0 = encoder0.eval().to(device)
        with torch.no_grad():
            fmap0 = encoder0.forward_features(imgs_t)    # [N, C_f, h, w]
            vec0  = fmap0.mean(dim=[2,3])               # [N, C_f]
            node0 = vec0.mean(dim=1)                    # [N]
            mu, sigma = node0.mean().item(), node0.std().item()
            # guard sigma against zero
            eps = 1e-6
            if sigma < eps:
                print(f"WARNING: static_sigma too small ({sigma}); setting to {eps}")
                sigma = eps
        # 6a-continued: compute init_alpha = train_ts_std / node_feat_std
        init_alpha = args.train_ts_std / sigma
        args.init_alpha = init_alpha
        # free GPU memory
        del encoder0, fmap0, vec0, node0
        torch.cuda.empty_cache()
        # pass μ,σ on to prepare_model via args
        args.static_mu    = mu
        args.static_sigma = sigma

        # now tile as before...

        # tile them into batches
        B_tr, B_va, B_te = x_tr.size(0), x_va.size(0), x_te.size(0)
        E_tr = imgs_t.unsqueeze(0).repeat(B_tr,1,1,1,1)  # [B_tr, N,3,H,W]
        E_va = imgs_t.unsqueeze(0).repeat(B_va,1,1,1,1)
        E_te = imgs_t.unsqueeze(0).repeat(B_te,1,1,1,1)

        train_ds = utils.data.TensorDataset(x_tr, E_tr, y_tr)
        val_ds   = utils.data.TensorDataset(x_va, E_va, y_va)
        test_ds  = utils.data.TensorDataset(x_te, E_te, y_te)
    else:
        train_ds = utils.data.TensorDataset(x_tr, y_tr)
        val_ds   = utils.data.TensorDataset(x_va, y_va)
        test_ds  = utils.data.TensorDataset(x_te, y_te)

    # ── 7) Build DataLoaders ────────────────────────────────────────
    train_loader = utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = utils.data.DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = utils.data.DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    return n_vertex, scaler, train_loader, val_loader, test_loader

def prepare_model(args, blocks, n_vertex, device):
    # loss_fn = nn.MSELoss()
    loss_fn = nn.SmoothL1Loss()
    es = earlystopping.EarlyStopping(delta=0.0,
                                     patience=args.patience,
                                     verbose=True,
                                     path=f"C-STGCN_{args.dataset}{args.pol}.pt")

    if args.use_cnn:
        model = STGCN_CNN_Finetune(args, blocks, n_vertex,
                                   cnn_model=args.cnn_model
                                  ).to(device)
    else:
        if args.graph_conv_type=='cheb_graph_conv':
            model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
        else:
            model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)

    # optimizer & scheduler
    if args.opt=="adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay_rate)
    elif args.opt=="nadamw":
        optimizer = optim.NAdam(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay_rate)
    else:  # lion
        optimizer = opt.Lion(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.step_size,
                                          gamma=args.gamma)
    return loss_fn, es, model, optimizer, scheduler

def train(args, device, model, loss_fn, optimizer, scheduler, es, train_loader, val_loader):
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0; n_steps = 0
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            if args.use_cnn:
                x_ts, x_img, y = batch
                x_ts, x_img, y = x_ts.to(device), x_img.to(device), y.to(device)
                out = model(x_ts, x_img)
            else:
                x_ts, y = batch
                x_ts, y = x_ts.to(device), y.to(device)
                out = model(x_ts)

            pred = out.view(out.size(0), -1)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item() * y.size(0)
            n_steps += y.size(0)

        scheduler.step()
        train_loss /= n_steps

        # validation
        model.eval()
        val_loss = 0.0; v_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                if args.use_cnn:
                    x_ts, x_img, y = batch
                    x_ts, x_img, y = x_ts.to(device), x_img.to(device), y.to(device)
                    out = model(x_ts, x_img)
                else:
                    x_ts, y = batch
                    x_ts, y = x_ts.to(device), y.to(device)
                    out = model(x_ts)

                pred = out.view(out.size(0), -1)
                l = loss_fn(pred, y)
                val_loss += l.item() * y.size(0)
                v_steps += y.size(0)
        val_loss /= v_steps

        print(f"Epoch {epoch+1}/{args.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        es(val_loss, model)
        if es.early_stop:
            print("Early stopping triggered.")
            break

@torch.no_grad()
def test(args, device, model, loss_fn, scaler, test_loader):
    model.load_state_dict(torch.load(f"C-STGCN_{args.dataset}{args.pol}.pt"))
    model.eval()

    # compute MSE
    total_mse = 0.0; n=0
    with torch.no_grad():
        for batch in test_loader:
            if args.use_cnn:
                x_ts, x_img, y = batch
                x_ts, x_img, y = x_ts.to(device), x_img.to(device), y.to(device)
                out = model(x_ts, x_img)
            else:
                x_ts, y = batch
                x_ts, y = x_ts.to(device), y.to(device)
                out = model(x_ts)

            pred = out.view(out.size(0), -1)
            mse = loss_fn(pred, y)
            total_mse += mse.item() * y.size(0)
            n += y.size(0)
    print(f"Test MSE: {total_mse/n:.6f}")

    # your original metrics (MAE, RMSE, WMAPE)
    mae, rmse, wmape = utility.evaluate_metric_cnn(model, test_loader, scaler)
    print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, WMAPE: {wmape:.6f}")

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args, device, blocks = get_parameters()
    n_vertex, scaler, train_loader, val_loader, test_loader = data_preparate(args, device)
    loss_fn, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex, device)
    train(args, device, model, loss_fn, optimizer, scheduler, es, train_loader, val_loader)
    test(args, device, model, loss_fn, scaler, test_loader)
