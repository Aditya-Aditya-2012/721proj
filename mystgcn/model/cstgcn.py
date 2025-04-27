import torch
import torch.nn as nn
import timm
from model.models import STGCNChebGraphConv
import torch.nn.functional as F

class STGCN_CNN_Finetune(nn.Module):
    def __init__(self, args, blocks, n_vertex,
                 cnn_model='efficientnet_b0'):
        super().__init__()
        # 1) static‐feature stats (from your data_preparate)
        self.register_buffer('mu',    torch.tensor(args.static_mu,    dtype=torch.float32))
        self.register_buffer('sigma', torch.tensor(args.static_sigma, dtype=torch.float32))

        # 2) frozen ImageNet encoder
        self.encoder = timm.create_model(cnn_model, pretrained=True)
        # strip away its pooling & classifier
        self.encoder.global_pool = nn.Identity()
        self.encoder.classifier  = nn.Identity()
        for p in self.encoder.parameters():  # freeze everything
            p.requires_grad = False

        # 3) small refinement head over the raw embeddings
        #    embed_dim == self.encoder.num_features
        embed_dim = self.encoder.num_features
        self.refine = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # 4) learnable fusion weight, init from args
        self.alpha = nn.Parameter(torch.tensor(args.init_alpha, dtype=torch.float32))

        # 5) the STGCN backbone
        self.stgcn = STGCNChebGraphConv(args, blocks, n_vertex)

    def forward(self, x_ts, x_img):
        B, Cchan, T, N = x_ts.shape  # Cchan==1

        # ─── 1) Flatten & upsample patches ─────────────────────────────
        # [B,N,3,64,64] → [B*N,3,64,64]
        x_imgs = x_img.view(B * N, 3, 64, 64)

        # Upsample to (128×128) or (224×224)
        # Try 128 first for memory; you can bump to 224 if you have headroom.
        x_imgs = F.interpolate(x_imgs,
                               size=(128,128),
                               mode='bilinear',
                               align_corners=False)

        # ─── 2) ImageNet‐style normalization ───────────────────────────
        # (mean & std are broadcast over (B*N,3,128,128))
        mean = x_imgs.new_tensor([0.485,0.456,0.406]).view(1,3,1,1)
        std  = x_imgs.new_tensor([0.229,0.224,0.225]).view(1,3,1,1)
        x_imgs = (x_imgs - mean) / std

        # ─── 3) Debug check: no NaNs going into the CNN? ───────────────
        
        # ─── 4) CNN forward & pool ────────────────────────────────────
        fmap = self.encoder.forward_features(x_imgs)       # [B*N, C_f, h', w']
        vec  = fmap.mean(dim=[2,3])                        # [B*N, C_f]

        # ─── 5) Reshape back & reduce to scalar per node ─────────────
        emb = vec.view(B, N, -1)                           # [B, N, C_f]
        emb_t = emb.permute(0, 2, 1)                  # [B, C_f, N]
        emb_r = self.refine(emb_t)                   # [B, C_f, N]
        emb   = emb_r.permute(0, 2, 1)                # [B, N, C_f]

        # ─── 5) Node‐scalar & standardize
        node_raw  = emb.mean(dim=2)                  # [B, N]
        sigma     = torch.clamp(self.sigma, min=1e-6)
        node_feat = (node_raw - self.mu) / sigma     # [B, N]
        # 3) build a 4-D feature map to match x_ts
        #    expand to shape [B,1,T,N]
        # E_seq = node_feat.unsqueeze(1).unsqueeze(2).expand(-1, C, T, -1)

        # 4) fuse and forward
        fused_ts = x_ts + self.alpha * node_feat.unsqueeze(1).unsqueeze(2).expand(-1, Cchan, T, -1)                     # [B,1,T,N]
        out      = self.stgcn(fused_ts)                  # [B,1_pred,T_pred,N] or similar
        return out