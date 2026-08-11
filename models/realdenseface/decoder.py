import torch
import numpy as np
import torch.nn as nn


def generate_positional_encoding(h, w, d_model):
    y = torch.linspace(-1, 1, h)
    x = torch.linspace(-1, 1, w)
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    
    gy = gy.unsqueeze(-1) 
    gx = gx.unsqueeze(-1)
    
    num_freqs = d_model // 4

    max_freq_log2 = np.log2(max(h, w)) - 1
    
    freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, num_freqs)
    
    freqs = freq_bands * np.pi 
    
    pe_x_sin = torch.sin(gx * freqs)
    pe_x_cos = torch.cos(gx * freqs)
    pe_y_sin = torch.sin(gy * freqs)
    pe_y_cos = torch.cos(gy * freqs)
    
    pe = torch.cat([pe_x_sin, pe_x_cos, pe_y_sin, pe_y_cos], dim=-1)
    
    pe = pe.permute(2, 0, 1).unsqueeze(0) # [1, d_model, h, w]
    return pe


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            dropout=attn_drop, 
            batch_first=True 
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, context_k=None, context_v=None):
        """
        Args:
            x: [Batch, Seq_Len_Q, Dim]
            context: [Batch, Seq_Len_KV, Dim] (Optional)
        """
        
        residual = x
        x = self.norm1(x)
        
        if context_k is None or context_v is None:
            # --- Self Attention ---
            x, _ = self.attn(query=x, key=x, value=x, need_weights=False)
        else:
            # --- Cross Attention ---
            x, _ = self.attn(query=x, key=context_k, value=context_v, need_weights=False)
            
        x = residual + x
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderStage(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.cross_blk = Block(dim, num_heads, mlp_ratio, drop, attn_drop)
        self.self_blk = Block(dim, num_heads, mlp_ratio, drop, attn_drop)


    def forward(self, x, context, pos_embed=None):
        cross_att_v = context if pos_embed is None else context + pos_embed
        x = self.cross_blk(x, context, cross_att_v)
        x = self.self_blk(x)   
        return x


class ViTDecoder(nn.Module):
    def __init__(
        self, 
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4., 
        num_stages=4,
        add_pe=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.add_pe = add_pe

        self.stages = nn.ModuleList([
            DecoderStage(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_stages)
        ])
        self.enc_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) 
            for _ in range(num_stages)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

        pos_embed = generate_positional_encoding(32, 32, embed_dim).flatten(2).transpose(1, 2).contiguous()
        self.register_buffer('pos_embed', pos_embed)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, uv_map, enc_feats):
        assert len(self.stages) == len(enc_feats)
        B, C, H, W = uv_map.shape
        
        x = uv_map.flatten(2).transpose(1, 2).contiguous() # [B, C, H, W] -> [B, L, C]

        outputs = []
        for i, (stage, enc_norm) in enumerate(zip(self.stages, self.enc_norms)):
            # enc_feat = enc_feats[-(i+1)] # deep to shallow
            enc_feat = enc_feats[i] # reversed outside
            enc_feat = enc_feat.flatten(2).transpose(1, 2).contiguous() # [B, C, H, W] -> [B, L, C]
            x = stage(x, enc_norm(enc_feat), self.pos_embed if self.add_pe else None)
            outputs.append(x)
            
        for i in range(len(outputs)):
            out = outputs[i]
            out = self.norm(out)
            out = out.transpose(1, 2).view(B, C, H, W).contiguous() # [B, L, C] -> [B, C, H, W]
            outputs[i] = out
        return outputs