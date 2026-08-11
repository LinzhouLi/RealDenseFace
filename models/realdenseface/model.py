import timm
import torch
from torch import nn
import torch.nn.functional as F

from .dpt_head import DPTHead, FlexDPTHead, DualFlexDPTHead
from .decoder import ViTDecoder, generate_positional_encoding


class UVFeatureMap(nn.Module):
    def __init__(self, D_uv=32, dim=384):
        super(UVFeatureMap, self).__init__()
        self.D_uv = D_uv
        channels_per_level = dim // 4
        
        resolutions = [D_uv // (2**i) for i in range(4)]
        
        self.pyramid_params = nn.ParameterList()
        
        for res in resolutions:
            pe_init = generate_positional_encoding(res, res, channels_per_level)
            self.pyramid_params.append(nn.Parameter(pe_init))

        dim = channels_per_level * 4
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

        self.register_buffer('cached_feature', None, persistent=False)


    def _generate_feature_map(self):
        upsampled_features = []
        target_size = (self.D_uv, self.D_uv)
        
        for param in self.pyramid_params:
            feat = F.interpolate(param, size=target_size, mode='bilinear', align_corners=True)
            upsampled_features.append(feat)
        
        combined_features = torch.cat(upsampled_features, dim=1)
            
        out = self.mlp(combined_features)
        return out
    

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.cached_feature = None


    @torch.no_grad()
    def build_cache(self):
        self.cached_feature = self._generate_feature_map().detach().contiguous()


    def forward(self, batch_size=1):
        if self.training:
            out = self._generate_feature_map()
        else:
            if self.cached_feature is None:
                with torch.no_grad():
                    self.build_cache()
            out = self.cached_feature
        if batch_size > 1: out = out.expand(batch_size, -1, -1, -1)
        return out


class DinoEncoder(nn.Module):
    def __init__(self, 
        model_name='vit_small_plus_patch16_dinov3.lvd1689m',
        img_size=512, 
        n_out_layers=4,
        pretrained=True,
        freeze=True
    ):
        super(DinoEncoder, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, img_size=img_size, dynamic_img_size=False)
        self.freeze = freeze
        
        # set output layer indices
        n_blocks = len(self.model.blocks)
        n_intervals = n_blocks // n_out_layers
        offset = n_intervals - 1 + n_blocks % n_out_layers
        out_indices = [i * n_intervals + offset for i in range(n_out_layers)]
        if n_blocks == 24 and n_out_layers == 4: out_indices = [4, 11, 17, 23]
        self.out_indices = out_indices
        # print("Dino encoder feature output indices:", self.out_indices)

        # freeze the model
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    
    def _forward(self, x):
        features = self.model.forward_intermediates(
            x,
            indices=self.out_indices,
            return_prefix_tokens=False,
            norm=True,
            output_fmt='NCHW',
            intermediates_only=True
        )
        return features


    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                return self._forward(x)
        else:
            return self._forward(x)
       

class RealDenseFace(nn.Module):
    def __init__(self,
        dino_encoder='vit_small_plus_patch16_dinov3.lvd1689m',
        dino_pretrained=True,
        dino_frozen=True,
        dpt_feature_dim=128,
        decoder_blocks=4,
        decoder_heads=6,
        target_size=512,

        dpt_mode='a',
        use_dual_dpt=False,
        reverse_enc_feats=True,
        reverse_dec_feats=True,
        add_dec_pe=True,
        **kwargs
    ):
        super(RealDenseFace, self).__init__()
        self.reverse_enc_feats = reverse_enc_feats
        self.reverse_dec_feats = reverse_dec_feats

        self.encoder = DinoEncoder(
            model_name=dino_encoder, 
            n_out_layers=decoder_blocks, 
            pretrained=dino_pretrained, 
            freeze=dino_frozen
        )
        enc_feat_dim = self.encoder.model.num_features
        patch_size = self.encoder.model.patch_embed.patch_size[0]

        self.decoder_coord = ViTDecoder(
            embed_dim=enc_feat_dim,
            num_heads=decoder_heads,
            num_stages=decoder_blocks,
            add_pe=add_dec_pe
        )

        self.decoder_depth = ViTDecoder(
            embed_dim=enc_feat_dim,
            num_heads=decoder_heads,
            num_stages=decoder_blocks,
            add_pe=False
        )

        dpt_mid_channels = []
        for i in range(decoder_blocks):
            d_scale = 2 ** min(4, decoder_blocks - i - 1)
            dpt_mid_channels.append(enc_feat_dim // d_scale)
        # print("DPT head mid channels:", dpt_mid_channels)

        self.dpt_head_coord = FlexDPTHead(
            in_channels=enc_feat_dim,
            out_channels=dpt_feature_dim // 2,
            features=dpt_feature_dim,
            mid_channels=dpt_mid_channels,
            dpt_mode=dpt_mode
        )

        self.dpt_head_depth = FlexDPTHead(
            in_channels=enc_feat_dim,
            out_channels=dpt_feature_dim // 2,
            features=dpt_feature_dim,
            mid_channels=dpt_mid_channels,
            dpt_mode=dpt_mode
        )

        self.uv_feat_map_coord = UVFeatureMap(D_uv=target_size // patch_size, dim=enc_feat_dim)
        self.uv_feat_map_depth = UVFeatureMap(D_uv=target_size // patch_size, dim=enc_feat_dim)
        self.coord_output = nn.Sequential(
            nn.Conv2d(dpt_feature_dim // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0)
        )
        self.depth_output = nn.Sequential(
            nn.Conv2d(dpt_feature_dim // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        )
        self.coord_log_var_output = nn.Sequential(
            nn.Conv2d(dpt_feature_dim // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        )
        self.depth_log_var_output = nn.Sequential(
            nn.Conv2d(dpt_feature_dim // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        )


    @torch.no_grad()
    def build_inference_cache(self):
        self.uv_feat_map_coord.build_cache()
        self.uv_feat_map_depth.build_cache()


    def forward(self, x):
        B, C, H, W = x.shape
        uv_feat_map_coord = self.uv_feat_map_coord(batch_size=B)
        uv_feat_map_depth = self.uv_feat_map_depth(batch_size=B)
        enc_feats = self.encoder(x)

        if self.reverse_enc_feats: enc_feats = enc_feats[::-1]
        dec_feats_coord = self.decoder_coord(uv_feat_map_coord, enc_feats) # from deep to shallow
        dec_feats_depth = self.decoder_depth(uv_feat_map_depth, enc_feats) # from deep to shallow

        if self.reverse_dec_feats: 
            dec_feats_coord = dec_feats_coord[::-1]
            dec_feats_depth = dec_feats_depth[::-1]
        dpt_feat_coord = self.dpt_head_coord(dec_feats_coord) # from shallow to deep
        dpt_feat_depth = self.dpt_head_depth(dec_feats_depth) # from shallow to deep

        coords = self.coord_output(dpt_feat_coord)
        depths = self.depth_output(dpt_feat_depth)
        coord_log_vars = self.coord_log_var_output(dpt_feat_coord)
        depth_log_vars = self.depth_log_var_output(dpt_feat_depth)
        return coords, coord_log_vars, depths, depth_log_vars
    

