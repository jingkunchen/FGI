from typing import Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import BasicUNet


class TextFusionModule(nn.Module):
    def __init__(self, feature_dim, text_dim, num_heads=8, img_h=14, img_w=14):
       
        super(TextFusionModule, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self.text_proj = nn.Linear(text_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.img_h = img_h
        self.img_w = img_w
        self.num_tokens = img_h * img_w
        # learnable �~M置�~V�| ~A�~L形�~J� [num_tokens, feature_dim]
        self.pos_embedding = nn.Parameter(torch.zeros(self.num_tokens, feature_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, feature, text_embedding):
       
        B, C, H, W = feature.shape

        feat_seq = feature.view(B, C, H * W).permute(2, 0, 1)
        pos_embed = self.pos_embedding.unsqueeze(1).expand(self.num_tokens, B, C)
        feat_seq = feat_seq + pos_embed
        text_feat = self.text_proj(text_embedding.squeeze(1)).permute(1, 0, 2)
        attn_out, _ = self.attn(query=feat_seq, key=text_feat, value=text_feat)
        fused_seq = self.norm(feat_seq + attn_out)
        fused_feature = fused_seq.permute(1, 2, 0).view(B, C, H, W)
        return fused_feature

class DistillProjectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(DistillProjectionBlock, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        return self.proj(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feat_dim=128):
        super().__init__()

        self.feat_dim = feat_dim

        self.encoder = BasicUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=feat_dim,
            features=(64, 128, 256, 512, 1024, 128),
            norm=("group", {"num_groups": 4}),
        )

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=feat_dim, out_channels=feat_dim, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=feat_dim, out_channels=feat_dim, kernel_size=1),
        )

        self.classifier = nn.Conv2d(in_channels=feat_dim, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        feat = self.encoder(x)
        logits = self.classifier(feat)

        return {"logits": logits, "feature": feat}
    
class UNet_student_or(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feat_dim=128):
        super().__init__()

        self.feat_dim = feat_dim

        self.encoder = BasicUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=feat_dim,
            features=(32, 64, 128, 256, 512, 32),
            norm=("group", {"num_groups": 4}),
        )

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=feat_dim, out_channels=feat_dim, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=feat_dim, out_channels=feat_dim, kernel_size=1),
        )

        self.classifier = nn.Conv2d(in_channels=feat_dim, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        feat = self.encoder(x)
        logits = self.classifier(feat)

        return {"logits": logits, "feature": feat}


class UNet_teacher(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feat_dim=128):
        super().__init__()
        self.feat_dim = feat_dim


        self.encoder = BasicUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=feat_dim,
            features=(64, 128, 256, 512, 1024, 128),
            norm=("group", {"num_groups": 4}),
        )

        # head ~C~H~F
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=feat_dim, out_channels=feat_dim, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=feat_dim, out_channels=feat_dim, kernel_size=1),
        )

        self.fusion_x1 = TextFusionModule(feature_dim=128, text_dim=768, num_heads=8, img_h=112, img_w=112)
        self.fusion_x2 = TextFusionModule(feature_dim=256, text_dim=768, num_heads=8, img_h=56, img_w=56)
        self.fusion_x3 = TextFusionModule(feature_dim=512, text_dim=768, num_heads=8, img_h=28, img_w=28)
        self.fusion_x4 = TextFusionModule(feature_dim=1024, text_dim=768, num_heads=8, img_h=14, img_w=14)

        self.distill_proj_x1 = DistillProjectionBlock(128, 128)
        self.distill_proj_x2 = DistillProjectionBlock(256, 256)
        self.distill_proj_x3 = DistillProjectionBlock(512, 512)
        self.distill_proj_x4 = DistillProjectionBlock(1024, 1024)
        self.classifier = nn.Conv2d(in_channels=feat_dim, out_channels=out_channels, kernel_size=1)

    def forward(self, x, joint_embedding):

        x0 = self.encoder.conv_0(x)       # hape: (B, 64, H, W)
        x1 = self.encoder.down_1(x0)        # (B, 128, H/2, W/2)
        x2 = self.encoder.down_2(x1)        # (B, 256, H/4, W/4)
        x3 = self.encoder.down_3(x2)        # (B, 512, H/8, W/8)
        x4 = self.encoder.down_4(x3)        # (B, 1024, H/16, W/16)

        fused_x1 = self.fusion_x1(x1, joint_embedding)  # [B, 128, 112, 112]
        fused_x2 = self.fusion_x2(x2, joint_embedding)         # [B, 256, 56, 56]
        fused_x3 = self.fusion_x3(x3, joint_embedding)         # [B, 512, 28, 28]
        fused_x4 = self.fusion_x4(x4, joint_embedding)         # [B, 1024, 14, 14]

        x1_final = x1 + fused_x1
        x2_final = x2 + fused_x2
        x3_final = x3 + fused_x3
        x4_final = x4 + fused_x4

        distill_x1 = self.distill_proj_x1(x1_final)  # [B, 128, 112, 112]
        distill_x2 = self.distill_proj_x2(x2_final)  # [B, 256, 56, 56]
        distill_x3 = self.distill_proj_x3(x3_final)  # [B, 512, 28, 28]
        distill_x4 = self.distill_proj_x4(x4_final)  # [B, 1024, 14, 14]
        x1_final = distill_x1 + x1
        x2_final = distill_x2 + x2
        x3_final = distill_x3 + x3
        x4_final = distill_x4 + x4


        u4 = self.encoder.upcat_4(x4_final, x3_final)   # (B, 512, H/8, W/8)
        u3 = self.encoder.upcat_3(u4, x2_final)   # (B, 256, H/4, W/4)
        u2 = self.encoder.upcat_2(u3, x1_final)   # (B, 128, H/2, W/2)
        u1 = self.encoder.upcat_1(u2, x0)   # (B, 128, H, W)


        feat = self.encoder.final_conv(u1)  # (B, feat_dim, H, W)


        head_out = self.head(feat)
        logits = self.classifier(head_out)

        return {"logits": logits, "feature": feat, "encoder_features": {"x1": distill_x1, "x2": distill_x2, "x3": distill_x3, "x4": distill_x4}}

class UNet_student(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feat_dim=128):
        super().__init__()
        self.feat_dim = feat_dim


        self.encoder = BasicUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=feat_dim,
            features=(32, 64, 128, 256, 512, 32),
            norm=("group", {"num_groups": 4}),
        )

        self.distill_proj_x1 = DistillProjectionBlock(64, 64)
        self.distill_proj_x2 = DistillProjectionBlock(128, 128)
        self.distill_proj_x3 = DistillProjectionBlock(256, 256)
        self.distill_proj_x4 = DistillProjectionBlock(512, 512)
      
        self.student_adapter_x1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.student_adapter_x2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.student_adapter_x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.student_adapter_x4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # head ~C~H~F
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=feat_dim, out_channels=feat_dim, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=feat_dim, out_channels=feat_dim, kernel_size=1),
        )

        # ~H~F__~Y~C~H~F
        self.classifier = nn.Conv2d(in_channels=feat_dim, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # ~[~N~C~T encoder ~F~E~C~P~D~B__~N~W__~W~I~A
        x0 = self.encoder.conv_0(x)       # ~S~G shape: (B, 64, H, W)
        x1 = self.encoder.down_1(x0)        # (B, 128, H/2, W/2)
        x2 = self.encoder.down_2(x1)        # (B, 256, H/4, W/4)
        x3 = self.encoder.down_3(x2)        # (B, 512, H/8, W/8)
        x4 = self.encoder.down_4(x3)        # (B, 1024, H/16, W/16)
                # 学�~T~_�~Z~D distill �~I��~A
        distill_x1 = self.distill_proj_x1(x1)  # [B, 64, H/2, W/2]
        distill_x2 = self.distill_proj_x2(x2)  # [B, 128, H/4, W/4]
        distill_x3 = self.distill_proj_x3(x3)  # [B, 256, H/8, W/8]
        distill_x4 = self.distill_proj_x4(x4)  # [B, 512, H/16, W/16]
        # 学�~T~_ distill �~I��~A�~O�~G adapter head 转�~M��~H��~U~Y�~H对�~T�~B�~Z~D维度
        aligned_x1 = self.student_adapter_x1(distill_x1)  # [B, 128, H/2, W/2]
        aligned_x2 = self.student_adapter_x2(distill_x2)  # [B, 256, H/4, W/4]
        aligned_x3 = self.student_adapter_x3(distill_x3)  # [B, 512, H/8, W/8]
        aligned_x4 = self.student_adapter_x4(distill_x4)  # [B, 1024, H/16, W/16]
        # decoder ~C~H~F
        u4 = self.encoder.upcat_4(x4, x3)   # (B, 512, H/8, W/8)
        u3 = self.encoder.upcat_3(u4, x2)   # (B, 256, H/4, W/4)
        u2 = self.encoder.upcat_2(u3, x1)   # (B, 128, H/2, W/2)
        u1 = self.encoder.upcat_1(u2, x0)   # (B, 128, H, W)

        # ~\~@~H~I~A~H~M BasicUNet ~\~@~P~N~Z~D~S~G~I
        feat = self.encoder.final_conv(u1)  # (B, feat_dim, H, W)

        # ~O~G head ~R~L classifier ~W~H~\~@~H logits
        head_out = self.head(feat)
        logits = self.classifier(head_out)

        # ~T~[~^~W~E~L~E__ "logits" ~X~\~@~H~D~K~L
        # "feature" ~X encoder ~Z~D~\~@~H~S~G~L
        # "encoder_features" ~H~Y~L~E~P~F~P~D~B~Z~D__~W~I~A
        return {"logits": logits, "feature": feat, "encoder_features": {"x1": aligned_x1, "x2": aligned_x2, "x3": aligned_x3, "x4": aligned_x4}}