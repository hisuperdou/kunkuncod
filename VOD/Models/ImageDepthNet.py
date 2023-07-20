import torch.nn as nn
# from .tem import RF
from .t2t_vit import T2t_vit_t_14
from .Transformer import Transformer
from .Transformer import token_Transformer
from .Decoder import Decoder
import torch
from .tem import RF
from .tem import ASPP

class ImageDepthNet(nn.Module):
    def __init__(self, args):
        super(ImageDepthNet, self).__init__()

        # VST Encoder
        # self.RF(3, 3)


        self.rgb_backbone = T2t_vit_t_14(pretrained=True, args=args)
        self.depth_backbone = T2t_vit_t_14(pretrained=True, args=args)


        #自定义部分


        # VST Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)


        # 自定义部分
        # ASPP module
        atrous_rates = [6, 12, 18, 24]  # Atrous rates for the ASPP module
        self.aspp = ASPP(in_channels=384, out_channels=384, atrous_rates=atrous_rates)


        #自定义结束


        # VST Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=args.img_size)

    def forward(self, image_Input, depth_Input):
        B, _, _, _ = image_Input.shape
        # image_Input [B, 3, 224, 224]
        # VST Encoder

        # VST Encoder

        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4 = self.rgb_backbone(image_Input)
        depth_fea_1_16, _, _ = self.depth_backbone(depth_Input)

        # VST Convertor
        rgb_fea_1_16, depth_fea_1_16 = self.transformer(rgb_fea_1_16, depth_fea_1_16)

        # rgb_fea_1_16 [B, 14*14, 384]   depth_fea_1_16 [B, 14*14, 384]


        #自定义部分--------------------------
        # 将 rgb_fea_1_16 重塑为匹配 ASPP 模块期望输入形状的大小
        rgb_fea_1_16 = rgb_fea_1_16.view(B, 384, 14, 14)
        depth_fea_1_16 = depth_fea_1_16.view(B, 384, 14, 14)
        # 将 ASPP 模块应用于 rgb_fea_1_16

        rgb_fea_1_16 = self.aspp(rgb_fea_1_16)
        depth_fea_1_16 =self.aspp(depth_fea_1_16)
        

        # 将 rgb_fea_1_16 重新重塑回原始形状
        rgb_fea_1_16 = rgb_fea_1_16.view(B, -1, 384)
        depth_fea_1_16 = depth_fea_1_16.view(B, -1, 384)

        




        #---------------------------------

        # VST Decoder
        saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens = self.token_trans(rgb_fea_1_16,
                                                                                                          depth_fea_1_16)
        # saliency_fea_1_16 [B, 14*14, 384]
        # fea_1_16 [B, 1 + 14*14 + 1, 384]
        # saliency_tokens [B, 1, 384]
        # contour_fea_1_16 [B, 14*14, 384]
        # contour_tokens [B, 1, 384]

        outputs = self.decoder(saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens,
                               rgb_fea_1_8, rgb_fea_1_4)

        return outputs
