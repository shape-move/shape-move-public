import torch
import torch.nn as nn

from model.nnutils.base import Encoder, Decoder
from model.nnutils.quantizer import QuantizeEMAReset, FSQ
    
class SCMotionVAE(nn.Module):
    def __init__(self,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 mu=0.99,
                 input_dim=263,
                 debug=False,
                 quantizer_type='ema_reset',
                 shape_dim=32,
                 p=0,
                 level=None):
        super().__init__()

        self.quantizer_type = quantizer_type
        self.code_dim = code_dim
        self.encoder = Encoder(input_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, debug=debug)
        de_output_emb_width = output_emb_width+shape_dim
        self.decoder = Decoder(input_dim, de_output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, debug=debug)
        
        in_dim = 10
        layer = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, shape_dim),
            nn.Dropout(0.0),
        ]

        self.embedding = nn.Sequential(*layer)


        if self.quantizer_type == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, mu)
        elif self.quantizer_type == "fsq":
            self.quantizer = FSQ(level, output_emb_width)
        else:
            raise NotImplementedError
    
    def preprocess(self, x):
        # Motion data, (b, T, J*3) -> (b, J*3, T)
        x = x.permute(0,2,1).float()

        return x

    def postprocess(self, x):
        # Motion data, (b, J*3, T) -> (b, T, J*3)
        x = x.permute(0,2,1)

        return x

    def encode(self, x):
        N = x.shape[0]
        x = self.preprocess(x)
        x = self.encoder(x)
        if self.quantizer_type != "fsq":
            x_mid = self.postprocess(x)
            x_mid = x_mid.contiguous().view(-1, x_mid.shape[-1])
        else:
            x_mid = x
        code_idx = self.quantizer.quantize(x_mid)
        code_idx = code_idx.view(N, -1)
        
        return code_idx
    
    def forward(self, x, beta, loss_dict):
        x = self.preprocess(x)
        x_mid = self.encoder(x)

        code_idx, loss, preplexity = self.quantizer(x_mid)

        shape_embed = self.embedding(beta)[...,None] # ([256, 32, 1])

        # to match the time dim
        shape_embed = shape_embed.tile((1,1,code_idx.shape[-1])) # ([256, 32, 16])

        code_idx = torch.cat([code_idx, shape_embed], dim=1)

        x_decode = self.decoder(code_idx)
        x_out = self.postprocess(x_decode)

        x_decode_detach = self.decoder(code_idx.detach())
        x_out_detach = self.postprocess(x_decode_detach)
        loss_dict.update({'loss_commit': loss})

        return x_out, loss, preplexity, x_out_detach

    def decode(self, x, beta):
        x_decode = self.quantizer.dequantize(x)
        x_decode = x_decode.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()

        shape_embed = self.embedding(beta)[...,None]

        # to match the time dim
        shape_embed = shape_embed.tile((1,1,x_decode.shape[-1]))
        x_decode = torch.cat([x_decode, shape_embed], dim=1)
        x_decode = self.decoder(x_decode)
        x_out = self.postprocess(x_decode)

        return x_out


   