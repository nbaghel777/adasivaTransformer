import numpy as np
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat = None, out_feat = None, dropout = 0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.linear1 = nn.Linear(in_feat, hid_feat)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hid_feat, out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)

class Attention(nn.Module):
    """
    Implement multi head self attention layer using the "Einstein summation convention".

    Parameters
    ----------
    dim:
        Token's dimension, EX: word embedding vector size
    num_heads:
        The number of distinct representations to learn
    dim_head:
        The dimension of the each head
    discriminator:
        Used in discriminator or not.
    """
    def __init__(self, dim, num_heads = 4, dim_head = None, discriminator = False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = int(dim / num_heads) if dim_head is None else dim_head
        self.weight_dim = self.num_heads * self.dim_head
        self.to_qkv = nn.Linear(dim, self.weight_dim * 3, bias = False)
        self.scale_factor = dim ** -0.5
        self.discriminator = discriminator
        self.w_out = nn.Linear(self.weight_dim, dim, bias = True)

        if discriminator:
            u, s, v = torch.svd(self.to_qkv.weight)
            self.init_spect_norm = torch.max(s)

    def forward(self, x):
        assert x.dim() == 3

        if self.discriminator:
            try:
                #print('qkvshape',self.to_qkv.weight)
                u, s, v = torch.svd(self.to_qkv.weight+0.0001)
            except:
                # torch.svd may have convergence issues for GPU and CPU.
                #self.to_qkv.weight=self.to_qkv.weight + 0.001*torch.rand(1152, 384).to("cuda")
                u, s, v = torch.svd(self.to_qkv.weight + 0.001*torch.rand(1152, 384).to("cuda"))
            self.to_qkv.weight = torch.nn.Parameter(self.to_qkv.weight * self.init_spect_norm / torch.max(s))

        # Generate the q, k, v vectors
        qkv = self.to_qkv(x)
        q, k, v = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d', k = 3, h = self.num_heads))

        # Enforcing Lipschitzness of Transformer Discriminator
        # Due to Lipschitz constant of standard dot product self-attention
        # layer can be unbounded, so adopt the l2 attention replace the dot product.
        if self.discriminator:
            attn = torch.cdist(q, k, p = 2)
        else:
            attn = torch.einsum("... i d, ... j d -> ... i j", q, k)
        scale_attn = attn * self.scale_factor
        scale_attn_score = torch.softmax(scale_attn, dim = -1)
        result = torch.einsum("... i j, ... j d -> ... i d", scale_attn_score, v)

        # re-compose
        result = rearrange(result, "b h t d -> b t (h d)")
        return self.w_out(result)


class DEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads = 4, dim_head = None,
        dropout = 0., mlp_ratio = 4):
        super(DEncoderBlock, self).__init__()
        self.attn = Attention(dim, num_heads, dim_head, discriminator = True)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MLP(dim, dim * mlp_ratio, dropout = dropout)

    def forward(self, x):
        x1 = self.norm1(x)
        x = x + self.dropout(self.attn(x1))
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

class DTransformerEncoder(nn.Module):
    def __init__(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        super(DTransformerEncoder, self).__init__()
        self.blocks = self._make_layers(dim, blocks, num_heads, dim_head, dropout)

    def _make_layers(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        layers = []
        for _ in range(blocks):
            layers.append(DEncoderBlock(dim, num_heads, dim_head, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

def prepatch(img,dpatch_size):
    img_patches = F.unfold(img, kernel_size=dpatch_size,stride=dpatch_size)
    img_patches = img_patches.permute(0,2,1)
    return img_patches
    
class Discriminator(nn.Module):
    def __init__(self,
        in_channels = 3,
        patch_size = 8,
        extend_size = 2,
        dim = 192,
        blocks = 2,
        num_heads = 6,
        dim_head = None,
        dropout = 0
    ):
        super(Discriminator, self).__init__()
        #self.patch_size = patch_size + 2 * extend_size
        self.patch_size = patch_size 
        self.token_dim = in_channels * (self.patch_size ** 2)
        self.project_patches = nn.Linear(self.token_dim, dim)

        self.emb_dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emb1D = nn.Parameter(torch.randn(self.token_dim + 1, dim))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )
        self.Transformer_Encoder = DTransformerEncoder(dim, blocks, num_heads, dim_head, dropout)

    def forward(self, img):
        img_patches=prepatch(img,self.patch_size)
        # Generate overlappimg image patches
        #stride_h = (img.shape[2] - self.patch_size) // 8 + 1
        #stride_w = (img.shape[3] - self.patch_size) // 8 + 1
        #img_patches = img.unfold(2, self.patch_size, stride_h).unfold(3, self.patch_size, stride_w)
        #print(img_patches.shape,'q1')
        img_patches = img_patches.contiguous().view(
            img_patches.shape[0], img_patches.shape[2] * img_patches.shape[3], img_patches.shape[1] * img_patches.shape[4] * img_patches.shape[5]
        )
        #print(img_patches.shape,'q2')
        img_patches = self.project_patches(img_patches)
        batch_size, tokens, _ = img_patches.shape

        # Prepend the classifier token
        cls_token = repeat(self.cls_token, '() n d -> b n d', b = batch_size)
        img_patches = torch.cat((cls_token, img_patches), dim = 1)

        # Plus the positional embedding
        img_patches = img_patches + self.pos_emb1D[: tokens + 1, :]
        img_patches = self.emb_dropout(img_patches)

        result = self.Transformer_Encoder(img_patches)
        logits = self.mlp_head(result[:, 0, :])
        logits = nn.Sigmoid()(logits)
        return logits
