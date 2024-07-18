from improved_diffusion.unet import UNetModel
import torch
from torch import nn
import numpy as np



class SAMDModel(UNetModel):
    def __init__(self,, *args, **kwargs):
        super().__init__(,*args, **kwargs)

        self.vivit = ViVit()
    


    def forward(self, x, timesteps, cond=None):
        cond = self.vivit(cond)
        return super().forward(x, timesteps, cond)
    




class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_frames, depth=3, heads=3,
                 dim_head=8, dropout=0., emb_dropout=0.):
        super().__init__()

        assert all(np.mod(image_size , patch_size) == 0), 'Image dimensions must be divisible by the patch size.'
        d = image_size[0] // patch_size[0]
        h = image_size[1] // patch_size[1]
        w = image_size[2] // patch_size[2]
        num_patches = d*h*w
        patch_dim = int(np.prod(patch_size))
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (d p1) (h p2) (w p3) -> b t (d h w) (p1 p2 p3 c)',
                      p1=patch_size[0], p2=patch_size[1], p3=patch_size[2], d=d, h=h, w=w),
            nn.Linear(patch_dim, patch_dim//8)
        )

        temporal_dims = [patch_dim//(8**(i+1))for i in range(depth+1)]
        space_dims = [num_patches*num_frames for i in range(depth+2)]

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches, patch_dim//8))
        self.temporal_transformer = Transformer(temporal_dims, depth, heads, dim_head, dropout, direction="down")
        self.space_transformer = Transformer(space_dims, depth+1, heads, dim_head, dropout, direction="up")

        self.dropout = nn.Dropout(emb_dropout)

        self.conv3d = nn.Sequential(
            nn.Upsample(scale_factor=d*h*w, mode='nearest'),
            Rearrange('b (p1 p2 p3) (d h w c) -> b c (d p1) (h p2) (w p3)',
                      p1=patch_size[0], p2=patch_size[1], p3=patch_size[2], d=d, h=h, w=w),
            nn.Conv3d(in_channels=space_dims[-1], out_channels=1, kernel_size=3, padding="same")
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, d = x.shape
        x += self.pos_embedding
        x = self.dropout(x)
        x = rearrange(x, 'b t n d -> b (t n) d')
        x = self.temporal_transformer(x)
        x = rearrange(x, 'b (t n) d -> b d (t n)', b=b, t=t, n=n)
        x = self.space_transformer(x)
        return self.conv3d(x)
    

class Transformer(nn.Module):
    def __init__(self, dims, depth, heads, dim_head, dropout=0., direction='down'):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dims[-1])

        if direction == 'down':
            code_func = FeedForward
        elif direction == 'up':
            code_func = Upsample

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dims[i], Attention(dims[i], heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dims[i], FeedForward(dims[i], dims[i], dropout=dropout)),
                code_func(dims[i], dims[i+1], dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff1, ff2 in self.layers:
            x = attn(x) + x
            x = ff1(x) + x
            x = ff2(x)
        return self.norm(x)



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Upsample(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='nearest'),
            Rearrange('b t (u d) -> b (u t) d', d=dim, u=8)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out