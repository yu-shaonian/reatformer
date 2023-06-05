import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
from models.module import *


class PatchEmbedding(nn.Module):
    def __init__(self, channel, height, width, patch_size1,patch_size2, dim):
        super().__init__()
        # batch, channel, image_width, image_height = x.shape
        # assert image_width % patch_size == 0 and image_height % patch_size == 0, "Image size must be divided by the path size!"
        self.patch_size1 = patch_size1
        self.patch_size2 = patch_size2

        num_patches = (height * width) // (patch_size1 * patch_size2)
        patch_dim = channel * patch_size1 * patch_size2

        self.to_patch_embedding = nn.Linear(patch_dim, dim, bias=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

    def forward(self, x):
        b, c, w, h = x.shape
        x = rearrange(x, 'b c (w p1) (h p2) -> b (w h) (p1 p2 c)', p1=self.patch_size1,
                      p2=self.patch_size2)  # 1, 3, 224, 224 = 1 ,3 ,(14, 16),(14, 16) => 1, 196, 768
        x1 = self.to_patch_embedding(x)

        cls_token = repeat(self.cls_token, '() n d -> b n d', b=b)
        x2 = torch.cat([cls_token, x1], dim=1)
        x2 = x2 + self.pos_embedding

        return x2


class Attention(nn.Module):
    def __init__(self, dim):

        super().__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)


    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_score = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.dim)
        attention_score = nn.Softmax(dim=1)(attention_score)
        out = torch.bmm(attention_score, V)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=12, dropout=0.1, project_out=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(
        nn.Linear(dim, dim),
        nn.Dropout(dropout)
           )if project_out else nn.Identity()


    def forward(self, x):

        Q = rearrange(self.query(x), 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(self.key(x), 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(self.value(x), 'b n (h d) -> b h n d', h=self.num_heads)
        attention_score = torch.einsum('b h q d, b h k d -> b h q k', Q, K)
        attention_score = nn.Softmax(dim=1)(attention_score) / math.sqrt(self.dim)
        out = torch.einsum('b h a n, b h n v -> b h a v', attention_score, V)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, dim=768, expansion=4, dropout=0.1):

        super().__init__(
        nn.Linear(dim, expansion * dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(expansion * dim, dim),
        nn.Dropout(dropout)
        )

class FeedForwardBlock2(nn.Sequential):
    def __init__(self, dim=768, expansion=4, dropout=0.1):

        super().__init__(
        nn.Linear(dim, 50),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(expansion * dim, 10),
        nn.Dropout(dropout)
        )

class MvsformerEncoder(nn.Module):
    def __init__(self, dim=100,num_heads=10):

        super().__init__()
        self.LN = nn.LayerNorm(dim)
        self.MHA = MultiHeadAttention(dim=dim,num_heads=num_heads)
        self.FFB = FeedForwardBlock(dim=dim)
        self.PatchEmbedding = PatchEmbedding(channel=192*3, height=640,width=512, patch_size1=16,patch_size2=16,dim=100)

    def forward(self, x):

        x = self.PatchEmbedding(x)
        x1 = self.LN(x)
        x2 = self.MHA(x1)
        x2 = x2 + x
        x3 = self.LN(x2)
        x4 = self.FFB(x3)
        x4 = x4 + x2
        return x4


class MvsformerDecoder(nn.Module):
    def __init__(self, channel=192*3, height=640, width=512, patch_size1=16,patch_size2=16, dim=100):
        super().__init__()
        # batch, channel, image_width, image_height = x.shape
        # assert image_width % patch_size == 0 and image_height % patch_size == 0, "Image size must be divided by the path size!"
        self.patch_size1 = patch_size1
        self.patch_size2 = patch_size2

        self.num_patches = (height * width) // (patch_size1 * patch_size2)
        patch_dim = channel * patch_size1 * patch_size2
        self.embedding_to_patch = nn.Linear(dim, patch_dim, bias=False)
        self.conv1 = nn.Sequential(
                Conv2d(channel, channel, 3, stride=1, padding=1),
                Conv2d(channel , channel// 3, 3, 1, padding=1),
                Conv2d(channel// 3, channel// 3, 3, 1, padding=1))


    def forward(self, x):
        # x = self.embedding_to_patch(x)
        B, P ,E = x.shape
        x = self.embedding_to_patch(x)

        cls_token = x[:, -1, :]
        x_patch = x[:, :(P -1), :]

        x = x_patch + cls_token
        # [B, P-1,E]
        x = rearrange(x, 'b (w h) (p1 p2 c) -> b c (w p1) (h p2)', w=32, h=40, p1=self.patch_size1,
                      p2=self.patch_size2)  # 1, 3, 224, 224 = 1 ,3 ,(14, 16),(14, 16) => 1, 196, 768
        x = self.conv1(x)

        return x

if __name__ == '__main__':
    FFB = FeedForwardBlock2(dim=2000)
    import time
    start = time.time()
    device = torch.device('cuda:0')
    input6 = torch.rand(1,192*3,512,640).to(device)
    # input4 = torch.rand(1,4,448,448).to(device)


    # patch_embedding6 = PatchEmbedding(input6,patch_size1=7,patch_size2=7,dim=100).to(device)
    transformer6 = MvsformerEncoder().to(device)
    transformer7 = MvsformerDecoder().to(device)
    out = transformer6(input6)
    out = transformer7(out)
    print("test")

