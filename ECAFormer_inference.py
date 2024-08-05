import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fvcore.nn import FlopCountAnalysis
class ShallowDeepConv(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=3, padding=1, bias=True, groups=n_fea_in)
        )

    def forward(self, img):
        input = torch.cat([img, img.mean(dim=1).unsqueeze(1)], dim=1)
        x_1 = self.conv1(input)
        visual_feats = self.depth_conv(x_1)
        semantic_feats = self.conv2(visual_feats)
        return visual_feats, semantic_feats


class DMSA(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.fusion_x = nn.Sequential(
            nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1,stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1,stride=1)
        )
        self.fusion_y = nn.Sequential(
            nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1,stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels= dim, out_channels=dim, kernel_size=1,stride=1)
        )
        self.to_q = nn.Sequential(
            nn.Linear(dim, dim_head * heads, bias=True),
            nn.LeakyReLU(),
            nn.Linear(dim_head * heads, dim_head * heads, bias=False)
        )
        self.to_k = nn.Sequential(
            nn.Linear(dim, dim_head * heads, bias=True),
            nn.LeakyReLU(),
            nn.Linear(dim_head * heads, dim_head * heads, bias=False)
        )
        self.to_v = nn.Sequential(
            nn.Linear(dim, dim_head * heads, bias=True),
            nn.LeakyReLU(),
            nn.Linear(dim_head * heads, dim_head * heads, bias=False)
        )
        self.rescale_x = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescale_y = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj_x = nn.Linear(dim_head * heads, dim, bias=True)
        self.proj_y = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, y_in):
        b, h, w, c = x_in.shape
        fusion_k_x = self.fusion_x(torch.cat([x_in, y_in], dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(b,h * w,c)
        fusion_k_y = self.fusion_y(torch.cat([x_in, y_in], dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(b,h * w,c)
        x = x_in.reshape(b, h * w, c)
        y = y_in.reshape(b, h * w, c)
        q_inp_x = self.to_q(x)
        k_inp_x = self.to_k(fusion_k_x)
        v_inp_x = self.to_v(x)
        q_inp_y = self.to_q(y)
        k_inp_y = self.to_k(fusion_k_y)
        v_inp_y = self.to_v(y)
        q_x, k_x, v_x = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                            (q_inp_x, k_inp_x, v_inp_x,))
        q_y, k_y, v_y = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                            (q_inp_y, k_inp_y, v_inp_y,))
        q_x = q_x.transpose(-2, -1)
        k_x = k_x.transpose(-2, -1)
        v_x = v_x.transpose(-2, -1)
        q_y = q_y.transpose(-2, -1)
        k_y = k_y.transpose(-2, -1)
        v_y = v_y.transpose(-2, -1)
        q_x = F.normalize(q_x, dim=-1, p=2)
        k_x = F.normalize(k_x, dim=-1, p=2)
        q_y = F.normalize(q_y, dim=-1, p=2)
        k_y = F.normalize(k_y, dim=-1, p=2)

        attn_x = (k_y @ q_x.transpose(-2, -1))
        attn_x = attn_x * self.rescale_x
        attn_x = attn_x.softmax(dim=-1)
        x = attn_x @ v_x
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c_x = self.proj_x(x).view(b, h, w, c)
        out_p_x = self.pos_emb(v_inp_x.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out_x = out_c_x + out_p_x

        attn_y = (k_x @ q_y.transpose(-2, -1))
        attn_y = attn_y * self.rescale_y
        attn_y = attn_y.softmax(dim=-1)
        y = attn_y @ v_y
        y = y.permute(0, 3, 1, 2)
        y = y.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c_y = self.proj_y(y).view(b, h, w, c)
        out_p_y = self.pos_emb(v_inp_y.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out_y = out_c_y + out_p_y
        return out_x, out_y

class FeedForward(nn.Module):
    def __init__(self, dim, expand=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * expand, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * expand, dim * expand, 3, 1, 1,
                      bias=False, groups=dim * expand),
            GELU(),
            nn.Conv2d(dim * expand, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class DMSABlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                DMSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, y):
        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x_hat, y_hat = attn(x, y)
            x = x_hat + x
            x = ff(x) + x
        out_x = x.permute(0, 3, 1, 2)
        out_y = y_hat.permute(0, 3, 1, 2)
        return out_x, out_y


class CrossAttenUnet(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(CrossAttenUnet, self).__init__()
        self.dim = dim
        self.level = level
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                DMSABlock(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2
        self.bottleneck = DMSABlock(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                DMSABlock(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2
        self.mapping = nn.Conv2d(self.dim * 2, out_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, y):
        x_res = x
        x = self.embedding(x)
        fea_xlist = []
        fea_ylist = []
        for (DMSA, FeaDownSample0, FeaDownsample1) in self.encoder_layers:
            x, y = DMSA(x, y)
            fea_ylist.append(y)
            fea_xlist.append(x)
            x = FeaDownSample0(x)
            y = FeaDownsample1(y)
        x, y = self.bottleneck(x, y)
        for i, (FeaUpSample0, FeaUpSample1, FeaFusion0, FeaFusion1, DMSA) in enumerate(self.decoder_layers):
            x = FeaUpSample0(x)
            y = FeaUpSample1(y)
            x = FeaFusion0(torch.cat([x, fea_xlist[self.level - 1 - i]], dim=1))
            y = FeaFusion1(torch.cat([fea_ylist[self.level - 1 - i], y], dim=1))
            x, y = DMSA(x, y)
        out = self.mapping(torch.cat([x, y], dim=1))+x_res
        return out


class ECAFormer0(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, level=2, num_blocks=[1, 2, 2]):
        super().__init__()
        self.ShallowDeepConv = ShallowDeepConv(n_feat)
        self.CrossAttUnet = CrossAttenUnet(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level,
                                       num_blocks=num_blocks)

    def forward(self, img):
        visual_feat, semantic_feat = self.ShallowDeepConv(img)
        semantic_feat = img * semantic_feat + img
        output_img = self.CrossAttUnet(semantic_feat,visual_feat)

        return output_img

class ECAFormer(nn.Module):
    def __init__(self,in_channels=3, out_channels=3, n_feat=40, level=2, num_blocks=[1, 2, 2],stage=1):
        super().__init__()
        modules=[]
        for i in range(stage):
            modules.append(ECAFormer0(in_channels, out_channels, n_feat, level,num_blocks))
        self.body = nn.Sequential(*modules)
    def forward(self, img):
        for net in self.body:
            img = net(img)
        return img


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ECAFormer(stage=1,n_feat=40, num_blocks=[1, 2, 2]).to(device)
    inputs = torch.randn((1, 3, 256, 256)).to(device)
    flops = FlopCountAnalysis(model, inputs)
    n_param = sum([p.nelement() for p in model.parameters()])
    print(f'GMac:{flops.total() / (1024 * 1024 * 1024)}')
    print(f'Params:{n_param}')
    model.load_state_dict(torch.load("./LOL-v1.pth")['params'])