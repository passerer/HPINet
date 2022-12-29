import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def patch_divide(x, step, ps):
    """Crop image into patches.
    Args:
        x (Tensor): Input feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        crop_x (Tensor): Cropped patches.
        nh (int): Number of patches along the horizontal direction.
        nw (int): Number of patches along the vertical direction.
    """
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        nh += 1
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    crop_x = torch.stack(crop_x, dim=0)  # (n, b, c, ps, ps)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()  # (b, n, c, ps, ps)
    return crop_x, nh, nw


def patch_reverse(crop_x, x, step, ps):
    """Reverse patches into image.
    Args:
        crop_x (Tensor): Cropped patches.
        x (Tensor): Feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        ouput (Tensor): Reversed image.
    """
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2
    return output


class PreNorm(nn.Module):
    """Normalization layer.
    Args:
        dim (int): Base channels.
        fn (Module): Module after normalization.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FFN(nn.Module):
    """Feed Forward Network.
    Args:
        dim (int): Base channels.
        hidden_dim (int): Channels of hidden mlp.
    """

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

        self._init_weights()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

    def _init_weights(self):
        pass


class Attention(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
    """

    def __init__(self, dim, heads, qk_dim):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

    def forward(self, x, y=None):
        """Forward function.
        If 'y' is None, it performs self-attention; Otherwise it performs cross-attention.
        Args:
            x (Tensor): Input feature.
            y (Tensor): Support feature.
        Returns:
            out(Tensor): Output feature.
        """
        if y is None:
            q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        else:
            q, k, v = self.to_q(x), self.to_k(y), self.to_v(y)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out


class Block(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        num (int): Number of blocks.
        qk_dim (int): Channels of query and key in Attention.
        mlp_dim (int): Channels of hidden mlp in FFN.
        heads (int): Head numbers of Attention.
    """

    def __init__(self, dim, num, qk_dim, mlp_dim, heads=1):
        super(Block, self).__init__()
        self.num = num
        self.layers = nn.ModuleList([])
        for _ in range(num):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, qk_dim)),
                PreNorm(dim, FFN(dim, mlp_dim))
            ]))

    def forward(self, x, y=None):
        b, n, c, h, w = x.size()
        x = rearrange(x, 'b n c h w -> (b n) (h w) c')
        if y is not None:
            y = rearrange(y, 'b n c h w -> (b n) (h w) c')
        for i in range(self.num):
            attn, ff = self.layers[i]
            if i > 0:
                y = None
            x = attn(x, y=y) + x
            x = ff(x) + x
        x = rearrange(x, '(b n) (h w) c  -> b n c h w', n=n, w=w)
        return x


class Match(nn.Module):
    """Match module.
    Find the most correlated patch for each patch.
    Args:
        dim (int): Base channels.
    """

    def __init__(self, dim):
        super(Match, self).__init__()
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, n, c, h, w = x.size()
        avg_fea = x.mean(dim=(-2, -1))  # (b, n, c)
        q = self.norm(avg_fea)
        attn = torch.matmul(q, q.transpose(-1, -2))  # (b, n, n)
        attn = attn * self.scale
        # self exclusion
        attn = attn - 100 * torch.eye(n).unsqueeze(0).to(attn.device)
        if self.training:
            hard_attn = F.gumbel_softmax(attn, tau=1., hard=True, dim=-1)
            v = x.view(b, n, -1)
            y = torch.matmul(hard_attn, v)
            y = y.view(b, n, c, h, w)
        else:
            _, indices = torch.max(attn, dim=-1)  # (b, n)
            indices = indices.flatten()  # (b*n,)
            v = x.flatten(0, 1)
            y = v[indices]
            y = y.view(b, n, c, h, w)
        return y


class HPINet(nn.Module):
    """HPINet model for SISR.
    Paper:
        From Coarse to Fine: Hierarchical Pixel Integration for Lightweight Image Super-Resolution,
        AAAI, 2023
    Args:
        model_type(str): Support 'S(mall)', 'M(edium)', and 'L(arge)'.
        upscale(int): Upscale factor.
    """

    # Hyperparameter to build different kinds of model.
    # dim: Base channels of the network.
    # block_num: Block numbers of the net work.
    # heads: Head numbers of Attention.
    # qk_dim: Channels of query and key in Attention
    # mlp_dim: Channels of hidden mlp in FFN
    # patch_size: Patch size.
    model_settings = {
        'M': dict(dim=56, block_num=8, qk_dim=32, mlp_dim=100,
                  patch_size=[12, 16, 20, 24, 12, 16, 20, 24]),
        'S': dict(dim=40, block_num=8, qk_dim=24, mlp_dim=72,
                  patch_size=[12, 16, 20, 24, 12, 16, 20, 24]),
        'L': dict(dim=64, block_num=10, qk_dim=36, mlp_dim=128,
                  patch_size=[12, 14, 16, 20, 24, 12, 14, 16, 20, 24]),
    }

    def __init__(self, model_type: str, upscale: int):
        super(HPINet, self).__init__()
        model_type = model_type.upper()
        if model_type not in self.model_settings:
            raise KeyError('Undefined model type: {}'.format(model_type))
        self.model_type = model_type
        setting = self.model_settings[model_type]

        self.dim = setting['dim']
        self.block_num = setting['block_num']
        self.patch_size = setting['patch_size']
        self.qk_dim = setting['qk_dim']
        self.mlp_dim = setting['mlp_dim']
        self.upscale = upscale

        self.first_conv = nn.Conv2d(3, self.dim, 3, 1, 1)
        self.cross_match = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        for _ in range(self.block_num):
            self.cross_match.append(Match(self.dim))
            self.blocks.append(Block(dim=self.dim, num=3, qk_dim=self.qk_dim, mlp_dim=self.mlp_dim))
            self.mid_convs.append(nn.Conv2d(self.dim, self.dim, 3, 1, 1))

        if upscale == 4:
            self.upconv1 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif upscale == 2 or upscale == 3:
            self.upconv = nn.Conv2d(self.dim, self.dim * (upscale ** 2), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale)
        else:
            raise NotImplementedError(
                'Upscale factor is expected to be one of (2, 3, 4), but got {}'.format(upscale))
        self.last_conv = nn.Conv2d(self.dim, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, target=None):
        """Forward function.
        In traning mode, 'target' should be provided for loss calculation.
        Args:
            x (Tensor): Input image.
            target (Tensor): GT image.
        """
        b, _, h, w = x.size()
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        x = self.first_conv(x)
        for i in range(self.block_num):
            ps = self.patch_size[i]
            step = ps - 2
            crop_x, nh, nw = patch_divide(x, step, ps)  # (b, n, c, ps, ps)
            y = self.cross_match[i](crop_x)
            crop_x = self.blocks[i](crop_x, y)
            residual = patch_reverse(crop_x, x, step, ps)
            x = x + self.mid_convs[i](residual)
        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        else:
            out = self.lrelu(self.pixel_shuffle(self.upconv(x)))
        out = base + self.last_conv(out)
        out = out[..., :h * self.upscale, :w * self.upscale]
        if self.training:
            loss = F.l1_loss(out, target)
            return loss
        else:
            return out

    def __repr__(self):
        num_parameters = sum(map(lambda x: x.numel(), self.parameters()))
        return '#Params of {}-{}: {:<.4f} [K]'.format(self._get_name(), self.model_type,
                                                      num_parameters / 10 ** 3)
