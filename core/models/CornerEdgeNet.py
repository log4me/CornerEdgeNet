import torch
import torch.nn as nn

from .py_utils import TopPool, BottomPool, LeftPool, RightPool

from .py_utils.utils import convolution, residual, corner_pool
from .py_utils.losses import CornerNet_Loss, CornerEdgeNet_Loss
from .py_utils.modules import hg_module, hg


def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=20):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_clses = (topk_inds / (height * width)).int()
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    # mask = torch.tensor(batch, cat * height * width).scatter_(-1, topk_inds, topk_scores) > 0
    # mask = mask.reshape(batch, cat, height, width)
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def _gather_feat(feat, ind):
    dim = feat.size(2)


# TODO. _decode func
def _decode(tl_props, tl_offsets, tl_edges, K=100, kernel=1, score_threshold=1, num_dets=1000):
    batch, cat, height, width = tl_props.size()
    tl_props = torch.sigmoid(tl_props)

    tl_props = _nms(tl_props, kernel=kernel)
    # prop is of shape (N, C, H, W) , offset and edges is of shape (N, C, 2, H, W)
    tl_offsets = tl_offsets.permute(0, 1, 3, 4, 2)
    tl_edges = tl_edges.permute(0, 1, 3, 4, 2)
    topk_scores, topk_inds = tl_props.reshape(batch, -1).topk(-1, K)
    topk_clses = (topk_inds / (height * width)).int()
    topk_ys = ((topk_inds % (height * width)) / width).int().float()
    topk_xs = (topk_inds % (height * width) % width).int().float()

    double_indx = topk_inds.unsqueeze(-1).expand(batch, K, 2)
    topk_offsets = tl_offsets.reshape(batch, -1, 2).gather(1, double_indx)
    topk_edges = tl_edges.reshape(batch, -1, 2).gather(1, double_indx)

    x0 = topk_xs + topk_offsets[..., 0]
    y0 = topk_ys + topk_offsets[..., 1]
    x1 = x0 + tl_edges[..., 0]
    y1 = x1 + tl_edges[..., 1]

    bboxes = torch.stack([x0, y0, x1, y1, topk_scores, topk_clses])
    return bboxes



def make_pool_layer(dim):
    return nn.Sequential()

def make_hg_layer(inp_dim, out_dim, modules):
    layers  = [residual(inp_dim, out_dim, stride=2)]
    layers += [residual(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)

# This is the network of the detection network, which controls the arthecture of the
class hg_net_edge(nn.Module):
    def __init__(
        self, hg, tl_modules, tl_props, tl_offsets, tl_whs,
    ):
        super(hg_net_edge, self).__init__()

        self._decode = _decode

        self.hg = hg

        # corner pool
        self.tl_modules = tl_modules

        # class prop
        self.tl_props = tl_props

        # offsets
        self.tl_offsets = tl_offsets

        # width and height
        self.tl_width_height = tl_whs


    def _train(self, *xs):
        image = xs[0]
        cnvs  = self.hg(image)

        tl_modules = [tl_mod_(cnv) for tl_mod_, cnv in zip(self.tl_modules, cnvs)]

        tl_props   = [tl_props_(tl_mod) for tl_props_, tl_mod in zip(self.tl_props, tl_modules)]

        tl_offsets   = [tl_offsets_(tl_mod)  for tl_offsets_,  tl_mod in zip(self.tl_offsets,  tl_modules)]

        tl_width_height    = [tl_width_height_(tl_mod)  for tl_width_height_,  tl_mod in zip(self.tl_offs,  tl_modules)]
        return [tl_modules, tl_props, tl_offsets, tl_width_height]

    def _test(self, *xs, **kwargs):
        image = xs[0]
        cnvs  = self.hg(image)
        # TODO. understand why only use the last layer calculate the outputs.

        tl_mod = self.tl_modules[-1](cnvs[-1])
        tl_props = self.tl_props[-1](tl_mod)
        N, C, H, W = tl_props.shape
        tl_offsets = self.tl_offsets[-1](tl_mod)
        tl_offsets = tl_offsets.reshape(N, C, -1, H, W)
        tl_width_height = self.tl_width_height[-1](tl_mod)
        tl_width_height = tl_width_height.reshape(N, C, -1, H, W)

        outs = [tl_props, tl_offsets, tl_width_height]
        return self._decode(*outs, **kwargs), tl_props, tl_offsets, tl_width_height

    def forward(self, *xs, test=False, **kwargs):
        if not test:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)


class model(hg_net_edge):
    def _pred_mod(self, dim):
        return nn.Sequential(
            convolution(3, 256, 256, with_bn=False),
            nn.Conv2d(256, dim, (1, 1))
        )

    def _merge_mod(self):
        return nn.Sequential(
            nn.Conv2d(256, 256, (1, 1), bias=False),
            nn.BatchNorm2d(256)
        )

    def __init__(self):
        stacks  = 2
        pre     = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(128, 256, stride=2)
        )
        hg_mods = nn.ModuleList([
            hg_module(
                5, [256, 256, 384, 384, 384, 512], [2, 2, 2, 2, 2, 4],
                make_pool_layer=make_pool_layer,
                make_hg_layer=make_hg_layer
            ) for _ in range(stacks)
        ])
        cnvs    = nn.ModuleList([convolution(3, 256, 256) for _ in range(stacks)])
        inters  = nn.ModuleList([residual(256, 256) for _ in range(stacks - 1)])
        cnvs_   = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])
        inters_ = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])

        hgs = hg(pre, hg_mods, cnvs, inters, cnvs_, inters_)

        # corner pool
        tl_modules = nn.ModuleList([corner_pool(256, TopPool, LeftPool) for _ in range(stacks)])


        tl_props = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])
        tl_offsets = nn.ModuleList([self._pred_mod(80 * 2) for _ in range(stacks)])
        tl_whs = nn.ModuleList([self._pred_mod(80 * 2) for _ in range(stacks)])

        # TODO. init of added modules
        super(model, self).__init__(
            hgs, tl_modules, tl_props, tl_offsets, tl_whs
        )

        self.loss = CornerEdgeNet_Loss(pull_weight=1e-1, push_weight=1e-1)
