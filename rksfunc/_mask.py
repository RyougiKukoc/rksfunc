from vapoursynth import core, VideoNode
from functools import partial
from typing import Callable
from ._resample import *


def scanny(clip: VideoNode) -> VideoNode:
    """
    Create a simple TCanny deband mask. Stolen from LoliHouse.
    :param clip: input clip.
    :return: result of mask.
    """
    from vapoursynth import GRAY16
    
    c8 = yer(clip.fmtc.bitdepth(bits=8))
    TC = core.tcanny.TCanny
    masks = TC(c8, 0.8, op=2, mode=1).std.Expr("x 7 < 0 65535 ?", GRAY16)
    maskb = TC(c8, 1.3, t_h=6.5, op=2, planes=0)
    maskg = TC(c8, 1.1, t_h=5.0, op=2, planes=0)
    mask = core.std.Expr([maskg, maskb, masks, c8], "a 20 < 65535 a 48 < x 256 * a 96 < y 256 * z ? ? ?", GRAY16)
    return mask.std.Maximum(0).std.Maximum(0).std.Minimum(0).rgvs.RemoveGrain(20)


insane_deband_mask = scanny

rscaamask = partial(rescaley, maskmode=1)


def gamma_mask(
    clip: VideoNode, 
    gamma: float = .7, 
    mask_method: Callable = None,
    dtcargs: dict = {}, 
    btcargs: dict = {}, 
) -> VideoNode:
    TC = core.tcanny.TCanny
    y = yer(clip)
    g = gammarize(y, gamma)
    if mask_method is not None:
        _d_mask = mask_method(g)
        _b_mask = mask_method(y)
    else:
        dargs = dict(sigma=2, sigma_v=2, t_h=4, op=2)
        dargs.update(dtcargs)
        bargs = dict(sigma=2, sigma_v=2, t_h=3, op=2)
        bargs.update(btcargs)
        _d_mask = TC(g, **dargs)
        _b_mask = TC(y, **bargs)
    return core.std.Expr([_b_mask, _d_mask], 'x y max').std.Maximum().std.Maximum() \
        .std.Minimum().std.Inflate().std.Inflate()
