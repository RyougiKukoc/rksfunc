from vapoursynth import core, VideoNode
from functools import partial
from ._resample import *


def scanny(c16: VideoNode) -> VideoNode:
    """
    Create a simple TCanny deband mask.
    :param c16: input YUV420P16 clip.
    :return: result of mask.
    """
    from vapoursynth import YUV420P16
    
    c8 = c16.fmtc.bitdepth(bits=8)
    TC = core.tcanny.TCanny
    masks = TC(c8, 0.8, op=2, mode=1, planes=[0, 1, 2]) \
        .std.Expr(["x 7 < 0 65535 ?", "x 256 *"], YUV420P16)
    maskb = TC(c8, 1.3, t_h=6.5, op=2, planes=0)
    maskg = TC(c8, 1.1, t_h=5.0, op=2, planes=0)
    mask = core.std.Expr([maskg, maskb, masks, c8], \
        ["a 20 < 65535 a 48 < x 256 * a 96 < y 256 * z ? ? ?", "z"], YUV420P16)
    return mask.std.Maximum(0).std.Maximum(0).std.Minimum(0).rgvs.RemoveGrain([20, 0])


insane_deband_mask = scanny

rscaamask = partial(rescaley, maskmode=1)
