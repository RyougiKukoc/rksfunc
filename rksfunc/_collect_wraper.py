from vapoursynth import core, VideoNode
from typing import Union, Tuple


def SynDeband(
    cyuv16: VideoNode, 
    r1: int = 12, 
    y1: int = 64, 
    uv1: int = 48, 
    r2: int = 21,
    y2: int = 48, 
    uv2: int = 32, 
    mstr: int = 6000, 
    inflate: int = 2,
    include_mask: bool = False, 
    kill: VideoNode = None, 
    bmask: VideoNode = None,
    limit: bool = False,
    limit_thry: float = 0.6,
    limit_thrc: float = 0.5,
    limit_elast: float = 1.2,
) -> Union[VideoNode, Tuple[VideoNode, VideoNode]]:
    if kill is None:
        kill = cyuv16.rgvs.RemoveGrain([20, 11]).rgvs.RemoveGrain([20, 11])
    elif not kill:
        kill = cyuv16
    grain = core.std.MakeDiff(cyuv16, kill)
    f3kdb_params = {
        'grainy': 0,
        'grainc': 0,
        'sample_mode': 2,
        'blur_first': True,
        'dither_algo': 2,
    }
    f3k1 = kill.neo_f3kdb.Deband(r1, y1, uv1, uv1, **f3kdb_params)
    f3k2 = f3k1.neo_f3kdb.Deband(r2, y2, uv2, uv2, **f3kdb_params)
    if limit:
        from mvsfunc import LimitFilter
        f3k2 = LimitFilter(f3k2, kill, thr=limit_thry, thrc=limit_thrc, elast=limit_elast)
    if bmask is None:
        from kagefunc import retinex_edgemask as rtx
        bmask = rtx(kill).std.Binarize(mstr)
        for _ in range(inflate):
            bmask = bmask.std.Inflate()
    deband = core.std.MaskedMerge(f3k2, kill, bmask)
    deband = core.std.MergeDiff(deband, grain)
    if include_mask:
        return deband, bmask
    else:
        return deband


def USMDering(cyuv16: VideoNode, mat=None, mrad=3, mthr=50,
               include_mask=False) -> Union[VideoNode, Tuple[VideoNode, VideoNode]]:
    """
    Unsharp mask based dering method. Thanks to Jan: https://skyeysnow.com/forum.php?mod=viewthread&tid=32112.
    :param cyuv16: YUVP16 clip input.
    :param mat: matrix used to blur.
    :param mrad: param of havsfunc.HQDeringmod.
    :param mthr: param of havsfunc.HQDeringmod.
    :param include_mask: whether the mask is included to output.
    :return: result of dering.
    """
    from havsfunc import HQDeringmod, ContraSharpening

    if mat is None:
        mat = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    elif mat == 1:
        mat = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    blur = cyuv16.std.Convolution(mat)
    mask = HQDeringmod(cyuv16, show=True, mrad=mrad, mthr=mthr)
    dering_1 = core.std.MaskedMerge(cyuv16, blur, mask)
    contra_1 = ContraSharpening(dering_1, cyuv16, 2)
    dering_2 = core.rgvs.Repair(dering_1, contra_1, 23)
    if include_mask:
        return dering_2, mask
    else:
        return dering_2


def daaJanmod(clip: VideoNode, itr: int = 4, opencl: bool = True) -> VideoNode:
    # copy from Jan
    # Interpolation
    if opencl:
        nnedi3 = core.nnedi3cl.NNEDI3CL
    else:
        nnedi3 = core.znedi3.nnedi3
    nnargs = dict(nsize=4, nns=4, qual=2, pscrn=1)
    def _intra_aa(clip: VideoNode) -> VideoNode:
        nn0 = nnedi3(clip, field=0, dh=False, **nnargs)
        nn1 = nnedi3(clip, field=1, dh=False, **nnargs)
        tr = clip.std.Transpose()
        nn2 = nnedi3(tr, field=0, dh=False, **nnargs).std.Transpose()
        nn3 = nnedi3(tr, field=1, dh=False, **nnargs).std.Transpose()
        return core.akarin.Expr([nn0, nn1, nn2, nn3], 'x y z a sort4 dup1 r1! dup2 r2! drop4 r1@ r2@ + 2 /')

    # Contra-sharpen
    def _cs(flt: VideoNode, src: VideoNode, p: float = 2.35) -> VideoNode:
        from havsfunc import Gauss
        blur = Gauss(flt, p=p)
        sharp = core.std.Expr([flt, blur], 'x 2 * y -')
        return core.std.Expr([sharp, flt, src], 'x y z min max y z max min')

    res = clip
    for _ in range(itr):
        res = _cs(_intra_aa(res), res)
    return res


def AliceDeband(clip: VideoNode) -> VideoNode:
    # copy from Alice
    from mvsfunc import LimitFilter
    from kagefunc import retinex_edgemask as rtx
    from vsutil import iterate
    from ._resample import yer
    
    kill = clip.rgvs.RemoveGrain(20).rgvs.RemoveGrain(20)
    noise1 = core.std.MakeDiff(clip, kill, planes=[0, 1, 2])
    noise2 = noise1.rgvs.RemoveGrain(20)
    noise = LimitFilter(noise2, noise1, thr=0.8, brighten_thr=0.8, elast=1.5)
    noise3 = noise1.knlm.KNLMeansCL(1, 2, 4, 1.25)
    noise = LimitFilter(noise3, noise, thr=0.8, brighten_thr=0.8, elast=1.5)
    maskg = rtx(kill).std.Inflate().std.Maximum().std.Inflate().std.Binarize(6000)
    maskg = iterate(maskg, core.std.Minimum, 2)
    maskd = yer(kill.tcanny.TCanny(0.6, t_h=7, planes=0, op=2))
    maskl = yer(kill.tcanny.TCanny(0.7, t_h=8, planes=0, op=2))
    srcy8 = yer(clip).fmtc.bitdepth(bits=8)
    mask1 = core.std.Expr([maskd, maskl, srcy8], "z 28 < 65535 z 64 < x y ? ?")
    mask1 = iterate(mask1.std.Inflate(), core.std.Maximum, 3)
    mask1 = mask1.std.Inflate().std.Minimum()
    mask2 = core.std.Expr([mask1, maskg], "x y > x y ?")
    deband_mask = core.std.Expr([mask1, mask2, srcy8], "z 96 < x y ?")
    deband = kill.f3kdb.Deband(12, 80, 80, 80, 0, 0, output_depth=16)
    deband = deband.f3kdb.Deband(24, 60, 60, 60, 0, 0, output_depth=16)
    deband = LimitFilter(deband, kill, thr=0.6, brighten_thr=0.6, elast=1.2)
    deband = core.std.MaskedMerge(deband, kill, deband_mask, [0, 1, 2], True)
    deband = core.std.MergeDiff(deband, noise)
    return deband


def TAAWrapper(cyuv: VideoNode, ay: int, auv: int, cmask: VideoNode = None, rpmode: int = 13, 
            taa_args: dict = {}, nocmask: bool = False) -> VideoNode:
    from vsTAAmbk import TAAmbk
    from ._resample import RescaleLuma
    
    preargs = dict(aatype=ay, aatypeu=auv, aatypev=auv, opencl=True)
    preargs.update(taa_args)
    aa = TAAmbk(cyuv, **preargs)
    aa = core.rgvs.Repair(aa, cyuv, rpmode)
    if nocmask:
        return aa
    if cmask is None:
        nw, nh = cyuv.width * 3 // 4, cyuv.height * 3 // 4
        cmask = RescaleLuma(cyuv, nw, nh, 'bicubic', 1, 0, linemode=False, maskmode=1)
    aa = core.std.MaskedMerge(aa, cyuv, cmask)
    return aa
