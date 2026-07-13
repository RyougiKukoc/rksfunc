from vapoursynth import core, VideoNode
from typing import Union, Tuple, Callable, Literal


def F3kdbCompat(
    clip: VideoNode, 
    range: int | None = 15, 
    y: int | None = 64, 
    cb: int | None = 64, 
    cr: int | None = 64, 
    grainy: int | None = 0, 
    grainc: int | None = 0, 
    sample_mode: int | None = 2, 
    seed: int | None = 0, 
    blur_first: int | None = True, 
    dynamic_grain: int | None = False,
    keep_tv_range: int | None = False, 
    random_algo_ref: int | None = 1, 
    random_algo_grain: int | None = 1, 
    random_param_ref: float | None = 1.0, 
    random_param_grain: float | None = 1.0, 
    y_1: int | None = 64, 
    cb_1: int | None = 64, 
    cr_1: int | None = 64, 
    y_2: int | None = 64, 
    cb_2: int | None = 64, 
    cr_2: int | None = 64, 
    scale: bool | None = False,
    angle_boost: float | None = 1.5,
    max_angle: float | None = 0.15,
    **kwargs
) -> VideoNode:
    peak = float((1 << 16) - 1) if scale else float((1 << 14) - 1)
    cvt = lambda x: None if x is None else (x * 255 / peak)
    [y, cb, cr, y_1, cb_1, cr_1, y_2, cb_2, cr_2] = map(cvt, [y, cb, cr, y_1, cb_1, cr_1, y_2, cb_2, cr_2])
    return clip.vszip.Deband(
        range=range, thr=[y, cb, cr], grain=[grainy, grainc],
        sample_mode=sample_mode, seed=seed, blur_first=blur_first, 
        dynamic_grain=dynamic_grain, keep_tv_range=keep_tv_range, 
        random_algo_ref=random_algo_ref, random_algo_grain=random_algo_grain,
        random_param_ref=random_param_ref, random_param_grain=random_param_grain,
        thr1=[y_1, cb_1, cr_1], thr2=[y_2, cb_2, cr_2], 
        angle_boost=angle_boost, max_angle=max_angle,
    )


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
        kill = cyuv16.zsmooth.RemoveGrain([20, 11]).zsmooth.RemoveGrain([20, 11])
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
    f3k1 = F3kdbCompat(kill, r1, y1, uv1, uv1, **f3kdb_params)
    f3k2 = F3kdbCompat(f3k1, r2, y2, uv2, uv2, **f3kdb_params)
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


def SynDebandV2(
    clip: VideoNode,
    preset: Literal['low', 'high', 'mid'] = 'low',
    banding_mask: VideoNode = None,
    debander: Callable[[VideoNode], VideoNode] = None,
    killer: Callable[[VideoNode], VideoNode] = None,
    ampo: float = 1.0,
) -> Union[VideoNode, Tuple[VideoNode, VideoNode]]:
    if banding_mask is None:
        from vsdeband import deband_detail_mask
        banding_mask = deband_detail_mask(clip).std.Maximum().std.Maximum().std.Deflate()
    kill = clip if killer is None else killer(clip)
    if debander is None:
        _d = core.vszip.Deband
        match preset:
            case 'low':
                deband = _d(kill, range=12, thr=0.6, grain=0, sample_mode=7, thr1=1.9, thr2=1.2, angle_boost=1.9)
                deband = _d(deband, range=22, thr=0.5, grain=0, sample_mode=7, thr1=1.7, thr2=1.1, angle_boost=1.8)
            case 'mid':
                deband = _d(kill, range=12, thr=1.8, grain=0, sample_mode=7, thr1=4.0, thr2=2.0, angle_boost=1.6)
                deband = _d(deband, range=22, thr=1.6, grain=0, sample_mode=7, thr1=3.6, thr2=1.8, angle_boost=1.5)
            case 'high':
                deband = _d(kill, range=12, thr=3.4, grain=0, sample_mode=6, thr1=6.8, thr2=3.3)
                deband = _d(deband, range=12, thr=3.2, grain=0, sample_mode=6, thr1=6.4, thr2=3.1)
            case _:
                raise ValueError(f"{preset = } is not available.")
    else:
        deband = debander(kill)
    deband = core.std.MaskedMerge(deband, kill, banding_mask)
    if killer is not None:
        return deband.std.MergeFullDiff(clip.std.MakeFullDiff(kill))
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
    dering_2 = core.zsmooth.Repair(dering_1, contra_1, 23)
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
    
    kill = clip.zsmooth.RemoveGrain(20).zsmooth.RemoveGrain(20)
    noise1 = core.std.MakeDiff(clip, kill, planes=[0, 1, 2])
    noise2 = noise1.zsmooth.RemoveGrain(20)
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
    deband = _vszip_f3kdb(kill, 12, 80, 80, 80, 0, 0)
    deband = _vszip_f3kdb(deband, 24, 60, 60, 60, 0, 0)
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
    aa = core.zsmooth.Repair(aa, cyuv, rpmode)
    if nocmask:
        return aa
    if cmask is None:
        nw, nh = cyuv.width * 3 // 4, cyuv.height * 3 // 4
        cmask = RescaleLuma(cyuv, nw, nh, 'bicubic', 1, 0, linemode=False, maskmode=1)
    aa = core.std.MaskedMerge(aa, cyuv, cmask)
    return aa


def janaitrt(clip: VideoNode, model=None, backend=None) -> VideoNode:
    from vapoursynth import RGB, GRAYS
    from vsmlrt import RealESRGAN, RealESRGANModel, BackendV2
    from ._resample import torgbs, uvsr
    
    if clip.format.name.startswith("RGB"):
        crgbs = clip.fmtc.bitdepth(bits=32)
    elif clip.format.name.startswith("YUV444"):
        crgbs = torgbs(clip)
    elif clip.format.name.startswith("YUV420"):
        crgbs = torgbs(uvsr(clip))
    elif clip.format.name.startswith("Gray"):
        crgbs = core.std.ShufflePlanes(clip.fmtc.bitdepth(bits=32), [0]*3, RGB).std.SetFrameProps(_Matrix=0)
    
    j2x = RealESRGAN(
        clip=crgbs,
        model=RealESRGANModel.animejanaiV3_HD_L1 if model is None else model,
        backend=BackendV2.TRT(fp16=True) if backend is None else backend,
    )
    j1x = j2x.resize.Spline36(width=crgbs.width, height=crgbs.height, range_in=0, range=0)
    
    if clip.format.name.startswith("RGB"):
        return j1x.fmtc.bitdepth(bits=clip.format.bits_per_sample, dmode=0)
    elif clip.format.name.startswith("YUV"):
        return j1x.resize.Spline36(format=clip.format.id, matrix=1, dither_type='ordered')
    else:
        r, g, b = core.std.SplitPlanes(j1x)
        y = core.std.Expr([r, g, b], "x y z + + 3 /", GRAYS)
        return y.fmtc.bitdepth(bits=clip.format.bits_per_sample, dmode=0)
    