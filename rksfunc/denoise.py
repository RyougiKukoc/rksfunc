from vapoursynth import core, VideoNode
from .resample import *


def w2xtrt(clip: VideoNode, noise, test=False, o420p16=False, trtargs=dict(), w2b=None) -> VideoNode:
    from vsmlrt import Waifu2x, Waifu2xModel, Backend

    if clip.format.name == "RGBS":
        crgbs = clip
    elif clip.format.name.startswith("YUV444"):
        crgbs = torgbs(clip)
    elif clip.format.name.startswith("YUV420"):
        crgbs = torgbs(uvsr(clip, False))
    
    if test:
        w2x = crgbs.w2xnvk.Waifu2x(noise, 1, 2)
    else:
        w2m = Waifu2xModel.cunet
        if w2b is None:
            w, h = clip.width, clip.height
            preargs = dict(
                max_shapes = (w, h), 
                opt_shapes = (w, h), 
                fp16 = True, 
                workspace = 256, 
                use_cuda_graph = True, 
                num_streams = 2, 
                static_shape = True
            )
            preargs.update(trtargs)
            w2b = Backend.TRT(**preargs)
        w2x = Waifu2x(crgbs, noise, 1, model=w2m, backend=w2b)
    return to420p16(w2x) if o420p16 else w2x


def tempostab(clip: VideoNode) -> VideoNode:
    from vapoursynth import YUV, GRAY, Error
    from havsfunc import SMDegrain, QTGMC
    
    if clip.format.color_family != YUV and clip.format.color_family != GRAY:
        raise Error("rksfunc.tempostab: only YUV and GRAY format are supported.")
    origdep = clip.format.bits_per_sample
    if origdep != 16:
        clip = Depth(clip, 16)
    if clip.format.color_family == YUV:
        clip_y = gety(clip)
    else:
        clip_y = clip
    tref = QTGMC(clip_y, InputType=1, Sharpness=0, SourceMatch=3)
    smd = SMDegrain(clip_y, RefineMotion=True, dct=6, blksize=32, prefilter=tref)
    if clip.format.color_family == YUV:
        smd = mergeuv(smd, clip)
    if origdep != 16:
        smd = Depth(smd, origdep)
    return smd


def medium_vfinal(c420p16: VideoNode, s1=2.5, r1=1, bs1=3, br1=12, pn1=2, pr1=8, \
    s2=2, r2=0, bs2=4, br2=8, pn2=2, pr2=6, ref=None, dftsigma=4, dfttbsize=1, \
    bm3d="bm3dcuda_rtc", fast=True) -> VideoNode:
    from vapoursynth import YUV444PS
    from vsutil import split, join
    from importlib import import_module
    
    if bm3d == "bm3dcuda":
        B2 = core.bm3dcuda.BM3Dv2
    elif bm3d == "bm3dcuda_rtc":
        B2 = core.bm3dcuda_rtc.BM3Dv2
    else:
        B2 = core.bm3dcpu.BM3Dv2
    
    if ref is None:
        from havsfunc import EdgeCleaner
        try:
            dfttest2 = import_module('dfttest2')
            DFTTest = dfttest2.DFTTest
        except ModuleNotFoundError:
            DFTTest = core.dfttest.DFTTest
        ref = DFTTest(c420p16, sigma=dftsigma, tbsize=dfttbsize, planes=[0, 1, 2])
        ref = EdgeCleaner(ref)

    hw = c420p16.width / 2  # half width
    hh = c420p16.height / 2  # half height
    srcy_f, srcu_f, srcv_f = split(Depth(c420p16, 32))
    refy_f, refu_f, refv_f = split(Depth(ref, 32))
    vfinal_y = B2(srcy_f, refy_f, s1, bs1, br1, r1, pn1, pr1, fast=fast)
    vyhalf = vfinal_y.resize.Spline36(hw, hh, src_left=-0.5)
    ryhalf = refy_f.resize.Spline36(hw, hh, src_left=-0.5)
    srchalf_444 = join([vyhalf, srcu_f, srcv_f])
    refhalf_444 = join([ryhalf, refu_f, refv_f])
    srchalf_opp = rgb2opp(torgbs(srchalf_444))
    refhalf_opp = rgb2opp(torgbs(refhalf_444))
    vfinal_half = B2(srchalf_opp, refhalf_opp, s2, bs2, br2, r2, pn2, pr2, fast=fast)
    vfinal_half = opp2rgb(vfinal_half).resize.Spline36(format=YUV444PS, matrix=1)
    _, vfinal_u, vfinal_v = split(vfinal_half)
    vfinal = join([vfinal_y, vfinal_u, vfinal_v])
    return Depth(vfinal, 16)


def light_vfinal(c420p16: VideoNode, s1=1.2, r1=1, bs1=3, br1=12, pn1=2, pr1=8, ds1=0.5, \
    s2=2, r2=0, bs2=4, br2=8, pn2=2, pr2=6, ds2=1, bm3d="bm3dcuda_rtc", fast=True) -> VideoNode:
    from vapoursynth import YUV444PS
    from vsutil import split, join
    
    if bm3d == "bm3dcuda":
        B2 = core.bm3dcuda.BM3Dv2
    elif bm3d == "bm3dcuda_rtc":
        B2 = core.bm3dcuda_rtc.BM3Dv2
    else:
        B2 = core.bm3dcpu.BM3Dv2
    hw = c420p16.width / 2  # half width
    hh = c420p16.height / 2  # half height
    srcy_f, srcu_f, srcv_f = split(Depth(c420p16, 32))
    vbasic_y = B2(srcy_f, srcy_f, s1 + ds1, bs1, br1, r1, pn1, pr1, fast=fast)
    vfinal_y = B2(srcy_f, vbasic_y, s1, bs1, br1, r1, pn1, pr1, fast=fast)
    vyhalf = vfinal_y.resize.Spline36(hw, hh, src_left=-0.5)
    srchalf_444 = join([vyhalf, srcu_f, srcv_f])
    srchalf_opp = rgb2opp(torgbs(srchalf_444))
    vbasic_half = B2(srchalf_opp, srchalf_opp, s2 + ds2, bs2, br2, r2, pn2, pr2, fast=fast)
    vfinal_half = B2(srchalf_opp, vbasic_half, s2, bs2, br2, r2, pn2, pr2, fast=fast)
    vfinal_half = opp2rgb(vfinal_half).resize.Spline36(format=YUV444PS, matrix=1)
    _, vfinal_u, vfinal_v = split(vfinal_half)
    vfinal = join([vfinal_y, vfinal_u, vfinal_v])
    return Depth(vfinal, 16)


def denoise2(clip: VideoNode, level=0.5) -> VideoNode:
    from mvsfunc import LimitFilter

    nr16k = clip.knlm.KNLMeansCL(1, 2, 4, 1.25)
    nr16b = clip.bilateral.Bilateral(nr16k, level)
    nr16: VideoNode = LimitFilter(nr16k, nr16b, thr=1, elast=2)
    nr16_dn1 = nr16.knlm.KNLMeansCL(0, 2, 4, 1.5)
    nr16_dn2 = nr16_dn1.rgvs.RemoveGrain(20).rgvs.RemoveGrain(17).rgvs.RemoveGrain(4)
    noise1 = core.std.MakeDiff(nr16_dn1, nr16_dn2)
    dnoise1 = core.std.MakeDiff(nr16_dn2, clip)
    noise2 = core.std.MakeDiff(clip, nr16b)
    return LimitFilter(
        nr16.std.MergeDiff(noise2).std.MergeDiff(noise1).std.MergeDiff(dnoise1), 
        nr16, 
        thr=0.75, 
        elast=3.0, 
    )


def defilmgrain(clip: VideoNode, s1=16, s2=3, s3=3, g=1.5, dark=10000) -> VideoNode:
    from vapoursynth import YUV, GRAY, Error
    from havsfunc import QTGMC
    from vsutil import iterate
    from dfttest2 import DFTTest
    
    def gamma_curve(clip_y, gamma) -> VideoNode:
        def lut_y(x):
            floa = min(max(int(x / 56283 * 65536), 0), 65535) / 65535
            gammaed = floa ** gamma
            return min(max(int(gammaed * 56283 + 4112), 4112), 60395)
        
        return core.std.Lut(clip_y, planes=[0], function=lut_y)
    
    if clip.format.color_family != YUV and clip.format.color_family != GRAY:
        raise Error("rksfunc.defilmgrain: only YUV and GRAY format are supported.")
    origdep = clip.format.bits_per_sample
    if origdep != 16:
        clip = Depth(clip, 16)
    if clip.format.color_family == YUV:
        clip_y = gety(clip)
    else:
        clip_y = clip
    clip_yf = Depth(clip_y, 32)
    basic_f = clip_yf.bm3dcuda_rtc.BM3Dv2(clip_yf, s1, 3, 12, 0, 2, 6)
    final_f = clip_yf.bm3dcuda_rtc.BM3Dv2(basic_f, s1, 3, 12, 0, 2, 6)
    line_f = clip_yf.bm3dcuda_rtc.BM3Dv2(final_f, s2, 3, 15, 0, 2, 8)
    line = Depth(line_f, 16)
    clearplane = DFTTest(Depth(final_f, 16), tbsize=1)
    emask = gamma_curve(clearplane, g).tcanny.TCanny(1, mode=0, op=3)
    emask = iterate(emask, core.std.Maximum, 2)
    emask = iterate(emask, core.std.Inflate, 2)
    emask = core.std.Expr([emask, clearplane], f"y {dark} > 0 x ?")
    tref = QTGMC(clip_y, InputType=1, Sharpness=0, SourceMatch=3)
    ttp = clip_y.ttmpsm.TTempSmooth(3, 5, 3, 3, pfclip=tref)
    ttp = core.std.MaskedMerge(ttp, line, emask)
    ttp_f = Depth(ttp, 32)
    vfinal_f = ttp_f.bm3dcuda_rtc.BM3Dv2(clip_yf, s3, 3, 12, 1, 2, 8)
    vfinal = Depth(vfinal_f, 16)
    if clip.format.color_family == YUV:
        vfinal = mergeuv(vfinal, clip)
    if origdep != 16:
        vfinal = Depth(vfinal, origdep)
    return vfinal
