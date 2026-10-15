from vapoursynth import core, VideoNode


def DPIRMDegrain(
    clip: VideoNode, 
    dpir_args: dict = {},  # overwrite vsmlrt.DPIR kwargs
    limit_args: dict = {},  # overwrite vszip.LimitFilter kwargs
    mdg_args: dict = {},  # overwrite MCTD kwargs
    plane_qtg: bool = True, 
    grain_qtg: bool = False, 
    prot_dark: bool = False,  # use qtgmc(clip) instead of dpirmdegrain to protect dark scene
    prot_lthr: int = 8000,  # Y value under prot_lthr (in 16 bit) will be considered as dark scene
    prot_hthr: int = 12000,  # A tanh buffer will be created to smooth between prot_lthr and prot_hthr
    check: bool = False,  # return y_dp, y_lm, mdg instead
    prefer_jet: bool = False,  # prefer to use QTGMC in vsjetpack
    qtg_args: dict = {},  # overwrite QTGMC kwargs
) -> VideoNode:
    from vapoursynth import YUV, GRAY
    from vsmlrt import DPIR, DPIRModel, Backend
    from havsfunc import MCTemporalDenoise
    from math import tanh
    from ._resample import yer, mergeuv
    
    assert clip.format.color_family in [YUV, GRAY]
    origdep = clip.format.bits_per_sample
    if origdep != 16:
        clip = clip.fmtc.bitdepth(bits=16)
    is_yuv = clip.format.color_family == YUV
    y = yer(clip) if is_yuv else clip
    
    if prefer_jet:
        from vsdeinterlace import QTempGaussMC
        pre_mcargs = dict(
            source_match_mode=QTempGaussMC.SourceMatchMode.TWICE_REFINED, 
            sharpen_strength=0,
            analyze_refine=0,
        )
        pre_mcargs.update(qtg_args)
        qtgmc = QTempGaussMC(**pre_mcargs).deshimmer
    else:
        from havsfunc import QTGMC
        from functools import partial
        pre_mcargs = dict(
            InputType=1, 
            Sharpness=0, 
            SourceMatch=3, 
            mv_kernel='mvu'
        )
        pre_mcargs.update(qtg_args)
        qtgmc = partial(QTGMC, **pre_mcargs)
    dpir_preargs = dict(strength=10, backend=Backend.TRT(fp16=True))
    dpir_preargs.update(dpir_args)
    limit_preargs = dict(dark_thr=3, bright_thr=3, elast=2)
    limit_preargs.update(limit_args)
    mdg_preargs = dict(refine=True, DCT=5, blksize=32, limit=0, mv_kernel='mvu')
    mdg_preargs.update(mdg_args)
    
    # DPIR & Limit
    y_mc = qtgmc(y)
    dpi = (y_mc if plane_qtg else y).fmtc.bitdepth(bits=32)
    y_dp = DPIR(dpi, model=DPIRModel.drunet_gray, **dpir_preargs).fmtc.bitdepth(bits=16)
    y_lm = core.vszip.LimitFilter(y, y_dp, **limit_preargs)
    
    # MDegrain
    dif = core.std.MakeDiff(y, y_dp)
    dif_lm = core.std.MakeDiff(y_lm, y_dp)
    dif_lm = qtgmc(dif_lm) if grain_qtg else dif_lm
    dif_md = MCTemporalDenoise(dif, p=dif_lm, **mdg_preargs)
    mdg_l = core.std.MergeDiff(y_dp, dif_md)
    
    # protect dark scene
    if prot_dark:
        dif1 = core.std.MakeDiff(y, y_dp)
        dif2 = core.std.MakeDiff(y, mdg_l)
        neutral = 1 << 15
        mdg_d = core.std.Expr([dif1, dif2, y_mc, mdg_l], f'x {neutral} - abs y {neutral} - abs < z a ?')
        y_mid = (prot_lthr + prot_hthr) / 2
        y_len = (prot_hthr - prot_lthr) / 4
        lut_y = [(1 + tanh((x - y_mid) / y_len)) / 2 * 65535 for x in range(65536)]
        dmask = core.std.Lut(y_dp, lut=lut_y)
        mdg = core.std.MaskedMerge(mdg_d, mdg_l, dmask)
    else:
        mdg = mdg_l
    
    mdg = mergeuv(mdg, clip) if is_yuv else mdg
    mdg = mdg.fmtc.bitdepth(bits=origdep)
    if check:
        return y_dp, y_lm, mdg
    else:
        return mdg

    
def w2xtrt(
    clip: VideoNode, 
    noise: int, 
    test: bool = False,
    ofmt: bool = False, 
    w2xargs: dict = {}, 
    **args) -> VideoNode:
    from vapoursynth import RGB, GRAYS
    from vsmlrt import Waifu2x, Waifu2xModel, BackendV2
    from ._resample import torgbs, uvsr

    if clip.format.name.startswith("RGB"):
        crgbs = clip.fmtc.bitdepth(bits=32)
    elif clip.format.name.startswith("YUV444"):
        crgbs = torgbs(clip)
    elif clip.format.name.startswith("YUV420"):
        crgbs = torgbs(uvsr(clip))
    elif clip.format.name.startswith("Gray"):
        crgbs = core.std.ShufflePlanes(clip.fmtc.bitdepth(bits=32), [0]*3, RGB).std.SetFrameProps(_Matrix=0)
    
    if test:
        # w2x = crgbs.w2xnvk.Waifu2x(noise, 1, 2)
        w2xbe = args.get('w2b', BackendV2.NCNN_VK(fp16=True))
    else:
        w2xbe = args.get('w2b', BackendV2.TRT(fp16=True))
    preargs = {'backend': w2xbe}
    preargs.update(w2xargs)
    w2x = Waifu2x(crgbs, noise, 1, model=Waifu2xModel.cunet, **preargs)
    ofmt = args.get('o420p16', False) or ofmt  # history problem
    
    if ofmt:
        if clip.format.name.startswith("RGB"):
            return w2x.fmtc.bitdepth(bits=clip.format.bits_per_sample, dmode=0)
        elif clip.format.name.startswith("YUV"):
            return w2x.resize.Spline36(format=clip.format.id, matrix=1, dither_type='ordered')
        else:
            r, g, b = core.std.SplitPlanes(w2x)
            y = core.std.Expr([r, g, b], "x y z + + 3 /", GRAYS)
            return y.fmtc.bitdepth(bits=clip.format.bits_per_sample, dmode=0)
    else:
        return w2x


def TempoStab(
        clip: VideoNode, 
        mdargs: dict = {},  # Overwrite MCTD kwargs
        mcargs: dict = {},  # Overwrite QTGMC kwargs
        prefer_jet: bool = False,  # prefer to use QTGMC in vsjetpack
        **kwargs
) -> VideoNode:
    from vapoursynth import YUV, GRAY, Error
    from havsfunc import MCTemporalDenoise
    from ._resample import yer, mergeuv
    
    if clip.format.color_family != YUV and clip.format.color_family != GRAY:
        raise Error("rksfunc.tempostab: only YUV and GRAY format are supported.")
    origdep = clip.format.bits_per_sample
    if origdep != 16:
        clip = clip.fmtc.bitdepth(bits=16)
    if clip.format.color_family == YUV:
        clip_y = yer(clip)
    else:
        clip_y = clip

    if prefer_jet:
        from vsdeinterlace import QTempGaussMC
        pre_mcargs = dict(
            source_match_mode=QTempGaussMC.SourceMatchMode.TWICE_REFINED, 
            sharpen_strength=0,
            analyze_refine=0,
        )
        pre_mcargs.update(mcargs)
        tref = QTempGaussMC(**pre_mcargs).deshimmer(clip_y)
    else:
        from havsfunc import QTGMC
        pre_mcargs = dict(
            InputType=1, 
            Sharpness=0, 
            SourceMatch=3, 
            mv_kernel='mvu'
        )
        pre_mcargs.update(mcargs)
        tref = QTGMC(clip_y, **pre_mcargs)
    pre_mdargs = dict(
        p=tref,
        refine=True,
        blksize=32,
        limit=0,
        DCT=5,
    )
    pre_mdargs.update(mdargs)
    smd = MCTemporalDenoise(clip_y, **pre_mdargs)

    if clip.format.color_family == YUV:
        smd = mergeuv(smd, clip)
    if origdep != 16:
        smd = smd.fmtc.bitdepth(bits=origdep)
    return smd


def BM3Dvk2Wrapper(
    clip: VideoNode,
    sigma: tuple[float] = (3, 6, 6),
    radius_gray: int = 1,
    radius_opp: int = 0,
    sigma_basic: tuple[float] = (5, 10, 10),
    ref: VideoNode | None = None,
    **kwargs,
):
    from vapoursynth import GRAY, YUV, RGB
    from mvsfunc import ToRGB, ToYUV
    from ._resample import yer, half444
    make_tuple = lambda x: x if hasattr(x, "__iter__") else (x,)
    sigma = make_tuple(sigma)
    sigma_basic = make_tuple(sigma_basic)
    if clip.format.color_family == GRAY:
        y32, c32 = clip.vszip.Dither(32), None
    elif clip.format.color_family == RGB:
        y32, c32 = None, clip.vszip.Dither(32)
    elif clip.format.name.startswith("YUV444"):
        y32, c32 = None, ToRGB(clip, '709', 32)
    else:
        s32 = clip.vszip.Dither(32)
        y32, c32 = yer(s32), ToRGB(half444(s32), '709')
    if ref is None:
        yref, cref = None, None
    elif ref.format.color_family == GRAY:
        yref, cref = ref.vszip.Dither(32), None
    elif ref.format.color_family == RGB:
        cref = ref.vszip.Dither(32)
        yref = yer(ToYUV(cref, '709', '444'))
    elif ref.format.name.startswith("YUV444"):
        r32 = ref.vszip.Dither(32)
        yref, cref = yer(r32), ToRGB(r32, '709')
    else:
        r32 = ref.vszip.Dither(32)
        yref, cref = yer(r32), ToRGB(half444(r32), '709')
    bm3d_kwargs = dict(profile="vn")
    bm3d_kwargs.update(kwargs)
    if y32 is not None:
        if yref is None:
            yref = core.bm3dvk2.BM3D(y32, None, sigma=sigma_basic[0], radius=radius_gray, **bm3d_kwargs)
        else:
            assert y32.height == yref.height and y32.width == yref.width, \
                f"{y32.format = }\n{yref.format = }"
        y_final = core.bm3dvk2.BM3D(y32, yref, sigma=sigma[0], radius=radius_gray, **bm3d_kwargs)
    if c32 is not None:
        if cref is None:
            cref = core.bm3dvk2.BM3D(c32, None, sigma=sigma_basic, radius=radius_opp, **bm3d_kwargs)
        else:
            assert c32.height == cref.height and c32.width == cref.width, \
                f"{c32.format = }\n{cref.format = }"
        c_final = core.bm3dvk2.BM3D(c32, cref, sigma=sigma, radius=radius_opp, **bm3d_kwargs)
    obit = clip.format.bits_per_sample
    if clip.format.color_family == GRAY:
        return y_final.vszip.Dither(obit)
    elif clip.format.color_family == RGB:
        return c_final.vszip.Dither(obit)
    elif clip.format.name.startswith("YUV444"):
        return ToYUV(c_final, '709', '444', obit, dither=0)
    else:
        y_final = y_final.vszip.Dither(obit)
        c_final = ToYUV(c_final, '709', '420', obit, dither=0)
        return core.std.ShufflePlanes([y_final, c_final, c_final], [0, 1, 2], YUV, clip)


def BM3DRef(
    c420p16: VideoNode, ref: VideoNode,
    bm3d="bm3dcuda_rtc", chroma=True,
    sy=2, ry=1, bsy=4, bry=8, pny=2, pry=8,
    sc=2, rc=0, bsc=4, brc=8, pnc=2, prc=6, 
) -> VideoNode:
    from vapoursynth import YUV444PS
    from vsutil import split, join
    from ._resample import rgb2opp, opp2rgb, torgbs
    
    if isinstance(bm3d, str):
        Bv2 = getattr(core, bm3d).BM3Dv2
    else:
        Bv2 = bm3d.BM3Dv2
    hw = c420p16.width // 2  # half width
    hh = c420p16.height // 2  # half height
    srcy_f, srcu_f, srcv_f = split(c420p16.fmtc.bitdepth(bits=32))
    refy_f, refu_f, refv_f = split(ref.fmtc.bitdepth(bits=32))
    vfinal_y = Bv2(srcy_f, refy_f, sy, bsy, bry, ry, pny, pry)
    vyhalf = vfinal_y.resize.Spline36(hw, hh, src_left=-0.5)
    ryhalf = refy_f.resize.Spline36(hw, hh, src_left=-0.5)
    srchalf_444 = join([vyhalf, srcu_f, srcv_f])
    refhalf_444 = join([ryhalf, refu_f, refv_f])
    srchalf_opp = rgb2opp(torgbs(srchalf_444))
    refhalf_opp = rgb2opp(torgbs(refhalf_444))
    vfinal_half = Bv2(srchalf_opp, refhalf_opp, sc, bsc, brc, rc, pnc, prc, chroma)
    vfinal_half = opp2rgb(vfinal_half).resize.Spline36(format=YUV444PS, matrix=1)
    _, vfinal_u, vfinal_v = split(vfinal_half)
    vfinal = join([vfinal_y, vfinal_u, vfinal_v])
    return vfinal.fmtc.bitdepth(bits=16)


def BM3DWrapper(
    c420p16: VideoNode, 
    bm3d="bm3dcuda_rtc", chroma=True,
    sy=1.2, ry=1, bsy=4, bry=8, pny=2, pry=8, dsy=0.6,
    sc=2.4, rc=0, bsc=4, brc=8, pnc=2, prc=6, dsc=1.2,
) -> VideoNode:
    from vapoursynth import YUV444PS
    from vsutil import split, join
    from ._resample import rgb2opp, opp2rgb, torgbs
    
    if isinstance(bm3d, str):
        Bv2 = getattr(core, bm3d).BM3Dv2
    else:
        Bv2 = bm3d.BM3Dv2
    hw = c420p16.width // 2  # half width
    hh = c420p16.height // 2  # half height
    srcy_f, srcu_f, srcv_f = split(c420p16.fmtc.bitdepth(bits=32))
    vbasic_y = Bv2(srcy_f, srcy_f, sy+dsy, bsy, bry, ry, pny, pry)
    vfinal_y = Bv2(srcy_f, vbasic_y, sy, bsy, bry, ry, pny, pry)
    vyhalf = vfinal_y.resize.Spline36(hw, hh, src_left=-0.5)
    srchalf_444 = join([vyhalf, srcu_f, srcv_f])
    srchalf_opp = rgb2opp(torgbs(srchalf_444))
    vbasic_half = Bv2(srchalf_opp, srchalf_opp, sc+dsc, bsc, brc, rc, pnc, prc, chroma)
    vfinal_half = Bv2(srchalf_opp, vbasic_half, sc, bsc, brc, rc, pnc, prc, chroma)
    vfinal_half = opp2rgb(vfinal_half).resize.Spline36(format=YUV444PS, matrix=1)
    _, vfinal_u, vfinal_v = split(vfinal_half)
    vfinal = join([vfinal_y, vfinal_u, vfinal_v])
    return vfinal.fmtc.bitdepth(bits=16)


def DeFilmGrain(clip: VideoNode, s1=16, s2=3, s3=3, g=1.5, dark=10000) -> VideoNode:
    from vapoursynth import YUV, GRAY, Error
    from havsfunc import QTGMC
    from vsutil import iterate
    from dfttest2 import DFTTest
    from ._resample import yer, mergeuv
    
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
        clip = clip.fmtc.bitdepth(bits=16)
    if clip.format.color_family == YUV:
        clip_y = yer(clip)
    else:
        clip_y = clip
    clip_yf = clip_y.fmtc.bitdepth(bits=32)
    basic_f = clip_yf.bm3dcuda_rtc.BM3Dv2(clip_yf, s1, 3, 12, 0, 2, 6)
    final_f = clip_yf.bm3dcuda_rtc.BM3Dv2(basic_f, s1, 3, 12, 0, 2, 6)
    line_f = clip_yf.bm3dcuda_rtc.BM3Dv2(final_f, s2, 3, 15, 0, 2, 8)
    line = line_f.fmtc.bitdepth(bits=16)
    clearplane = DFTTest(final_f.fmtc.bitdepth(bits=16), tbsize=1)
    emask = gamma_curve(clearplane, g).tcanny.TCanny(1, mode=0, op=3)
    emask = iterate(emask, core.std.Maximum, 2)
    emask = iterate(emask, core.std.Inflate, 2)
    emask = core.std.Expr([emask, clearplane], f"y {dark} > 0 x ?")
    tref = QTGMC(clip_y, InputType=1, Sharpness=0, SourceMatch=3)
    ttp = clip_y.zsmooth.TTempSmooth(3, 5, 3, 3, pfclip=tref)
    ttp = core.std.MaskedMerge(ttp, line, emask)
    ttp_f = ttp.fmtc.bitdepth(bits=32)
    vfinal_f = ttp_f.bm3dcuda_rtc.BM3Dv2(clip_yf, s3, 3, 12, 1, 2, 8)
    vfinal = vfinal_f.fmtc.bitdepth(bits=16)
    if clip.format.color_family == YUV:
        vfinal = mergeuv(vfinal, clip)
    if origdep != 16:
        vfinal = vfinal.fmtc.bitdepth(bits=origdep)
    return vfinal


def ChromaDenoise(clip: VideoNode, sigma=1.2, bm3d="bm3dvk2", **w2xtrt_kwargs) -> VideoNode:
    from vapoursynth import YUV
    from functools import partial
    from mvsfunc import ToYUV, ToRGB
    from ._resample import half444
    w2xtrt_params = dict(noise=3)
    w2xtrt_params.update(w2xtrt_kwargs)
    if bm3d == "bm3dvk2":
        BM3D = partial(core.bm3dvk2.BM3D, profile='vn', sigma=sigma, radius=1)
        w2xtrt_params['ofmt'] = False
    else:
        BM3D = partial(getattr(core, bm3d).BM3Dv2, sigma=sigma, radius=1,
                       block_step=4, bm_range=8, ps_num=2, ps_range=8)
        w2xtrt_params['ofmt'] = True
    c16 = clip.vszip.Dither(16)
    c32 = half444(clip).vszip.Dither(32)
    w2x = w2xtrt(c32, **w2xtrt_params)
    if bm3d == "bm3dvk2":
        c32 = ToRGB(c32, '709')
    vfn = BM3D(c32, w2x).vszip.Dither(16)
    if bm3d == "bm3dvk2":
        vfn = ToYUV(vfn, '709', '444')
    return core.std.ShufflePlanes([c16, vfn], [0, 1, 2], YUV, c16)


def AdaptiveBM3D(clip: VideoNode) -> VideoNode:
    from ._mask import GammaMask
    from ._resample import mergeuv
    
    if hasattr(core, 'bm3dvk2'):
        degrain_l = BM3Dvk2Wrapper(clip)
    else:
        degrain_l = BM3DWrapper(clip)
    degrain_d = clip.vszip.Bilateral(degrain_l, 0.5)
    amask = degrain_l.zsmooth.RemoveGrain([20, 11]).zsmooth.RemoveGrain([20, 11])
    amask = amask.std.PlaneStats().adg.Mask(12)
    degrain = core.std.MaskedMerge(degrain_l, degrain_d, amask, first_plane=True)
    clear_edge = core.std.MaskedMerge(degrain, degrain_l, GammaMask(degrain_l).std.Maximum())
    return mergeuv(clear_edge, degrain_l)
