from vapoursynth import core, VideoNode
from typing import Union, Tuple


def Depth(clip: VideoNode, bits) -> VideoNode:
    return clip.fmtc.bitdepth(bits=bits)


def yer(clip: VideoNode) -> VideoNode:
    from vapoursynth import GRAY
    
    return clip.std.ShufflePlanes(0, GRAY)


def mergeuv(clipy: VideoNode, clipuv: VideoNode) -> VideoNode:
    from vapoursynth import YUV

    return core.std.ShufflePlanes([clipy, clipuv], [0, 1, 2], YUV)


def uvsr(c420p16: VideoNode, mode: Union[int, str] = -1, opencl: bool = True) -> VideoNode:
    """
    YUV420P16 -> YUV444P16
    :param c420p16: input VideoNode.
    :param mode: index or name in ['nnedi3', 'bicubic', 'krigbilateral']
    :param opencl: whether to use nnedi3cl, default znedi3.
    :return: the YUV444P16 form of input.
    """
    from vapoursynth import YUV444P16, YUV
    from functools import partial
    
    mode_list = ['nnedi3', 'bicubic', 'krigbilateral']
    if isinstance(mode, int):
        mode = mode_list[mode]
    elif isinstance(mode, str):
        mode = mode.lower()
    assert mode in mode_list
    c420p16 = c420p16.fmtc.bitdepth(bits=16)
    assert c420p16.format.name == 'YUV420P16'
    if mode == 'nnedi3':
        y, u, v = core.std.SplitPlanes(c420p16)
        u, v = map(partial(LeftChroma2x, opencl=opencl), [u, v])
        return core.std.ShufflePlanes([y, u, v], [0] * 3, YUV)
    elif mode == 'bicubic':
        return c420p16.resize.Bicubic(format=YUV444P16)
    elif mode == 'krigbilateral':
        return KrigBilateral(c420p16)


def LeftChroma2x(clip: VideoNode, opencl: bool) -> VideoNode:
    n2x = core.nnedi3cl.NNEDI3CL if opencl else core.znedi3.nnedi3
    h2x = n2x(clip, field=0, dh=True)
    w2x = n2x(h2x.std.Transpose().resize.Bicubic(src_left=.5), field=1, dh=True)
    return w2x.std.Transpose()

    
def KrigBilateral(c420p16: VideoNode, shader_fp: str = None) -> VideoNode:
    import os
    
    assert c420p16.format.name == 'YUV420P16'
    if shader_fp is None:
        shader_fp = os.path.join(os.path.dirname(__file__), 'KrigBilateral', 'KrigBilateral.glsl')
    return c420p16.placebo.Shader(shader=shader_fp)


def rgb2opp(clip: VideoNode, normalize: bool = False) -> VideoNode:
    # copy from yvsfunc
    from vapoursynth import RGBS, YUV, MATRIX_UNSPECIFIED
    from math import sqrt
    
    assert clip.format.id == RGBS
    if normalize:
        coef = [1/3, 1/3, 1/3, 0, 1/sqrt(6), -1/sqrt(6), 0, 0, 1/sqrt(18), 1/sqrt(18), -2/sqrt(18), 0]
    else:
        coef = [1/3, 1/3, 1/3, 0, 1/2, -1/2, 0, 0, 1/4, 1/4, -1/2, 0]
    opp = core.fmtc.matrix(clip, fulls=True, fulld=True, col_fam=YUV, coef=coef)
    opp = core.std.SetFrameProps(opp, _Matrix=MATRIX_UNSPECIFIED, BM3D_OPP=1)
    return opp


def opp2rgb(clip: VideoNode, normalize: bool = False) -> VideoNode:
    # copy from yvsfunc
    from vapoursynth import YUV444PS, MATRIX_RGB, RGB 
    from math import sqrt
    
    assert clip.format.id == YUV444PS
    if normalize:
        coef = [1, sqrt(3/2), 1/sqrt(2), 0, 1, -sqrt(3/2), 1/sqrt(2), 0, 1, 0, -sqrt(2), 0]
    else:
        coef = [1, 1, 2/3, 0, 1, -1, 2/3, 0, 1, 0, -4/3, 0]
    rgb = core.fmtc.matrix(clip, fulls=True, fulld=True, col_fam=RGB, coef=coef)
    rgb = core.std.SetFrameProps(rgb, _Matrix=MATRIX_RGB)
    rgb = core.std.RemoveFrameProps(rgb, 'BM3D_OPP')
    return rgb


def Crop420(clip: VideoNode, l=0, r=0, t=0, b=0) -> VideoNode:
    from functools import partial
    from vapoursynth import YUV
    
    if l == 0 and r == 0 and t == 0 and b == 0:
        return clip
    if not clip.format.name.startswith("YUV420"):
        raise ValueError("Input clip must be YUV420.")
    if (l + r) % 2 == 1 or (t + b) % 2 == 1:
        raise ValueError("You can't specify odd width or height.")
    y, u, v = clip.std.SplitPlanes()
    y = y.std.Crop(l, r, t, b)
    chroma_crop = partial(core.resize.Bicubic, width=y.width//2, height=y.height//2, \
        src_left=l/2, src_top=t/2, src_width=y.width//2, src_height=y.height//2)
    u, v = map(chroma_crop, [u, v])
    return core.std.ShufflePlanes([y, u, v], [0] * 3, YUV)


def Gammarize(clip: VideoNode, gamma, tvrange=False) -> VideoNode:
    def scale8(val, depth):
        return val * ((1 << depth) - 1) // 255
    
    if clip.format.name.startswith("YUV"):
        is_yuv = True
        y = yer(clip)
    elif clip.format.name.startswith("Gray"):
        is_yuv = False
        y = clip
    else:
        raise ValueError("Input clip must be either YUV or GRAY.")
    
    bits = y.format.bits_per_sample
    thrl = scale8(16, bits) if tvrange else scale8(0, bits)
    thrh = scale8(235, bits) if tvrange else scale8(255, bits)
    rng = scale8(235 - 16, bits) if tvrange else scale8(255 - 0, bits)
    corrected = y.std.Expr(f'x {rng} / {gamma} pow {rng} * {thrl} + {thrl} max {thrh} min')
    
    return mergeuv(corrected, clip) if is_yuv else corrected


def dvdto720(clip: VideoNode, width: int = 960) -> VideoNode:
    from nnedi3_rpow2 import nnedi3_rpow2 as nnr2
    from vapoursynth import RGB48, YUV420P16
    
    crgb = clip.resize.Bicubic(format=RGB48, matrix_in=5)
    c709 = crgb.resize.Spline36(format=YUV420P16, matrix=1)
    return nnr2(c709, 2, width, 720, nsize=4, nns=4, qual=2, pscrn=1)


def RescaleLuma(
    clip: VideoNode,
    w: int, 
    h: int, 
    kernel: str = 'bicubic', 
    b = None, 
    c = None, 
    taps = None, 
    linemode: bool = True, 
    maskmode: int = 0,  # 0: y_rescale only, 1: mask only, 2: return (y_rescale, mask)
    opencl: bool = True, 
    num_maximum: int = 3, 
    num_inflate: int = 3,
    thr = 10 
) -> Union[VideoNode, Tuple[VideoNode, VideoNode]]:
    from fvsfunc import Resize
    from nnedi3_rpow2 import nnedi3_rpow2 as nnr2
    from vsutil import iterate
    from vapoursynth import YUV, GRAY
    
    assert clip.format.color_family in [YUV, GRAY]
    is_yuv = clip.format.color_family == YUV
    y = yer(clip) if is_yuv else clip
    
    ow, oh = y.width, y.height
    maxv = (1 << 16) - 1
    thrl = 4 * maxv // 0xFF
    thrh = 24 * maxv // 0xFF
    thrdes = thr * maxv // 0xFF
    rf = 2
    while w * rf < ow:
        rf *= 2
    upsizer = "nnedi3cl" if opencl else "znedi3"
    
    descale = Resize(y, w, h, kernel=kernel, a1=b, a2=c, taps=taps, invks=True)
    rescale = nnr2(descale, rf, ow, oh, upsizer=upsizer, nsize=4, nns=4, qual=2)
    upscale = Resize(descale, ow, oh, kernel=kernel, a1=b, a2=c, taps=taps)
    dmask = core.std.Expr([y, upscale], 'x y - abs')
    dmask = iterate(dmask, core.std.Maximum, num_maximum)
    dmask = iterate(dmask, core.std.Inflate, num_inflate)
    
    if linemode:
        emask = rescale.std.Prewitt().std.Expr(f'x {thrh} >= {maxv} x {thrl} <= 0 x ? ?')
        cmask = core.std.Expr([dmask, emask], f'x {thrdes} >= 0 y ?').std.Inflate().std.Deflate()
        if maskmode == 1:
            return cmask
        yrs = core.std.MaskedMerge(y, rescale, cmask)
    else:
        cmask = dmask.std.Expr(f'x {thrdes} >= {maxv} 0 ?').std.Maximum().std.Maximum().std.Minimum().std.Inflate()
        if maskmode == 1:
            return cmask
        yrs = core.std.MaskedMerge(rescale, y, cmask)
        
    y_rescale = mergeuv(yrs, clip) if is_yuv else yrs
    return (y_rescale, cmask) if maskmode == 2 else y_rescale


def half444(c420: VideoNode) -> VideoNode:
    y = yer(c420)
    y_half = y.resize.Spline36(y.width // 2, y.height // 2, src_left=-.5)
    return mergeuv(y_half, c420)
