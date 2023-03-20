from vapoursynth import core, VideoNode, Error


def sourcer(fn: str, mode=1) -> VideoNode:
    if mode == 1:
        src = core.lsmas.LWLibavSource(fn)
    elif mode == 2:
        import sys, os, subprocess as sp
        dgi = fn + '.dgi'
        if not os.path.exists(dgi):
            os.environ['Path'] = os.environ['Path'] + ';' + sys.prefix + '\\x26x'
            cmd = f'DGIndexNV.exe -i "{fn}" -o "{dgi}" -h'
            p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT)
            ret = p.communicate()[0]
        if not hasattr(core, "dgdecodenv"):
            core.std.LoadPlugin(sys.prefix + '\\x26x\\DGDecodeNV.dll')
        try:
            src = core.dgdecodenv.DGSource(dgi)
        except Error:
            os.remove(dgi)
            os.environ['Path'] = os.environ['Path'] + ';' + sys.prefix + '\\x26x'
            cmd = f'DGIndexNV.exe -i "{fn}" -o "{dgi}" -h'
            p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT)
            ret = p.communicate()[0]
            src = core.dgdecodenv.DGSource(dgi)
    return core.std.SetFrameProps(src, Name="src")


def ivtcqtg(c8: VideoNode, withdaa: bool = True, opencl: bool = True) -> VideoNode:
    from havsfunc import QTGMC, daa
    from mvsfunc import FilterCombed
    from ._resample import depth
    
    field_match = c8.vivtc.VFM(order=1, mode=3, cthresh=10)
    deint = QTGMC(c8, "fast", TFF=True, FPSDivisor=2, opencl=True)
    ivtc = FilterCombed(field_match, deint).vivtc.VDecimate().std.SetFieldBased(0)
    return daa(depth(ivtc, 16), 4, 4, 2, 1, opencl=opencl) if withdaa else depth(ivtc, 16)


def ivtcdrb(clip: VideoNode, bifrost: bool = False, rainbowsmooth: bool = False, order=1) -> VideoNode:
    from havsfunc import daa
    from ._resample import depth
    
    if clip.format.bits_per_sample != 8:
        clip = depth(clip, 8)
    ivtc_filt = clip.tcomb.TComb(2)
    if bifrost:
        ivtc_filt = ivtc_filt.bifrost.Bifrost(interlaced=True)
    if rainbowsmooth:
        from RainbowSmooth import RainbowSmooth
        ivtc_filt = RainbowSmooth(ivtc_filt)
    ivtc16 = depth(ivtc_filt.vivtc.VFM(order, cthresh=10).vivtc.VDecimate(), 16)
    return daa(ivtc16, 4, 4, 2, 1, opencl=True)
