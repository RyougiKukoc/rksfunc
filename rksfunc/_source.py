from vapoursynth import core, VideoNode, Error


def sourcer(fn: str = None, mode: int = 1) -> VideoNode:
    import os
    if fn is None:
        for tfn in os.listdir():
            if os.path.isfile(tfn):
                if os.path.splitext(tfn)[-1] in ['.m2ts', '.hevc', '.264', '.avc', '.mkv', '.mp4']:
                    fn = tfn
                    break
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
            raise Error(f'Remove {dgi} then try again')
    else:
        raise ValueError("mode must be 1 or 2.")
    return core.std.SetFrameProps(src, Name=os.path.basename(fn))


def genqp(clip: VideoNode, qpfile_fp: str = None):
    if qpfile_fp is None:
        import os
        qpfile_fp = os.path.splitext(clip.get_frame(0).props['Name']) + '.qpfile'
    with open(qpfile, "r") as f:
        qpstr = f.readlines()
    qpstr = [i for i in qpstr if i != "\n"]  # delete blank line
    qpstr = [i if i.endswith("\n") else i + "\n" for i in qpstr]
    qpstr = [i[:-3] for i in qpstr]  # remove K\n
    qp = [int(i) for i in qpstr]
    if qp[0] != 0:
        qp = [0] + qp
    if qp[-1] != clip.num_frames - 1:
        qp = qp + [clip.num_frames - 1]
    return qp


def ivtcqtg(c8: VideoNode, withdaa: bool = True, opencl: bool = True) -> VideoNode:
    from havsfunc import QTGMC
    from yvsfunc import daa_mod
    from mvsfunc import FilterCombed
    from ._resample import depth
    
    field_match = c8.vivtc.VFM(order=1, mode=3, cthresh=10)
    deint = QTGMC(c8, "fast", TFF=True, FPSDivisor=2, opencl=opencl)
    ivtc = FilterCombed(field_match, deint).vivtc.VDecimate().std.SetFieldBased(0)
    return daa_mod(depth(ivtc, 16), opencl=opencl) if withdaa else depth(ivtc, 16)


def ivtcdrb(
    clip: VideoNode, 
    bifrost: bool = False, 
    rainbowsmooth: bool = False, 
    order: int = 1, 
    tcombmode: int = 2,
    opencl: bool = True
) -> VideoNode:
    from yvsfunc import daa_mod
    from ._resample import depth
    
    if clip.format.bits_per_sample != 8:
        clip = depth(clip, 8)
    ivtc_filt = clip.tcomb.TComb(tcombmode)
    if bifrost:
        ivtc_filt = ivtc_filt.bifrost.Bifrost(interlaced=True)
    if rainbowsmooth:
        from RainbowSmooth import RainbowSmooth
        ivtc_filt = RainbowSmooth(ivtc_filt)
    ivtc16 = depth(ivtc_filt.vivtc.VFM(order, cthresh=10).vivtc.VDecimate(), 16)
    return daa_mod(ivtc16, opencl=opencl)
