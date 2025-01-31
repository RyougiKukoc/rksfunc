from vapoursynth import core, VideoNode, Error


def sourcer(fn: str = None, mode: int = 1) -> VideoNode:
    import os
    import sys
    import subprocess as sp
    
    if fn is None:
        for tfn in os.listdir():
            if os.path.isfile(tfn):
                if os.path.splitext(tfn)[-1] in ['.m2ts', '.hevc', '.264', '.avc', '.mkv', '.mp4']:
                    fn = tfn
                    break
    if mode == 1:
        src = core.lsmas.LWLibavSource(fn)
    elif mode == 2:
        dgi = fn + '.dgi'
        cmd = ['DGIndexNV', '-i', fn, '-o', dgi, '-h']
        if not hasattr(core, "dgdecodenv"):
            core.std.LoadPlugin(os.path.join(sys.prefix, 'x26x', 'DGDecodeNV.dll'))
            cmd[0] = os.path.join(sys.prefix, 'x26x', 'DGIndexNV')
        if not os.path.exists(dgi):
            _ = sp.run(cmd)
        try:
            src = core.dgdecodenv.DGSource(dgi)
        except Error:
            raise Error(f'Remove {dgi} then try again')
    elif mode == 3:
        src = core.bs.VideoSource(fn, cachepath='/')
    else:
        raise ValueError("mode must be in [1, 2, 3].")
    return core.std.SetFrameProps(src, Name=os.path.basename(fn))


def GenQPList(qpfile_fp: str = None, clip: VideoNode = None, force_align: bool = False):
    if qpfile_fp is None:
        import os
        assert clip is not None
        qpfile_fp = os.path.splitext(clip.get_frame(0).props['Name'])[0] + '.qpfile'
    with open(qpfile_fp, "r") as f:
        qpstr = f.readlines()
    qpstr = [i for i in qpstr if i != "\n"]  # delete blank line
    qpstr = [i if i.endswith("\n") else i + "\n" for i in qpstr]
    qpstr = [i[:-3] for i in qpstr]  # remove K\n
    qp = [int(i) for i in qpstr]
    if force_align:
        assert clip is not None
        if qp[0] != 0:
            qp = [0] + qp
        if qp[-1] != clip.num_frames - 1:
            qp = qp + [clip.num_frames - 1]
    return qp


def IVTCDeRainbow(
    clip: VideoNode, 
    bifrost: bool = False, 
    rainbowsmooth: bool = False, 
    order: int = 1, 
    tcombmode: int = 2,
    opencl: bool = True
) -> VideoNode:
    from yvsfunc import daa_mod
    
    if clip.format.bits_per_sample != 8:
        clip = clip.fmtc.bitdepth(bits=8)
    if tcombmode:
        clip = clip.tcomb.TComb(tcombmode)
    if bifrost:
        clip = clip.bifrost.Bifrost(interlaced=True)
    if rainbowsmooth:
        from RainbowSmooth import RainbowSmooth
        clip = RainbowSmooth(clip)
    ivtc16 = clip.vivtc.VFM(order, cthresh=10).vivtc.VDecimate().fmtc.bitdepth(bits=16)
    return daa_mod(ivtc16, opencl=opencl)
