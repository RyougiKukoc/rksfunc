import vapoursynth as vs
from vapoursynth import core
from ._resample import yer, ToYUV
from functools import partial

def DescaleKernelTester(clip: vs.VideoNode, descale_args: dict, cache_lim: int = 10, topk: int = 5) -> vs.VideoNode:
    def transfoer_property(n, f):
        fout = f[0].copy()
        for df in f[1:]:
            fout.props[str(df.props['_Kernel'])] = df.props['PlaneStatsAverage']
        return fout
    
    def error_sorter(n, f, topk=5):
        fout = f.copy()
        test_result = []
        for k, v in f.props.items():
            if k.startswith("DescaleKernelTest_"):
                test_result.append((k, v))
        test_result.sort(key=lambda x: x[1], reverse=False)
        for i, (k, v) in enumerate(test_result):
            if i >= topk:
                del fout.props[k]
        return fout
    
    class CacheTransfer:
        def __init__(self, src_clip: vs.VideoNode, selector, cache_lim: int = 10):
            self.cache: list[vs.VideoNode] = []
            self.cache_lim = cache_lim
            self.src_clip = src_clip
            self.selector = selector
        
        def load(self) -> vs.VideoNode:
            if len(self.cache):
                self.src_clip = self.src_clip.std.ModifyFrame([self.src_clip] + self.cache, self.selector)
        
        def add(self, clip: vs.VideoNode):
            self.cache.append(clip)
            if len(self.cache) >= self.cache_lim:
                self.load()
                self.cache.clear()
        
        def output(self) -> vs.VideoNode:
            if len(self.cache):
                self.load()
                self.cache.clear()
            return self.src_clip
                
    if clip.format.name.startswith("YUV"):
        y32 = yer(clip).fmtc.bitdepth(bits=32)
    elif clip.format.name.startswith("RGB"):
        c444 = ToYUV(clip, css='444')
        y32 = yer(c444).fmtc.bitdepth(bits=32)
    else:
        y32 = clip.fmtc.bitdepth(bits=32)
    
    upscale_args = descale_args.copy()
    upscale_args['width'] = y32.width
    upscale_args['height'] = y32.height
    
    # Bilinear
    cacher = CacheTransfer(y32, transfoer_property, cache_lim=cache_lim)
    rsc = y32.descale.Debilinear(**descale_args).resize.Bilinear(**upscale_args)
    dif = core.std.Expr([y32, rsc], "x y - abs").std.PlaneStats() \
        .std.SetFrameProps(_Kernel="DescaleKernelTest_Bilinear")
    cacher.add(dif)
    
    # Bicubic
    base = 30
    for b in range(base + 1):
        for c in range(base + 1):
            dsc = y32.descale.Debicubic(**descale_args, b=b/base, c=c/base)
            rsc = dsc.resize.Bicubic(**upscale_args, filter_param_a=b/base, filter_param_b=c/base)
            dif = core.std.Expr([y32, rsc], "x y - abs").std.PlaneStats() \
                .std.SetFrameProps(_Kernel=f"DescaleKernelTest_Bicubic_{b}p{base}_{c}p{base}")
            cacher.add(dif)
    
    # Lanczos
    for taps in range(2, 11):
        rsc = y32.descale.Delanczos(**descale_args, taps=taps).resize.Lanczos(**upscale_args, filter_param_a=taps)
        dif = core.std.Expr([y32, rsc], "x y - abs").std.PlaneStats() \
            .std.SetFrameProps(_Kernel=f"DescaleKernelTest_Lanczos_{taps}")
        cacher.add(dif)
    
    # Spline
    rsc = y32.descale.Despline16(**descale_args).resize.Spline16(**upscale_args)
    dif = core.std.Expr([y32, rsc], "x y - abs").std.PlaneStats() \
        .std.SetFrameProps(_Kernel="DescaleKernelTest_Spline16")
    cacher.add(dif)
    rsc = y32.descale.Despline36(**descale_args).resize.Spline36(**upscale_args)
    dif = core.std.Expr([y32, rsc], "x y - abs").std.PlaneStats() \
        .std.SetFrameProps(_Kernel="DescaleKernelTest_Spline36")
    cacher.add(dif)
    rsc = y32.descale.Despline64(**descale_args).resize.Spline64(**upscale_args)
    dif = core.std.Expr([y32, rsc], "x y - abs").std.PlaneStats() \
        .std.SetFrameProps(_Kernel="DescaleKernelTest_Spline64")
    cacher.add(dif)
    
    out = cacher.output()
    return out.std.ModifyFrame(out, partial(error_sorter, topk=topk))