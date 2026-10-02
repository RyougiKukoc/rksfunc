Simple functions I use.

# Installation

Requires Python 3.10 or newer.

```shell
python -m pip install "rksfunc @ git+https://github.com/RyougiKukoc/rksfunc.git"
```

Runtime dependencies are intentionally not declared in `pyproject.toml`.
Install the VapourSynth modules and native plugins needed by your script
yourself. Normal Python imports report missing dependencies; the package does
not install or replace them automatically.

# Manual wheel synchronization

This repository remains the source of truth for the module and its version.
The central [wheels repository](https://github.com/AliceTeaParty/vapoursynth-api4-wheels)
checks out an exact commit under `modules/rksfunc/src/` and builds the wheel
from this repository's unchanged packaging metadata.

After updating the source and `project.version`, manually run **Sync wheel
preview** in this repository's Actions page. Choose a source ref; the notifier
resolves it to a fixed commit and submits a preview build. It does not publish
a Release. A central maintainer separately selects the successful build and
wheel SHA256 in **Publish - Python module**.

Configure the repository secret `WHEELS_UPDATE_TOKEN` with a credential limited
to the central repository's Actions write permission. The default `GITHUB_TOKEN`
cannot trigger a different repository. Without that secret, a maintainer can
run the preview directly in the central repository or submit a reviewed version
request PR through its [module interface](https://github.com/AliceTeaParty/vapoursynth-api4-wheels/tree/main/modules).

Once a wheel is explicitly published:

```shell
python -m pip install --extra-index-url https://aliceteaparty.github.io/vapoursynth-api4-wheels/simple/ rksfunc
```

# Bundled shader

`KrigBilateral.glsl` is distributed with the package. Its original attribution
and LGPL-3.0-or-later header are retained. See [LICENSES.md](LICENSES.md)
for the shader's provenance and included license texts. Those notices do not
declare a project-wide license for the Python source.
