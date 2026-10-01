# License and third-party notices

This project does not declare a project-wide license for its Python code.
The notices below describe the bundled third-party shader only; they do not
relicense the other source files.

## KrigBilateral shader

- File: `rksfunc/KrigBilateral.glsl`
- Original attribution in the shader: KrigBilateral by Shiandow.
- Referenced upstream: https://gist.github.com/igv/a015fc885d5c22e6891820ad89555637
- License stated in the file: GNU Lesser General Public License version 3,
  or, at your option, any later version (`LGPL-3.0-or-later`).

The shader's attribution and license header are retained unchanged. The
license texts are included in `LICENSES/LGPL-3.0-or-later.txt` and
`LICENSES/GPL-3.0-or-later.txt`; the LGPL incorporates the GPL's terms.
The copies were obtained from the SPDX license-list-data project:

- https://github.com/spdx/license-list-data/blob/main/text/LGPL-3.0-or-later.txt
- https://github.com/spdx/license-list-data/blob/main/text/GPL-3.0-or-later.txt

Source distributions include this notice, both license texts and the shader.
Wheels include the shader beside the Python package and the notices under
their `.dist-info/licenses/` directory. The shader remains available
in source form in the wheel; it is not converted to an embedded binary.

## Wheel hosting

The upstream maintainer has authorized building and hosting this module in
https://github.com/AliceTeaParty/vapoursynth-api4-wheels. The hosting arrangement
does not introduce a blanket license for the Python source or replace the
shader's license. Refer to this repository for source and future changes.
