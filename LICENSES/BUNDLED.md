# What NitroLearn's release carries from others, and under which licence

NitroLearn's GitHub release keeps its own copies of what an install would otherwise download from other
people's servers (`tools/bundles.py` makes them; the release's `BUNDLED-NOTICES.md` lists them for each
version). Each is passed on unchanged, and each licence below allows that.

| Copy on the release | What it is, where it comes from | Licence | Text |
|---|---|---|---|
| `uv-<version>-py3-none-*.whl` | uv by Astral, the wheels from PyPI | MIT or Apache-2.0 | `uv-MIT.txt`, `Apache-2.0.txt` (both also inside each wheel) |
| `python-<version>-<platform>.tar.gz` | Python 3.11 as python-build-standalone (Astral) builds it | Python Software Foundation License, with the licences of the libraries built in | `Python-PSF.txt`, and `python-build-standalone.md` |
| `rocketsim-recordings.zip` | RocketSim's five game recordings, from its repository | MIT | `RocketSim-MIT.txt` (also inside the archive) |
| `nitrolearn-crates-<version>.tar.gz` | the source code of every Rust crate the builds use | each crate's own: MIT, Apache-2.0, BSD, Zlib, ISC, Unicode, CC0 and the like | in each crate's folder; the list is `VENDORED-CRATES.md` inside |
| `nitrolearn-wheels-<version>-<platform>.zip` | every Python package an install takes from PyPI, except PyTorch | each package's own: MIT, BSD, Apache-2.0, PSF, MPL-2.0 (certifi) and the like | in each wheel's `.dist-info`; the list is `NOTICES.md` inside |
| `NitroLearn-Setup.exe` | the setup wizard, made with Inno Setup | Inno Setup's licence (any use; its notices kept) | `Inno-Setup.txt` |

The release workflow refuses a crate or a package whose licence does not allow passing it on.

Not carried, and why:
- **PyTorch**: its graphics-card builds are bigger than one file on a GitHub release may be (2 GiB) on
  Windows, and on Linux they need NVIDIA's own CUDA libraries, several GB under NVIDIA's terms. The installer
  downloads PyTorch from PyTorch's own index.
- **The arena meshes** (collision meshes): they are Rocket League's game data. NitroLearn never hosts them;
  the installer downloads them from RLGym's public package on PyPI, checked by hash.
- **Qt** (the dashboard's window on a Linux desktop only): GPL/LGPL software, installed from PyPI.
- **The NVIDIA driver and RLBot v5**: separate programs, from their makers.
- **Rust** (only for building from the code and for your own Rust rewards): from rustup, Rust's own installer.
