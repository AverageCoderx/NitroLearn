# The Python builds NitroLearn's release carries

The release's `python-<version>-<platform>.tar.gz` files are, unchanged, the "install_only_stripped" builds
of Python made by python-build-standalone (<https://github.com/astral-sh/python-build-standalone>, Astral),
which is what uv installs anyway. Their checksums are pinned in `tools/bundles.py`, `install.ps1` and
`install.sh`, and uv checks them again against its own list.

- **Python itself**: the Python Software Foundation License; its full text, with the licences of the code
  included in Python, is `Python-PSF.txt` (the Windows build's `LICENSE.txt`, which adds the terms for the
  Microsoft C runtime that comes with it). Each build carries it as `python/LICENSE.txt` (Windows) or
  `python/lib/python3.11/LICENSE.txt` (Linux).
- **The libraries built into it**, all under licences that allow passing them on: OpenSSL (Apache-2.0),
  libffi (MIT), zlib (Zlib), bzip2 (BSD-style), XZ Utils / liblzma (0BSD), SQLite (public domain), Tcl/Tk
  (BSD-style), mpdecimal (BSD-2-Clause), Expat (MIT); on Linux also libedit (BSD-3-Clause, used instead of
  GNU readline), ncurses (MIT-style), libuuid (BSD-3-Clause) and the X11 libraries Tk uses (MIT).
- **python-build-standalone's own build scripts** (not in the files): Mozilla Public License 2.0.
