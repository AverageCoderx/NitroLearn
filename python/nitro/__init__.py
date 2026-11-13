"""NitroLearn: a fast Rocket League bot training framework."""

import os as _os
from importlib.metadata import version as _distribution_version

from nitro import folder_env as _folder_env

__version__ = _distribution_version("nitrolearn")
# nitro.cmd and tools/env.ps1 cannot tell whether NVIDIA's driver can read the folder's CUDA cache path, so
# every NitroLearn process makes it readable (or drops it) before PyTorch starts the driver.
_folder_env.native_safe(_os.environ)
