"""Packaging entry point for the `sgl-deep-gemm` wheel.

`pyproject.toml` supplies the metadata (name, version, package-data); this file
exists to fix the *platform tag*.

The wheel ships a prebuilt `deep_gemm/_C.so`, which is a native AArch64/x86-64
ELF object linked against a specific torch. Left to itself, setuptools treats
every file as package-data, concludes the distribution is pure Python, and
stamps the wheel `py3-none-any`. That tag claims the wheel runs everywhere on
every interpreter -- false in both dimensions, and it lets pip install a Thor
build onto an x86 host, or onto a torch with a different `libc10_cuda` ABI.

Declaring `_C.so` through `ext_modules` makes setuptools advertise the real
platform. The extension is prebuilt rather than compiled here, so the build is
not repeated: `build_ext` is told the artefact already exists.
"""
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

EXTENSION = 'deep_gemm._C'


class PrebuiltExtension(build_ext):
    """Adopt the already-compiled `deep_gemm/_C.so` instead of compiling one."""

    def run(self):
        pass  # nothing to compile: build_sgl_deep_gemm.sh produced the artefact

    def build_extension(self, ext):
        pass

    def get_ext_fullpath(self, ext_name):
        path = Path(self.build_lib) / Path(*ext_name.split('.'))
        return str(path.with_suffix('.so'))

    def get_output_mapping(self):
        source = Path(self.build_lib) / 'deep_gemm' / '_C.so'
        return {str(source): str(source)}


setup(
    ext_modules=[Extension(EXTENSION, sources=[])],
    cmdclass={'build_ext': PrebuiltExtension},
)
