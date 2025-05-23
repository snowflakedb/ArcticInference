# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


class CMakeExtension(Extension):

    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):

    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG",
                                   0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [
                item for item in os.environ["CMAKE_ARGS"].split(" ") if item
            ]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [
            f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"
        ]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator
                                for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [
                    "-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))
                ]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(["cmake", ext.sourcedir, *cmake_args],
                       cwd=build_temp,
                       check=True)
        subprocess.run(["cmake", "--build", ".", *build_args],
                       cwd=build_temp,
                       check=True)


def build_custom_kernels():
    compiled_lib_filename = "libCustomOps.so"
    c_source_base_subdir = "csrc"
    custom_ops_module_subdir = "custom_ops"
    build_artefacts_subdir = "build"
    python_package_name = "arctic_inference"  # Your package directory name

    try:
        project_root_dir = Path(__file__).resolve().parent
    except NameError:
        project_root_dir = Path.cwd()

    cpp_custom_ops_source_dir = project_root_dir / c_source_base_subdir / custom_ops_module_subdir
    build_output_dir = cpp_custom_ops_source_dir / build_artefacts_subdir
    target_so_path_in_package_source = project_root_dir / python_package_name / compiled_lib_filename

    target_so_path_in_package_source.parent.mkdir(parents=True, exist_ok=True)
    build_output_dir.mkdir(parents=True, exist_ok=True)

    torch_cmake_prefix = torch.utils.cmake_prefix_path
    cmake_configure_command = [
        "cmake", f"-DTORCH_CMAKE_PREFIX_PATH={torch_cmake_prefix}", ".."
    ]
    subprocess.run(cmake_configure_command,
                   cwd=build_output_dir,
                   check=True,
                   capture_output=True)

    num_cpu_cores = os.cpu_count() or 1
    make_build_command = ["make", f"-j{num_cpu_cores}"]
    subprocess.run(make_build_command,
                   cwd=build_output_dir,
                   check=True,
                   capture_output=True)

    compiled_library_file_path = build_output_dir / compiled_lib_filename

    if not compiled_library_file_path.exists():
        raise FileNotFoundError(
            f"Compiled library {compiled_library_file_path.resolve()} not found after build."
        )

    if target_so_path_in_package_source.exists(
    ) or target_so_path_in_package_source.is_symlink():
        try:
            target_so_path_in_package_source.unlink(missing_ok=True)
        except TypeError:
            if target_so_path_in_package_source.exists(
            ) or target_so_path_in_package_source.is_symlink():
                target_so_path_in_package_source.unlink()
        except OSError as e:
            if target_so_path_in_package_source.is_dir():
                shutil.rmtree(target_so_path_in_package_source)
            else:
                raise OSError(
                    f"Error removing {target_so_path_in_package_source.resolve()}: {e}"
                ) from e

    shutil.copy2(compiled_library_file_path, target_so_path_in_package_source)

    return compiled_lib_filename


setup(
    ext_modules=[
        CMakeExtension("arctic_inference.common.suffix_cache._C",
                       "csrc/suffix_cache")
    ],
    cmdclass={"build_ext": CMakeBuild},
    package_data={"arctic_inference": [build_custom_kernels()]},
)
