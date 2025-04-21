import os
import pathlib
import subprocess
import platform, shutil, subprocess, pathlib
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class build_ext(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        if platform.system() != "Windows":
            super().run()

    def build_extension(self, ext):        # skip MSVC’s empty‑objects link
        if platform.system() == "Windows":
            return
        return super().build_extension(ext)

    def build_cmake(self, ext):
        root = pathlib.Path(ext.sourcedir).resolve()     
        build_dir = root / "build"                      
        build_dir.mkdir(parents=True, exist_ok=True)

        extdir = pathlib.Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir.parent.mkdir(parents=True, exist_ok=True)

        cfg = "Debug" if self.debug else "Release"

        print("\n▶ Conan install …")
        subprocess.check_call([
            "conan", "install", str(root),
            "--output-folder", str(build_dir),
            "--build=missing", "--profile=default"
        ])

        toolchain = build_dir / "conan_toolchain.cmake"

        print("▶ CMake configure …")
        cmake_cfg = [
            "cmake",
            "-S", str(root),
            "-B", str(build_dir),
            f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir.parent}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir.parent}",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        subprocess.check_call(cmake_cfg)

        if platform.system() == "Windows":
            if not (build_dir / "CMakeCache.txt").exists():
                # honour an env override, else default to VS 2022
                gen = os.getenv("CMAKE_GENERATOR", "Visual Studio 17 2022")
                cmake_cfg += ["-G", gen, "-A", "x64"]     # arch explicit
        subprocess.check_call(cmake_cfg)

        print("▶ CMake build …")
        if not self.dry_run:
            subprocess.check_call(["cmake", "--build", str(build_dir),
                               "--config", cfg])

setup(
    name='capnhook_ml',
    version='0.1',
    author='ismaeelbashir03',
    include_package_data=True,
    ext_modules=[CMakeExtension('capnhook_ml', sourcedir='.')],
    cmdclass={
        'build_ext': build_ext,
    }
)