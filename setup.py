import os
import pathlib
import subprocess
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
        super().run()

    def build_cmake(self, ext):
        root = pathlib.Path(ext.sourcedir).resolve()     
        build_dir = root / "build"                      
        build_dir.mkdir(parents=True, exist_ok=True)

        extdir = pathlib.Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir.parent.mkdir(parents=True, exist_ok=True)

        cfg = "Debug" if self.debug else "Release"

        print("\n▶ Conan install …")
        self.spawn([
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
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        self.spawn(cmake_cfg)

        print("▶ CMake build …")
        if not self.dry_run:
            self.spawn(["cmake", "--build", str(build_dir), "--config", cfg])


setup(
    name='capnhook_ml',
    version='0.1',
    author='ismaeelbashir03',
    ext_modules=[CMakeExtension('capnhook_ml', sourcedir='.')],
    cmdclass={
        'build_ext': build_ext,
    }
)