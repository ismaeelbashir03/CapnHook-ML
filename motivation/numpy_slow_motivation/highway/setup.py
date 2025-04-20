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
        cwd = pathlib.Path().absolute()
        
        source_dir = pathlib.Path(ext.sourcedir or cwd)
        csrc_dir = source_dir / 'csrc'
        
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.parent.mkdir(parents=True, exist_ok=True)

        config = 'Debug' if self.debug else 'Release'
        
        print("Running conan install...")
        os.chdir(str(csrc_dir))
        self.spawn([
            'conan', 'install', '.', 
            '--output-folder=build', 
            '--build=missing', 
            '--profile=default'
        ])
        
        print("Configuring CMake...")
        build_dir = csrc_dir / 'build'
        build_dir.mkdir(exist_ok=True)
        os.chdir(str(build_dir))
        
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
            '-DCMAKE_BUILD_TYPE=' + config,
        ]
        
        self.spawn(['cmake', '--preset', 'conan-release'] + cmake_args + ['..'])
        
        print("Building with CMake...")
        if not self.dry_run:
            self.spawn(['cmake', '--build', '--preset', 'conan-release'])
        
        os.chdir(str(cwd))

setup(
    name='capnhook_ml_cpp',
    version='0.1',
    author='ismaeelbashir03',
    packages=[],
    ext_modules=[CMakeExtension('capnhook_ext', sourcedir='.')],
    cmdclass={
        'build_ext': build_ext,
    }
)