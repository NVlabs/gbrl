import os
import sys
import subprocess
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils import log

class CMakeExtension(Extension):
    """Extension to integrate CMake build"""
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    """Build extension using CMake"""
    user_options = build_ext.user_options + [
        ('cmake-verbose', None, 'Enable verbose output from CMake'),
    ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.cmake_verbose = False

    def finalize_options(self):
        build_ext.finalize_options(self)
        self.cmake_verbose = os.getenv('DEBUG', '0') == '1'

    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = 'Debug' if self.cmake_verbose else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCMAKE_BUILD_TYPE=' + cfg
        ]

        build_args = ['--config', cfg]
        if self.cmake_verbose:
            cmake_args.append('-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON')
            build_args.append('--verbose')

        if 'CPU_ONLY' not in os.environ or os.environ('CPU_ONLY', '1'):
            print("found cuda_home")
            cmake_args.append('-DUSE_CUDA=ON')
        
        build_temp = self.build_temp
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        # Run cmake configuration
        self.run_subprocess(['cmake', ext.sourcedir] + cmake_args, build_temp)
        # Build the extension
        self.run_subprocess(['cmake', '--build', '.'] + build_args, build_temp)

    def run_subprocess(self, cmd, cwd):
        log.info('Running command: {}'.format(' '.join(cmd)))
        try:
            subprocess.check_call(cmd, cwd=cwd)
        except subprocess.CalledProcessError as e:
            log.error(f"Command {cmd} failed with error code {e.returncode}")
            log.error(e.output)
            raise

    def move_built_library(self, build_temp, extdir):
        built_objects = []
        for root, _, files in os.walk(build_temp):
            for file in files:
                if file.endswith(('.so', '.pyd', '.dll', '.dylib')):
                    built_objects.append(os.path.join(root, file))

        if not built_objects:
            raise RuntimeError(f"Cannot find built library in {build_temp}")

        for built_object in built_objects:
            dest_path = os.path.join(extdir, os.path.basename(built_object))
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            log.info(f'Moving {built_object} to {dest_path}')
            self.copy_file(built_object, dest_path)

setup(
    name="gbrl",
    ext_modules=[CMakeExtension('gbrl/gbrl_cpp', sourcedir='.')],
    cmdclass=dict(build_ext=CMakeBuild),
    language='c++',
    # packages=setuptools.find_packages(exclude=["gbrl.src.cpp", "gbrl.src.cuda"]),
    packages=setuptools.find_packages(),
)
