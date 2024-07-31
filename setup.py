import os
import sys
import subprocess
import platform
import shutil
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils import log
import sysconfig


class CMakeExtension(Extension):
    """Extension to integrate CMake build"""
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        print(self.sourcedir)

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

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = 'Debug' if self.cmake_verbose else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DPYTHON_INCLUDE_DIR=' + sysconfig.get_path('include'),
            '-DCMAKE_BUILD_TYPE=' + cfg,
        ]   
        if os.environ.get('COVERAGE', '0') == '1':
             cmake_args.append('-DCOVERAGE=ON')
        if sysconfig.get_config_var('LIBRARY') is not None:
            cmake_args.append('-DPYTHON_LIBRARY=' + sysconfig.get_config_var('LIBRARY'))
        if 'CC' in os.environ:
            cmake_args.append('-DCMAKE_C_COMPILER=' + os.environ['CC'])
        if 'CXX' in os.environ:
            cmake_args.append('-DCMAKE_CXX_COMPILER=' + os.environ['CXX'])
        build_args = ['--config', cfg]
        if self.cmake_verbose:
            cmake_args.append('-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON')
            build_args.append('--verbose')

        if ('CPU_ONLY' not in os.environ and platform.system() != 'Darwin') or ('CPU_ONLY' in os.environ and os.environ['CPU_ONLY'] != '1'):
            cmake_args.append('-DUSE_CUDA=ON')
            if 'CUDACXX' in os.environ:
                cmake_args.append('-DCMAKE_CUDA_COMPILER=' + os.environ['CUDACXX'])
        
        build_temp = self.build_temp
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        # Run cmake configuration
        self.run_subprocess(['cmake', ext.sourcedir] + cmake_args, build_temp)
        # Build the extension
        self.run_subprocess(['cmake', '--build', '.'] + build_args, build_temp)

        self.move_built_library(extdir)

    def run_subprocess(self, cmd, cwd):
        log.info('Running command: {}'.format(' '.join(cmd)))
        try:
            subprocess.check_call(cmd, cwd=cwd)
        except subprocess.CalledProcessError as e:
            log.error(f"Command {cmd} failed with error code {e.returncode}")
            log.error(e.output)
            raise

    def move_built_library(self, build_temp):
        built_objects = []
        for root, _, files in os.walk(build_temp):
            for file in files:
                if file.endswith(('.so', '.pyd', '.dll', '.dylib')):
                    built_objects.append(os.path.join(root, file))

        if not built_objects:
            raise RuntimeError(f"Cannot find built library in {build_temp}")
        for built_object in built_objects:
            dest_path = os.path.join(os.path.dirname(__file__), 'gbrl')
            log.info(f'Moving {built_object} to {dest_path}')
            self.copy_file(built_object, dest_path)

setup(
    name="gbrl",
    ext_modules=[CMakeExtension('gbrl/gbrl_cpp', sourcedir='.')],
    cmdclass=dict(build_ext=CMakeBuild),
    packages=find_packages(include=["gbrl"]),  # List of all packages to include
    include_package_data=True,
)
