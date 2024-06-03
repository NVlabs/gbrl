
import os
import subprocess
from pathlib import Path

import platform

import pybind11
from pathlib import Path
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

DEBUG = False

system = platform.system()
os_arch = platform.machine()

# Function to check if CUDA is available
def find_cuda():
    CUDA_PATH = None
    if system == 'Darwin':
        return None
    if 'CUDA_PATH' in os.environ:
        CUDA_PATH = os.environ.get('CUDA_PATH', None)
    elif 'CUDA_HOME' in os.environ:
        CUDA_PATH = os.environ.get('CUDA_HOME', None)
    elif system == 'Linux':
        print("Could not find CUDA_PATH in environment variables. Defaulting to /usr/local/cuda")
        CUDA_PATH = "/usr/local/cuda"
    if not CUDA_PATH or not os.path.isdir(CUDA_PATH):
        print("Compiling for CPU only")
        return None
    print('CUDA PATH found compiling for cpu and gpu')
    return CUDA_PATH

def get_cuda_version(nvcc_path):
    try:
        output = subprocess.check_output([nvcc_path, '--version'], encoding='utf-8')
        version_line = [line for line in output.split('\n') if 'release' in line][0]
        # Example output line: "Cuda compilation tools, release 10.1, V10.1.243"
        version_number = version_line.split('release')[1].split(',')[0].strip().split(' ')[0]
        return version_number
    except subprocess.CalledProcessError as e:
        print("Failed to execute nvcc to determine CUDA version.")
        return None

def get_gencode_options(cuda_version):
    version_major = int(cuda_version.split('.')[0])
    gencode_options = [
        '-gencode=arch=compute_60,code=sm_60',
        '-gencode=arch=compute_70,code=sm_70',
    ]
    if version_major >= 10:
        # Add options for CUDA Toolkit 10.x and above
        gencode_options.extend([
            '-gencode=arch=compute_75,code=sm_75',
        ])
    if version_major >= 11:
        # Add options for CUDA Toolkit 11.x and above
        gencode_options.extend([
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
        ])
    if version_major >= 12:
        # Add options for CUDA Toolkit 11.x and above
        gencode_options.extend([
            '-gencode=arch=compute_90,code=sm_90',
        ])
    # Note: Adjust the conditions and gencode options based on actual CUDA Toolkit version support
    return gencode_options

# Function to check if Graphviz is available
def find_graphviz():
    for path in os.environ["PATH"].split(os.pathsep):
        graphviz_path = os.path.join(path, 'dot')
        if os.path.exists(graphviz_path):
            return True
    print('Could not find Graphviz -> compiling without it. Tree visualization will not be available')
    return False

cuda_home = find_cuda() if system != 'Darwin' else None
graphviz_available = find_graphviz() if system != 'Darwin' else None

# Define base directories as Path objects
base_dir = Path("gbrl/src/cpp")
cuda_dir = Path("gbrl/src/cuda")
source_files =  [

    "gbrl_binding.cpp", 
    "gbrl.cpp", 
    "types.cpp", 
    "optimizer.cpp", 
    "scheduler.cpp", 
    "node.cpp", 
    "utils.cpp", 
    "fitter.cpp", 
    "predictor.cpp", 
    "split_candidate_generator.cpp", 
    "loss.cpp", 
    "math_ops.cpp", 
]

source_files = [str(base_dir / src_file) for src_file in source_files]

include_paths = [
    str(pybind11.get_include()), 
    str(base_dir), 
]


extra_link_args = []
extra_compile_args=[]

if system == 'Linux':
    extra_link_args.append('-fopenmp' )
    extra_compile_args.extend([
    "-O3",
    '-fopenmp',
    '-std=c++14',
    "-Wall",
    "-Wpedantic",
    '-march=native',
    "-Wextra"]
    )
elif system == 'Darwin':
    # extra_link_args.append('-lomp')
    extra_link_args.append( "-L/usr/local/lib")
    extra_compile_args.extend([
    "-O3" ,
    "-Xpreprocessor",
    '-fopenmp',
    '-std=c++14',
    "-Wall",
    "-Wpedantic",
    "-Wextra"])
    include_paths.append("/usr/local/include")
elif system == 'Windows':
    extra_compile_args.extend(['/std:c++14', '/O2', '/W3'])


if DEBUG:
    extra_compile_args.append('-g')
# Define macros based on availability
define_macros = [('MODULE_NAME', 'gbrl_cpp')] # Add this line]
cuda_source_files = []
if cuda_home:
    define_macros.append(('USE_CUDA', None))
    cuda_src_files = ['cuda_predictor.cu', 
                              'cuda_fitter.cu',
                              'cuda_loss.cu',
                              'cuda_types.cu',
                              'cuda_utils.cu',
                              'cuda_preprocess.cu']

    cuda_source_files.extend([str(cuda_dir / src_file) for src_file in cuda_src_files])
    if system == 'Linux':
        extra_link_args.extend([
        f'-L{cuda_home}/lib64',
        '-lcudart'])
    elif system == 'Windows':
        cuda_lib_path = os.path.join(cuda_home, 'lib', 'x64')
        extra_link_args.extend(['/LIBPATH:' + cuda_lib_path, 'cudart.lib'])
    cuda_include_path = os.path.join(cuda_home, 'include')
    include_paths.append(str(cuda_dir))
    include_paths.append(cuda_include_path)
    
    
if DEBUG:
    define_macros.append(('DEBUG', None))
if graphviz_available:
    define_macros.append(('USE_GRAPHVIZ', None))
    if system == 'Linux':
        include_paths.append("/usr/include/graphviz")
    elif system == 'Darwin':
        graphviz_base_path = subprocess.getoutput("brew --prefix graphviz")
        include_paths.append(os.path.join(graphviz_base_path, "include", "graphviz"))
    extra_link_args.extend(["-lgvc", "-lcgraph"])

# Custom build_ext subclass
class CustomBuildExt(build_ext):
    def build_extensions(self):

        if system == 'Darwin':
            llvm_path = subprocess.getoutput("brew --prefix llvm")
            # os_arch = platform.machine()
            # Configure compiler paths
            os.environ["CC"] = f"{llvm_path}/bin/clang"
            os.environ["CXX"] = f"{llvm_path}/bin/clang++"

            # Initially configure LDFLAGS and CPPFLAGS with LLVM paths
            os.environ["LDFLAGS"] = (f"{os.getenv('LDFLAGS', '')} "
                                     f"-L{llvm_path}/lib -L{llvm_path}/lib/c++ "
                                     f"-Wl,-rpath,{llvm_path}/lib "
                                     f"-Wl,-rpath,{llvm_path}/lib/c++ -lomp")
            os.environ["CPPFLAGS"] = f"{os.getenv('CPPFLAGS', '')} -I{llvm_path}/include"
            #
        # Compile CUDA code if available
        if cuda_home and cuda_source_files:
            self.compile_cuda()
        super().build_extensions()

    def compile_cuda(self):
        nvcc_path = cuda_home + '/bin/nvcc'  # NVCC path
        cuda_version = get_cuda_version(nvcc_path)
        gencode = get_gencode_options(cuda_version)
        nvcc_compile_args = []
        nvcc_compile_args.extend(gencode)
        
        if system == 'Linux':
            nvcc_compile_args.append('--compiler-options')
            nvcc_compile_args.append("'-fPIC'")
        nvcc_compile_args.extend([
            "--extended-lambda",
            "-O3",
            '-I' + str(pybind11.get_include()),
            f'-I{str(base_dir)}',
            f'-I{str(cuda_dir)}',
        ])
        if system == 'Windows':
            nvcc_compile_args.append('-Xcompiler=/MD')

        if DEBUG:
            nvcc_compile_args.append('-G')
            nvcc_compile_args.append('-DDEBUG')

        # Compile each .cu file
        build_dir = self.build_temp
        Path(build_dir).mkdir( parents=True, exist_ok=True )
        # for source in ext.sources:
        for source in cuda_source_files:
            if source.endswith('.cu'):
                target = source.replace('.cu', '.o')
                target = os.path.join(build_dir, target)  # Place .o file in the specified build directory
                Path(os.path.dirname(target)).mkdir(parents=True, exist_ok=True)
                command = [nvcc_path] + nvcc_compile_args + ['-c', source, '-o', target]
                print(' '.join(command))
                subprocess.check_call(command)
        # Manually adding the compiled CUDA object files to be linked
        for idx, _ in enumerate(self.extensions):
            cuda_extras = [os.path.join(build_dir, f.replace('.cu', '.o')) for f in cuda_source_files]
            self.extensions[idx].extra_objects.extend(cuda_extras)


setup(
    name="gbrl",
    packages=find_packages(include=["gbrl", "gbrl.*"], exclude=("tests*",)),
    package_data={
        'gbrl_cpp': ['*.so', '*.pyd'] # Include SO/PYD files
    },
    ext_modules=[
        Extension(
            "gbrl.gbrl_cpp",
            source_files,
            extra_compile_args=extra_compile_args, 
            extra_link_args=extra_link_args,
            include_dirs=include_paths,
            language='c++',
            define_macros=define_macros
        ),
    ],
    cmdclass={
        'build_ext': CustomBuildExt,
    }
)
