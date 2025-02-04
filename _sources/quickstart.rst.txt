Quickstart
==========

Prerequisites
-------------

python3.7 or higher 

Installation
------------

To install GBRL via pip, use the following command:

.. code-block:: console
   
   pip install gbrl

To install the CPU-only version, even with a valid CUDA installation, set the CPU_ONLY environment variable:

.. code-block:: console

   export CPU_ONLY=1 
   pip install gbrl

Dependencies 
------------

macOS
~~~~~~

GBRL requires LLVM and OpenMP. These can be installed using Homebrew:

.. code-block:: console

   brew install libomp llvm


CUDA
~~~~ 

GBRL compiles CUDA and requires NVCC. 

Ensure that ``CUDA_PATH`` is set. Verify that NVCC exists by running the command

.. code-block:: console
   
   nvcc --version

And set ``CUDACXX`` to the location of NVCC.

.. note:: 

   CUDA installation via Anaconda may not install the full CUDAToolkit.  
   
   Make sure that Anaconda is up-to-date and run ``conda install cuda -c nvidia`` to install the full CUDAToolkit.

   Anaconda might still cause issues with CUDA. In such case, set ``CUDACXX`` and ``CUDA_PATH`` to the non-conda location. For example on Linux, the non-conda location of NVCC can be found by running ``which nvcc`` while Anaconda is deactivated.

.. note::

   For integration with Microsoft Visual Studio, copy the following files:

   .. code-block:: console

      CUDA <cuda_version>.props
      CUDA <cuda_version>.targets
      CUDA <cuda_version>.xml
      Nvda.Build.CudaTasks.v<cuda_version>.dll
      cudart.lib


   into ``<visual studio path>\BuildTools\MSBuild\Microsoft\VC\<visual studio version>\BuildCustomizations``.

After installing GBRL, verify that CUDA is enabled:
 
.. code-block:: python

   import gbrl

   print(gbrl.cuda_available())


Environment Variables
~~~~~~~~~~~~~~~~~~~~~

After installation, you may need to set environment variables to ensure that your system correctly locates all necessary files. Here are examples of setting these variables for different operating systems.

### Windows PowerShell using Visual Studio 16 2019 and CUDA 12.4

.. code-block:: console

   # PowerShell script to set environment variables
   $env:CMAKE_GENERATOR = "Visual Studio 16 2019" # Adjust version as necessary
   $env:VS160COMNTOOLS = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\Tools"  # Adjust path as necessary
   $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
   $env:CUDA_PATH_V12_4 = $env:CUDA_PATH  # Adjust CUDA version as necessary
   $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
   $env:INCLUDE = "$env:CUDA_PATH\include;$env:INCLUDE"
   $env:LIB = "$env:CUDA_PATH\lib\x64;$env:LIB"
   $env:CUDACXX = "$env:CUDA_PATH\bin\nvcc.exe"

### macOS

.. code-block:: bash

   export PATH="$(brew --prefix llvm)/bin:$PATH"
   export LDFLAGS="-L$(brew --prefix libomp)/lib -L$(brew --prefix llvm)/lib -L$(brew --prefix llvm)/lib/c++ -Wl,-rpath,$(brew --prefix llvm)/lib/c++"
   export CPPFLAGS="-I$(brew --prefix libomp)/include -I$(brew --prefix llvm)/include"
   export CC="$(brew --prefix llvm)/bin/clang"
   export CXX="$(brew --prefix llvm)/bin/clang++"
   export DYLD_LIBRARY_PATH="$(brew --prefix llvm)/lib:$(brew --prefix libomp)/lib"

### Linux

.. code-block:: bash

   export CUDA_HOME=/usr/local/cuda
   export PATH=$PATH:$CUDA_HOME/bin:/usr/local/bin
   export CUDACXX=$CUDA_HOME/bin/nvcc
   export CC=/usr/bin/gcc
   export CXX=/usr/bin/g++

Explanation:
- `CMAKE_GENERATOR` and `CMAKE_GENERATOR_PLATFORM` are used by CMake to specify the build system.
- `CUDA_PATH` (or `CUDA_HOME` for consistency with CUDA-related tools) specifies the location of the CUDA Toolkit.
- `PATH` is updated to include the CUDA binaries.
- `INCLUDE` and `LIB` are updated to include CUDA headers and libraries.
- `CUDACXX` specifies the location of, NVCC, the CUDA compiler.


Graphviz (optional)
~~~~~~~~~~~~~~~~~~~

To enable tree visualization, ensure  `Graphviz <https://graphviz.org/download//>`__  and its development headers are installed before compiling.





 
