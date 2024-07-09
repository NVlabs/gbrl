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

Ensure that ``CUDA_HOME`` is set. Verify that NVCC exists by running the command

.. code-block:: console
   
   nvcc --version

And set ``CUDACXX`` to the location of NVCC.

.. note:: 

   CUDA installation via Anaconda may not install the full CUDAToolkit.  
   
   Make sure that Anaconda is up-to-date and run ``conda install cuda -c nvidia`` to install the full CUDAToolkit.

   Anaconda might still cause issues with CUDA. In such case, set ``CUDACXX`` and ``CUDA_HOME`` to the non-conda location. For example on Linux, the non-conda location of NVCC can be found by running ``which nvcc`` while Anaconda is deactivated.

For integration with Microsoft Visual Studio, copy the following files:

.. code-block:: console

   CUDA <cuda_version>.props
   CUDA <cuda_version>.targets
   CUDA <cuda_version>.xml
   Nvda.Build.CudaTasks.v<cuda_version>.dll
   cudart.lib


into ``<visual studio path>\BuildTools\MSBuild\Microsoft\VC\<visual studio version>\BuildCustomizations``.

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

After installation, set the necessary environment variables. For example:
   Setting enviornment variables can be cumbersome. Here are a few reference examples:

   Windows powershell:

   ..code-block:: console
   
      # PowerShell script to set environment variables
      $env:CMAKE_GENERATOR = "Visual Studio 16 2019" # change version
      $env:CMAKE_GENERATOR_PLATFORM = "x64"
      $env:VS160COMNTOOLS =" C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\Tools"  # change to the correct path
      $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
      $env:CUDA_PATH_V12_4 = $env:CUDA_PATH # Change to the relevant cuda version
      $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
      $env:INCLUDE = "$env:CUDA_PATH\include;$env:INCLUDE"
      $env:LIB = "$env:CUDA_PATH\lib\x64;$env:LIB"
      $env:CUDACXX = "$env:CUDA_PATH\bin\nvcc.exe"

   macOS

   .. code-block:: bash

      export PATH="$(brew --prefix llvm)/bin:$PATH"
      export LDFLAGS="-L$(brew --prefix libomp)/lib -L$(brew --prefix llvm)/lib -L$(brew --prefix llvm)/lib/c++ -Wl,-rpath,$(brew --prefix llvm)/lib/c++"
      export CPPFLAGS="-I$(brew --prefix libomp)/include -I$(brew --prefix llvm)/include"
      export CC="$(brew --prefix llvm)/bin/clang"
      export CXX="$(brew --prefix llvm)/bin/clang++"
      export DYLD_LIBRARY_PATH="$(brew --prefix llvm)/lib:$(brew --prefix libomp)/lib" 

   
   Linux 

   .. code-block:: bash

   export PATH="$(brew --prefix llvm)/bin:$PATH"
   export LDFLAGS="-L$(brew --prefix libomp)/lib -L$(brew --prefix llvm)/lib -L$(brew --prefix llvm)/lib/c++ -Wl,-rpath,$(brew --prefix llvm)/lib/c++"
   export CPPFLAGS="-I$(brew --prefix libomp)/include -I$(brew --prefix llvm)/include"
   export CC="$(brew --prefix llvm)/bin/clang"
   export CXX="$(brew --prefix llvm)/bin/clang++"
   export DYLD_LIBRARY_PATH="$(brew --prefix llvm)/lib:$(brew --prefix libomp)/lib" 


After installing GBRL, verify that CUDA is enabled:
 
.. code-block:: python

   import gbrl

   print(gbrl.cuda_available())


Graphviz (optional)
~~~~~~~~~~~~~~~~~~~

To enable tree visualization, ensure  `Graphviz <https://graphviz.org/download//>`__  and its development headers are installed before compiling.





 
