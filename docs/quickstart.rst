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


After installation, set the necessary environment variables:

.. code-block:: bash

   export PATH="$(brew --prefix llvm)/bin:$PATH"
   export LDFLAGS="-L$(brew --prefix libomp)/lib -L$(brew --prefix llvm)/lib -L$(brew --prefix llvm)/lib/c++ -Wl,-rpath,$(brew --prefix llvm)/lib/c++"
   export CPPFLAGS="-I$(brew --prefix libomp)/include -I$(brew --prefix llvm)/include"
   export CC="$(brew --prefix llvm)/bin/clang"
   export CXX="$(brew --prefix llvm)/bin/clang++"
   export DYLD_LIBRARY_PATH="$(brew --prefix llvm)/lib:$(brew --prefix libomp)/lib" 

CUDA
~~~~ 

GBRL compiles CUDA and requires NVCC. 

Ensure that ``CUDA_HOME`` is set. Verify that NVCC exists by running the command

.. code-block:: console
   
   nvcc --version

And set ``CUDACXX`` to the location of NVCC.

.. note:: 

   CUDA installation via anaconda may not install the full CUDAToolkit.  

   Run ``conda install cuda -c nvidia`` to install the full CUDAToolkit.

For integration with Microsoft Visual Studio, copy the following files:

.. code-block:: console

   CUDA <cuda_version>.props
   CUDA <cuda_version>.targets
   CUDA <cuda_version>.xml
   Nvda.Build.CudaTasks.v<cuda_version>.dll
   cudart.lib


into ``<visual studio path>\BuildTools\MSBuild\Microsoft\VC\v160\BuildCustomizations``.

After installing GBRL, verify that CUDA is enabled:
 
.. code-block:: python

   import gbrl

   print(gbrl.cuda_available())


Graphviz (optional)
~~~~~~~~~~~~~~~~~~~

To enable tree visualization, ensure Graphviz is installed before compiling.


 
