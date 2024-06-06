Quickstart
==========

Install GBRL via pip:

.. code-block:: console
   
   pip install gbrl

CPU only version despite having a valid CUDA installation by setting `CUDA_ONLY=1` as an environment variable:

.. code-block:: console
   export CPU_ONLY=1 
   pip install gbrl

Dependencies 
============ 

macOS
~~~~~~

GBRL is dependent on LLVM and OpenMP. 

These dependencies can be installed via Homebrew:

.. code-block:: console

   brew install libomp llvm


Once installed make sure that the appropriate environment variables are set:

.. code-block:: bash

   export PATH="$(brew --prefix llvm)/bin:$PATH"
   export LDFLAGS="-L$(brew --prefix libomp)/lib -L$(brew --prefix llvm)/lib -L$(brew --prefix llvm)/lib/c++ -Wl,-rpath,$(brew --prefix llvm)/lib/c++"
   export CPPFLAGS="-I$(brew --prefix libomp)/include -I$(brew --prefix llvm)/include"
   export CC="$(brew --prefix llvm)/bin/clang"
   export CXX="$(brew --prefix llvm)/bin/clang++"
   export DYLD_LIBRARY_PATH="$(brew --prefix llvm)/lib:$(brew --prefix libomp)/lib" 

CUDA
~~~~ 

Make sure that ``CUDA_HOME`` is set. 
For integration with Microsoft Visual Studio make sure to copy the following files:

.. code-block:: console
   
   CUDA <cuda_version>.props
   CUDA <cuda_version>.targets
   CUDA <cuda_version>.xml
   Nvda.Build.CudaTasks.v<cuda_version>.dll
   cudart.lib


into ``<visual studio path>\BuildTools\MSBuild\Microsoft\VC\v160\BuildCustomizations``.

Once GBRL is installed, verify that CUDA is enabled by running
 
.. code-block:: python
   # Verify that GPU is visible by running
   import gbrl

   print(gbrl.cuda_available())


Graphviz (optional)
~~~~~~~~~~~~~~~~~~~
For tree visualization make sure graphviz is installed before compilation. 


 
