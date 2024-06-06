Quickstart
==========

Install GBRL via pip:

.. code-block:: console
   
   pip install gbrl

CPU only version is installed with the following command:

.. code-block:: console

   CPU_ONLY=1 pip install gbrl

Dependencies 
============ 

MAC OS
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

.. code-block:: python
   # Verify that GPU is visible by running
   import gbrl

   print(gbrl.cuda_available())


Graphviz
~~~~~~~~

*OPTIONAL*  
For tree visualization make sure graphviz is installed before compilation. 