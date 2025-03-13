##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html 
#
##############################################################################
__version__ = "1.0.12"

import importlib.util
import os
import platform
import sys

_loaded_cpp_module = None

def load_cpp_module():
    global _loaded_cpp_module
    module_name = "gbrl_cpp"
    python_version = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    python_version_short = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if platform.system() == "Windows":
        ext = ".pyd"
    elif platform.system() == "Darwin":  # macOS
        ext = ".dylib"
    else:  # Assume Linux/Unix
        ext = ".so"
    possible_paths = [
        os.path.join(os.path.dirname(__file__)),  # Current directory
        os.path.join(os.path.dirname(__file__), "Release"),  # Release folder
    ]
    for dir_path  in possible_paths:
        if os.path.exists(dir_path):
        # Scan for files that match the module name and extension
            for file_name in os.listdir(dir_path):
                if file_name.startswith(module_name) and file_name.endswith(ext) and (python_version in file_name or python_version_short in file_name):
                    # Dynamically load the matching shared library
                    file_path = os.path.join(dir_path, file_name)
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    _loaded_cpp_module = module = module
                    return module
                
    if platform.system() == "Darwin":  # check for .so on Darwin
        ext = ".so"
        for dir_path  in possible_paths:
            if os.path.exists(dir_path):
            # Scan for files that match the module name and extension
                for file_name in os.listdir(dir_path):
                    if file_name.startswith(module_name) and file_name.endswith(ext) and (python_version in file_name or python_version_short in file_name):
                        # Dynamically load the matching shared library
                        file_path = os.path.join(dir_path, file_name)
                        spec = importlib.util.spec_from_file_location(module_name, file_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        _loaded_cpp_module = module = module
                        return module
    raise ImportError(f"Could not find {module_name}{ext} in any of the expected locations: {possible_paths}")


# Load the C++ module dynamically
_gbrl_cpp_module = load_cpp_module()

# Create a global alias for the GBRL class
GBRL_CPP = _gbrl_cpp_module.GBRL

cuda_available = GBRL_CPP.cuda_available

