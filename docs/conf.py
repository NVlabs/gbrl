# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os 

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))) 

from unittest.mock import MagicMock
class Mock(MagicMock):
    __all__ = []

    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

sys.modules['gbrl.gbrl_cpp'] = Mock()
# List of modules to mock
MOCK_MODULES = ['numpy', 'torch', 'gymnasium', 'sklearn']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
project = 'GBRL'
copyright = '2024, NVIDIA Corporation'
author = 'Benjamin Fuhrer, Chen Tessler, Gal Dalal'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
]

nbsphinx_execute = 'never'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
