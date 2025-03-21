# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_PATH) 
sys.path.insert(0, ROOT_PATH + '/gbrl')

with open(ROOT_PATH + '/gbrl/__init__.py') as file_handler:
    __version__ = file_handler.readlines()[8].split('"')[1]

from unittest.mock import MagicMock


class Mock(MagicMock):
    __all__ = []

    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


sys.modules['gbrl.gbrl_cpp'] = Mock()
project = 'GBRL'
copyright = '2024, NVIDIA Corporation'
author = 'Benjamin Fuhrer, Chen Tessler, Gal Dalal'
release = __version__
version = "master (" + __version__ + " )"

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
    'sphinx.ext.githubpages',
    'sphinx.ext.ifconfig',
]

nbsphinx_execute = 'never'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []
# html_static_path = ['_static']

html_context = {
    "display_github": True,
    "version": version
}

html_theme_options = {
    "language_selector": True,
    "version_selector": True,
}
