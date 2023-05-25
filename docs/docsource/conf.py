import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Nova Server'
copyright = '2023, Dominik Schiller'
author = 'Dominik Schiller'
release = '0.1.3'


import pytorch_sphinx_theme

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'myst_parser'

]
source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'press'
#html_static_path = ['_static']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_default_options = {
    'undoc-members': True,
    #'special-members': True
}

autosummary_mock_imports = [
    'hcai_datasets.hcai_affectnet',
    'hcai_datasets.hcai_audioset',
    'hcai_datasets.hcai_ckplus',
    'hcai_datasets.hcai_faces',
    'hcai_datasets.hcai_is2021_ess',
    'hcai_datasets.hcai_librispeech'
]