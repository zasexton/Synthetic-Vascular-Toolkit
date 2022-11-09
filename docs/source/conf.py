# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0,os.path.abspath('.'))

project = 'SVCCO'
copyright = '2022, Zachary Sexton'
author = 'Zachary Sexton'
release = '0.5.52'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon",
              "sphinx.ext.autodoc",
              "sphinx.ext.viewcode"]

napoleon_google_docstring = False

templates_path = ['_templates']
exclude_patterns = []

root_doc = 'index'
language = 'python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'classic'
html_static_path = ['_static']
plotly_include_source = True
plotly_include_directive_source = True
