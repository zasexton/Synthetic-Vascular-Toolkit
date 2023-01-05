#!/usr/bin/env python3
"""
This module provides a flexible package for autogenerating vascular networks.

It provides:
  - constrained constructive optimization routines for vasuclar construction
  - implicit surface/volume handling for anatomic and engineered shapes
  - integration with open-source vascular simulation software SimVascular
  - gCode and Stl 3D printing file creation

"""
import re
from pathlib import Path
from setuptools import setup, find_packages
import os
import glob
#from tqdm import tqdm
#from importlib_metadata import version

here = Path(__file__).parent

long_description = (here / "README.md").read_text("utf8")

#VERSION = re.search(
#    r'__version__ = "(.+?)"', (here / "svcco" / "__init__.py").read_text("utf8")
#).group(1)
VERSION = '0.6.21'

CLASSIFIERS = ['Intended Audience :: Science/Research',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8',
               'Programming Language :: Python :: 3.9',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX :: Linux',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS']

ALL_FILE_LIST = glob.glob("./**/*.py",recursive=True)
ALL_FILE_LIST = [file for file in ALL_FILE_LIST if "__init__" not in file and "CCO_" not in file]
MODULES = []
"""
for filename in tqdm(ALL_FILE_LIST):
    file = open(filename,'r')
    lines = [line for line in file.readlines() if 'import' in line and line[0] != '#']
    for line in lines:
        words = line.split(" ")
        if words[0] == "import" or words[0] == "from":
            if words[1][0] != ".":
                mod = words[1].split(".")[0].replace('\n','')
                if mod not in MODULES:
                    MODULES.append(mod)
    file.close()

print(MODULES)
INSTALL_REQUIREMENTS = []
for mod in MODULES:
    if mod == 'svcco' or mod == 'importlib_metadata' or mod == 'pip' or mod == 'svZeroDSolver' or mod == 'svzerodsolver':
        continue
    try:
        install_version = version(mod)
        INSTALL_REQUIREMENTS.append(mod+'>='+install_version)
    except:
        pass
print(INSTALL_REQUIREMENTS)
"""
# temporarily change pyvista version to 0.34.2 for colab integration, original version = 0.33.3
INSTALL_REQUIREMENTS = ['pyvista==0.34.2', 'numpy==1.21.0', 'setuptools>=62.3.3', 'tqdm>=4.63.0',
                        'matplotlib>=3.3.4', 'vtk>=9.0.0', 'binarytree>=6.5.1', 'memory_profiler>=0.60.0',
                        'networkx>=2.5.1', 'pydotplus>=2.0.2', 'numba>=0.55.1', 'seaborn>=0.11.2',
                        'tetgen>=0.6.0', 'scipy>=1.6.0', 'pymeshfix>=0.15.0', 'geomdl>=5.3.1',
                        'sympy>=1.9', 'plotly>=5.1.0','scikit-image>=0.16.1','meshio==5.3.4','importlib_metadata',
                        'nlopt==2.7.0']

PACKAGES = find_packages(include=["svcco","svcco.*"]) #['svcco']+['svcco.'+ pkg for pkg in find_packages('svcco')]
print(PACKAGES)
OPTIONS  = None
"""
INSTALL_REQUIREMENTS = ['numpy>=1.16.0',
                        'numba>=0.53.0',
                        'scipy>=1.5.0',
                        'pyvista==0.30.1',
                        'matplotlib>=3.3.4',
                        'seaborn>=0.11.0',
                        'sympy>=1.8.0',
                        'tqdm>=4.61.0',
                        'vtk>=9.0.0',
                        'scikit-image>=0.16.1',
                        'pandas>=1.3.0',
                        'plotly>=5.1.0',
                        'pymeshfix>=0.15.0',
                        'tetgen>=0.6.0']
"""
PROJECT_URLS = {}
#PACKAGE_DATA = {}
setup_info = dict(
    name='svcco',
    version=VERSION,
    author='Zachary Sexton',
    author_email='zsexton@stanford.edu',
    url="https://zasexton.github.io/svcco/",
    description='Automated Vascular Generation and CFD Simulation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    python_requires='>=3.7',
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIREMENTS,
    packages=PACKAGES,
    include_package_data=True,
    zip_safe=False,
    package_data={'':['svcco/utils/remeshing/Linux/mmg*']}
    )

setup(**setup_info)
