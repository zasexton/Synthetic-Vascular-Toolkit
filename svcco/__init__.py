r'''
# Automated Vascularization & Fluid Simulation for Engineered Tissues

`svcco` falls under the open-source SimVascular project which seeks to
understand and quantify hemodynamics in health and disease.
This python module creates synthetic microvascular networks for given
perfusion territories, a common problem when creating engineered tissues.

## Installing SVCCO

Conda install
```bash
# Best practice to use an environment rather than installing at the base
# By default we will specify the python environment to 3.7
# however 3.8 and 3.9 are valid
conda create -n svcco-env python=3.7
conda activate svcco-env
# becuase conda allows pip installing
pip install svcco
```

pip install

```bash
pip install svcco
```
'''
#from __future__ import annotations

__version__ = "0.5.30"

import traceback
import warnings
from pathlib import Path
from typing import overload

#from svcco import implicit, collision, branch_addition, sv_interface, forest_utils, utils

#@overload
#def svcco(*modules: Path | str, output_directory: None = None,
#          format):
#    pass
from . import implicit
from . import collision
from . import branch_addition
from . import sv_interface
from . import forest_utils
from . import utils

from .utils.remeshing import remesh
from .utils.gcode import gcode
from .utils.fluid_analysis import reynolds, wss
from .utils.gui import gui_helper

from .tree import tree, forest, perfusion_territory, get
from .implicit.implicit import surface
from .implicit.tests.bumpy_sphere import bumpy_sphere
from .implicit.tests.heart import *
from .implicit.visualize.visualize import plot_volume

#from .sv_interface.ROM import centerlines, generate_1d_mesh, io_1d, mesh, models, parameters
#else:
#    implicit = _ModuleProxy('implicit')
#    collision = _ModuleProxy('collision')
#    branch_addition = _ModuleProxy('branch_addition')
#    sv_interface = _ModuleProxy('sv_interface')
