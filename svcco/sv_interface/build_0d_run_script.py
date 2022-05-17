run_0d_script="""import os
import sys
sys.path.append('{}')
from svZeroDSolver import svzerodsolver
svzerodsolver.solver.set_up_and_run_0d_simulation('{}')
"""
