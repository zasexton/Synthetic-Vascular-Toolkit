"Locate Solver Files"
import os
import platform

def locate_0d_solver(windows_drive="C",linux_drive=os.sep):
    path = None
    if platform.system() == "Windows":
        for root, subdirs, files in os.walk(windows_drive+":"+os.sep):
            for d in subdirs:
                if d == 'svZeroDSolver' and '.git' not in root:
                    path = root
                    escape = True
                    break
                else:
                    escape = False
            if escape:
                break
    else:
        for root, subdirs, files in os.walk(linux_drive):
            for d in subdirs:
                if d == 'svZeroDSolver' and '.git' not in root:
                    path = root
                    escape = True
                    break
                else:
                    escape = False
            if escape:
                break
    return path

def locate_1d_solver(windows_drive="C",linux_drive=os.sep):
    path = None
    if platform.system() == "Windows":
        for root, subdirs, files in os.walk(windows_drive+":"+os.sep):
            for f in files:
                if f == 'svOneDSolver.exe':
                    path = root+os.sep+d+os.sep+f
                    escape = True
                    break
                else:
                    escape = False
            if escape:
                break
    else:
        for root, subdirs, files in os.walk(linux_drive):
            for f in files:
                if f == 'OneDSolver':
                    path = root+os.sep+d+os.sep+f
                    escape = True
                    break
                else:
                    escape = False
            if escape:
                break
    return path
