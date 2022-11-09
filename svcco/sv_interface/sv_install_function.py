"""
Function for installing extra external packages from within the
sv python interpreter
"""

install_function="""
from pip._internal import main as pipmain
def install(packagename,version=None):
    try:
        if version is None:
            pipmain(['install',packagename])
        else:
            pipmain(['install',packagename+'=='+version])
    except:
        print('package installation failed')
"""

uninstall_function="""
def uninstall(packagename):
    try:
        pipmain(['uninstall',packagename])
    except:
        print('package uninstallation failed')
"""
