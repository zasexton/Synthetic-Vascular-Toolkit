Installation
^^^^^^^^^^^^

**Installation via pip is the recommended method for all platforms.**
Manual methods are only recommended for advanced users and developers.

.. note::

    This package contains dependancies which rely on python versions 3.6, 3.7,
    3.8, or 3.9 in order function properly. Users should consider building a
    virtual python environment to accommodate python requirements without affecting
    other installed packages.

Install via pip
===============

The easiest method to install or upgrade SVCCO is using `pip <https://pip.pypa.io/en/stable/>`_.
The following commands will download and install the SVCCO module from the
Python Package Index (`PyPi <https://pypi.org/project/svcco/>`_).

.. code-block:: console

    $ pip install --user svcco

Upgrade a current version of SVCCO to the latest version

.. code-block:: console

    $ pip install svcco --upgrade

Install a specific version

.. code-block:: console

    $ pip install svcco==0.5.50

Manual Installation
===================

To manually install the module first clone the repository via ``git`` or download
a ZIP archive from the `project repository page <https://github.com/zasexton/Synthetic-Vascular-Toolkit>`_
on GitHub. The package will include a *setup.py* script which will automatically
handle dependancy fetching, linking, and building of required files to your Python
distribution's *site-packages* directory.

.. code-block:: console

    $ pip install --user .

To upgrade a current manual build, please pull the latest commits from the repository
via ``git pull --rebase`` and then re-execute the above command to rebuild the module.

.. note::

    Manual installation will only allow a user to build the latest version of the
    module release. If older versions are desired, users should refer to pip installation
    of a specific version.

Checking Installation
=====================

If you need to verify your current installation, you may print the current module
version within the python interpreter using ``svcco.__version__`` variable after
importing the module. The following blocks of code demonstrate this for Windows
and Unix command platforms, respectively.

For Windows::

    Windows PowerShell
    Copyright (C) Microsoft Corporation. All rights reserved.

    PS C:\> python
    Python 3.7.9 (tags/v3.7.9:13c94747c7, Aug 17 2020, 18:58:18) [MSC v.1900 64 bit (AMD64)] on win32
    Type "help", "copyright", "credits" or "license" for more information.
    >>>import svcco
    >>>svcco.__version__
    '0.5.30'
    >>>

For Unix::

    $python
    Python 3.7.13 (defualt, Oct 18 2022, 18:57:03)
    [GCC 11.2.0] :: on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>>import svcco
    >>>svcco.__version__
    '0.5.30'
    >>>

Installing into an Anaconda Environment
=======================================

Because of limitations in Python version that must be used with this module, it
may be beneficial for users to install SVCCO within a virtual python environment.
A common platform to for creating and managing python environments is with ``conda``
using the `miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
or full `Anaconda <https://www.anaconda.com/>`_ distributions. ``conda`` works
across Windows, macOS, and Linux platforms.

After installation of miniconda or Anaconda, users should have access to the
command line tools ``conda`` which can create new virtual python environments.
If already within the conda base environment, users will notice "(base)" in front
of the working directory in the command line as shown below:

.. code-block::

    (base) username:~$

If not in the base environment, use ``conda activate`` to enter the conda base
environment. To see more information about conda command line tools, please refer
to the `documentation <https://docs.conda.io/projects/conda/en/latest/commands.html>`_.
From here, users can create a new environment suitable for SVCCO
with the following command:

.. code-block::

    (base) username:~$conda create -n cco python=3.7

The following message will be prompted to allow additional packages
to be installed for setting up the virtual environment. If reasonable, proceed.

.. code-block::

    Collecting package metadata (current_repodata.json): done
    Solving environment: done


    ==> WARNING: A newer version of conda exists. <==
      current version: 4.10.3
      latest version: 22.9.0

    Please update conda by running

        $ conda update -n base -c defaults conda



    ## Package Plan ##

      environment location: /home/zack/anaconda3/envs/test

      added / updated specs:
        - python=3.7


    The following packages will be downloaded:

        package                    |            build
        ---------------------------|-----------------
        setuptools-65.5.0          |   py37h06a4308_0         1.1 MB
        ------------------------------------------------------------
                                               Total:         1.1 MB

    The following NEW packages will be INSTALLED:

      _libgcc_mutex      pkgs/main/linux-64::_libgcc_mutex-0.1-main
      _openmp_mutex      pkgs/main/linux-64::_openmp_mutex-5.1-1_gnu
      ca-certificates    pkgs/main/linux-64::ca-certificates-2022.10.11-h06a4308_0
      certifi            pkgs/main/linux-64::certifi-2022.9.24-py37h06a4308_0
      ld_impl_linux-64   pkgs/main/linux-64::ld_impl_linux-64-2.38-h1181459_1
      libffi             pkgs/main/linux-64::libffi-3.3-he6710b0_2
      libgcc-ng          pkgs/main/linux-64::libgcc-ng-11.2.0-h1234567_1
      libgomp            pkgs/main/linux-64::libgomp-11.2.0-h1234567_1
      libstdcxx-ng       pkgs/main/linux-64::libstdcxx-ng-11.2.0-h1234567_1
      ncurses            pkgs/main/linux-64::ncurses-6.3-h5eee18b_3
      openssl            pkgs/main/linux-64::openssl-1.1.1q-h7f8727e_0
      pip                pkgs/main/linux-64::pip-22.2.2-py37h06a4308_0
      python             pkgs/main/linux-64::python-3.7.13-haa1d7c7_1
      readline           pkgs/main/linux-64::readline-8.2-h5eee18b_0
      setuptools         pkgs/main/linux-64::setuptools-65.5.0-py37h06a4308_0
      sqlite             pkgs/main/linux-64::sqlite-3.39.3-h5082296_0
      tk                 pkgs/main/linux-64::tk-8.6.12-h1ccaba5_0
      wheel              pkgs/main/noarch::wheel-0.37.1-pyhd3eb1b0_0
      xz                 pkgs/main/linux-64::xz-5.2.6-h5eee18b_0
      zlib               pkgs/main/linux-64::zlib-1.2.13-h5eee18b_0


    Proceed ([y]/n)?

After building the new virtual environment, users should activate the environment
with the following code (if not already within the new environment):

.. code-block:
    (base) username:~$conda activate cco
    (cco) username:~$

To verify that the command line is within the newly created environment, simply
check the name of the environment within parenthesis to the far left of the command
line where "(base)" was located. Now the environment is ready to install SVCCO.
This can be installed through the regular pip process within this new environment.

.. code-block::

    (cco) username:~$ pip install svcco
