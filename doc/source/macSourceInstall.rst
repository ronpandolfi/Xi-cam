OSX Source Install
==================

.. highlight:: bash
    :linenothreshold: 2

Preparing OSX
-------------

1. Install the Homebrew package manager::

    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)

2.  Install libraries with brew::

    brew install python openssl cmake homebrew/science/hdf5 libjpeg pkg-config libfreetype libpng cartr/qt4/qt

General source install
----------------------

1.  Clone the Xi-cam repository and enter it:
.. code-block:: bash
    git clone https://github.com/ronpandolfi/Xi-cam.git && cd Xi-cam
2.  Install the virtualenv python package and activate it:
.. code-block:: bash
    pip install virtualenv && source venv/bin/activate
3.  Create a virtual environment:
.. code-block:: bash
    virtualenv venv --system-site-packages
4.  Upgrade pip:
.. code-block:: bash
    pip install --upgrade pip
5.  Install numpy:
.. code-block:: bash
    pip install --ignore-installed numpy
6.  Install Xi-cam
.. code-block:: bash
    pip install .



NOTES
-----

- Do not create the virtual environment using PyCharm; this will use an internal python resulting in broken links to the global site-packages.
- Do not use the Pycharm terminal console; this console runs a unique shell which is missing path variables, resulting in failed installations of PySide etc.