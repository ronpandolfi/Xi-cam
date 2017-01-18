OSX Source Install
==================

.. highlight:: bash
    :linenothreshold: 2

Preparing OSX
-------------

1. Install Command Line Tools:

.. code-block:: bash

    xcode-select --install

1. Install the Homebrew package manager:

.. code-block:: bash

    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)

2.  Install libraries with brew:

.. code-block:: bash

    brew install python openssl cmake homebrew/science/hdf5 libjpeg pkg-config freetype libpng cartr/qt4/qt cartr/qt4/pyside cartr/qt4/shiboken

General source install
----------------------

1.  Clone the Xi-cam repository and enter it:

.. code-block:: bash

    git clone https://github.com/ronpandolfi/Xi-cam.git && cd Xi-cam

2.  Install the virtualenv python package:

.. code-block:: bash

    pip install virtualenv

3.  Create a virtual environment and activate it:

.. code-block:: bash

    virtualenv venv --system-site-packages && source venv/bin/activate

4.  Upgrade pip:

.. code-block:: bash

    pip install --upgrade pip

5.  Install numpy:

.. code-block:: bash

    pip install --ignore-installed numpy

6.  Install Xi-cam:

.. code-block:: bash

    pip install .

NOTES
-----

- Do not create the virtual environment using PyCharm; this will use an internal python resulting in broken links to the global site-packages.
- Do not use the Pycharm terminal console; this console runs a unique shell which is missing path variables, resulting in failed installations of PySide etc.