"""
Usage: python setup.py install
       python setup.py bdist_wheel
       python setup.py sdist bdist_egg
       twine upload dist/*
"""

# if __name__ == "__main__":
#     print 'You should run his with pip instead!\n Try this:\n\tpip install .\n...or...\n\tpip install xicam'
#     exit(0)

try:
    import numpy as np
except ImportError:
    print 'Error: Numpy is not installed. Install numpy first!'
    exit(1)

from setuptools import setup, find_packages
from codecs import open
from os import path
import glob
from numpy.distutils.core import Extension


print find_packages(exclude=['contrib', 'docs', 'tests'])

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:  # rst?
    long_description = f.read()

EXT = Extension(name='pipeline.cWarpImage',
                sources=['cext/cWarpImage.cc', 'cext/remesh.cc'],
                extra_compile_args=['-O3', '-ffast-math'],  # '-fopenmp',, '-I/opt/local/include'
                # extra_link_args=['-fopenmp'],
                include_dirs=[np.get_include()],
                )
setup(
    name='xicam',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='1.2.10',

    description='A synchrotron data analysis interface',
    long_description=long_description,

    # The project's main homepage.
    url='https://bitbucket.org/ronpandolfi/xicam',

    # Author details
    author='Ronald J Pandolfi',
    author_email='ronpandolfi@lbl.gov',

    # Choose your license
    license='BSD',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7'
    ],

    # What does your project relate to?
    keywords='synchrotron analysis x-ray scattering',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    #package_dir={'xicam.plugins':'xicam.plugins'},

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['scipy', 'Cython', 'pyFAI', 'fabio', 'h5py', 'Shiboken', 'PySide', 'pyqtgraph', 'QDarkStyle',
                      'nexusformat', 'Pillow', 'pyfits', 'PyOpenGL', 'PyYAML', 'qtconsole','tifffile','pysftp','requests','dask','distributed','appdirs','futures','scikit-image','imageio','vispy'],

    setup_requires=['numpy', 'cython'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        # 'dev': ['check-manifest'],
        # 'test': ['coverage'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
         'gui': ['gui'],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[('lib/python2.7/site-packages/gui', glob.glob('gui/*')),
                ('lib/python2.7/site-packages/yaml/tomography',glob.glob('yaml/tomography/*'))],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'gui_scripts': [
            'xicam=xicamlauncher.main:main',
        ],
    },

    ext_modules=[EXT],
    include_package_data=True
)
