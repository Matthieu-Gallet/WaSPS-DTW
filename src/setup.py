from __future__ import print_function
import os.path
import sys
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)


DISTNAME = 'soft-dtw'
DESCRIPTION = "Python implementation of soft-DTW"
LONG_DESCRIPTION = open('README.rst').read()
MAINTAINER = 'Mathieu Blondel'
MAINTAINER_EMAIL = ''
URL = 'https://github.com/mblondel/soft-dtw/'
LICENSE = 'BSD-3-Clause'
DOWNLOAD_URL = 'https://github.com/mblondel/soft-dtw/'
VERSION = '0.1.dev0'

# Extensions Cython
extensions = [
    Extension(
        "sdtw.soft_dtw_fast",
        ["sdtw/soft_dtw_fast.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    ),
    Extension(
        "sdtw.wasserstein_fast",
        ["sdtw/wasserstein_fast.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ],
    ext_modules=cythonize(extensions, language_level=3),
    install_requires=['numpy', 'scipy', 'scikit-learn'],
)
