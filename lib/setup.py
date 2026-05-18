"""Build Cython extensions for the sdtw package."""
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

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
    ),
]

setup(
    name="sdtw",
    packages=find_packages(),
    ext_modules=cythonize(extensions, language_level=3),
    install_requires=["numpy", "scipy", "scikit-learn"],
)
