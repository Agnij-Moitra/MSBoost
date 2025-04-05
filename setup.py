from setuptools import setup
from Cython.Build import cythonize
import numpy
import os

# Enable parallel compilation
os.environ['CYTHON_BUILD_PARALLEL'] = '1'

# Set optimization flags
extra_compile_args = ['-O3', '-march=native', '-flto']
extra_link_args = ['-flto']

setup(
    ext_modules=cythonize(
        "MSBoost.pyx",              # Your Cython file
        annotate=True,              # Optional: generates HTML annotation file
        compiler_directives={'language_level': '3'},
        build_dir="build",
        cache=True,                 # Enable caching
    ),
    include_dirs=[numpy.get_include()],  # Include NumPy headers if needed
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)