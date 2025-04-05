from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os
from Cython.Compiler import Options

Options.docstrings = False
os.environ['CYTHON_BUILD_PARALLEL'] = '1'

extensions = [
    Extension(
        "MSBoost",
        ["MSBoost.pyx"],
        extra_compile_args=['-O3', '-march=native', '-flto'],
        extra_link_args=['-flto'],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        annotate=True,
        compiler_directives={
        'language_level': "3",                     # Use Python 3 syntax
        'boundscheck': False,                      # Disable bounds checking
        'wraparound': False,                       # Disable negative indexing
        'nonecheck': False,                        # Disable None checks
        'initializedcheck': False,                 # Skip uninitialized variable checks
        'cdivision': True,                         # Enable C-style division
        'overflowcheck': False,                    # Disable integer overflow checks
        'infer_types': True,                       # Enable type inference
        'autotestdict': False,                     # Skip dictionary type checks
        'optimize.use_switch': True,               # Use switch statements for optimization
        'optimize.unpack_method_calls': True,      # Optimize method calls
        'binding': False,                          # Disable Python function bindings
        'profile': False,                          # Disable profiling support
        'linetrace': False,                        # Disable line tracing for debugging
        'embedsignature': False                    # Prevent embedding function signatures
    },
        build_dir="build",
        cache=True,
    ),
    install_requires=["numpy", "pandas", "scikit-learn", "scipy"],
    python_requires=">=3.6",
)
