from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

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
        compiler_directives={'language_level': '3'},
        build_dir="build",
        cache=True,
    ),
    install_requires=["numpy", "pandas", "scikit-learn", "scipy"],
    python_requires=">=3.6",
)
