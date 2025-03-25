from setuptools import setup, find_packages, Extension

try:
    from Cython.Build import cythonize
    import numpy as np
    
    # Define the extension modules
    extensions = [
        Extension(
            "MSBoostCy",  # Name of the extension
            ["MSBoost.pyx"],  # Source file(s)
            include_dirs=[np.get_include()],  # Include directories for numpy
            extra_compile_args=["-O3"],  # Optimization flags
        )
    ]
    
    ext_modules = cythonize(extensions, language_level=3)
except ImportError:
    # If Cython is not available, use the pure Python version
    ext_modules = []

setup(
    name="MSBoost",
    version="0.1.0",
    description="Model Selection with Multiple Base Estimators for Gradient Boosting",
    author="MSBoost Authors",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "Cython>=0.29.0",  # Add Cython as a dependency
    ],
    python_requires=">=3.6",
)