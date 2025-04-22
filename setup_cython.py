"""
Run:
python setup_cython.py build_ext --inplace
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "my_module_app",
    ext_modules = cythonize("utils_cython.pyx",
    compiler_directives={'language_level': 3}),
    zip_safe=False,
)