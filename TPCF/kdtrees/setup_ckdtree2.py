from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

ext_modules = [Extension("ckdtree2", ["ckdtree2.pyx"])]

setup(
  name = 'ckdtree app',
  #ext_modules = cythonize("ckdtree2.pyx"),
  cmdclass = {'build_ext': build_ext},
  include_dirs=[numpy.get_include()],
  ext_modules = ext_modules
)