from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "territories",
        ["src/env/bindings.cpp", "src/env/env.cpp", "src/env/vector_env.cpp"],
        include_dirs=["src/env"],
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
