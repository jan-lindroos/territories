cmake_minimum_required(VERSION 3.15...4.0)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(territories
  src/env/bindings.cpp
  src/env/env.cpp
  src/env/vector_env.cpp)

target_include_directories(territories PRIVATE src/env)

install(TARGETS territories LIBRARY DESTINATION .)
install(FILES src/simulation.py src/ui.py src/cli.py DESTINATION .)
