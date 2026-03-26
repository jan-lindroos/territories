#include "env.h"
#include "vector_env.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace territories;

PYBIND11_MODULE(territories, m) {
  py::class_<Snake>(m, "Snake")
      .def_readwrite("x", &Snake::x)
      .def_readwrite("y", &Snake::y)
      .def_readwrite("alive", &Snake::alive);

  py::class_<Env>(m, "Env")
      .def(py::init<int, int, int, int, int>(), py::arg("num_agents"),
           py::arg("width"), py::arg("height"), py::arg("window_radius"),
           py::arg("seed") = -1)
      .def("reset", &Env::Reset)
      .def("step", &Env::Step)
      .def("get_observations", &Env::GetObservations)
      .def_readonly("num_agents", &Env::num_agents_)
      .def_readonly("width", &Env::width_)
      .def_readonly("height", &Env::height_)
      .def_readonly("window_radius", &Env::window_radius_)
      .def_readwrite("snakes", &Env::snakes_)
      .def_readwrite("territory_grid", &Env::territory_grid_)
      .def_readwrite("trail_grid", &Env::trail_grid_);

  py::class_<VectorEnv>(m, "VectorEnv")
      .def(py::init<int, int, int, int, int, int>(), py::arg("num_envs"),
           py::arg("num_agents"), py::arg("width"), py::arg("height"),
           py::arg("window_radius"), py::arg("seed") = -1)
      .def("reset", &VectorEnv::Reset)
      .def("step", &VectorEnv::Step)
      .def("get_observations", &VectorEnv::GetObservations)
      .def("get_env", &VectorEnv::GetEnv, py::return_value_policy::reference)
      .def_readonly("num_envs", &VectorEnv::num_envs_);
}
