#include "env.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace territories;

PYBIND11_MODULE(territories, m) {
    py::class_<Agent>(m, "Agent")
        .def_readonly("x", &Agent::x)
        .def_readonly("y", &Agent::y)
        .def_readonly("is_alive", &Agent::is_alive)
        .def_readonly("alive_steps", &Agent::alive_steps)
        .def_readonly("kills", &Agent::kills)
        .def_readonly("territory", &Agent::territory)
        .def_readonly("has_trail", &Agent::has_trail);

    py::class_<Env>(m, "Env")
        .def(py::init<int, int, int, int, int>(),
             py::arg("num_agents"),
             py::arg("width"),
             py::arg("height"),
             py::arg("window_radius"),
             py::arg("seed"))
        .def("reset", &Env::reset)
        .def("step", &Env::step)
        .def("get_obs", [](Env& self) {
            const auto& obs = self.get_obs();
            return py::array_t<int>(
                {static_cast<py::ssize_t>(obs.size())},
                obs.data(),
                py::cast(self)  // prevent gc
            );
        })
        .def("get_agents", &Env::get_agents);
}
