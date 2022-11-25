//
// Created by iffi on 10/27/22.
//

#include "VX_Renderer.h"

PYBIND11_MODULE(voxcraft_viz, m) {
    py::class_<VX_HistoryRenderer> vxhr(m, "VXHistoryRenderer",
                                        R"(History renderer.)");

    // Define enum first to let pybind11 know about it
    py::enum_<VX_HistoryRenderer::ViewMode>(vxhr, "ViewMode")
            .value("HISTORY", VX_HistoryRenderer::ViewMode::HISTORY)
            .value("HISTORY_ELECTRICAL", VX_HistoryRenderer::ViewMode::HISTORY_ELECTRICAL)
            .value("HISTORY_ROTATION", VX_HistoryRenderer::ViewMode::HISTORY_ROTATION)
            .export_values();

    vxhr.def(py::init<std::string, int, int>(),
             py::arg("history") = "",
             py::arg("width") = 640,
             py::arg("height") = 480);
    vxhr.def("open", &VX_HistoryRenderer::Open);
    vxhr.def("render", &VX_HistoryRenderer::Render,
             py::arg("mode") = VX_HistoryRenderer::HISTORY);
    vxhr.def("get_frames", &VX_HistoryRenderer::GetFrames);
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}