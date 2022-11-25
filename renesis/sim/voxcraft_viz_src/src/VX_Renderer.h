/*******************************************************************************
Copyright (c) 2010, Jonathan Hiller (Cornell University)
If used in publication cite "J. Hiller and H. Lipson "Dynamic Simulation of Soft Heterogeneous Objects" In press. (2011)"

This file is part of Voxelyze.
Voxelyze is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your option) any later version. Voxelyze is distributed in the hope
that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details. See <http://www.opensource.org/licenses/lgpl-3.0.html> for license details.
*******************************************************************************/

#ifndef VX_RENDERER_H
#define VX_RENDERER_H

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "VX_GL_Utils.h"

namespace py = pybind11;

class VX_HistoryRenderer {
public:
    enum ViewMode {
        HISTORY,            //!< Draws only history, experimental for Sida
        HISTORY_ELECTRICAL, //!< Draws history with electrical colors
        HISTORY_ROTATION,
    };
public:
    VX_HistoryRenderer() = delete;

    explicit VX_HistoryRenderer(const std::string &history = "", int width = 640, int height = 480);

    ~VX_HistoryRenderer();

    void Open(const std::string &filename);

    void Render(ViewMode mode = HISTORY);

    // Do not call this function unless the module is used as a python module
    py::array GetFrames() const;

    size_t GetFrameNumber() const;

    std::tuple<int, int> GetFrameSize() const;

    const uint8_t* GetRawFrame(size_t frame) const;

private:
    void InitGL();

    void ClearScreen();

    void ParseSettings();

    bool RenderHistoryFrame(const std::string& recordLine, ViewMode mode = HISTORY);

    static void RenderFloor(vfloat hexTileEdgeLength);

    void SaveFrame();

#ifdef USE_SOFTWARE_GL
    OSMesaContext ctx;
    std::unique_ptr<uint8_t[]> frameBuffer;
#else
    int window = -1;
#endif
    int width, height;
    Vec3D<> historyCoM; // Center of mass
    Vec3D<> historyBound;

    // Configurations in the history file
    double rescale = 1.0;
    double voxelSize = 0.01;
    std::map<int, CColor> matColors;

    // The input history
    std::stringstream historyStream;
    // Rendered frames
    size_t renderedFrameNum = 0;
    std::vector<std::unique_ptr<uint8_t[]>> frames;

    CColor defaultColor{0.2, 0.2, 0.2, 0.2};
};

#endif // VX_RENDERER_H
