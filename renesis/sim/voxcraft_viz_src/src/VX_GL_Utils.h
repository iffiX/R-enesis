/*******************************************************************************
Copyright (c) 2010, Jonathan Hiller (Cornell University)
If used in publication cite "J. Hiller and H. Lipson "Dynamic Simulation of Soft Heterogeneous Objects" In press. (2011)"

This file is part of Voxelyze.
Voxelyze is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your option) any later version. Voxelyze is distributed in the hope
that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details. See <http://www.opensource.org/licenses/lgpl-3.0.html> for license details.
*******************************************************************************/

#ifndef VX_GL_Utils_H
#define VX_GL_Utils_H

#include "VX_GLInclude.h"
#include "VX_Vec3D.h"

#ifndef vfloat
#define vfloat double
#endif

struct CColor {
    vfloat r, g, b, a;

    CColor() : r(-1), g(-1), b(-1), a(-1) {}

    CColor(const CColor &c) = default;

    CColor(vfloat r, vfloat g, vfloat b, vfloat a = 1.0) : r(r), g(g), b(b), a(a) {}

    bool isValid() const {
        return r >= 0.0 && r <= 1.0 && g >= 0.0 && g <= 1.0 && b >= 0.0 && b <= 1.0 && a >= 0.0 && a <= 1.0;
    };
};

/** @struct CJetScale
 *  The "Jet" color scale, see https://www.mathworks.com/help/matlab/ref/jet.html
 */
struct CJetScale {
    inline CColor operator()(vfloat val) {
        if (val > 1.0)
            return CColor(1.0, 0.0, 0.0, 1.0);
        else if (val > 0.75)
            return CColor(1, 4 - val * 4, 0, 1.0);
        else if (val > 0.5)
            return CColor(val * 4 - 2, 1, 0, 1.0);
        else if (val > 0.25)
            return CColor(0, 1, 2 - val * 4, 1.0);
        else if (val > 0)
            return CColor(0, val * 4, 1, 1.0);
        else
            return CColor(0, 0, 1.0, 1.0);
    };
};

// OpenGL primitives drawing class
class CGL_Utils {
public:
    // 3D
    static void DrawCube(bool Faces = true, bool Edges = true, float LineWidth = 0.0,
                         const CColor &Color = CColor(),
                         bool Topless = false); // draws unit cube. changes glColor if Color != NULL
    static void
    DrawCube(const Vec3D<> &Center, vfloat Dim, const Vec3D<> &Squeeze = Vec3D<>(1.0, 1.0, 1.0), bool Faces = true,
             bool Edges = true, float LineWidth = 0.0, const CColor &Color = CColor(),
             bool Topless = false); // Draws scaled and translated cube
    static void
    DrawCube(const Vec3D<> &V1, const Vec3D<> &V2, bool Faces = true, bool Edges = true, float LineWidth = 0.0,
             const CColor &Color = CColor(), bool Topless = false); // Draws arbitrary rectangular prism
    static void DrawCubeFace(bool Topless = false);

    static void DrawCubeEdge();

    static void DrawSphere(const CColor &Color = CColor()); // draws unit sphere (no need for lines...)
    static void DrawSphere(const Vec3D<> &p, vfloat Rad, const Vec3D<> &Squeeze = Vec3D<>(1.0, 1.0, 1.0),
                           const CColor &Color = CColor()); // arbitrary sphere
    static void DrawSphereFace();

    static void DrawCylinder(float taper = 1.0, bool Faces = true, bool Edges = true, float LineWidth = 0.0,
                             const CColor &Color = CColor()); // draws unit cylinder in +X direction
    static void
    DrawCylinder(const Vec3D<> &v0, const Vec3D<> &v1, float Rad, const Vec3D<> &Squeeze = Vec3D<>(1.0, 1.0, 1.0),
                 bool Faces = true, bool Edges = true, float LineWidth = 0.0, const CColor &Color = CColor());

    static void DrawCylinder(const Vec3D<> &v0, const Vec3D<> &v1, float Rad, float Rad2,
                             const Vec3D<> &Squeeze = Vec3D<>(1.0, 1.0, 1.0),
                             bool Faces = true, bool Edges = true, float LineWidth = 0.0,
                             const CColor &Color = CColor());

    static void DrawCone(const Vec3D<> &v0, const Vec3D<> &v1, float Rad, bool Faces = true, bool Edges = true,
                         float LineWidth = 0.0,
                         const CColor &Color = CColor());

    static void DrawCylFace(float taper = 1.0);

    static void DrawCylEdge(float taper = 1.0);

    // 2D
    static void DrawRectangle(bool Fill = false, float LineWidth = 0.0,
                              const CColor &Color = CColor()); // draw unit circle in Z Plane
    static void DrawRectangle(const Vec3D<> &Center, vfloat Dim, const Vec3D<> &Normal,
                              const Vec3D<> &Squeeze = Vec3D<>(1.0, 1.0, 1.0),
                              bool Fill = false, float LineWidth = 0.0, const CColor &Color = CColor()); // draw square
    static void DrawRectangle(const Vec3D<> &V1, const Vec3D<> &V2, bool Fill = false, float LineWidth = 0.0,
                              const CColor &Color = CColor()); // draw rectangle

    static void
    DrawCircle(bool Fill = false, float LineWidth = 0.0, const CColor &Color = CColor()); // draw unit circle in Z Plane
    static void
    DrawCircle(const Vec3D<> &p, vfloat Rad, const Vec3D<> &Normal, const Vec3D<> &Squeeze = Vec3D<>(1.0, 1.0, 1.0),
               bool Fill = false, float LineWidth = 0.0, const CColor &Color = CColor()); // draw circle

    // compound objetc:
    static void DrawArrow(
            const CColor &Color = CColor());                                        // draw unit arrow from origin to ZPlane
    static void DrawArrowD(const Vec3D<> &vO, const Vec3D<> &vP,
                           const CColor &Color = CColor()); // draws 3D arrow Directly from vO to vP
    static void DrawArrow(const Vec3D<> &vO, const Vec3D<> &Dir,
                          const CColor &Color = CColor()); // draws 3D arrow from vO with direction&magnitude Dir

    static void DrawLineArrowD(const Vec3D<> &vO, const Vec3D<> &vP, vfloat LineWidth = 1.0,
                               const CColor &Color = CColor()); // draws 3D arrow Directly from vO to vP

protected: // vector functions
    static void AlignWith(const Vec3D<> &Base, const Vec3D<> &target, Vec3D<> &rotax,
                          vfloat &rotamt); // find rotation axis and angle to align base vector with target vector
};

#endif // VX_GL_Utils_H
