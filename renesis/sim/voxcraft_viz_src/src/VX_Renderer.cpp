/*******************************************************************************
Copyright (c) 2010, Jonathan Hiller (Cornell University)
If used in publication cite "J. Hiller and H. Lipson "Dynamic Simulation of Soft
Heterogeneous Objects" In press. (2011)"

This file is part of Voxelyze.
Voxelyze is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. Voxelyze is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
details. See <http://www.opensource.org/licenses/lgpl-3.0.html> for license
details.
*******************************************************************************/
#include "VX_Renderer.h"
#include "VX_GL_Utils.h"
#include <iostream>
#include <boost/property_tree/ptree.hpp>
#include <utility>
#include <boost/property_tree/xml_parser.hpp>

namespace pt = boost::property_tree;
using namespace std;

vector<string> split(const string &text, const string &sep) {
    vector<string> tokens;
    size_t start = 0, end = 0;
    while ((end = text.find(sep, start)) != string::npos) {
        tokens.push_back(text.substr(start, end - start));
        start = end + sep.length();
    }
    tokens.push_back(text.substr(start));
    return move(tokens);
}

VX_HistoryRenderer::VX_HistoryRenderer(const string &history, int width, int height)
        : width(width), height(height) {
    historyStream << history;
}

VX_HistoryRenderer::~VX_HistoryRenderer() {
#ifdef USE_SOFTWARE_GL
    OSMesaDestroyContext(ctx);
#else
    if (window != -1) {
        glutDestroyWindow(window);
    }
#endif
}

void VX_HistoryRenderer::Open(const string &filename) {
    ifstream historyFile(filename);
    if (!historyFile.is_open())
        throw std::invalid_argument("Invalid file.");
    historyStream << historyFile.rdbuf();
}


void VX_HistoryRenderer::Render(ViewMode mode) {
    InitGL();
    ParseSettings();
    string line;
    while (getline(historyStream, line)) {
        ClearScreen();
        // UpdateLighting();
        RenderFloor(voxelSize * 2);
        if(RenderHistoryFrame(line, mode)){
#ifndef USE_SOFTWARE_GL
            glutSwapBuffers();
#endif
            SaveFrame();
        }
        line.clear();
    }
}

py::array VX_HistoryRenderer::GetFrames() const {
    if (frames.empty())
        return py::array();
    else {
        auto array = new uint8_t[width * height * 4 * frames.size()];
        size_t framePixelNum = width * height * 4;
        for (size_t f = 0; f < frames.size(); f++) {
            memcpy(array + framePixelNum * f, frames[f].get(), framePixelNum * sizeof(uint8_t));
        }
        auto capsule = py::capsule(array, [](void *v) { delete [] (uint8_t*)v; std::cout << "Array Freed" << std::endl;});
        return std::move(py::array(
                {frames.size(), (size_t) height, (size_t) width, (size_t) 4},
                {size_t(height * width * 4 * sizeof(uint8_t)),
                 size_t(width * 4 * sizeof(uint8_t)),
                 size_t(4 * sizeof(uint8_t)),
                 size_t(sizeof(uint8_t))},
                array, capsule));
    }
}

size_t VX_HistoryRenderer::GetFrameNumber() const {
    return frames.size();
}

std::tuple<int, int> VX_HistoryRenderer::GetFrameSize() const {
    return move(make_tuple(width, height));
}

const uint8_t *VX_HistoryRenderer::GetRawFrame(size_t frame) const {
    if (frame >= frames.size())
        throw std::invalid_argument("Invalid frame index.");
    return frames[frame].get();
}

void VX_HistoryRenderer::InitGL() {
#ifdef USE_SOFTWARE_GL
    ctx = OSMesaCreateContextExt( OSMESA_RGBA, 16, 0, 0, NULL );
    if (!ctx) {
        throw runtime_error("OSMesaCreateContext failed!");
    }
    frameBuffer = make_unique<uint8_t[]>(width * height * sizeof(GLubyte) * 4);
    if (!OSMesaMakeCurrent(ctx, frameBuffer.get(), GL_UNSIGNED_BYTE, width, height)) {
        throw runtime_error("OSMesaMakeCurrent failed!");
    }
#else
    // TODO: add check for successful init
    int fakeArgc = 0;
    glutInit(&fakeArgc, NULL);
    // Note: don't use GLUT_RGBA, it is the same as GLUT_RGB
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_ALPHA);
    glutInitWindowPosition(80, 80);
    glutInitWindowSize(width, height);
    window = glutCreateWindow("");
    glutHideWindow();
#endif

    // Set the camera lens to have
    // a 60 degree (vertical) field of view, an aspect ratio of width/height,
    // and have everything closer than 1 unit to the
    // camera and greater than 100 units distant clipped away.
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, float(width) / float(height), 0.01, 100);

    // Turn on backface culling
    glFrontFace(GL_CCW);
    glCullFace(GL_BACK);

    // Turn on depth testing
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    // Enable opacity
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    //		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    //		glEnable(GL_LINE_SMOOTH);
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_POLYGON_SMOOTH);
    glEnable(GL_NORMALIZE);
    glPolygonOffset(1.0, 2);
    glEnable(GL_POLYGON_OFFSET_FILL);
}

void VX_HistoryRenderer::UpdateLighting() {
    glShadeModel(GL_SMOOTH); // smooth surfaces
    glEnable(GL_LIGHTING);   // global lighting

    float AmbientLight[] = {0.9f, 0.9f, 0.9f, 1.0f};
    glLightfv(GL_LIGHT0, GL_AMBIENT, AmbientLight); // Doesn't do anything unless we have at least one light!

    float d[4], s[4], p[4];
    glEnable(GL_LIGHT0);
    d[0] = 0.35, d[1] = 0.35, d[2] = 0.35, d[3] = 1;
    s[0] = 0.16, s[1] = 0.16, s[2] = 0.16, s[3] = 1;
    p[0] = -0.5f * 100 + historyCoM.x;
    p[1] = 0.5f * 100 + historyCoM.y;
    p[2] = 2.0f * 100 + historyCoM.z;
    p[3] = 1;
    glLightfv(GL_LIGHT0, GL_DIFFUSE, d);
    glLightfv(GL_LIGHT0, GL_SPECULAR, s);
    glLightfv(GL_LIGHT0, GL_POSITION, p);

    glEnable(GL_LIGHT1);
    d[0] = 0.235, d[1] = 0.235, d[2] = 0.235, d[3] = 1;
    s[0] = 0.08, s[1] = 0.08, s[2] = 0.08, s[3] = 1;
    p[0] = 2.0f * 100 + historyCoM.x;
    p[1] = -0.5f * 100 + historyCoM.y;
    p[2] = 1.0f * 100 + historyCoM.z;
    p[3] = 1;
    glLightfv(GL_LIGHT1, GL_DIFFUSE, d);
    glLightfv(GL_LIGHT1, GL_SPECULAR, s);
    glLightfv(GL_LIGHT1, GL_POSITION, p);

    glEnable(GL_LIGHT2);
    d[0] = 0.35, d[1] = 0.35, d[2] = 0.35, d[3] = 1;
    s[0] = 0.08, s[1] = 0.08, s[2] = 0.08, s[3] = 1;
    p[0] = 1.0f * 100 + historyCoM.x;
    p[1] = -1.0f * 100 + historyCoM.y;
    p[2] = -1.0f * 100 + historyCoM.z;
    p[3] = 1;
    glLightfv(GL_LIGHT2, GL_DIFFUSE, d);
    glLightfv(GL_LIGHT2, GL_SPECULAR, s);
    glLightfv(GL_LIGHT2, GL_POSITION, p);

    // Global scene lighing setup
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); // default, but verify

    GLfloat mat_specular[] = {1.0f, 1.0f, 1.0f, 1.0f}; // Specular (highlight)
    GLfloat mat_shininess[] = {70};                    // Shininess (size of highlight)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);

    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE); // Use ambient and diffuse
    glEnable(GL_COLOR_MATERIAL);                                // enable color tracking

    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE); // for accurate light reflections, mu
}

void VX_HistoryRenderer::ClearScreen() {
    // Set the current clear color to deep grey
    glClearColor(0.3, 0.3, 0.3, 1);

    // Clear color buffer and depth buffer to prepare for next rendering
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void VX_HistoryRenderer::ParseSettings() {
    size_t j;
    string line;
    string tokenSetting = "{{{setting}}}";
    while (getline(historyStream, line)) {
        if ((j = line.find(tokenSetting)) != string::npos) {
            std::istringstream is(line.substr(j + tokenSetting.length()));

            pt::ptree tree;
            pt::read_xml(is, tree);
            auto materialID = tree.get_child_optional("matcolor.id");
            if (materialID) {
                auto id = tree.get<int>("matcolor.id", 0);
                auto r = tree.get<double>("matcolor.r", 0);
                auto g = tree.get<double>("matcolor.g", 0);
                auto b = tree.get<double>("matcolor.b", 0);
                auto a = tree.get<double>("matcolor.a", 0);
                matColors[id] = CColor(r, g, b, a);
                continue;
            }
            // rescale the whole space. so history file can contain less digits. ( e.g. not 0.000221, but 2.21 )
            auto op_rescale = tree.get_child_optional("rescale");
            if (op_rescale) {
                rescale = tree.get("rescale", 1.0);
            }
            // voxel size
            auto op_voxel_size = tree.get_child_optional("voxel_size");
            if (op_voxel_size) {
                voxelSize = tree.get("voxel_size", 0.01);
            }
        } else {
            break;
        }
    }
}

void VX_HistoryRenderer::RenderFloor(vfloat size) {
    double z = 0.0;

    vfloat sX = 1.5 * size;  // distance in x for each 2-hex pattern
    vfloat sY = .866 * size; // distance in y for each 2-hex pattern

    /*      __
     *   __/  \
     *  /  \__/
     *  \__/
     *
     *  Draw 2 hexagons in the above way and
     *  create repetitive tiled floors
    */
    GLuint floorList = glGenLists(1);
    glNewList(floorList, GL_COMPILE);
    for (int i = -20; i <= 30; i++) {
        for (int j = -40; j <= 60; j++) {
            // Draw hexagons with different colors
            double colorFloor =
                    0.8 + 0.1 * ((int) (1000 * sin((float) ((i + 110) * (j + 106) * (j + 302)))) % 10) / 10.0;
            glColor4d(colorFloor, colorFloor, colorFloor + 0.1, 1.0);
            glBegin(GL_TRIANGLE_FAN);
            glVertex3d(i * sX, j * sY, z);
            glVertex3d(i * sX + 0.5 * size, j * sY, z);
            glVertex3d(i * sX + 0.25 * size, j * sY + 0.433 * size, z);
            glVertex3d(i * sX - 0.25 * size, j * sY + 0.433 * size, z);
            glVertex3d(i * sX - 0.5 * size, j * sY, z);
            glVertex3d(i * sX - 0.25 * size, j * sY - 0.433 * size, z);
            glVertex3d(i * sX + 0.25 * size, j * sY - 0.433 * size, z);
            glVertex3d(i * sX + 0.5 * size, j * sY, z);
            glEnd();

            colorFloor = 0.8 + 0.1 * ((int) (1000 * sin((float) ((i + 100) * (j + 103) * (j + 369)))) % 10) / 10.0;
            glColor4d(colorFloor, colorFloor, colorFloor + 0.1, 1.0);

            glBegin(GL_TRIANGLE_FAN);
            glVertex3d(i * sX + .75 * size, j * sY + 0.433 * size, z);
            glVertex3d(i * sX + 1.25 * size, j * sY + 0.433 * size, z);
            glVertex3d(i * sX + size, j * sY + 0.866 * size, z);
            glVertex3d(i * sX + 0.5 * size, j * sY + 0.866 * size, z);
            glVertex3d(i * sX + 0.25 * size, j * sY + 0.433 * size, z);
            glVertex3d(i * sX + 0.5 * size, j * sY, z);
            glVertex3d(i * sX + size, j * sY, z);
            glVertex3d(i * sX + 1.25 * size, j * sY + 0.433 * size, z);
            glEnd();
        }
    }

    glEndList();
    glCallList(floorList);
    glDeleteLists(floorList, 1);
}

bool VX_HistoryRenderer::RenderHistoryFrame(const string &recordLine, ViewMode mode) {
    if (recordLine.find("<<<") == string::npos)
        return false;
    // A temporary variable for finding substrings
    size_t j = 0;

    // Each record line has 2 parts, the voxels and the links, separated by "|"
    auto recordParts = split(recordLine, "|");
    if (recordParts.size() > 1) {
        if ((j = recordParts[1].find("]]]")) != string::npos) {
            glBegin(GL_LINE_STRIP);
            float prevLineWidth;
            float x1, y1, z1, x2, y2, z2;
            glColor4f(0.97f, 0.55f, 0.19f, 1.0f);
            glGetFloatv(GL_LINE_WIDTH, &prevLineWidth);
            glLineWidth(3.0);

            auto links = split(recordParts[1].substr(j + 3, recordParts[1].length() - j - 10), ";");
            for (auto &link : links) {
                auto pos = split(link, ",");
                if (pos.size() <= 1)
                    continue;
                if (pos.size() < 6) {
                    cerr << "ERROR: a link has pos size " << pos.size() << " less than 6. Link: " << link;
                    continue;
                }
                x1 = stof(pos[0]);
                y1 = stof(pos[1]);
                z1 = stof(pos[2]);
                x2 = stof(pos[3]);
                y2 = stof(pos[4]);
                z2 = stof(pos[5]);

                glBegin(GL_LINES);
                glVertex3f(x1, y1, z1);
                glVertex3f(x2, y2, z2);
                glEnd();
            }
            glEnd();
            glLineWidth(prevLineWidth);
        }
    }
    if ((j = recordParts[0].find(">>>")) != string::npos) {
        auto voxels = split(recordParts[0].substr(j + 3, recordParts[0].length() - j - 10), ";");
        double voltage;
        double p1, p2, p3;
        double angle, r1, r2, r3;
        int materialID;

        Vec3D<> CoM, boundMin, boundMax;
        size_t validVoxelNum = 0;

        for (auto &voxel : voxels) {
            auto pos = split(voxel, ",");
            if (pos.size() <= 1)
                continue;
            if (pos.size() < 15) {
                cerr << "ERROR: a voxel has pos size " << pos.size() << " less than 14. Voxel" << voxel;
                continue;
            }
            glPushMatrix();
            auto constIterator = pos.cbegin();
            p1 = stod(*constIterator) * rescale;
            constIterator++;
            p2 = stod(*constIterator) * rescale;
            constIterator++;
            p3 = stod(*constIterator) * rescale;
            constIterator++;
            angle = stod(*constIterator);
            constIterator++;
            r1 = stod(*constIterator);
            constIterator++;
            r2 = stod(*constIterator);
            constIterator++;
            r3 = stod(*constIterator);
            constIterator++;
            Vec3D<double> nnn, ppp;
            nnn.x = stod(*constIterator) * rescale;
            constIterator++;
            nnn.y = stod(*constIterator) * rescale;
            constIterator++;
            nnn.z = stod(*constIterator) * rescale;
            constIterator++;
            ppp.x = stod(*constIterator) * rescale;
            constIterator++;
            ppp.y = stod(*constIterator) * rescale;
            constIterator++;
            ppp.z = stod(*constIterator) * rescale;
            constIterator++;
            materialID = stoi(*constIterator);
            if (materialID < 0 || materialID >= 10) {
                materialID = 0;
            }
            constIterator++;
            voltage = stod(*constIterator);
            glTranslated(p1, p2, p3);
            glRotated(angle, r1, r2, r3);
            if (nnn.Dist2(ppp) < 1) {
                CColor c;
                if (mode == HISTORY_ELECTRICAL) {
                    c = CJetScale()(voltage / 100.0);
                } else if (mode == HISTORY_ROTATION) {
                    c = CColor(0, 0, 0, 0.8);
                    c.b = angle / 60;
                    if (c.b > 1) c.b = 1;
                    c.r = 1 - c.b;
                    c.g = (1 - c.b) * 0.5;
                } else {
                    if (matColors.find(materialID) != matColors.end()) {
                        c = matColors[materialID];
                    } else {
                        cerr << "Color of material with id " << materialID << " is not found, using default.";
                        c = defaultColor;
                    }
                }
                CGL_Utils::DrawCube(nnn, ppp, true, true, 0.0, c, false);
            }
            glPopMatrix();

            Vec3D<> center = Vec3D<>(p1, p2, p3);
            CoM += center;
            if (validVoxelNum == 0) {
                boundMin = center;
                boundMax = center;
            }
            boundMin = boundMin.Min(center);
            boundMax = boundMax.Max(center);
            validVoxelNum++;
        }
        CoM /= vfloat(validVoxelNum);
        Vec3D<> boundDiff = boundMax - boundMin;
        // Update camera view center, but gently.
        if (renderedFrameNum == 0) {
            // Initialize mass center
            historyCoM = CoM;
            historyBound = boundDiff;
        } else {
            // Keep z the same
            vfloat z = historyCoM.z;
            historyCoM = historyCoM * 0.95 + CoM * 0.05;
            historyCoM.z = z;
            historyBound = historyBound * 0.98 + boundDiff * 0.02;
        }


        Vec3D<> camPos = historyCoM + Vec3D<>(historyBound.x, -historyBound.y, historyBound.z) * 2;

        // Set camera position and orientation
        // Position camera at historyCoM + 1 looking at historyCoM with the vector
        // <0, 0, 1> pointing upward.
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(camPos.x, camPos.y, camPos.z, historyCoM.x, historyCoM.y, historyCoM.z, 0, 0, 1);
    }
    renderedFrameNum++;
    return true;
}

void VX_HistoryRenderer::SaveFrame() {
    // The buffer is aligned since we are reading 4 channels
    auto raw = new uint8_t[width * height * 4];
    auto buffer = make_unique<uint8_t[]>(width * height * 4);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, raw);

    // Inverse y axis since opengl y axis goes from bottom to top
    for (size_t row = 0; row < height; row++) {
        memcpy(buffer.get() + row * width * 4,
               raw + (height - row - 1) * width * 4,
               width * 4 * sizeof(uint8_t));
    }

    frames.emplace_back(move(buffer));
}