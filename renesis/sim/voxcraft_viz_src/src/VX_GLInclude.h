//
// Created by iffi on 10/26/22.
//

#ifndef GLINCLUDE_H
#define GLINCLUDE_H

#ifdef USE_SOFTWARE_GL
#include <OSMesa/gl.h>
#include <OSMesa/glu.h>
#include <OSMesa/osmesa.h>
#else
#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#ifdef _WIN32
#include <windows.h>
#include <GL/glut.h>
#endif
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif
#endif


#endif //GLINCLUDE_H
