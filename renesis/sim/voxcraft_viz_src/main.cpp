//
// Created by iffi on 10/27/22.
//
#include <fstream>
#include <iostream>
#include "VX_Renderer.h"

using namespace std;

void WriteImage(const string &path, const uint8_t *imageData, int width, int height) {
    ofstream myImage(path);
    if (myImage.fail())
        throw std::runtime_error("Unable to create image.ppm");

    //Image header - Need this to start the image properties
    myImage << "P3" << endl;						//Declare that you want to use ASCII colour values
    myImage << width << " " << height << endl;		//Declare w & h
    myImage << "255" << endl;						//Declare max colour ID


    //Image Body - outputs imageData array to the .ppm file, creating the image
    for (size_t x = 0; x < width * height * 4; x++) {
        if (not (x % 4 == 3)) {
            myImage << int(imageData[x]) << " " << endl;		//Sets 3 bytes of colour to each pixel
        }
    }
    myImage.close();
}

int main() {
    VX_HistoryRenderer renderer("", 640, 480);
    renderer.Open("demos/test.history");
    renderer.Render();
    auto frameSize = renderer.GetFrameSize();
    WriteImage("test.ppm", renderer.GetRawFrame(100), get<0>(frameSize), get<1>(frameSize));
}