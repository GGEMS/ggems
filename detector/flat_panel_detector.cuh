// This file is part of GGEMS
//
// GGEMS is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// GGEMS is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with GGEMS.  If not, see <http://www.gnu.org/licenses/>.
//
// GGEMS Copyright (C) 2013-2014 Julien Bert

#ifndef FLAT_PANEL_DETECTOR_CUH
#define FLAT_PANEL_DETECTOR_CUH

#include "constants.cuh"
#include "global.cuh"

// Struct that handle an image detector
struct ImageDetector {
    unsigned int geometry_id;
    unsigned int nx, ny, nz;
    unsigned int nb_voxels;
    unsigned int countp;
    float sx, sy, sz;
    float xmin, xmax, ymin, ymax, zmin, zmax;
    float *data;
};


// Flat panel detector
class FlatPanelDetector {
    public:
        FlatPanelDetector();
        void attach_to(unsigned int geometry_id);
        void set_resolution(float sx, float sy, float sz);
        void init(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax);

        void save_image(std::string outputname);

        ImageDetector panel_detector;

    private:
};

#endif
