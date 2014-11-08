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

#ifndef BASE_OBJECT_CUH
#define BASE_OBJECT_CUH

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "global.cuh"

// Class that define the base of every object in GGEMS
class BaseObject {
    public:
        BaseObject();

        void set_material(std::string mat_name);
        void set_name(std::string obj_name);
        void set_color(float r, float g, float b);
        void set_transparency(float val);

        // Bounding box
        float xmin, xmax, ymin, ymax, zmin, zmax;
        // Viewing
        Color color;
        float transparency;
        // Property
        std::string material_name;
        std::string object_name;

    private:
};

#endif
