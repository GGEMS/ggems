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

#ifndef SOURCE_BUILDER_CUH
#define SOURCE_BUILDER_CUH

#include <vector>
#include <string>
#include "point_source.cuh"

#define POINT_SOURCE 0

struct Sources {
    unsigned int *ptr_sources;
    unsigned int *data_size_sources;
    float *data_sources;
    unsigned int nb_sources;
    unsigned int nb_data_elements;
};

// Class to manage sources on the simulation
class SourceBuilder {
    public:
        SourceBuilder();
        unsigned int add_source(PointSource src);

        //void save_ggems_geometry(std::string filename);

        Sources sources;

    private:        



};

#endif
