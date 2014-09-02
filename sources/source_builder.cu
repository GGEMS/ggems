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

#ifndef SOURCE_BUILDER_CU
#define SOURCE_BUILDER_CU

#include "source_builder.cuh"

///////// Source builder class ////////////////////////////////////////////////////

SourceBuilder::SourceBuilder() {
    sources.ptr_sources = NULL;
    sources.data_size_sources = NULL;
    sources.data_sources = NULL;
    sources.nb_sources = 0;
    sources.nb_data_elements = 0;
}

// Add a point source on the simulation
void SourceBuilder::add_source(PointSource src) {
    sources.nb_sources++;

    // Store the address to access to this source
    sources.ptr_sources = (unsigned int*)realloc(sources.ptr_sources,
                                                 sources.nb_sources*sizeof(unsigned int));
    sources.ptr_sources[sources.nb_sources-1] = sources.nb_data_elements;

    // Store the size of the data needs for this source
    sources.data_size_sources = (unsigned int*)realloc(sources.data_size_sources,
                                                       sources.nb_sources*sizeof(unsigned int));
    sources.data_size_sources[sources.nb_sources-1] = 5;

    // Finally store all parameters
    sources.nb_data_elements += 5;
    sources.data_sources = (float*)realloc(sources.data_sources, sources.nb_data_elements*sizeof(float));

    sources.data_size_sources[sources.nb_data_elements-5] = POINT_SOURCE;
    sources.data_size_sources[sources.nb_data_elements-4] = src.px;
    sources.data_size_sources[sources.nb_data_elements-3] = src.py;
    sources.data_size_sources[sources.nb_data_elements-2] = src.pz;
    sources.data_size_sources[sources.nb_data_elements-1] = src.energy;
}

#endif
