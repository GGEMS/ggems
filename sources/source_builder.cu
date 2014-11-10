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
    //sources.size_of_sources = NULL;
    sources.data_sources = NULL;
    sources.seeds = NULL;
    sources.nb_sources = 0;
    sources.ptr_sources_dim = 0;
    //sources.size_of_sources_dim = 0;
    sources.data_sources_dim = 0;
    sources.seeds_dim = 0;
}

// Add a point source on the simulation
void SourceBuilder::add_source(PointSource src) {
    sources.nb_sources++;

    // Store the address to access to this source
    array_push_back(&sources.ptr_sources, sources.ptr_sources_dim, sources.data_sources_dim);

    // Store information of this source
    array_push_back(&sources.data_sources, sources.data_sources_dim, (float)POINT_SOURCE);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.geometry_id);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.px);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.py);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.pz);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.energy);

    // Save the seed
    array_push_back(&sources.seeds, sources.seeds_dim, src.seed);

}

// Add a cone beam source
void SourceBuilder::add_source(ConeBeamSource src) {
    sources.nb_sources++;

    // Store the address to access to this source
    array_push_back(&sources.ptr_sources, sources.ptr_sources_dim, sources.data_sources_dim);

    // Store information of this source
    array_push_back(&sources.data_sources, sources.data_sources_dim, (float)CONE_BEAM_SOURCE);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.geometry_id);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.px);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.py);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.pz);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.phi);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.theta);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.psi);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.aperture);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.energy);

    // Save the seed
    array_push_back(&sources.seeds, sources.seeds_dim, src.seed);

}

#endif
















