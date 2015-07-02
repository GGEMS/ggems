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


///////// Host/Device function ////////////////////////////////////////////////////

__host__ __device__ void get_primaries(Sources sources, ParticleStack &particles, ui32 id_src, ui32 id_part) {

    // Read the address source
    ui32 adr = sources.ptr_sources[id_src];
    // Read the kind of sources
    ui32 type = (ui32)(sources.data_sources[adr+ADR_SRC_TYPE]);
    ui32 geom_id = (ui32)(sources.data_sources[adr+ADR_SRC_GEOM_ID]);
    
    f32 energy;
    
    // Point Source
    if (type == POINT_SOURCE) {
        f32 px = sources.data_sources[adr+ADR_POINT_SRC_PX];
        f32 py = sources.data_sources[adr+ADR_POINT_SRC_PY];
        f32 pz = sources.data_sources[adr+ADR_POINT_SRC_PZ];
           
        f32 nb_peak = sources.data_sources[adr+ADR_POINT_SRC_NB_PEAK];
        
        if (nb_peak == 1) {
            energy = sources.data_sources[adr+ADR_POINT_SRC_ENERGY];
        }
        if (nb_peak == 2) {
          
            f32 p1, p2;
            
            p1 = sources.data_sources[adr+ADR_POINT_SRC_PARTPDEC + 1];
            p2 = sources.data_sources[adr+ADR_POINT_SRC_PARTPDEC + 2];
            
            //printf("p1 %f p2 %f \n", p1, p2);
           
            //f32 rnd = (f32) rand()/RAND_MAX;
            f32 rnd = JKISS32(particles,id_part);
            
           // printf("p1 %f p2 %f -- rnd %f \n", p1, p2, rnd);
            
            f32 p = p1/ (f32)(p1+p2);
            
            //printf("p1 %f p2 %f -- rnd %f p %f \n", p1, p2, rnd, p);
            
            if (rnd < p) {
                energy = sources.data_sources[adr+ADR_POINT_SRC_ENERGY];
            } else {
                energy = sources.data_sources[adr+ADR_POINT_SRC_ENERGY + 1];
            }
        } 
        
       // printf("energy %f \n", energy);
        
        point_source_primary_generator(particles, id_part, px, py, pz, energy, PHOTON, geom_id);

    } else if (type == CYLINDER_SOURCE) {
        f32 px = sources.data_sources[adr+ADR_CYLINDER_SRC_PX];
        f32 py = sources.data_sources[adr+ADR_CYLINDER_SRC_PY];
        f32 pz = sources.data_sources[adr+ADR_CYLINDER_SRC_PZ];
        f32 rad = sources.data_sources[adr+ADR_CYLINDER_SRC_RAD];
        f32 length = sources.data_sources[adr+ADR_CYLINDER_SRC_LEN];
        f32 energy = sources.data_sources[adr+ADR_CYLINDER_SRC_ENERGY];
        cylinder_source_primary_generator(particles, id_part, px, py, pz, rad, length,
                                           energy, PHOTON, geom_id);
    } else if (type == PLANAR_SOURCE) {
        f32 px = sources.data_sources[adr+ADR_PLANAR_SRC_PX];
        f32 py = sources.data_sources[adr+ADR_PLANAR_SRC_PY];
        f32 pz = sources.data_sources[adr+ADR_PLANAR_SRC_PZ];
        f32 width = sources.data_sources[adr+ADR_PLANAR_SRC_WID];
        f32 length = sources.data_sources[adr+ADR_PLANAR_SRC_LEN];
        f32 energy = sources.data_sources[adr+ADR_PLANAR_SRC_ENERGY];
        planar_source_primary_generator(particles, id_part, px, py, pz, width, length,
                                           energy, PHOTON, geom_id);
    } else if (type == CONE_BEAM_SOURCE) {
        f32 px = sources.data_sources[adr+ADR_CONE_BEAM_SRC_PX];
        f32 py = sources.data_sources[adr+ADR_CONE_BEAM_SRC_PY];
        f32 pz = sources.data_sources[adr+ADR_CONE_BEAM_SRC_PZ];
        f32 phi = sources.data_sources[adr+ADR_CONE_BEAM_SRC_PHI];
        f32 theta = sources.data_sources[adr+ADR_CONE_BEAM_SRC_THETA];
        f32 psi = sources.data_sources[adr+ADR_CONE_BEAM_SRC_PSI];
        f32 aperture = sources.data_sources[adr+ADR_CONE_BEAM_SRC_APERTURE];
        f32 energy = sources.data_sources[adr+ADR_CONE_BEAM_SRC_ENERGY];

        cone_beam_source_primary_generator(particles, id_part, px, py, pz,
                                           phi, theta, psi, aperture, energy, PHOTON, geom_id);
    } else if (type == VOXELIZED_SOURCE) {
        f32 px = sources.data_sources[adr+ADR_VOX_SOURCE_PX];
        f32 py = sources.data_sources[adr+ADR_VOX_SOURCE_PY];
        f32 pz = sources.data_sources[adr+ADR_VOX_SOURCE_PZ];

        f32 nb_vox_x = sources.data_sources[adr+ADR_VOX_SOURCE_NB_VOX_X];
        f32 nb_vox_y = sources.data_sources[adr+ADR_VOX_SOURCE_NB_VOX_Y];
        f32 nb_vox_z = sources.data_sources[adr+ADR_VOX_SOURCE_NB_VOX_Z];

        f32 sx = sources.data_sources[adr+ADR_VOX_SOURCE_SPACING_X];
        f32 sy = sources.data_sources[adr+ADR_VOX_SOURCE_SPACING_Y];
        f32 sz = sources.data_sources[adr+ADR_VOX_SOURCE_SPACING_Z];

        f32 energy = sources.data_sources[adr+ADR_VOX_SOURCE_ENERGY];

        f32 nb_acts = sources.data_sources[adr+ADR_VOX_SOURCE_NB_CDF];

        f32 emission_type = sources.data_sources[adr+ADR_VOX_SOURCE_EMISSION_TYPE];

        f32 *cdf_index = &(sources.data_sources[adr+ADR_VOX_SOURCE_CDF_INDEX]);
        ui32 adr_cdf_act = adr+nb_acts;
        f32 *cdf_act = &(sources.data_sources[adr_cdf_act+ADR_VOX_SOURCE_CDF_INDEX]);

        if (emission_type == EMISSION_BACK2BACK) {

            // Back2back fills the particle' stack with two particles, we need to
            // adjust the ID to be the ID of event (half size) and not the ID of particles
            // Consequently only even ID generate back2back

            if ((id_part&1)==0) {
                // Is even
                voxelized_source_primary_generator(particles, id_part,
                                                   cdf_index, cdf_act, nb_acts,
                                                   px, py, pz, nb_vox_x, nb_vox_y, nb_vox_z,
                                                   sx, sy, sz, energy, PHOTON, geom_id);
            }

        } else if (emission_type == EMISSION_MONO) {
            //printf("ERROR: voxelized source, emission 'MONO' is not impleted yet!\n");
            //exit(EXIT_FAILURE);
            // TODO
        }

    }

}

///////// Kernel ////////////////////////////////////////////////////

// Kernel to create new particles (sources manager)
__global__ void kernel_get_primaries(Sources sources, ParticleStack particles, ui32 isrc) {

    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particles.size) return;

    // Get new particles
    get_primaries(sources, particles, isrc, id);

}



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

    tot_activity = 0;
}

// Add a point source on the simulation
void SourceBuilder::add_source(PointSource src) {
    sources.nb_sources++;
    
    ui32 n = src.energy_hist.size();
    f32 *newhist = (f32*)malloc(sizeof(f32) * n);
    ui32 i=0; while (i < n) {
        newhist[i] = src.energy_hist[i];
        ++i;
    }
    
    f32 *newpartpdec = (f32*)malloc(sizeof(f32) * n);
    i=0; while (i < n) {
        newpartpdec[i] = src.partpdec[i];
        ++i;
    }

    // Store the address to access to this source
    array_push_back(&sources.ptr_sources, sources.ptr_sources_dim, sources.data_sources_dim);

    // Store information of this source
    array_push_back(&sources.data_sources, sources.data_sources_dim, (f32)POINT_SOURCE);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.geometry_id);
    
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.px);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.py);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.pz);
   
    array_push_back(&sources.data_sources, sources.data_sources_dim, n);

    array_append_array(&sources.data_sources, sources.data_sources_dim, &(newhist), n);
    array_append_array(&sources.data_sources, sources.data_sources_dim, &(newpartpdec), n);
    
    array_push_back(&sources.data_sources, sources.data_sources_dim, 2*n + SIZE_POINT_SRC);
    
    // Save the seed
    //array_push_back(&sources.seeds, sources.seeds_dim, src.seed);
    
    free(newhist);
    free(newpartpdec);
}

// Add a cylinder source on the simulation
void SourceBuilder::add_source(CylinderSource src) {
    sources.nb_sources++;

    // Store the address to access to this source
    array_push_back(&sources.ptr_sources, sources.ptr_sources_dim, sources.data_sources_dim);

    // Store information of this source
    array_push_back(&sources.data_sources, sources.data_sources_dim, (f32)CYLINDER_SOURCE);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.geometry_id);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.px);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.py);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.pz);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.rad);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.length);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.energy);

    // Save the seed
    array_push_back(&sources.seeds, sources.seeds_dim, src.seed);

}

// Add a planar source on the simulation
void SourceBuilder::add_source(PlanarSource src) {
    sources.nb_sources++;
    // Store the address to access to this source
    array_push_back(&sources.ptr_sources, sources.ptr_sources_dim, sources.data_sources_dim);

    // Store information of this source
    array_push_back(&sources.data_sources, sources.data_sources_dim, (f32)PLANAR_SOURCE);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.geometry_id);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.px);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.py);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.pz);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.width);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.length);
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
    array_push_back(&sources.data_sources, sources.data_sources_dim, (f32)CONE_BEAM_SOURCE);
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

// Add a voxelized source
void SourceBuilder::add_source(VoxelizedSource src) {
    sources.nb_sources++;
  
    
    // Store the address to access to this source
    array_push_back(&sources.ptr_sources, sources.ptr_sources_dim, sources.data_sources_dim);

    // Store information of this source
    array_push_back(&sources.data_sources, sources.data_sources_dim, (f32)VOXELIZED_SOURCE);

    array_push_back(&sources.data_sources, sources.data_sources_dim, src.geometry_id);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.px);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.py);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.pz);

    array_push_back(&sources.data_sources, sources.data_sources_dim, (f32)src.nb_vox_x);
    array_push_back(&sources.data_sources, sources.data_sources_dim, (f32)src.nb_vox_y);
    array_push_back(&sources.data_sources, sources.data_sources_dim, (f32)src.nb_vox_z);

    array_push_back(&sources.data_sources, sources.data_sources_dim, src.spacing_x);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.spacing_y);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.spacing_z);

    array_push_back(&sources.data_sources, sources.data_sources_dim, src.energy);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.tot_activity);
    array_push_back(&sources.data_sources, sources.data_sources_dim, src.activity_size); // Nb CDF

    // Emission type
    if (src.source_type == "mono") {
        array_push_back(&sources.data_sources, sources.data_sources_dim, (f32)EMISSION_MONO);
    } else if (src.source_type == "back2back") {
        array_push_back(&sources.data_sources, sources.data_sources_dim, (f32)EMISSION_BACK2BACK);
    }

     
    // Store index to access to the CDF
    array_append_array(&sources.data_sources, sources.data_sources_dim, &(src.activity_index), src.activity_size);
    // Store the CDF of the activities
    array_append_array(&sources.data_sources, sources.data_sources_dim, &(src.activity_cdf), src.activity_size);

    // Save the seed
    array_push_back(&sources.seeds, sources.seeds_dim, src.seed);

    // Count the activity of this source to the total activity
    tot_activity += src.tot_activity;
}


// Copy source data to the GPU
void SourceBuilder::copy_source_cpu2gpu() {

    // First allocate the GPU mem for the scene
    HANDLE_ERROR( cudaMalloc((void**) &dsources.ptr_sources, sources.ptr_sources_dim*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dsources.data_sources, sources.data_sources_dim*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dsources.seeds, sources.seeds_dim*sizeof(ui32)) );

    // Copy data to the GPU
    dsources.nb_sources = sources.nb_sources;
    dsources.ptr_sources_dim = sources.ptr_sources_dim;
    dsources.data_sources_dim = sources.data_sources_dim;
    dsources.seeds_dim = sources.seeds_dim;

    HANDLE_ERROR( cudaMemcpy(dsources.ptr_sources, sources.ptr_sources,
                             sources.ptr_sources_dim*sizeof(ui32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dsources.data_sources, sources.data_sources,
                             sources.data_sources_dim*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dsources.seeds, sources.seeds,
                             sources.seeds_dim*sizeof(ui32), cudaMemcpyHostToDevice) );
}

















#endif
















