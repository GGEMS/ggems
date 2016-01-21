// This file is part of GGEMS
//
// FIREwork is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FIREwork is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with FIREwork.  If not, see <http://www.gnu.org/licenses/>.
//
// GGEMS Copyright (C) 2013-2014 Julien Bert

#ifndef PHOTON_NAVIGATOR_CUH
#define PHOTON_NAVIGATOR_CUH

#include "constants.cuh"
#include "particles.cuh"
#include "cross_sections_builder.cuh"
#include "materials.cuh"
#include "global.cuh"
#include "digitizer.cuh"

#include "geometry_builder.cuh"
#include "aabb.cuh"
#include "sphere.cuh"
#include "meshed.cuh"
#include "voxelized.cuh"

#include "flat_panel_detector.cuh"

#include "vector.cuh"
#include "prng.cuh"

#include "photon.cuh"


__host__ __device__ void photon_navigator_simple(ParticleStack &particles, ui32 part_id,
                          Scene geometry, MaterialsTable materials,
                          PhotonCrossSectionTable photon_CS_table,
                          GlobalSimulationParameters parameters,
                          Pulses &pulses);

#ifndef SINGLE_PRECISION
    // Add function with double precision
__host__ __device__ void photon_navigator_double(ParticleStack &particles, ui32 part_id,
                          Scene geometry, MaterialsTable materials,
                          PhotonCrossSectionTable photon_CS_table,
                          GlobalSimulationParameters parameters,
                          Pulses &pulses);

__host__ __device__ void photon_navigator_raytracing_colli(ParticleStack &particles, ui32 part_id,
                          Scene geometry, MaterialsTable materials,
                          PhotonCrossSectionTable photon_CS_table,
                          GlobalSimulationParameters parameters,
                          Pulses &pulses);
#endif

#endif
