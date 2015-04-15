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

#ifndef MAIN_NAVIGATOR_CU
#define MAIN_NAVIGATOR_CU

#include "main_navigator.cuh"

// Kernel to track photon particles
__global__ void kernel_photon_navigator(ParticleStack particles, Scene geometry, MaterialsTable materials,
                                        PhotonCrossSectionTable photon_CS_table,
                                        GlobalSimulationParameters parameters, Pulses pulses) {

    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particles.size) return;

    // Stepping loop, iterate the particle until the end
    ui32 istep = 0;
    while (particles.endsimu[id] == PARTICLE_ALIVE) {

        // Track photon
        photon_navigator(particles, id, geometry, materials, photon_CS_table,
                         parameters, pulses);
        
        //// SPECIAL CASE FOR RAYTRACING
        //photon_navigator_raytracing_colli(particles, id, geometry, materials, photon_CS_table,
          //                       parameters, pulses);

    }

    istep++;

}


void cpu_main_navigator(ParticleStack &particles, Scene geometry,
                        MaterialsTable materials, PhotonCrossSectionTable photon_CS_table,
                        GlobalSimulationParameters parameters, Pulses &pulses,
                        HistoryBuilder &history) {

#ifdef DEBUG
        printf("CPU: photon navigator\n");
#endif

    // For each particle
    ui32 id = 0;
    while (id < particles.size) {

        // Stepping loop, iterate the particle until the end
        ui32 istep = 0;
        while (particles.endsimu[id] == PARTICLE_ALIVE) {

            // If a photon
            if (particles.pname[id] == PHOTON) {
                photon_navigator(particles, id, geometry, materials, photon_CS_table,
                                 parameters, pulses);
              
                //// SPECIAL CASE FOR RAYTRACING
                //photon_navigator_raytracing_colli(particles, id, geometry, materials, photon_CS_table,
                  //               parameters, pulses);

                // Record this step if required
                if (history.record_flag == ENABLED) {
                    history.cpu_record_a_step(particles, id);
                }

            }

            istep++;
            
            #ifdef DEBUG
                printf("part %d >>>>>> step %i\n", id, istep);
            #endif
            
        } // istep

        // next particle
        ++id;

    } // id

}


void gpu_main_navigator(ParticleStack &particles, Scene geometry,
                        MaterialsTable materials, PhotonCrossSectionTable photon_CS_table,
                        GlobalSimulationParameters parameters, Pulses &pulses, ui32 gpu_block_size) {


#ifdef DEBUG
        printf("GPU: kernel photon navigator\n");
#endif

    // Kernel
    dim3 threads, grid;
    threads.x = gpu_block_size;
    grid.x = (particles.size + gpu_block_size - 1) / gpu_block_size;
    kernel_photon_navigator<<<grid, threads>>>(particles, geometry, materials, photon_CS_table,
                                               parameters, pulses);
    cuda_error_check("Error ", " Kernel_photon_navigator");

}




#endif
