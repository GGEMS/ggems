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

void cpu_main_navigator(ParticleStack &particles, Scene geometry,
                        MaterialsTable materials, PhotonCrossSectionTable photon_CS_table,
                        GlobalSimulationParameters parameters, Singles &singles,
                        HistoryBuilder &history) {

    // For each particle
    ui32 id = 0;
    while (id < particles.size) {

#ifdef DEBUG
        printf(">>>> Particle %i\n", id);
#endif

        // Stepping loop, iterate the particle until the end
        ui32 istep = 0;
        while (particles.endsimu[id] == PARTICLE_ALIVE) {

            // If a photon
            if (particles.pname[id] == PHOTON) {
                cpu_photon_navigator(particles, id, geometry, materials, photon_CS_table,
                                     parameters, singles, history);
            }

            istep++;

            //printf(">>>>>> step %i\n", istep);

        } // istep

        // next particle
        ++id;

    } // id

}


void gpu_main_navigator(ParticleStack &particles, Scene geometry,
                        MaterialsTable materials, PhotonCrossSectionTable photon_CS_table,
                        GlobalSimulationParameters parameters, Singles &singles, ui32 gpu_block_size) {

    /*
    // Kernel
    dim3 threads, grid;
    threads.x = gpu_block_size;
    grid.x = (particles.size + gpu_block_size - 1) / gpu_block_size;
    kernel_photon_navigator<<<grid, threads>>>(particles, geometry, materials, photon_CS_table,
                                               parameters, singles);
    cuda_error_check("Error ", " Kernel_photon_navigator");
    */
}




#endif
