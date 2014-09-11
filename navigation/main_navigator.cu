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

void cpu_main_navigator(ParticleBuilder particles, GeometryBuilder geometry,
                        MaterialBuilder materials, SimulationParameters parameters) {

    // For each particle
    unsigned int id = 0;
    while (id < particles.stack.size) {

        // Type of particle
        if (particles.stack.pname[id] == PHOTON) {
            cpu_photon_navigator(particles, id, geometry, materials, parameters);
        }

        // next particle
        ++id;

    } // id

}

#endif
