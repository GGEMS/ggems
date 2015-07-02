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

#ifndef CYLINDER_SOURCE_CUH
#define CYLINDER_SOURCE_CUH

#include "global.cuh"
#include "particles.cuh"
#include "prng.cuh"
#include "constants.cuh"

// External function
__host__ __device__ void cylinder_source_primary_generator(ParticleStack particles, ui32 id,
                                                        f32 px, f32 py, f32 pz,
                                                        f32 rad, f32 length, f32 energy,
                                                        ui8 type, ui32 geom_id);

// Sphere
class CylinderSource {
    public:
        CylinderSource(f32 vpx, f32 vpy, f32 vpz, f32 vrad, f32 vlen, 
                       ui32 vseed, std::string vname, ui32 vgeom_id);
        CylinderSource();
        void set_position(f32 vpx, f32 vpy, f32 vpz);
        void set_histpoint(f32 venergy, f32 vpart);
        void set_radius(f32 vrad);
        void set_length(f32 vlen);
        void set_energy(f32 venergy);
        void set_seed(ui32 vseed);
        void set_in_geometry(ui32 vgeometry_id);
        void set_source_name(std::string vsource_name);

        f32 px, py, pz, rad, length, energy;
        ui32 seed, geometry_id;
        std::string source_name;

        std::vector<f32> energy_hist, partpdec;
        
    private:
};

#endif
