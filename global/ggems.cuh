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

#ifndef GGEMS_CUH
#define GGEMS_CUH

#include <vector>
#include <string>

#include "../processes/constants.cuh"
#include "../processes/particles.cuh"
#include "../geometry/geometry_builder.cuh"
#include "../geometry/materials.cuh"
#include "../sources/source_builder.cuh"

#include "../geometry/aabb.cuh"
#include "../geometry/sphere.cuh"
#include "../geometry/meshed.cuh"
#include "../geometry/voxelized.cuh"
#include "../sources/point_source.cuh"


// Simulation parameters
struct SimulationParameters {
    char physics_list[NB_PROCESSES];
    char secondaries_list[NB_PARTICLES];
    int nb_of_particles;
    int nb_iterations;
    float time;
    int seed;
};


// Class to manage the hierarchical structure of the world
class SimulationBuilder {
    public:
        SimulationBuilder();

        void set_geometry(Geometry obj);
        void set_materials(MaterialsTable tab);
        void set_sources(Sources src);
        void set_particles(ParticleStack p);

        void set_hardware_target(std::string value);
        void set_process(std::string process_name);
        void set_secondary(std::string pname);
        void set_number_of_particles(unsigned int nb);
        void set_max_number_of_iterations(unsigned int nb);

        void init_simulation();
        void start_simulation();

    private:
        unsigned short int target;
        unsigned int nb_of_particles;
        unsigned int nb_of_iterations;
        unsigned int max_iteration;

        // Main elements of the simulation
        ParticleBuilder particles;
        GeometryBuilder geometry;
        MaterialBuilder materials;
        SourceBuilder sources;
        SimulationParameters parameters;





};



#endif
