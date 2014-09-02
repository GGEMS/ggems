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

#ifndef GGEMS_CU
#define GGEMS_CU

#include "ggems.cuh"

///////// Simulation Builder class ////////////////////////////////////////////////

SimulationBuilder::SimulationBuilder() {
    kind_of_device = CPU_DEVICE;
}

// Set the geometry of the simulation
void SimulationBuilder::set_geometry(Geometry obj) {
    Geometry = obj;
}

// Set the materials definition associated to the geometry
void SimulationBuilder::set_materials(MaterialsTable tab) {
    materials = tab;
}

// Set the hardware used for the simulation CPU or GPU (CPU by default)
void SimulationBuilder::set_hardware_target(std::string value) {
    if (value == "GPU") {
        kind_of_device = GPU_DEVICE;
    } else {
        kind_of_device = CPU_DEVICE;
    }
}

// Add a process to the physics list
void SimulationBuilder::set_process(std::string process_name) {

    if (process_name == "Compton")
    {
        m_physics_list[PHOTON_COMPTON] = ENABLED;
        // printf("add Compton\n");
    }
    else if (process_name == "PhotoElectric")
    {
        m_physics_list[PHOTON_PHOTOELECTRIC] = ENABLED;
        // printf("add photoelectric\n");
    }
    else if (process_name == "eIonisation")
    {
        m_physics_list[ELECTRON_IONISATION] = ENABLED;
        // printf("add photoelectric\n");
    }
    else if (process_name == "eBremsstrahlung")
    {
        m_physics_list[ELECTRON_BREMSSTRAHLUNG] = ENABLED;
        // printf("add photoelectric\n");
    }
    else if (process_name == "eMultipleScattering")
    {
        m_physics_list[ELECTRON_MSC] = ENABLED;
        // printf("add photoelectric\n");
    }
    else
    {
        printf("[\033[31;03mWARNING\033[00m] \"%s\" is an unknow physics process\n",process_name.c_str());
        printf("process available are : \n-Compton\n-PhotoElectric\n-eIonisation\n-eBremsstrahlung\n-eMultipleScattering\n");
        cancel_simulation();
    }

}

// Abort the current simulation
void SimulationBuilder::cancel_simulation() {
    printf("[\033[31;03mSimulation aborded\033[00m]\n");
    exit(EXIT_FAILURE);
}

#endif
