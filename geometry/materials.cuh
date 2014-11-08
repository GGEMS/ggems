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

#ifndef MATERIALS_H
#define MATERIALS_H

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <map>
#include <algorithm>
#include <cfloat>

#include "G4Material.hh"
//#include "G4EmCalculator.hh"
//#include "G4ParticleTable.hh"
#include "G4PhysicalConstants.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"

#include "geometry_builder.cuh"
#include "constants.cuh"

// To handle one material
class Material {
    public:
        Material();
        std::vector<std::string> mixture_Z;
        std::vector<float> mixture_f;
        std::string name;
        float density;
        unsigned short int nb_elements;
};

// Open and load the material database
class MaterialDataBase {
    public:
        MaterialDataBase();
        void load_materials(std::string);
        void load_elements(std::string);

        std::map<std::string, Material> materials;
        std::map<std::string, unsigned short int>  elements_Z;
        std::map<std::string, float> elements_A;

    private:
        void skip_comment(std::istream &);
        std::string remove_white_space(std::string txt);

        std::string read_element_name(std::string);
        int read_element_Z(std::string);
        float read_element_A(std::string);

        std::string read_material_name(std::string);
        float read_material_density(std::string);
        unsigned short int read_material_nb_elements(std::string);
        std::string read_material_element(std::string);
        float read_material_fraction(std::string);
};

// Table containing every definition of the materials used in the world
struct MaterialsTable {
    unsigned int nb_materials;              // n
    unsigned int nb_elements_total;         // k

    unsigned short int *nb_elements;        // n
    unsigned short int *index;              // n

    unsigned short int *mixture;            // k
    float *atom_num_dens;                   // k

    float *nb_atoms_per_vol;                // n
    float *nb_electrons_per_vol;            // n
    float *electron_mean_excitation_energy; // n
    float *rad_length;                      // n

    //parameters of the density correction
    float *fX0;                             // n
    float *fX1;
    float *fD0;
    float *fC;
    float *fA;
    float *fM;

  // parameters of the energy loss fluctuation model:
    float *fF1;
    float *fF2;
    float *fEnergy0;
    float *fEnergy1;
    float *fEnergy2;
    float *fLogEnergy1;
    float *fLogEnergy2;
    float *fLogMeanExcitationEnergy;

    float *density;
};

// This class is used to build the material table
class MaterialBuilder {
    public:
        MaterialBuilder();
        void load_elements_database(std::string filename);
        void load_materials_database(std::string filename);

        void get_materials_table_from_world(GeometryBuilder World);

        MaterialsTable materials_table;

    private:
        MaterialDataBase material_db;


};


#endif
