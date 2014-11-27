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


#include "G4Material.hh"
//#include "G4EmCalculator.hh"
//#include "G4ParticleTable.hh"
#include "G4PhysicalConstants.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"

#include "geometry_builder.cuh"
#include "constants.cuh"
#include "global.cuh"

// To handle one material
class Material {
    public:
        Material();
        std::vector<std::string> mixture_Z;
        std::vector<f32> mixture_f;
        std::string name;
        f32 density;
        ui16 nb_elements;
};

// Open and load the material database
class MaterialDataBase {
    public:
        MaterialDataBase();
        void load_materials(std::string);
        void load_elements(std::string);

        std::map<std::string, Material> materials;
        std::map<std::string, ui16>  elements_Z;
        std::map<std::string, f32> elements_A;

    private:
        void skip_comment(std::istream &);
        std::string remove_white_space(std::string txt);

        std::string read_element_name(std::string);
        i32 read_element_Z(std::string);
        f32 read_element_A(std::string);

        std::string read_material_name(std::string);
        f32 read_material_density(std::string);
        ui16 read_material_nb_elements(std::string);
        std::string read_material_element(std::string);
        f32 read_material_fraction(std::string);
};

// Table containing every definition of the materials used in the world
struct MaterialsTable {
    ui32 nb_materials;              // n
    ui32 nb_elements_total;         // k

    ui16 *nb_elements;        // n
    ui16 *index;              // n

    ui16 *mixture;            // k
    f32 *atom_num_dens;                   // k

    f32 *nb_atoms_per_vol;                // n
    f32 *nb_electrons_per_vol;            // n
    f32 *electron_mean_excitation_energy; // n
    f32 *rad_length;                      // n

    //parameters of the density correction
    f32 *fX0;                             // n
    f32 *fX1;
    f32 *fD0;
    f32 *fC;
    f32 *fA;
    f32 *fM;

  // parameters of the energy loss fluctuation model:
    f32 *fF1;
    f32 *fF2;
    f32 *fEnergy0;
    f32 *fEnergy1;
    f32 *fEnergy2;
    f32 *fLogEnergy1;
    f32 *fLogEnergy2;
    f32 *fLogMeanExcitationEnergy;

    f32 *density;
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
