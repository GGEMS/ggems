// GGEMS Copyright (C) 2015

#ifndef MATERIALS_CUH
#define MATERIALS_CUH


#include "G4Material.hh"
#include "G4PhysicalConstants.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"

#include "global.cuh"
#include "txt_reader.cuh"


// To handle one material
class aMaterial {
    public:
        aMaterial() {}
        ~aMaterial() {}
        std::vector<std::string> mixture_Z;
        std::vector<f32> mixture_f;
        std::string name;
        f32 density;
        ui16 nb_elements;
};

// Open and load the material database
class MaterialsDataBase {
    public:
        MaterialsDataBase();
        void load_materials(std::string);
        void load_elements(std::string);

        std::map<std::string, aMaterial> materials;
        std::map<std::string, ui16>  elements_Z;
        std::map<std::string, f32> elements_A;

    private:
        TxtReader m_txt_reader;

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

    // Cut
    f32 *photon_energy_cut;               // n
    f32 *electron_energy_cut;             // n

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
class Materials {
    public:
        Materials();
        // Load default data from GGEMS
        void load_elements_database();
        void load_materials_database();
        // Load data provided by the user
        void load_elements_database(std::string filename);
        void load_materials_database(std::string filename);

        //void add_materials_and_update_indices(std::vector<std::string> mats_list, ui16 *data, ui32 ndata);

        void initialize(GlobalSimulationParameters params, std::vector<std::string> mats_list);

        MaterialsTable data_h;
        MaterialsTable data_d;

    private:
        //ui16 m_get_material_index(std::string material_name);

        bool m_check_mandatory();
        void m_copy_materials_table_cpu2gpu();
        void m_build_materials_table(GlobalSimulationParameters params, std::vector<std::string> mats_list);
        //void m_free_materials_table();

        ui32 m_nb_materials;              // n
        ui32 m_nb_elements_total;         // k

        MaterialsDataBase m_material_db;

};


#endif
