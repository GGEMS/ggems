// GGEMS Copyright (C) 2015

#ifndef MATERIALS_CUH
#define MATERIALS_CUH


#include "global.cuh"
#include "txt_reader.cuh"
#include "range_cut.cuh"

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
        void load_elements();
        f32 get_density( std::string mat_name );
        ui16 get_nb_elements( std::string mat_name );
        std::string get_element_name( std::string mat_name, ui16 index);
        f32 get_atom_num_dens(std::string mat_name, ui16 index );
        f32 get_mass_fraction( std::string mat_name, ui16 index);
        ui16 get_element_Z(std::string elt_name);
        f32 get_element_A(std::string elt_name);
        f32 get_element_pot( std::string elt_name );
        f32 get_rad_len( std::string mat_name );
        ui8 get_element_state( ui16 Z );

        void compute_ioni_parameters( std::string mat_name );
        f32 get_mean_excitation();
        f32 get_X0_density();
        f32 get_X1_density();
        f32 get_D0_density();
        f32 get_C_density();
        f32 get_A_density();
        f32 get_M_density();
        f32 get_F1_fluct();
        f32 get_F2_fluct();
        f32 get_Energy0_fluct();
        f32 get_Energy1_fluct();
        f32 get_Energy2_fluct();
        f32 get_LogEnergy1_fluct();
        f32 get_LogEnergy2_fluct();

        std::map<std::string, aMaterial> materials;

    private:
        TxtReader m_txt_reader;
        void m_add_elements( std::string elt_name, ui16 elt_Z, f32 elt_A, f32 elt_pot );
        std::map<std::string, ui16>  elements_Z;
        std::map<std::string, f32> elements_A;
        std::map<std::string, f32> elements_pot;

        f32 m_read_X0_density( ui16 Z );
        f32 m_read_X1_density( ui16 Z );
        f32 m_read_D0_density( ui16 Z );
        f32 m_read_C_density( ui16 Z );
        f32 m_read_A_density( ui16 Z );
        f32 m_read_M_density( ui16 Z );

        // Ioni params
        f32 m_MeanExcEnergy, m_LogMeanExcEnergy, m_TotNbOfElectPerVolume,
            m_X0, m_X1, m_D0, m_C, m_A, m_M, m_rad_len;
        f32 m_F1fluct, m_F2fluct, m_Energy0fluct, m_Energy1fluct,
            m_Energy2fluct, m_LogEnergy1fluct, m_LogEnergy2fluct;

};

// Table containing every definition of the materials used in the world
struct MaterialsTable {
    ui32 nb_materials;              // n
    ui32 nb_elements_total;         // k

    ui16 *nb_elements;        // n
    ui16 *index;              // n

    ui16 *mixture;            // k
    f32 *atom_num_dens;       // k
    f32 *mass_fraction;       // k

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

// Struct that handle CPU&GPU CS data
struct MaterialsData {
    MaterialsTable data_h;
    MaterialsTable data_d;
};


// This class is used to build the material table
class Materials {
    public:
        Materials();
        // Load data provided by the user        
        void load_materials_database(std::string filename);        
        void initialize(std::vector<std::string> mats_list, GlobalSimulationParameters params);

        MaterialsData tables;

        void print();
        
    private:
        //ui16 m_get_material_index(std::string material_name);

        bool m_check_mandatory();
        void m_copy_materials_table_cpu2gpu();
        void m_build_materials_table(GlobalSimulationParameters params, std::vector<std::string> mats_list);
        //void m_free_materials_table();

        ui32 m_nb_materials;              // n
        ui32 m_nb_elements_total;         // k

        MaterialsDataBase m_material_db;
        std::vector<std::string> m_materials_list_name;
        RangeCut m_rangecut;

};


#endif
