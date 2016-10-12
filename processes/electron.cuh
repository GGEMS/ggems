// GGEMS Copyright (C) 2015

#include "global.cuh"
#include "materials.cuh"

#ifndef ELECTRON_CUH
#define ELECTRON_CUH

// Cross section table for electron and positrons
struct ElectronsCrossSectionTable
{
    f32* E;                    // n*k
    f32* eIonisationdedx;      // n*k
    f32* eIonisationCS;        // n*k
    f32* eBremdedx;            // n*k
    f32* eBremCS;              // n*k
    f32* eMSC;                 // n*k
    f32* eRange;               // n*k
    f32* eIonisation_E_CS_max; // k  |_ For CS from eIoni
    f32* eIonisation_CS_max;   // k  |
    f32 E_min;
    f32 E_max;
    ui32 nb_bins;       // n
    ui32 nb_mat;        // k
    f32 cutEnergyElectron;
    f32 cutEnergyGamma;   
};

// Struct that handle CPU&GPU CS data
struct ElectronsCrossSection
{
    ElectronsCrossSectionTable data_h;
    ElectronsCrossSectionTable data_d;
};

f32 ElectronIonisation_DEDX( MaterialsTable materials, f32 Ekine, ui8 mat_id );
f32 ElectronIonisation_CS( MaterialsTable materials, f32 Ekine, ui16 mat_id );
f32 ElectronBremsstrahlung_DEDX( MaterialsTable materials, f32 Ekine, ui8 mat_id );
__host__ __device__ f32 ElectronBremmsstrahlung_CSPA( f32 Z, f32 cut, f32 Ekine );
__host__ __device__ f32 ElectronBremmsstrahlung_CS( const MaterialsTable &materials, f32 Ekine, f32 min_E, ui8 mat_id );
f32 ElectronMultipleScattering_CS( MaterialsTable material, f32 Ekine, ui8 mat_id);

/*

class ElectronCrossSection
{
public :
    ElectronCrossSection() {}
    ~ElectronCrossSection() {}

    void initialize ( GlobalSimulationParameters params, MaterialsTable materials );

    void generateTable();

    void printElectronTables ( std::string );
    
    void m_copy_cs_table_cpu2gpu();

    inline ElectronsCrossSectionTable get_data_h()
    {
        return data_h;
    }
    inline ElectronsCrossSectionTable get_data_d()
    {
        return data_d;
    }

    ElectronsCrossSectionTable data_h;
    ElectronsCrossSectionTable data_d;

private :

    void Energy_table();
    void Range_table ( int id_mat );
    f32 GetDedx ( f32 Energy,int material );


    // eIoni functions
    void eIoni_DEDX_table ( int id_mat );
    f32 eIoniDEDX ( f32 Ekine,int id_mat );
    f32 DensCorrection ( f32 x, int id_mat );
    void eIoni_CrossSection_table ( int id_mat );
    f32 eIoniCrossSection ( int id_mat, f32 Ekine );
    f32 eIoniCrossSectionPerAtom ( int index, f32 Ekine );

    // Bremsthralung functions
    void eBrem_DEDX_table ( int id_mat );
    f32 eBremDEDX ( f32 Ekine,int id_mat );
    f32 eBremLoss ( f32 Z,f32 T,f32 Cut );
    void eBrem_CrossSection_table ( int );
    f32 eBremCrossSection ( f32 ,int );
    f32 eBremCrossSectionPerVolume ( f32 , int );
    f32 eBremCrossSectionPerAtom ( f32, f32, f32 );

    // MSC functions
    void eMSC_CrossSection_table ( int id_mat );
    f32 eMscCrossSection ( f32 Ekine, int id_mat );
    f32 eMscCrossSectionPerAtom ( f32 Ekine,  unsigned short int AtomNumber );



    ui32 nb_bins;         // n
    ui32 nb_mat;          // k
    f32 cutEnergyElectron;
    f32 cutEnergyGamma;

    f32 MaxKinEnergy;
    f32 MinKinEnergy;

    GlobalSimulationParameters parameters;
    MaterialsTable myMaterials;






};

*/

#endif
