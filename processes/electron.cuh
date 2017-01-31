// GGEMS Copyright (C) 2017

/*!
 * \file electron.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.2
 * \date 18 novembre 2015
 *
 * v0.2: JB - Change all structs and remove CPU exec
 *
 */

#include "global.cuh"
#include "materials.cuh"

#ifndef ELECTRON_CUH
#define ELECTRON_CUH

// Cross section table for electron and positrons
struct ElectronsCrossSectionData
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

f32 ElectronIonisation_DEDX(const MaterialsData *h_materials, f32 Ekine, ui8 mat_id );
f32 ElectronIonisation_CS( const MaterialsData *h_materials, f32 Ekine, ui16 mat_id );
f32 ElectronBremsstrahlung_DEDX( const MaterialsData *h_materials, f32 Ekine, ui8 mat_id );
__host__ __device__ f32 ElectronBremmsstrahlung_CSPA( f32 Z, f32 cut, f32 Ekine );
__host__ __device__ f32 ElectronBremmsstrahlung_CS( const MaterialsData *h_materials, f32 Ekine, f32 min_E, ui8 mat_id );
f32 ElectronMultipleScattering_CS( const MaterialsData *h_material, f32 Ekine, ui8 mat_id);

#endif
