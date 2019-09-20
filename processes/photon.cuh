// GGEMS Copyright (C) 2017

/*!
 * \file photon.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.2
 * \date 18 novembre 2015
 *
 * v0.2: JB - Change all structs and remove CPU exec
 *
 */

#ifndef PHOTON_CUH
#define PHOTON_CUH

#include "particles.cuh"
#include "materials.cuh"
#include "global.cuh"
#include "physical_constants.cuh"
#include "prng.cuh"
#include "fun.cuh"
#include "sandia_table.cuh"
#include "shell_data.cuh"
#include "vector.cuh"

// Cross section table for photon particle
struct PhotonCrossSectionData{
    f32* E_bins;                // n

    f32* Compton_Std_CS;        // n*k

    f32* Photoelectric_Std_CS;  // n*k
    f32* Photoelectric_Std_xCS; // n*101 (Nb of Z)

    f32* Rayleigh_Lv_CS;        // n*k
    f32* Rayleigh_Lv_SF;        // n*101 (Nb of Z)
    f32* Rayleigh_Lv_xCS;       // n*101 (Nb of Z)

    f32 E_min;
    f32 E_max;
    ui32 nb_bins;         // n
    ui32 nb_mat;          // k
};

// Utils
__host__ __device__ f32 get_CS_from_table(f32 *E_bins, f32 *CSTable, f32 energy,
                                          ui32 E_index, ui32 CS_index);

// Compton - model standard G4
__host__ __device__ f32 Compton_CSPA_standard(f32 E, ui16 Z);
__host__ __device__ f32 Compton_CS_standard(const MaterialsData *materials, ui16 mat, f32 E);

__host__ __device__ SecParticle Compton_SampleSecondaries_standard(ParticlesData *particles,
                                                                   f32 cutE,
                                                                   ui8 flag_electron,
                                                                   ui32 id );
__host__ __device__ void Compton_standard(ParticlesData *particles,
                                          ui32 id);

//

// PhotoElectric - model standard G4
__host__ __device__ f32 Photoelec_CSPA_standard(f32 E, ui16 Z);
__host__ __device__ f32 Photoelec_CS_standard(const MaterialsData *materials,
                                              ui16 mat, f32 E);
__host__ __device__ SecParticle Photoelec_SampleSecondaries_standard(ParticlesData *particles,
                                                                     const MaterialsData *mat,
                                                                     const PhotonCrossSectionData *photon_CS_table,
                                                                     ui32 E_index,
                                                                     f32 cutE,
                                                                     ui16 matindex, ui8 flag_electron,
                                                                     ui32 id);

//

// Rayleigh scattering - model Livermore G4
__host__ __device__ ui16 Rayleigh_LV_CS_CumulIntervals(ui32 pos);
__host__ __device__ ui16 Rayleigh_LV_CS_NbIntervals(ui32 pos);
__host__ __device__ ui16 Rayleigh_LV_SF_CumulIntervals(ui32 pos);
__host__ __device__ ui16 Rayleigh_LV_SF_NbIntervals(ui32 pos);

f32* Rayleigh_CS_Livermore_load_data();
f32* Rayleigh_SF_Livermore_load_data();
__host__ __device__ f32 Rayleigh_CSPA_Livermore(f32* rayl_cs, f32 E, ui16 Z);
__host__ __device__ f32 Rayleigh_CS_Livermore(const MaterialsData *materials,
                                              f32* rayl_cs, ui16 mat, f32 E);
__host__ __device__ f32 Rayleigh_SF_Livermore(f32* rayl_sf, f32 E, i32 Z);
__host__ __device__ void Rayleigh_SampleSecondaries_Livermore(ParticlesData *particles,
                                                              const MaterialsData *mat,
                                                              const PhotonCrossSectionData *photon_CS_table,
                                                              ui32 E_index,
                                                              ui16 matindex,
                                                              ui32 id);
/*
__host__ __device__ void _Rayleigh_SampleSecondaries_Livermore(ParticlesData particles,
                                                              const MaterialsData *mat,
                                                              PhotonCrossSectionTable photon_CS_table,
                                                              ui32 E_index,
                                                              ui16 matindex,
                                                              ui32 id);
*/

#endif
