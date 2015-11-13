// GGEMS Copyright (C) 2015

#ifndef PHOTON_CUH
#define PHOTON_CUH

#include "particles.cuh"
#include "materials.cuh"
#include "global.cuh"
#include "prng.cuh"
#include "fun.cuh"
#include "constants.cuh"
#include "sandia_table.cuh"
#include "shell_data.cuh"

// Cross section table for photon particle
struct PhotonCrossSectionTable{
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
                                            ui32 E_index, ui32 mat_index, ui32 nb_bins);

// Compton - model standard G4
__host__ __device__ f32 Compton_CSPA_standard(f32 E, ui16 Z);
__host__ __device__ f32 Compton_CS_standard(MaterialsTable materials, ui16 mat, f32 E);

__host__ __device__ SecParticle Compton_SampleSecondaries_standard(ParticleStack particles,
                                                                   f32 cutE,
                                                                   ui32 id,
                                                                   GlobalSimulationParameters parameters);
//

// PhotoElectric - model standard G4
__host__ __device__ f32 Photoelec_CSPA_standard(f32 E, ui16 Z);
__host__ __device__ f32 Photoelec_CS_standard(MaterialsTable materials,
                                                ui16 mat, f32 E);
__host__ __device__ SecParticle Photoelec_SampleSecondaries_standard(ParticleStack particles,
                                                                     MaterialsTable mat,
                                                                     PhotonCrossSectionTable photon_CS_table,
                                                                     ui32 E_index,
                                                                     f32 cutE,
                                                                     ui16 matindex,
                                                                     ui32 id,
                                                                     GlobalSimulationParameters parameters);

//

// Rayleigh scattering - model Livermore G4
__host__ __device__ ui16 Rayleigh_LV_CS_CumulIntervals(ui32 pos);
__host__ __device__ ui16 Rayleigh_LV_CS_NbIntervals(ui32 pos);
__host__ __device__ ui16 Rayleigh_LV_SF_CumulIntervals(ui32 pos);
__host__ __device__ ui16 Rayleigh_LV_SF_NbIntervals(ui32 pos);

f32* Rayleigh_CS_Livermore_load_data();
f32* Rayleigh_SF_Livermore_load_data();
__host__ __device__ f32 Rayleigh_CSPA_Livermore(f32* rayl_cs, f32 E, ui16 Z);
__host__ __device__ f32 Rayleigh_CS_Livermore(MaterialsTable materials,
                                                f32* rayl_cs, ui16 mat, f32 E);
__host__ __device__ f32 Rayleigh_SF_Livermore(f32* rayl_sf, f32 E, i32 Z);
__host__ __device__ void Rayleigh_SampleSecondaries_Livermore(ParticleStack particles,
                                                              MaterialsTable mat,
                                                              PhotonCrossSectionTable photon_CS_table,
                                                              ui32 E_index,
                                                              ui16 matindex,
                                                              ui32 id);


#endif
