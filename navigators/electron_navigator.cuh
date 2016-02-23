// GGEMS Copyright (C) 2015

/*!
 * \file electron_navigator.cuh
 * \brief
 * \author Yannick Lemaréchal <yannick.lemarechal@univ-brest.fr>
 * \version 0.1
 * \date 20 novembre 2015
 *
 *
 *
 */

#ifndef ELECTRON_NAVIGATOR_CUH
#define ELECTRON_NAVIGATOR_CUH

#include "electron.cuh"
#include "dose_calculator.cuh"
#include "voxelized.cuh"
#include "raytracing.cuh"
#include "transport_navigator.cuh"


__host__ __device__ void e_read_CS_table (//                             ParticleStack particles,
    //ui32 id,
    ui16 mat, //material
    f32 energy, //energy of particle
    ElectronsCrossSectionTable electron_CS_table,
    ui8 &next_discrete_process, //next discrete process id
    ui32 &table_index,
    f32 & next_interaction_distance,
    f32 & dedxeIoni,
    f32 & dedxeBrem,
    f32 & erange,
    f32 & lambda,
    f32 randomnumbereBrem,
    f32 randomnumbereIoni,
    GlobalSimulationParametersData parameters );

__host__ __device__ f32 compute_lambda_for_scaled_energy(f32 CS, f32 e, ElectronsCrossSectionTable table, ui16 mat_id);

__host__ __device__ f32 StepFunction ( f32 Range );

__host__ __device__ f32 gTransformToGeom ( f32 TPath,f32 currentRange,f32 currentLambda,f32 currentEnergy,
                                           f32 &par1, f32 &par2, ElectronsCrossSectionTable electron_CS_table, ui8 mat );

__host__ __device__ f32 GetEnergy ( f32 Range, ElectronsCrossSectionTable d_table, ui8 mat );

__host__ __device__ f32 GetLambda ( f32 Range, ElectronsCrossSectionTable d_table, ui8 mat );

//__host__ __device__ f32 gTransformToGeom ( f32 TPath,f32 currentRange,f32 currentLambda,f32 currentEnergy,f32 *par1,f32 *par2, ElectronsCrossSectionTable electron_CS_table, int mat );

__host__ __device__ f32 eLoss (f32 LossLength, f32 Ekine, f32 dedxeIoni, f32 dedxeBrem, f32 erange,
                               ElectronsCrossSectionTable d_table, ui8 mat, MaterialsTable materials, ParticlesData &particles,
                               GlobalSimulationParametersData parameters, ui32 id );

__host__ __device__ f32 eFluctuation (f32 meanLoss, f32 Ekine, MaterialsTable materials, ParticlesData &particles, ui32 id, ui8 id_mat );

__host__ __device__ f32 GlobalMscScattering (f32 GeomPath, f32 cutstep, f32 CurrentRange, f32 CurrentEnergy, f32 CurrentLambda, f32 dedxeIoni,
                                             f32 dedxeBrem, ElectronsCrossSectionTable d_table, ui8 mat, ParticlesData &particles, ui32 id,
                                             f32 par1, f32 par2, MaterialsTable materials, DoseData &dosi, ui32xyzw index_phantom,
                                             VoxVolumeData phantom, GlobalSimulationParametersData parameters );

__host__ __device__ void eMscScattering (f32 tPath, f32 zPath, f32 currentRange, f32 currentLambda, f32 currentEnergy, f32 par1, f32 par2,
                                         ParticlesData &particles, ui32 id, MaterialsTable materials, ui8 mat,
                                         VoxVolumeData phantom, ui32xyzw index_phantom );

__host__ __device__ void gLatCorrection (f32xyz currentDir, f32 tPath, f32 zPath, f32 currentTau, f32 phi, f32 sinth,
                                         ParticlesData &particles, ui32 id, f32 safety );

__host__ __device__ f32 eCosineTheta ( f32 trueStep,f32 currentRange,f32 currentLambda,f32 currentEnergy,f32 *currentTau,
                                       f32 par1,f32 par2, MaterialsTable materials, ui8 id_mat, ui32 id, ParticlesData &particles );

__host__ __device__ f32 eSimpleScattering (f32 xmeanth, f32 x2meanth, ui32 id, ParticlesData &particles );

__host__ __device__ f32 gGeomLengthLimit ( f32 gPath, f32 currentLambda, f32 currentRange, f32 par1, f32 par3 );

__host__ __device__ SecParticle eSampleSecondarieElectron (f32 CutEnergy, ParticlesData &particles, ui32 id);

__host__ __device__
void eSampleSecondarieGamma ( f32 minEnergy, f32 maxEnergy, ParticlesData &particles, ui32 id, MaterialsTable materials, ui8 id_mat );

__host__ __device__ f32xyz CorrUnit ( f32xyz u, f32xyz v,f32 uMom, f32 vMom );




#endif
