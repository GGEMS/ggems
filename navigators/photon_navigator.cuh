// GGEMS Copyright (C) 2015

/*!
 * \file photon_navigator.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 20 novembre 2015
 *
 *
 *
 */

#ifndef PHOTON_NAVIGATOR_CUH
#define PHOTON_NAVIGATOR_CUH

#include "photon.cuh"


__host__ __device__ void photon_get_next_interaction ( ParticlesData &particles,
                                                       const GlobalSimulationParametersData parameters,
                                                       const PhotonCrossSectionTable photon_CS_table,
                                                       ui16 mat_id, ui32 part_id );

__host__ __device__ SecParticle photon_resolve_discrete_process (ParticlesData &particles,
                                                                 const GlobalSimulationParametersData parameters,
                                                                 const PhotonCrossSectionTable photon_CS_table,
                                                                 const MaterialsTable materials,
                                                                 ui16 mat_id, ui32 part_id );

#endif
