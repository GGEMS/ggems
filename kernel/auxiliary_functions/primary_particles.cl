#ifndef GUARD_GGEMS_AUXILIARYFUNCTIONS_PRIMARYPARTICLES_CL
#define GUARD_GGEMS_AUXILIARYFUNCTIONS_PRIMARYPARTICLES_CL

/*!
  \file primary_particles.cl

  \brief Structure storing primary particles on device

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thursday October 10, 2019
*/

#include "GGEMS/global/ggems_configuration.hh"

/*!
  \struct PrimaryParticles_t
  \brief Structure storing informations about primary particles in OpenCL device
*/
typedef struct __attribute__((aligned (1))) PrimaryParticles_t
{
  unsigned long number_of_primaries_;
  
  float p_E_[MAXIMUM_PARTICLES];
  float p_dx_[MAXIMUM_PARTICLES];
  float p_dy_[MAXIMUM_PARTICLES];
  float p_dz_[MAXIMUM_PARTICLES];
  float p_px_[MAXIMUM_PARTICLES];
  float p_py_[MAXIMUM_PARTICLES];
  float p_pz_[MAXIMUM_PARTICLES];
  float p_tof_[MAXIMUM_PARTICLES];

  unsigned int p_prng_state_1_[MAXIMUM_PARTICLES];
  unsigned int p_prng_state_2_[MAXIMUM_PARTICLES];
  unsigned int p_prng_state_3_[MAXIMUM_PARTICLES];
  unsigned int p_prng_state_4_[MAXIMUM_PARTICLES];
  unsigned int p_prng_state_5_[MAXIMUM_PARTICLES];

  unsigned int p_geometry_id_[MAXIMUM_PARTICLES];
  unsigned short p_E_index_[MAXIMUM_PARTICLES];
  unsigned short p_scatter_order_[MAXIMUM_PARTICLES];

  float p_next_interaction_distance_[MAXIMUM_PARTICLES];
  unsigned char p_next_discrete_process_[MAXIMUM_PARTICLES];

  unsigned char p_status_[MAXIMUM_PARTICLES];
  unsigned char p_level_[MAXIMUM_PARTICLES];
  unsigned char p_pname_[MAXIMUM_PARTICLES];
}PrimaryParticles;

#endif // End of GUARD_GGEMS_AUXILIARYFUNCTIONS_PRIMARYPARTICLES_HH
