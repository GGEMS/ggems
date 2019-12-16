#ifndef GUARD_GGEMS_PROCESSES_PRIMARY_PARTICLES
#define GUARD_GGEMS_PROCESSES_PRIMARY_PARTICLES

/*!
  \file primary_particles.hh

  \brief Structure storing the primary particle buffers for both OpenCL and GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday December 16, 2019
*/

#include "GGEMS/global/ggems_configuration.hh"
#include "GGEMS/opencl/types.hh"

/*!
  \struct PrimaryParticles_t
  \brief Structure storing informations about primary particles
*/
#ifdef OPENCL_COMPILER
typedef struct __attribute__((aligned (1))) PrimaryParticles_t
#else
#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
typedef struct PACKED PrimaryParticles_t
#endif
{
  f32cl_t p_E_[MAXIMUM_PARTICLES]; /*!< Energies of particles */
  f32cl_t p_dx_[MAXIMUM_PARTICLES]; /*!< Position of the particle in x */
  f32cl_t p_dy_[MAXIMUM_PARTICLES]; /*!< Position of the particle in y */
  f32cl_t p_dz_[MAXIMUM_PARTICLES]; /*!< Position of the particle in z */
  f32cl_t p_px_[MAXIMUM_PARTICLES]; /*!< Momentum of the particle in x */
  f32cl_t p_py_[MAXIMUM_PARTICLES]; /*!< Momentum of the particle in y */
  f32cl_t p_pz_[MAXIMUM_PARTICLES]; /*!< Momentum of the particle in z */
  f32cl_t p_tof_[MAXIMUM_PARTICLES]; /*!< Time of flight */

  uintcl_t p_geometry_id_[MAXIMUM_PARTICLES]; /*!< current geometry crossed by the particle */
  ushortcl_t p_E_index_[MAXIMUM_PARTICLES]; /*!< Energy index within CS and Mat tables */
  ushortcl_t p_scatter_order_[MAXIMUM_PARTICLES]; /*!< Scatter order, usefull for the imagery */

  f32cl_t p_next_interaction_distance_[MAXIMUM_PARTICLES]; /*!< Distance to the next interaction */
  ucharcl_t p_next_discrete_process_[MAXIMUM_PARTICLES]; /*!< Next process */

  ucharcl_t p_status_[MAXIMUM_PARTICLES]; /*!< */
  ucharcl_t p_level_[MAXIMUM_PARTICLES]; /*!< */
  ucharcl_t* p_pname_[MAXIMUM_PARTICLES]; /*!< particle name (photon, electron, etc) */
} PrimaryParticles;
#ifndef OPENCL_COMPILER
#ifdef _MSC_VER
#pragma pack(pop)
#endif
#endif

#endif // GUARD_GGEMS_PROCESSES_PRIMARY_PARTICLES
