#ifndef GUARD_GGEMS_PHYSICS_GGEMSPRIMARYPARTICLES_HH
#define GUARD_GGEMS_PHYSICS_GGEMSPRIMARYPARTICLES_HH

/*!
  \file GGEMSPrimaryParticles.hh

  \brief Structure storing the primary particle buffers for both OpenCL and GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday December 16, 2019
*/

#include "GGEMS/global/GGEMSConfiguration.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \struct GGEMSPrimaryParticles_t
  \brief Structure storing informations about primary particles
*/
#ifdef OPENCL_COMPILER
typedef struct __attribute__((aligned (1))) GGEMSPrimaryParticles_t
#else
#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
typedef struct PACKED GGEMSPrimaryParticles_t
#endif
{
  GGfloat p_E_[MAXIMUM_PARTICLES]; /*!< Energies of particles */
  GGfloat p_dx_[MAXIMUM_PARTICLES]; /*!< Position of the particle in x */
  GGfloat p_dy_[MAXIMUM_PARTICLES]; /*!< Position of the particle in y */
  GGfloat p_dz_[MAXIMUM_PARTICLES]; /*!< Position of the particle in z */
  GGfloat p_px_[MAXIMUM_PARTICLES]; /*!< Momentum of the particle in x */
  GGfloat p_py_[MAXIMUM_PARTICLES]; /*!< Momentum of the particle in y */
  GGfloat p_pz_[MAXIMUM_PARTICLES]; /*!< Momentum of the particle in z */
  GGfloat p_tof_[MAXIMUM_PARTICLES]; /*!< Time of flight */

  GGuint p_geometry_id_[MAXIMUM_PARTICLES]; /*!< current geometry crossed by the particle */
  GGushort p_E_index_[MAXIMUM_PARTICLES]; /*!< Energy index within CS and Mat tables */
  GGuchar p_scatter_order_[MAXIMUM_PARTICLES]; /*!< Scatter order, usefull for the imagery */

  GGfloat p_next_interaction_distance_[MAXIMUM_PARTICLES]; /*!< Distance to the next interaction */
  GGuchar p_next_discrete_process_[MAXIMUM_PARTICLES]; /*!< Next process */

  GGuchar p_status_[MAXIMUM_PARTICLES]; /*!< */
  GGuchar p_level_[MAXIMUM_PARTICLES]; /*!< */
  GGuchar p_pname_[MAXIMUM_PARTICLES]; /*!< particle name (photon, electron, etc) */
} GGEMSPrimaryParticles;
#ifndef OPENCL_COMPILER
#ifdef _MSC_VER
#pragma pack(pop)
#endif
#endif

#endif // GUARD_GGEMS_PHYSICS_GGEMSPRIMARYPARTICLES_HH
