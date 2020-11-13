#ifndef GUARD_GGEMS_PHYSICS_GGEMSPRIMARYPARTICLES_HH
#define GUARD_GGEMS_PHYSICS_GGEMSPRIMARYPARTICLES_HH

// ************************************************************************
// * This file is part of GGEMS.                                          *
// *                                                                      *
// * GGEMS is free software: you can redistribute it and/or modify        *
// * it under the terms of the GNU General Public License as published by *
// * the Free Software Foundation, either version 3 of the License, or    *
// * (at your option) any later version.                                  *
// *                                                                      *
// * GGEMS is distributed in the hope that it will be useful,             *
// * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
// * GNU General Public License for more details.                         *
// *                                                                      *
// * You should have received a copy of the GNU General Public License    *
// * along with GGEMS.  If not, see <https://www.gnu.org/licenses/>.      *
// *                                                                      *
// ************************************************************************

/*!
  \file GGEMSPrimaryParticlesStack.hh

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
#pragma pack(push, 1)
typedef struct GGEMSPrimaryParticles_t
{
  GGfloat E_[MAXIMUM_PARTICLES]; /*!< Energies of particles */
  GGfloat dx_[MAXIMUM_PARTICLES]; /*!< Direction of the particle in x */
  GGfloat dy_[MAXIMUM_PARTICLES]; /*!< Direction of the particle in y */
  GGfloat dz_[MAXIMUM_PARTICLES]; /*!< Direction of the particle in z */
  GGfloat px_[MAXIMUM_PARTICLES]; /*!< Position of the particle in x */
  GGfloat py_[MAXIMUM_PARTICLES]; /*!< Position of the particle in y */
  GGfloat pz_[MAXIMUM_PARTICLES]; /*!< Position of the particle in z */
  GGfloat tof_[MAXIMUM_PARTICLES]; /*!< Time of flight for photon only, it time since generation of photon */

  //GGuint geometry_id_[MAXIMUM_PARTICLES]; /*!< current geometry crossed by the particle */
  GGint E_index_[MAXIMUM_PARTICLES]; /*!< Energy index within CS and Mat tables */
  GGint solid_id_[MAXIMUM_PARTICLES]; /*!< current solid crossed by the particle */

  GGfloat particle_solid_distance_[MAXIMUM_PARTICLES]; /*!< Distance from previous position to next position, OUT_OF_WORLD if no next position */
  GGfloat next_interaction_distance_[MAXIMUM_PARTICLES]; /*!< Distance to the next interaction */
  GGchar next_discrete_process_[MAXIMUM_PARTICLES]; /*!< Next process */

  GGchar status_[MAXIMUM_PARTICLES]; /*!< Status of the particle */
  GGchar level_[MAXIMUM_PARTICLES]; /*!< Level of the particle */
  GGchar pname_[MAXIMUM_PARTICLES]; /*!< particle name (photon, electron, etc) */

  GGint particle_tracking_id; /*!< Particle id for tracking */
} GGEMSPrimaryParticles; /*!< Using C convention name of struct to C++ (_t deletion) */
#pragma pack(pop)

#endif // GUARD_GGEMS_PHYSICS_GGEMSPRIMARYPARTICLESSTACK_HH
