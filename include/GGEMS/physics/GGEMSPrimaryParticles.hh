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
  \file GGEMSPrimaryParticles.hh

  \brief Structure storing the primary particle buffers for both OpenCL and GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday December 16, 2019
*/

#include "GGEMS/physics/GGEMSParticleConstants.hh"

/*!
  \struct GGEMSPrimaryParticles_t
  \brief Structure storing informations about primary particles
*/
typedef struct GGEMSPrimaryParticles_t
{
  GGint particle_tracking_id; /*!< Particle id for tracking */

  GGfloat E_[MAXIMUM_PARTICLES]; /*!< Energies of particles */
  GGfloat dx_[MAXIMUM_PARTICLES]; /*!< Direction of the particle in x */
  GGfloat dy_[MAXIMUM_PARTICLES]; /*!< Direction of the particle in y */
  GGfloat dz_[MAXIMUM_PARTICLES]; /*!< Direction of the particle in z */
  GGfloat px_[MAXIMUM_PARTICLES]; /*!< Position of the particle in x */
  GGfloat py_[MAXIMUM_PARTICLES]; /*!< Position of the particle in y */
  GGfloat pz_[MAXIMUM_PARTICLES]; /*!< Position of the particle in z */
  GGchar scatter_[MAXIMUM_PARTICLES]; /*!< Index of scattered photon */

  GGint E_index_[MAXIMUM_PARTICLES]; /*!< Energy index within CS and Mat tables */
  GGint solid_id_[MAXIMUM_PARTICLES]; /*!< current solid crossed by the particle */

  GGfloat particle_solid_distance_[MAXIMUM_PARTICLES]; /*!< Distance from previous position to next position, OUT_OF_WORLD if no next position */
  GGfloat next_interaction_distance_[MAXIMUM_PARTICLES]; /*!< Distance to the next interaction */
  GGchar next_discrete_process_[MAXIMUM_PARTICLES]; /*!< Next process */

  GGchar status_[MAXIMUM_PARTICLES]; /*!< Status of the particle */
  GGchar level_[MAXIMUM_PARTICLES]; /*!< Level of the particle */
  GGchar pname_[MAXIMUM_PARTICLES]; /*!< particle name (photon, electron, etc) */

  GGfloat px_gl_[MAXIMUM_DISPLAYED_PARTICLES*MAXIMUM_INTERACTIONS]; /*!< Position in X of primary particles interactions */
  GGfloat py_gl_[MAXIMUM_DISPLAYED_PARTICLES*MAXIMUM_INTERACTIONS]; /*!< Position in Y of primary particles interactions */
  GGfloat pz_gl_[MAXIMUM_DISPLAYED_PARTICLES*MAXIMUM_INTERACTIONS]; /*!< Position in Z of primary particles interactions */
  GGint stored_particles_gl_[MAXIMUM_DISPLAYED_PARTICLES]; /*!< index to current interaction particle to store */
} GGEMSPrimaryParticles; /*!< Using C convention name of struct to C++ (_t deletion) */

#endif // GUARD_GGEMS_PHYSICS_GGEMSPRIMARYPARTICLESSTACK_HH
