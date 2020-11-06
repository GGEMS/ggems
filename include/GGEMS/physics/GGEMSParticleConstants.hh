#ifndef GUARD_GGEMS_PHYSICS_GGEMSPARTICLECONSTANTS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSPARTICLECONSTANTS_HH

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
  \file GGEMSParticleConstants.hh

  \brief Storing particle states for GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday April 13, 2020
*/

#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

constant GGchar PRIMARY = 0; /*!< Primary particle */
constant GGchar GEOMETRY_BOUNDARY = 99; /*!< Particle on the boundary */
constant GGchar ALIVE = 0; /*!< Particle alive */
constant GGchar DEAD = 1; /*!< Particle dead */
constant GGchar FREEZE = 2; /*!< Particle freeze */
constant GGfloat OUT_OF_WORLD = FLT_MAX; /*!< Particle out of world */

constant GGchar PHOTON = 0; /*!< Photon particle */
constant GGchar ELECTRON = 1; /*!< Electron particle */
constant GGchar POSITRON = 2; /*!< Positron particle */

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSPARTICLECONSTANTS_HH