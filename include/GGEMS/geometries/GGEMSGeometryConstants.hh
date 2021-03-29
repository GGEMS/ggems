#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSGEOMETRYCONSTANTS_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSGEOMETRYCONSTANTS_HH

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
  \file GGEMSGeometryConstants.hh

  \brief Geometry tolerances for navigation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday May 27, 2020
*/

#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

__constant GGfloat EPSILON2 = 1.0e-02f; /*!< Epsilon of 0.01 */
__constant GGfloat EPSILON3 = 1.0e-03f; /*!< Epsilon of 0.001 */
__constant GGfloat EPSILON6 = 1.0e-06f; /*!< Epsilon of 0.000001 */
__constant GGfloat GEOMETRY_TOLERANCE = 1.0e-04f; /*!< Geometry tolerance, 100 nm */

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSPARTICLECONSTANTS_HH