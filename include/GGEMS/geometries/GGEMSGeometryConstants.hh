#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSGEOMETRYCONSTANTS_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSGEOMETRYCONSTANTS_HH

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