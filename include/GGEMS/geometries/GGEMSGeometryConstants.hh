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

/*!
  \namespace GGEMSTolerance
  \brief Namespace storing the tolerance for the float computations
*/
#ifndef OPENCL_COMPILER
namespace GGEMSTolerance
{
#endif
  __constant GGfloat EPSILON2 = 1.0e-02f; /*!< Epsilon of 0.01 */
  __constant GGfloat EPSILON3 = 1.0e-03f; /*!< Epsilon of 0.001 */
  __constant GGfloat EPSILON6 = 1.0e-06f; /*!< Epsilon of 0.000001 */
  __constant GGfloat GEOMETRY = 100.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::nm; /*!< Tolerance for the geometry navigation */
  #else
  1.e-6f; /*!< Tolerance for the geometry navigation */
  #endif
#ifndef OPENCL_COMPILER
}
#endif

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSPARTICLECONSTANTS_HH