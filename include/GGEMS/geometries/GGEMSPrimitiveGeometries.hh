#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSPRIMITIVEGEOMETRIES_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSPRIMITIVEGEOMETRIES_HH

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
  \file GGEMSPrimitiveGeometries.hh

  \brief Structure storing some primitive geometries

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 22, 2020
*/

#include "GGEMS/maths/GGEMSMatrixTypes.hh"

/*!
  \struct GGEMSOBB_t
  \brief Structure storing OBB (Oriented Bounding Box) geometry
*/
typedef struct GGEMSOBB_t
{
  GGfloat44 matrix_transformation_; /*!< Matrix of transformation including angle of rotation */
  GGfloat3 border_min_xyz_; /*!< Min. of border in X, Y and Z */
  GGfloat3 border_max_xyz_; /*!< Max. of border in X, Y and Z */
} GGEMSOBB; /*!< Using C convention name of struct to C++ (_t deletion) */

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \struct GGEMSPoint3_t
  \brief Structure storing Point geometry
*/
typedef struct GGEMSPoint3_t
{
  GGfloat x_; /*!< X position of point */
  GGfloat y_; /*!< Y position of point */
  GGfloat z_; /*!< Z position of point */
} GGEMSPoint3; /*!< Using C convention name of struct to C++ (_t deletion) */

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \struct GGEMSVector3_t
  \brief Structure storing Vector geometry
*/
typedef struct GGEMSVector3_t
{
  GGfloat x_; /*!< X position of vector */
  GGfloat y_; /*!< Y position of vector */
  GGfloat z_; /*!< Z position of vector */
} GGEMSVector3; /*!< Using C convention name of struct to C++ (_t deletion) */

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \struct GGEMSSphere3_t
  \brief Structure storing Sphere geometry
*/
typedef struct GGEMSSphere3_t
{
  GGEMSPoint3 center_; /*!< Center of sphere */
  GGfloat     radius_; /*!< Radius of sphere */
} GGEMSSphere3; /*!< Using C convention name of struct to C++ (_t deletion) */

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \struct GGEMSPlane3_t
  \brief Structure storing Plane geometry
*/
typedef struct GGEMSPlane3_t
{
  GGEMSVector3 n_; /*!< normal plane. Points x on the plane satisfy dot(n, x) = d */
  GGfloat      d_; /*!< d = dot(n, p) for a given point p on the plane */
} GGEMSPlane3; /*!< Using C convention name of struct to C++ (_t deletion) */

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \struct GGEMSTriangle3_t
  \brief Structure storing Triangle geometry
*/
typedef struct GGEMSTriangle3_t
{
  GGEMSPoint3              pts_[3]; /*!< 3 points describing triangle */
  GGEMSSphere3             bounding_sphere_; // sphere around triangle */
  struct GGEMSTriangle3_t* next_triangle_; // use of next triangle (useful for meshed navigation) */
} GGEMSTriangle3; /*!< Using C convention name of struct to C++ (_t deletion) */

#endif // GUARD_GGEMS_GEOMETRIES_GGEMSPRIMITIVEGEOMETRIES_HH
