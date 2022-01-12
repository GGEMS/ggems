#ifndef GUARD_GGEMS_MATHS_GGEMSMATRIXOPERATIONS_HH
#define GUARD_GGEMS_MATHS_GGEMSMATRIXOPERATIONS_HH

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
  \file GGEMSMatrixOperations.hh

  \brief Definitions of functions using matrix

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday December 16, 2019
*/

#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

#include "GGEMS/maths/GGEMSMatrixTypes.hh"

 #ifdef __OPENCL_C_VERSION__

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat3 GGfloat44MultGGfloat3(global GGfloat44 const* matrix, GGfloat3 const* point)
  \param matrix - A matrix (4x4)
  \param point - Point in 3D (x, y, z)
  \return a vector 3x1
  \brief Compute the multiplication of matrix 4x4 and a point 3x1
*/
inline GGfloat3 GGfloat44MultGGfloat3(global GGfloat44 const* matrix, GGfloat3 const* point)
{
  GGfloat4 point4D;
  point4D.x = point->x; point4D.y = point->y; point4D.z = point->z; point4D.w = 1.0f;

  GGfloat4 row0;
  row0.x = matrix->m0_[0]; row0.y = matrix->m0_[1]; row0.z = matrix->m0_[2]; row0.w = matrix->m0_[3];

  GGfloat4 row1;
  row1.x = matrix->m1_[0]; row1.y = matrix->m1_[1]; row1.z = matrix->m1_[2]; row1.w = matrix->m1_[3];

  GGfloat4 row2;
  row2.x = matrix->m2_[0]; row2.y = matrix->m2_[1]; row2.z = matrix->m2_[2]; row2.w = matrix->m2_[3];

  GGfloat3 vector = {
    dot(row0, point4D),
    dot(row1, point4D),
    dot(row2, point4D)
  };

  return vector;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat3 GGfloat33MultGGfloat3(global GGfloat44 const* matrix, GGfloat3 const* point)
  \param matrix - A matrix (3x3)
  \param point - Point in 3D (x, y, z)
  \return a vector 3x1
  \brief Compute the multiplication of matrix 3x3 and a point 3x1
*/
inline GGfloat3 GGfloat33MultGGfloat3(global GGfloat44 const* matrix, GGfloat3 const* point)
{
  GGfloat3 point3D;
  point3D.x = point->x; point3D.y = point->y; point3D.z = point->z;

  GGfloat3 row0;
  row0.x = matrix->m0_[0]; row0.y = matrix->m0_[1]; row0.z = matrix->m0_[2];

  GGfloat3 row1;
  row1.x = matrix->m1_[0]; row1.y = matrix->m1_[1]; row1.z = matrix->m1_[2];

  GGfloat3 row2;
  row2.x = matrix->m2_[0]; row2.y = matrix->m2_[1]; row2.z = matrix->m2_[2];

  GGfloat3 vector = {
    dot(row0, point3D),
    dot(row1, point3D),
    dot(row2, point3D)
  };

  return vector;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat3 GGfloat33TransposeMultGGfloat3(global GGfloat44 const* matrix, GGfloat3 const* point)
  \param matrix - A matrix (3x3)
  \param point - Point in 3D (x, y, z)
  \return a vector 3x1
  \brief Compute the multiplication of matrix 3x3 transpose and a point 3x1
*/
inline GGfloat3 GGfloat33TransposeMultGGfloat3(global GGfloat44 const* matrix, GGfloat3 const* point)
{
  GGfloat3 point3D;
  point3D.x = point->x; point3D.y = point->y; point3D.z = point->z;

  GGfloat3 row0;
  row0.x = matrix->m0_[0]; row0.y = matrix->m1_[0]; row0.z = matrix->m2_[0];

  GGfloat3 row1;
  row1.x = matrix->m0_[1]; row1.y = matrix->m1_[1]; row1.z = matrix->m2_[1];

  GGfloat3 row2;
  row2.x = matrix->m0_[2]; row2.y = matrix->m1_[2]; row2.z = matrix->m2_[2];

  GGfloat3 vector = {
    dot(row0, point3D),
    dot(row1, point3D),
    dot(row2, point3D)
  };

  return vector;
}

#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat44 GGfloat44MultGGfloat44(GGfloat44 const* mat1, GGfloat44 const* mat2)
  \param mat1 - first matrix
  \param mat2 - second matrix
  \brief Perform the matrix (4x4) multiplication
  \return New matrix AxB
*/
inline GGfloat44 GGfloat44MultGGfloat44(GGfloat44 const* mat1, GGfloat44 const* mat2)
{
  GGfloat44 tmp;

  // Row 1
  tmp.m0_[0] = mat1->m0_[0]*mat2->m0_[0] + mat1->m0_[1]*mat2->m1_[0] + mat1->m0_[2]*mat2->m2_[0] + mat1->m0_[3]*mat2->m3_[0];
  tmp.m0_[1] = mat1->m0_[0]*mat2->m0_[1] + mat1->m0_[1]*mat2->m1_[1] + mat1->m0_[2]*mat2->m2_[1] + mat1->m0_[3]*mat2->m3_[1];
  tmp.m0_[2] = mat1->m0_[0]*mat2->m0_[2] + mat1->m0_[1]*mat2->m1_[2] + mat1->m0_[2]*mat2->m2_[2] + mat1->m0_[3]*mat2->m3_[2];
  tmp.m0_[3] = mat1->m0_[0]*mat2->m0_[3] + mat1->m0_[1]*mat2->m1_[3] + mat1->m0_[2]*mat2->m2_[3] + mat1->m0_[3]*mat2->m3_[3];

  // Row 2
  tmp.m1_[0] = mat1->m1_[0]*mat2->m0_[0] + mat1->m1_[1]*mat2->m1_[0] + mat1->m1_[2]*mat2->m2_[0] + mat1->m1_[3]*mat2->m3_[0];
  tmp.m1_[1] = mat1->m1_[0]*mat2->m0_[1] + mat1->m1_[1]*mat2->m1_[1] + mat1->m1_[2]*mat2->m2_[1] + mat1->m1_[3]*mat2->m3_[1];
  tmp.m1_[2] = mat1->m1_[0]*mat2->m0_[2] + mat1->m1_[1]*mat2->m1_[2] + mat1->m1_[2]*mat2->m2_[2] + mat1->m1_[3]*mat2->m3_[2];
  tmp.m1_[3] = mat1->m1_[0]*mat2->m0_[3] + mat1->m1_[1]*mat2->m1_[3] + mat1->m1_[2]*mat2->m2_[3] + mat1->m1_[3]*mat2->m3_[3];

  // Row 3
  tmp.m2_[0] = mat1->m2_[0]*mat2->m0_[0] + mat1->m2_[1]*mat2->m1_[0] + mat1->m2_[2]*mat2->m2_[0] + mat1->m2_[3]*mat2->m3_[0];
  tmp.m2_[1] = mat1->m2_[0]*mat2->m0_[1] + mat1->m2_[1]*mat2->m1_[1] + mat1->m2_[2]*mat2->m2_[1] + mat1->m2_[3]*mat2->m3_[1];
  tmp.m2_[2] = mat1->m2_[0]*mat2->m0_[2] + mat1->m2_[1]*mat2->m1_[2] + mat1->m2_[2]*mat2->m2_[2] + mat1->m2_[3]*mat2->m3_[2];
  tmp.m2_[3] = mat1->m2_[0]*mat2->m0_[3] + mat1->m2_[1]*mat2->m1_[3] + mat1->m2_[2]*mat2->m2_[3] + mat1->m2_[3]*mat2->m3_[3];

  // Row 4
  tmp.m3_[0] = mat1->m3_[0]*mat2->m0_[0] + mat1->m3_[1]*mat2->m1_[0] + mat1->m3_[2]*mat2->m2_[0] + mat1->m3_[3]*mat2->m3_[0];
  tmp.m3_[1] = mat1->m3_[0]*mat2->m0_[1] + mat1->m3_[1]*mat2->m1_[1] + mat1->m3_[2]*mat2->m2_[1] + mat1->m3_[3]*mat2->m3_[1];
  tmp.m3_[2] = mat1->m3_[0]*mat2->m0_[2] + mat1->m3_[1]*mat2->m1_[2] + mat1->m3_[2]*mat2->m2_[2] + mat1->m3_[3]*mat2->m3_[2];
  tmp.m3_[3] = mat1->m3_[0]*mat2->m0_[3] + mat1->m3_[1]*mat2->m1_[3] + mat1->m3_[2]*mat2->m2_[3] + mat1->m3_[3]*mat2->m3_[3];

  return tmp;
}

#endif // End of GUARD_GGEMS_MATHS_MATRIX_FUNCTIONS_HH
