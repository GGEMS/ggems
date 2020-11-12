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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat3 GGfloat44MultGGfloat3(GGfloat44 const* matrix, GGfloat3 const* point)
  \param matrix - A matrix (4x4)
  \param point - Point in 3D (x, y, z)
  \return a vector 3x1
  \brief Compute the multiplication of matrix 4x4 and a point 3x1
*/
inline GGfloat3 GGfloat44MultGGfloat3(GGfloat44 const* matrix, GGfloat3 const* point)
{
  GGfloat4 point4D = {point->x, point->y, point->z, 1.0f};
  GGfloat4 row0 = {matrix->m0_[0], matrix->m0_[1], matrix->m0_[2], matrix->m0_[3]};
  GGfloat4 row1 = {matrix->m1_[0], matrix->m1_[1], matrix->m1_[2], matrix->m1_[3]};
  GGfloat4 row2 = {matrix->m2_[0], matrix->m2_[1], matrix->m2_[2], matrix->m2_[3]};

  #ifdef __OPENCL_C_VERSION__
  GGfloat3 vector = {
    dot(row0, point4D),
    dot(row1, point4D),
    dot(row2, point4D)
  };
  #else
  GGfloat3 vector = {
    row0.s0*point4D.s0 + row0.s1*point4D.s1 + row0.s2*point4D.s2 + row0.s3*point4D.s3,
    row1.s0*point4D.s0 + row1.s1*point4D.s1 + row1.s2*point4D.s2 + row1.s3*point4D.s3,
    row2.s0*point4D.s0 + row2.s1*point4D.s1 + row2.s2*point4D.s2 + row2.s3*point4D.s3
  };
  #endif

  return vector;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat3 GGfloat33MultGGfloat3(GGfloat33 const* matrix, GGfloat3 const* point)
  \param matrix - A matrix (3x3)
  \param point - Point in 3D (x, y, z)
  \return a vector 3x1
  \brief Compute the multiplication of matrix 3x3 and a point 3x1
*/
inline GGfloat3 GGfloat33MultGGfloat3(GGfloat33 const* matrix, GGfloat3 const* point)
{
  GGfloat3 point3D = {point->x, point->y, point->z};
  GGfloat3 row0 = {matrix->m0_[0], matrix->m0_[1], matrix->m0_[2]};
  GGfloat3 row1 = {matrix->m1_[0], matrix->m1_[1], matrix->m1_[2]};
  GGfloat3 row2 = {matrix->m2_[0], matrix->m2_[1], matrix->m2_[2]};

  #ifdef __OPENCL_C_VERSION__
  GGfloat3 vector = {
    dot(row0, point3D),
    dot(row1, point3D),
    dot(row2, point3D)
  };
  #else
  GGfloat3 vector = {
    row0.s0*point3D.s0 + row0.s1*point3D.s1 + row0.s2*point3D.s2,
    row1.s0*point3D.s0 + row1.s1*point3D.s1 + row1.s2*point3D.s2,
    row2.s0*point3D.s0 + row2.s1*point3D.s1 + row2.s2*point3D.s2
  };
  #endif

  return vector;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat44 GGfloat44MultGGfloat44(GGfloat44 const* A, GGfloat44 const* B)
  \param A - first matrix
  \param B - second matrix
  \brief Perform the matrix (4x4) multiplication
  \return New matrix AxB
*/
inline GGfloat44 GGfloat44MultGGfloat44(GGfloat44 const* A, GGfloat44 const* B)
{
  GGfloat44 tmp;

  // Row 1
  tmp.m0_[0] = A->m0_[0]*B->m0_[0] + A->m0_[1]*B->m1_[0] + A->m0_[2]*B->m2_[0] + A->m0_[3]*B->m3_[0];
  tmp.m0_[1] = A->m0_[0]*B->m0_[1] + A->m0_[1]*B->m1_[1] + A->m0_[2]*B->m2_[1] + A->m0_[3]*B->m3_[1];
  tmp.m0_[2] = A->m0_[0]*B->m0_[2] + A->m0_[1]*B->m1_[2] + A->m0_[2]*B->m2_[2] + A->m0_[3]*B->m3_[2];
  tmp.m0_[3] = A->m0_[0]*B->m0_[3] + A->m0_[1]*B->m1_[3] + A->m0_[2]*B->m2_[3] + A->m0_[3]*B->m3_[3];

  // Row 2
  tmp.m1_[0] = A->m1_[0]*B->m0_[0] + A->m1_[1]*B->m1_[0] + A->m1_[2]*B->m2_[0] + A->m1_[3]*B->m3_[0];
  tmp.m1_[1] = A->m1_[0]*B->m0_[1] + A->m1_[1]*B->m1_[1] + A->m1_[2]*B->m2_[1] + A->m1_[3]*B->m3_[1];
  tmp.m1_[2] = A->m1_[0]*B->m0_[2] + A->m1_[1]*B->m1_[2] + A->m1_[2]*B->m2_[2] + A->m1_[3]*B->m3_[2];
  tmp.m1_[3] = A->m1_[0]*B->m0_[3] + A->m1_[1]*B->m1_[3] + A->m1_[2]*B->m2_[3] + A->m1_[3]*B->m3_[3];

  // Row 3
  tmp.m2_[0] = A->m2_[0]*B->m0_[0] + A->m2_[1]*B->m1_[0] + A->m2_[2]*B->m2_[0] + A->m2_[3]*B->m3_[0];
  tmp.m2_[1] = A->m2_[0]*B->m0_[1] + A->m2_[1]*B->m1_[1] + A->m2_[2]*B->m2_[1] + A->m2_[3]*B->m3_[1];
  tmp.m2_[2] = A->m2_[0]*B->m0_[2] + A->m2_[1]*B->m1_[2] + A->m2_[2]*B->m2_[2] + A->m2_[3]*B->m3_[2];
  tmp.m2_[3] = A->m2_[0]*B->m0_[3] + A->m2_[1]*B->m1_[3] + A->m2_[2]*B->m2_[3] + A->m2_[3]*B->m3_[3];

  // Row 4
  tmp.m3_[0] = A->m3_[0]*B->m0_[0] + A->m3_[1]*B->m1_[0] + A->m3_[2]*B->m2_[0] + A->m3_[3]*B->m3_[0];
  tmp.m3_[1] = A->m3_[0]*B->m0_[1] + A->m3_[1]*B->m1_[1] + A->m3_[2]*B->m2_[1] + A->m3_[3]*B->m3_[1];
  tmp.m3_[2] = A->m3_[0]*B->m0_[2] + A->m3_[1]*B->m1_[2] + A->m3_[2]*B->m2_[2] + A->m3_[3]*B->m3_[2];
  tmp.m3_[3] = A->m3_[0]*B->m0_[3] + A->m3_[1]*B->m1_[3] + A->m3_[2]*B->m2_[3] + A->m3_[3]*B->m3_[3];

  return tmp;
}

#endif // End of GUARD_GGEMS_MATHS_MATRIX_FUNCTIONS_HH
