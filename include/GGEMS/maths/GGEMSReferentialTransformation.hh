#ifndef GUARD_GGEMS_MATHS_GGEMSREFERENTIALTRANSFORMATION_HH
#define GUARD_GGEMS_MATHS_GGEMSREFERENTIALTRANSFORMATION_HH

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
  \file GGEMSReferentialTransformation.hh

  \brief Definitions of functions changing referential computation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday November 5, 2020
*/

#include "GGEMS/maths/GGEMSMatrixTypes.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifdef __OPENCL_C_VERSION__

/*!
  \fn inline GGfloat3 RotateUnitZ(GGfloat3 vector, GGfloat3 const new_uz)
  \param vector - vector to change
  \param new_uz - new direction
  \return a vector of 3x1 float
  \brief rotateUz, function from CLHEP
*/
inline GGfloat3 RotateUnitZ(GGfloat3 vector, GGfloat3 const new_uz)
{
  GGfloat u1 = new_uz.x;
  GGfloat u2 = new_uz.y;
  GGfloat u3 = new_uz.z;

  GGfloat up = u1*u1 + u2*u2;
  if (up > 0) {
    up = sqrt(up);
    GGfloat px = vector.x,  py = vector.y, pz = vector.z;
    vector.x = (u1*u3*px - u2*py) /up + u1*pz;
    vector.y = (u2*u3*px + u1*py) /up + u2*pz;
    vector.z =    -up*px +             u3*pz;
  }
  else if (u3 < 0.) {
    vector.x = -vector.x;    // phi=0  theta=gpu_pi
    vector.z = -vector.z;
  }

  return {vector.x, vector.y, vector.z};
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

inline GGfloat3 GlobalToLocalPosition(global GGfloat44 const* matrix, GGfloat3 const point)
{
  // Extract translation
  GGfloat3 translation;
  //translation.x = matrix->m0_.x;
  //= {matrix->m0_.w, matrix->m1_.w, matrix->m2_.w};

  return translation;
}

/*__host__ __device__ f32xyz fxyz_global_to_local_position( const f32matrix44 &G, f32xyz u)
{
    // first, extract the translation
    f32xyz T = { G.m03, G.m13, G.m23 };
    // Then the sub matrix (R and P)
    f32matrix33 g = { G.m00, G.m01, G.m02,
                      G.m10, G.m11, G.m12,
                      G.m20, G.m21, G.m22 };
    // Inverse transform
    f32matrix33 ginv = fmatrix_trans( g );
    u = fxyz_sub( u, T );
    u = fmatrix_mul_fxyz( ginv, u );

    return u;
}*/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
 \fn inline GGfloat3 LocalToGlobalPosition(global GGfloat44 const* matrix, GGfloat3 const point)
 \param matrix - A matrix (4x4)
 \param point - Point in 3D (x, y, z)
 \return The point expresses in the global frame
 \brief Transform a 3D point from local to global frame
*/
inline GGfloat3 LocalToGlobalPosition(global GGfloat44 const* matrix, GGfloat3 const point)
{
  return GGfloat44MultGGfloat3(matrix, point);
}

#endif

#endif // End of GUARD_GGEMS_MATHS_GGEMSREFERENTIALTRANSFORMATION_HH
