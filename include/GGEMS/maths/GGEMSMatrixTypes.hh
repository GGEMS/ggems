#ifndef GUARD_GGEMS_MATHS_GGEMSMATRIXTYPES_HH
#define GUARD_GGEMS_MATHS_GGEMSMATRIXTYPES_HH

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
  \file GGEMSMatrixTypes.hh

  \brief Class managing the matrix types

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday November 13, 2019
*/

#include "GGEMS/global/GGEMSConfiguration.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \struct GGfloat33_t
  \brief Structure storing float 3 x 3 matrix
*/
#ifdef __OPENCL_C_VERSION__
typedef struct __attribute__((aligned (1))) GGfloat33_t
#else
typedef struct PACKED GGfloat33_t
#endif
{
  GGfloat3 m0_; /*!< Row 0 of matrix */
  GGfloat3 m1_; /*!< Row 1 of matrix */
  GGfloat3 m2_; /*!< Row 2 of matrix */
} GGfloat33; /*!< Using C convention name of struct to C++ (_t deletion) */

/*!
  \struct GGfloat44_t
  \brief Structure storing float 4 x 4 matrix
*/
#ifdef __OPENCL_C_VERSION__
typedef struct __attribute__((aligned (1))) GGfloat44_t
#else
typedef struct PACKED GGfloat44_t
#endif
{
  GGfloat4 m0_; /*!< Row 0 of matrix */
  GGfloat4 m1_; /*!< Row 1 of matrix */
  GGfloat4 m2_; /*!< Row 2 of matrix */
  GGfloat4 m3_; /*!< Row 3 of matrix */
} GGfloat44; /*!< Using C convention name of struct to C++ (_t deletion) */

#endif // End of GUARD_GGEMS_MATHS_GGEMSMATRIXTYPES_HH
