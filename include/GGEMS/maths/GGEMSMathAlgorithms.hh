#ifndef GUARD_GGEMS_MATHS_GGEMSMATHALGORITHMS_HH
#define GUARD_GGEMS_MATHS_GGEMSMATHALGORITHMS_HH

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
  \file GGEMSMathAlgorithms.hh

  \brief Definitions of miscellaneous mathematical functions

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday December 18, 2019
*/

#include "GGEMS/tools/GGEMSSystemOfUnits.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \fn inline GGint BinarySearchLeft(GGfloat const key, global GGfloat const* array, GGuint const size, GGuint const offset, GGuint min)
  \param key - value in p_array to find
  \param array - p_array where is the key value
  \param size - size of p_array, number of elements
  \param offset - apply offset when searching index (optionnal)
  \param min - apply a min index (optionnal)
  \return index of key value in p_array buffer
  \brief Find the index of the key value in the p_array buffer
*/
#ifdef __OPENCL_C_VERSION__
inline GGint BinarySearchLeft(GGfloat const key, global GGfloat const* array, GGint const size, GGint const offset, GGint min)
#else
inline GGint BinarySearchLeft(GGfloat const key, GGfloat const* array, GGint const size, GGint const offset, GGint min)
#endif
{
  GGint max = size - 1, mid = 0; // Max element, and median element
  GGint min_check = min; // Min element

  while (min < max) {
    // Computing median index
    mid = (min + max) >> 1;
    if (key == array[mid + offset]) {
      return mid;
    }
    else if (key > array[mid + offset]) {
      min = mid + 1;
    }
    else {
      max = mid;
    }
  }

  // Checking the min elements
  if (min > min_check) min--;

  // Return the min element
  return min;
}

/*!
  \fn inline GGfloat LinearInterpolation(GGfloat xa, GGfloat ya, GGfloat xb, GGfloat yb, GGfloat x)
  \param xa - Coordinate x of point A
  \param ya - Coordinate y of point A
  \param xb - Coordinate x of point B
  \param yb - Coordinate y of point B
  \param x - value to interpolate
  \return the interpolated value
  \brief interpolate the x value between point A and B
*/
inline GGfloat LinearInterpolation(GGfloat const xa, GGfloat const ya, GGfloat const xb, GGfloat const yb, GGfloat const x)
{
  // Taylor young 1st order
  // if ( xa > x ) return ya;
  // if ( xb < x ) return yb;
  if (xa > xb) return yb;
  if (xa >= x) return ya;
  if (xb <= x) return yb;

  return ya + (x - xa)*(yb - ya)/(xb - xa);
}

/*!
  \fn inline GGfloat LogLogInterpolation(GGfloat x, GGfloat x0, GGfloat y0, GGfloat x1, GGfloat y1)
  \param x0 - Coordinate x0 of point A
  \param y0 - Coordinate y0 of point A
  \param x1 - Coordinate x1 of point B
  \param y1 - Coordinate y1 of point B
  \param x - value to interpolate
  \return the loglog interpolated value
  \brief log log interpolation of the x value between point (x0,y0) and (x1,y1)
*/
inline GGfloat LogLogInterpolation(GGfloat x, GGfloat x0, GGfloat y0, GGfloat x1, GGfloat y1)
{
  if (x < x0) return y0;
  if (x > x1) return y1;

  x0 = 1.0f / x0;

  return pow(10.0f, log10(y0) + log10(y1/y0) * (log10(x*x0) / log10(x1*x0)));
}

#endif // GUARD_GGEMS_MATHS_GGEMSMATHALGORITHMS_HH
