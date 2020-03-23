#ifndef GUARD_GGEMS_MATHS_GGEMSMATHALGORITHMS_HH
#define GUARD_GGEMS_MATHS_GGEMSMATHALGORITHMS_HH

/*!
  \file GGEMSMathAlgorithms.hh

  \brief Definitions of miscellaneous mathematical functions

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday December 18, 2019
*/

#include "GGEMS/tools/GGEMSTypes.hh"


#ifdef OPENCL_COMPILER
/*!
  \fn inline GGuint BinarySearchLeft(GGfloat const key, __global GGfloat const* array, GGuint const size, GGuint const offset, GGuint min)
  \param key - value in p_array to find
  \param array - p_array where is the key value
  \param size - size of p_array, number of elements
  \param offset - apply offset when searching index (optionnal)
  \param min - apply a min index (optionnal)
  \return index of key value in p_array buffer
  \brief Find the index of the key value in the p_array buffer
*/
inline GGuint BinarySearchLeft(GGfloat const key, __global GGfloat const* array, GGuint const size, GGuint const offset, GGuint min)
#else
/*!
  \fn inline GGuint BinarySearchLeft(GGfloat const key, GGfloat const* array, GGuint const size, GGuint const offset, GGuint min)
  \param key - value in p_array to find
  \param array - p_array where is the key value
  \param size - size of p_array, number of elements
  \param offset - apply offset when searching index (optionnal)
  \param min - apply a min index (optionnal)
  \return index of key value in p_array buffer
  \brief Find the index of the key value in the p_array buffer
*/
inline GGuint BinarySearchLeft(GGfloat const key, GGfloat const* array, GGuint const size, GGuint const offset, GGuint min)
#endif
{
  GGuint max = size - 1, mid = 0; // Max element, and median element
  GGuint const kMinCheck = min; // Min element

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
  if (min > kMinCheck) min--;

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

#endif // GUARD_GGEMS_MATHS_GGEMSMATHALGORITHMS_HH
