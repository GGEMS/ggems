#ifndef GUARD_GGEMS_MATHS_MATH_FUNCTIONS_HH
#define GUARD_GGEMS_MATHS_MATH_FUNCTIONS_HH

/*!
  \file math_functions.hh

  \brief Definitions of miscellaneous mathematical functions

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday December 18, 2019
*/

#include "GGEMS/opencl/types.hh"

/*!
  \fn inline uintcl_t binary_search_left(f64cl_t const key, f64cl_t const* p_array, uintcl_t const size, uintcl_t const offset = 0, uintcl_t min = 0)
  \param key - value in p_array to find
  \param p_array - p_array where is the key value
  \param size - size of p_array, number of elements
  \param offset - apply offset when searching index (optionnal)
  \param min - apply a min index (optionnal)
  \return index of key value in p_array buffer
  \brief Find the index of the key value in the p_array buffer
*/
#ifdef OPENCL_COMPILER
inline uintcl_t binary_search_left(f64cl_t const key,
  __global f64cl_t const* p_array, uintcl_t const size, uintcl_t const offset,
  uintcl_t min)
#else
inline uintcl_t binary_search_left(f64cl_t const key, f64cl_t const* p_array,
  uintcl_t const size, uintcl_t const offset, uintcl_t min)
#endif
{
  uintcl_t max = size - 1, mid = 0; // Max element, and median element
  uintcl_t const kMinCheck = min; // Min element

  while (min < max) {
    // Computing median index
    mid = (min + max) >> 1;
    if (key == p_array[mid + offset]) {
      return mid;
    }
    else if (key > p_array[mid + offset]) {
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
  \fn inline f64cl_t linear_interpolation(f64cl_t xa, f64cl_t ya, f64cl_t xb, f64cl_t yb, f64cl_t x)
  \param xa - Coordinate x of point A
  \param ya - Coordinate y of point A
  \param xb - Coordinate x of point B
  \param yb - Coordinate y of point B
  \param x - value to interpolate
  \return the interpolated value
  \brief interpolate the x value between point A and B
*/
inline f64cl_t linear_interpolation(f64cl_t const xa, f64cl_t const ya,
  f64cl_t const xb, f64cl_t const yb, f64cl_t const x)
{
  // Taylor young 1st order
  // if ( xa > x ) return ya;
  // if ( xb < x ) return yb;
  if (xa > xb) return yb;
  if (xa >= x) return ya;
  if (xb <= x) return yb;

  return ya + (x - xa)*(yb - ya)/(xb - xa);
}

#endif // GUARD_GGEMS_MATHS_MATH_FUNCTIONS_HH
