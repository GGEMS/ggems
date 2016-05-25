// GGEMS Copyright (C) 2015

/*!
 * \file vector.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 * \todo a) Use template here
 *
 */

#ifndef VECTOR_H
#define VECTOR_H

/////////////////////////////////////////////////////////////////////////////
// Maths
/////////////////////////////////////////////////////////////////////////////

#include "global.cuh"

/// Single precision functions //////////////////////////////////////////////

#ifndef F32MATRIX33
#define F32MATRIX33
struct f32matrix33 {
    f32 m00, m01, m02;
    f32 m10, m11, m12;
    f32 m20, m21, m22;
};
#endif

#ifndef F32MATRIX44
#define F32MATRIX44
struct f32matrix44
{
    f32 m00, m01, m02, m03;
    f32 m10, m11, m12, m13;
    f32 m20, m21, m22, m23;
    f32 m30, m31, m32, m33;

    friend std::ostream& operator<< ( std::ostream& os, const f32matrix44 m )
    {
        os  << std::fixed << std::setprecision ( 2 );
        //os  << "Particle state : " << std::endl;

        os << "| " << m.m00 << " " << m.m01 << " " << m.m02 << " " << m.m03 << " |" << std::endl;
        os << "| " << m.m10 << " " << m.m11 << " " << m.m12 << " " << m.m13 << " |" << std::endl;
        os << "| " << m.m20 << " " << m.m21 << " " << m.m22 << " " << m.m23 << " |" << std::endl;
        os << "| " << m.m30 << " " << m.m31 << " " << m.m32 << " " << m.m33 << " |" << std::endl;

        return os;

    }

};
#endif

// r = u - v
__host__ __device__ f32xyz fxyz_sub(f32xyz u, f32xyz v);
// r = u + v
__host__ __device__ f32xyz fxyz_add(f32xyz u, f32xyz v);
// r = u * v
__host__ __device__ f32xyz fxyz_mul(f32xyz u, f32xyz v);
// r = u / v
__host__ __device__ f32xyz fxyz_div(f32xyz u, f32xyz v);
// r = u * s
__host__ __device__ f32xyz fxyz_scale(f32xyz u, f32 s);
// r = u . v
__host__ __device__ f32 fxyz_dot(f32xyz u, f32xyz v);
// r = u x v
__host__ __device__ f32xyz fxyz_cross(f32xyz u, f32xyz v);
// r = m * u
__host__ __device__ f32xyz fmatrix_mul_fxyz(f32matrix33 matrix, f32xyz u);
__host__ __device__ f32xyz fmatrix_mul_fxyz(f32matrix44 matrix, f32xyz u);
// r = m * n
__host__ __device__ f32matrix44 fmatrix_mul( f32matrix44 m, f32matrix44 n );
// r= m^T
__host__ __device__ f32matrix44 fmatrix_trans( f32matrix44 m );
__host__ __device__ f32matrix33 fmatrix_trans( f32matrix33 m );
// return an unitary vector
__host__ __device__ f32xyz fxyz_unit(f32xyz u);
// return the magnitude of the vector
__host__ __device__ f32 fxyz_mag( f32xyz u );
// rotate a vector u
__host__ __device__ f32xyz fxyz_rotate_euler(f32xyz u, f32xyz EulerAngles); // phi, theta, psi
// rotate a vector u around the x-axis
__host__ __device__ f32xyz fxyz_rotate_x_axis(f32xyz u, f32 angle);
// rotate a vector u around the y-axis
__host__ __device__ f32xyz fxyz_rotate_y_axis(f32xyz u, f32 angle);
// rotate a vector u around the z-axis
__host__ __device__ f32xyz fxyz_rotate_z_axis(f32xyz u, f32 angle);
// return abs
__host__ __device__ f32xyz fxyz_abs(f32xyz u);
// return 1/u
__host__ __device__ f32xyz fxyz_inv(f32xyz u);

//// Struct that handle nD variable
static __inline__ __host__ __device__ f32xy make_f32xy(f32 vx, f32 vy) {
    f32xy t; t.x = vx; t.y = vy; return t;
}

static __inline__ __host__ __device__ f32xyz make_f32xyz(f32 vx, f32 vy, f32 vz) {
    f32xyz t; t.x = vx; t.y = vy; t.z = vz; return t;
}

static __inline__ __host__ __device__ f64xyz make_f64xyz(f64 vx, f64 vy, f64 vz) {
    f64xyz t; t.x = vx; t.y = vy; t.z = vz; return t;
}

static __inline__ __host__ __device__ i32xyz make_i32xyz(i32 vx, i32 vy, i32 vz) {
    i32xyz t; t.x = vx; t.y = vy; t.z = vz; return t;
}

static __inline__ __host__ __device__ ui32xy make_ui32xy(ui32 vx, ui32 vy) {
    ui32xy t; t.x = vx; t.y = vy; return t;
}

static __inline__ __host__ __device__ ui32xyz make_ui32xyz(ui32 vx, ui32 vy, ui32 vz) {
    ui32xyz t; t.x = vx; t.y = vy; t.z = vz; return t;
}

static __inline__ __host__ __device__ f32matrix33 make_f32matrix33( f32 m00, f32 m01, f32 m02,
                                                                    f32 m10, f32 m11, f32 m12,
                                                                    f32 m20, f32 m21, f32 m22 )
{
    f32matrix33 M;
    M.m00 = m00; M.m01 = m01; M.m02 = m02;
    M.m10 = m10; M.m11 = m11; M.m12 = m12;
    M.m20 = m20; M.m21 = m21; M.m22 = m22;
    return M;
}

static __inline__ __host__ __device__ f32matrix44 make_f32matrix44( f32 m00, f32 m01, f32 m02, f32 m03,
                                                                    f32 m10, f32 m11, f32 m12, f32 m13,
                                                                    f32 m20, f32 m21, f32 m22, f32 m23,
                                                                    f32 m30, f32 m31, f32 m32, f32 m33 )
{
    f32matrix44 M;
    M.m00 = m00; M.m01 = m01; M.m02 = m02; M.m03 = m03;
    M.m10 = m10; M.m11 = m11; M.m12 = m12; M.m13 = m13;
    M.m20 = m20; M.m21 = m21; M.m22 = m22; M.m23 = m23;
    M.m30 = m30; M.m31 = m31; M.m32 = m32; M.m33 = m33;
    return M;
}

static __inline__ __host__ __device__ f32xyz cast_ui32xyz_to_f32xyz(ui32xyz u)
{
    return make_f32xyz( f32( u.x ), f32( u.y), f32( u.z) );
}

/// Tranformation class //////////////////////////////////////////////////////

/*!
 * \fn __host__ __device__ f32xyz fxyz_local_to_global_frame( f32matrix44 G, f32xyz u )
 * \brief Transform a 3D point from local to global frame
 * \param G Global transformation matrix (4x4)
 * \param u Point in 3D (x, y, z)
 * \return The point expresses in the global frame
 */
__host__ __device__ f32xyz fxyz_local_to_global_frame( f32matrix44 G, f32xyz u );

/*!
 * \fn __host__ __device__ f32xyz fxyz_global_to_local_frame( f32matrix44 G, f32xyz u )
 * \brief Transform a 3D point from global to local frame
 * \param G Global transformation matrix (4x4)
 * \param u Point in 3D (x, y, z)
 * \return The point expresses in the local frame
 */
__host__ __device__ f32xyz fxyz_global_to_local_frame( f32matrix44 G, f32xyz u );

/*!
  \class TransformCalculator
  \brief This class handles everything about geometry transformation
*/
class TransformCalculator
{
public:
    /*!
     * \brief TransformCalculator contructor
     */
    TransformCalculator();

    /*!
     * \brief TransformCalculator destructor
     */
    ~TransformCalculator() {}

    /*!
     * \fn void set_translation( f32 tx, f32 ty, f32 tz )
     * \brief Set the translation part of the transformation
     * \param tx Translation along the x-axis
     * \param ty Translation along the y-axis
     * \param tz Translation along the z-axis
     */
    void set_translation( f32 tx, f32 ty, f32 tz );

    /*!
     * \fn void set_translation( f32xyz t )
     * \brief Set the translation part of the transformation
     * \param t Translation vector
     */
    void set_translation( f32xyz t );

    /*!
     * \fn void set_rotation( f32 rx, f32 ry, f32 rz )
     * \brief Set the rotation part of the transformation
     * \param rx Rotation angle around the x-axis
     * \param ry Rotation angle around the y-axis
     * \param rz Rotation angle around the z-axis
     */
    void set_rotation( f32 rx, f32 ry, f32 rz );

    /*!
     * \fn void set_rotation( f32xyz angles )
     * \brief Set the rotation part of the transformation ( angle x-axis, angle y-axis, angle z-axis)
     * \param angles Angles around the x, y, and z-axis
     */
    void set_rotation( f32xyz angles );

    /*!
     * \fn void set_axis_transformation( f32matrix33 P )
     * \brief Set the transformation of the frame, usefull for mirroring or convert 3D to 2D
     * \param P Matrix (3x3) that contains the mapping of the coordinates (ex. x becomes y and vice-versa). Values are 0, 1 or -1.
     */
    void set_axis_transformation( f32matrix33 P );

    /*!
     * \fn void set_axis_transformation( f32 m00, f32 m01, f32 m02, f32 m10, f32 m11, f32 m12, f32 m20, f32 m21, f32 m22 )
     * \brief Set the transformation of the frame, usefull for mirroring or convert 3D to 2D
     * \param m00 Element of the matrix that map the coordinates. Values are 0, 1 or -1.
     * \param m01 Element of the matrix that map the coordinates. Values are 0, 1 or -1.
     * \param m02 Element of the matrix that map the coordinates. Values are 0, 1 or -1.
     * \param m10 Element of the matrix that map the coordinates. Values are 0, 1 or -1.
     * \param m11 Element of the matrix that map the coordinates. Values are 0, 1 or -1.
     * \param m12 Element of the matrix that map the coordinates. Values are 0, 1 or -1.
     * \param m20 Element of the matrix that map the coordinates. Values are 0, 1 or -1.
     * \param m21 Element of the matrix that map the coordinates. Values are 0, 1 or -1.
     * \param m22 Element of the matrix that map the coordinates. Values are 0, 1 or -1.
     */
    void set_axis_transformation( f32 m00, f32 m01, f32 m02,
                                  f32 m10, f32 m11, f32 m12,
                                  f32 m20, f32 m21, f32 m22 );

    /*!
     * \fn f32matrix44 get_translation_matrix()
     * \brief Get the translation matrix
     * \return Homogeneous matrix (4x4)
     */
    f32matrix44 get_translation_matrix();

    /*!
     * \fn f32matrix44 get_rotation_matrix()
     * \brief Get the rotation matrix
     * \return Homogeneous matrix (4x4)
     */
    f32matrix44 get_rotation_matrix();

    /*!
     * \fn f32matrix44 get_projection_matrix()
     * \brief Get the projection matrix
     * \return Homogeneous matrix (4x4)
     */
    f32matrix44 get_projection_matrix();

    /*!
     * \fn f32matrix44 get_transformation_matrix()
     * \brief Get the global transformation matrix
     * \return Homogeneous matrix (4x4)
     */
    f32matrix44 get_transformation_matrix();

private:
    f32matrix44 m_T;      /*!< Translation matrix */
    f32matrix44 m_R;      /*!< Rotation matrix */
    f32matrix44 m_P0;     /*!< Orthographic projection matrix */
    //f32matrix44 m_G;      /*!< Global transformation matrix */
};




/// Double precision functions ///////////////////////////////////////////////

/*

#ifndef SINGLE_PRECISION
    // Add function with double precision

#ifndef F64MATRIX33
#define F64MATRIX33
struct f64matrix33 {
    f64 a, b, c, d, e, f, g, h, i;
};
#endif

// r = u - v
__host__ __device__ f64xyz fxyz_sub(f64xyz u, f64xyz v);
// r = u + v
__host__ __device__ f64xyz fxyz_add(f64xyz u, f64xyz v);
// r = u * v
__host__ __device__ f64xyz fxyz_mul(f64xyz u, f64xyz v);
// r = u / v
__host__ __device__ f64xyz fxyz_div(f64xyz u, f64xyz v);
// r = u * s
__host__ __device__ f64xyz fxyz_scale(f64xyz u, f64 s);
// r = u . v
__host__ __device__ f64 fxyz_dot(f64xyz u, f64xyz v);
// r = u x v
__host__ __device__ f64xyz fxyz_cross(f64xyz u, f64xyz v);
// r = m * u
__host__ __device__ f64xyz fmatrixfxyz_mul(f64matrix33 matrix, f64xyz u);
// return an unitary vector
__host__ __device__ f64xyz fxyz_unit(f64xyz u);
// rotate a vector u
__host__ __device__ f64xyz fxyz_rotate_euler(f64xyz u, f64xyz EulerAngles); // phi, theta, psi
// rotate a vector u around the x-axis
__host__ __device__ f64xyz fxyz_rotate_x_axis(f64xyz u, f64 angle);
// rotate a vector u around the y-axis
__host__ __device__ f64xyz fxyz_rotate_y_axis(f64xyz u, f64 angle);
// rotate a vector u around the z-axis
__host__ __device__ f64xyz fxyz_rotate_z_axis(f64xyz u, f64 angle);
// return abs
__host__ __device__ f64xyz fxyz_abs(f64xyz u);
// return 1/u
__host__ __device__ f64xyz fxyz_inv(f64xyz u);

inline __host__ __device__  f64 f64xyz_mul(f64xyz v){ return v.x * v.y * v.z;}

#endif
*/




#endif
