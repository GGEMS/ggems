#ifndef GUARD_GGEMS_TOOLS_MATRIX_HH
#define GUARD_GGEMS_TOOLS_MATRIX_HH

/*!
  \file matrix.hh

  \brief Class managing the matrix computation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday November 13, 2019
*/

#include "GGEMS/global/ggems_configuration.hh"
#include "GGEMS/global/opencl_manager.hh"

/*!
  \namespace Matrix
  \brief namespace storing miscellaneous functions for Matrix
*/
namespace Matrix
{
  /*!
    \struct float3x3_t
    \brief Structure storing float 3 x 3 matrix
  */
  #ifdef _MSC_VER
  #pragma pack(push, 1)
  #endif
  typedef struct PACKED float3x3_t
  {
    cl_float m00_, m01_, m02_;
    cl_float m10_, m11_, m12_;
    cl_float m20_, m21_, m22_;
  } float3x3;
  #ifdef _MSC_VER
  #pragma pack(pop)
  #endif

  /*!
    \struct float4x4_t
    \brief Structure storing float 4 x 4 matrix
  */
  #ifdef _MSC_VER
  #pragma pack(push, 1)
  #endif
  typedef struct PACKED float4x4_t
  {
    cl_float m00_, m01_, m02_, m03_;
    cl_float m10_, m11_, m12_, m13_;
    cl_float m20_, m21_, m22_, m23_;
    cl_float m30_, m31_, m32_, m33_;
  } float4x4;
  #ifdef _MSC_VER
  #pragma pack(pop)
  #endif

  /*!
    \fn inline cl_float3 MakeFloatXYZ(float const& x, float const& y, float const& z)
    \param x - x parameter
    \param y - y parameter
    \param z - z parameter
    \brief Make a float X, Y and Z with custom values
  */
  inline cl_float3 MakeFloatXYZ(float const& x, float const& y, float const& z)
  {
    cl_float3 tmp;
    tmp.s[0] = x;
    tmp.s[1] = y;
    tmp.s[2] = z;
    return tmp;
  }

  /*!
    \fn inline cl_float3 MakeFloatXYZZeros()
    \brief Make a float X, Y and Z with zeros for value
  */
  inline cl_float3 MakeFloatXYZZeros()
  {
    cl_float3 tmp;
    tmp.s[0] = 0.0f;
    tmp.s[1] = 0.0f;
    tmp.s[2] = 0.0f;
    return tmp;
  }

  /*!
    \fn inline float3x3 MakeFloat3x3(float const& m00, float const& m01, float const& m02, float const& m10, float const& m11, float const& m12, float const& m20, float const& m21, float const& m22)
    \param m00 - Element 0,0 in the matrix 3x3 for local axis
    \param m01 - Element 0,1 in the matrix 3x3 for local axis
    \param m02 - Element 0,2 in the matrix 3x3 for local axis
    \param m10 - Element 1,0 in the matrix 3x3 for local axis
    \param m11 - Element 1,1 in the matrix 3x3 for local axis
    \param m12 - Element 1,2 in the matrix 3x3 for local axis
    \param m20 - Element 2,0 in the matrix 3x3 for local axis
    \param m21 - Element 2,1 in the matrix 3x3 for local axis
    \param m22 - Element 2,2 in the matrix 3x3 for local axis
    \brief Make a float3x3 with custom values
  */
  inline float3x3 MakeFloat3x3(
    float const& m00, float const& m01, float const& m02,
    float const& m10, float const& m11, float const& m12,
    float const& m20, float const& m21, float const& m22)
  {
    float3x3 tmp;
    // Row 1
    tmp.m00_ = m00; tmp.m01_ = m01; tmp.m02_ = m02;
    // Row 2
    tmp.m10_ = m10; tmp.m11_ = m11; tmp.m12_ = m12;
    // Row 3
    tmp.m20_ = m20; tmp.m21_ = m21; tmp.m22_ = m22;
    return tmp;
  }

  /*!
    \fn inline float3x3 MakeFloat3x3Zeros()
    \brief Make a float3x3 with zeros for value
  */
  inline float3x3 MakeFloat3x3Zeros()
  {
    float3x3 tmp;
    // Row 1
    tmp.m00_ = 0.0f; tmp.m01_ = 0.0f; tmp.m02_ = 0.0f;
    // Row 2
    tmp.m10_ = 0.0f; tmp.m11_ = 0.0f; tmp.m12_ = 0.0f;
    // Row 3
    tmp.m20_ = 0.0f; tmp.m21_ = 0.0f; tmp.m22_ = 0.0f;
    return tmp;
  }

  /*!
    \fn inline float4x4 MakeFloat4x4(float const& m00, float const& m01, float const& m02, float const& m03, float const& m10, float const& m11, float const& m12, float const& m13, float const& m20, float const& m21, float const& m22, float const& m23, float const& m30, float const& m31, float const& m32, float const& m33)
    \param m00 - Element 0,0 in the matrix 4x4 for local axis
    \param m01 - Element 0,1 in the matrix 4x4 for local axis
    \param m02 - Element 0,2 in the matrix 4x4 for local axis
    \param m03 - Element 0,3 in the matrix 4x4 for local axis
    \param m10 - Element 1,0 in the matrix 4x4 for local axis
    \param m11 - Element 1,1 in the matrix 4x4 for local axis
    \param m12 - Element 1,2 in the matrix 4x4 for local axis
    \param m13 - Element 1,3 in the matrix 4x4 for local axis
    \param m20 - Element 2,0 in the matrix 4x4 for local axis
    \param m21 - Element 2,1 in the matrix 4x4 for local axis
    \param m22 - Element 2,2 in the matrix 4x4 for local axis
    \param m23 - Element 2,3 in the matrix 4x4 for local axis
    \param m30 - Element 3,0 in the matrix 4x4 for local axis
    \param m31 - Element 3,1 in the matrix 4x4 for local axis
    \param m32 - Element 3,2 in the matrix 4x4 for local axis
    \param m33 - Element 3,3 in the matrix 4x4 for local axis
    \brief Make a float4x4 with custom values
  */
  inline float4x4 MakeFloat4x4(
    float const& m00, float const& m01, float const& m02, float const& m03,
    float const& m10, float const& m11, float const& m12, float const& m13,
    float const& m20, float const& m21, float const& m22, float const& m23,
    float const& m30, float const& m31, float const& m32, float const& m33)
  {
    float4x4 tmp;
    // Row 1
    tmp.m00_ = m00; tmp.m01_ = m01; tmp.m02_ = m02; tmp.m03_ = m03;
    // Row 2
    tmp.m10_ = m10; tmp.m11_ = m11; tmp.m12_ = m12; tmp.m13_ = m13;
    // Row 3
    tmp.m20_ = m20; tmp.m21_ = m21; tmp.m22_ = m22; tmp.m23_ = m23;
    // Row 4
    tmp.m30_ = m30; tmp.m31_ = m31; tmp.m32_ = m32; tmp.m33_ = m33;
    return tmp;
  }

  /*!
    \fn inline float4x4 MakeFloat3x3Zeros()
    \brief Make a float4x4 with zeros for value
  */
  inline float4x4 MakeFloat4x4Zeros()
  {
    float4x4 tmp;
    // Row 1
    tmp.m00_ = 0.0f; tmp.m01_ = 0.0f; tmp.m02_ = 0.0f; tmp.m03_ = 0.0f;
    // Row 2
    tmp.m10_ = 0.0f; tmp.m11_ = 0.0f; tmp.m12_ = 0.0f; tmp.m13_ = 0.0f;
    // Row 3
    tmp.m20_ = 0.0f; tmp.m21_ = 0.0f; tmp.m22_ = 0.0f; tmp.m23_ = 0.0f;
    // Row 4
    tmp.m30_ = 0.0f; tmp.m31_ = 0.0f; tmp.m32_ = 0.0f; tmp.m33_ = 0.0f;
    return tmp;
  }

  inline float4x4 MatrixMult4x4(float4x4 const& A, float4x4 const& B)
  {
    float4x4 tmp = MakeFloat4x4Zeros();

    // Row 1
    tmp.m00_ = A.m00_*B.m00_ + A.m01_*B.m10_ + A.m02_*B.m20_ + A.m03_*B.m30_;
    tmp.m01_ = A.m00_*B.m01_ + A.m01_*B.m11_ + A.m02_*B.m21_ + A.m03_*B.m31_;
    tmp.m02_ = A.m00_*B.m02_ + A.m01_*B.m12_ + A.m02_*B.m22_ + A.m03_*B.m32_;
    tmp.m03_ = A.m00_*B.m03_ + A.m01_*B.m13_ + A.m02_*B.m23_ + A.m03_*B.m33_;

    // Row 2
    tmp.m10_ = A.m10_*B.m00_ + A.m11_*B.m10_ + A.m12_*B.m20_ + A.m13_*B.m30_;
    tmp.m11_ = A.m10_*B.m01_ + A.m11_*B.m11_ + A.m12_*B.m21_ + A.m13_*B.m31_;
    tmp.m12_ = A.m10_*B.m02_ + A.m11_*B.m12_ + A.m12_*B.m22_ + A.m13_*B.m32_;
    tmp.m13_ = A.m10_*B.m03_ + A.m11_*B.m13_ + A.m12_*B.m23_ + A.m13_*B.m33_;

    // Row 3
    tmp.m20_ = A.m20_*B.m00_ + A.m21_*B.m10_ + A.m22_*B.m20_ + A.m23_*B.m30_;
    tmp.m21_ = A.m20_*B.m01_ + A.m21_*B.m11_ + A.m22_*B.m21_ + A.m23_*B.m31_;
    tmp.m22_ = A.m20_*B.m02_ + A.m21_*B.m12_ + A.m22_*B.m22_ + A.m23_*B.m32_;
    tmp.m23_ = A.m20_*B.m03_ + A.m21_*B.m13_ + A.m22_*B.m23_ + A.m23_*B.m33_;

    // Row 4
    tmp.m30_ = A.m30_*B.m00_ + A.m31_*B.m10_ + A.m32_*B.m20_ + A.m33_*B.m30_;
    tmp.m31_ = A.m30_*B.m01_ + A.m31_*B.m11_ + A.m32_*B.m21_ + A.m33_*B.m31_;
    tmp.m32_ = A.m30_*B.m02_ + A.m31_*B.m12_ + A.m32_*B.m22_ + A.m33_*B.m32_;
    tmp.m33_ = A.m30_*B.m03_ + A.m31_*B.m13_ + A.m32_*B.m23_ + A.m33_*B.m33_;

    return tmp;
  }
}

/*!
  \class TransformCalculator
  \brief This class handles everything about geometry transformation
*/
class TransformCalculator
{
  public:
    /*!
      \brief TransformCalculator constructor
    */
    TransformCalculator(void);

    /*!
      \brief TransformCalculator destructor
    */
    ~TransformCalculator(void);

  public:
    /*!
      \fn TransformCalculator(TransformCalculator const& transform_calculator) = delete
      \param ggems_manager - reference on the ggems manager
      \brief Avoid copy of the class by reference
    */
    TransformCalculator(
      TransformCalculator const& transform_calculator) = delete;

    /*!
      \fn TransformCalculator& operator=(TransformCalculator const& transform_calculator) = delete
      \param ggems_manager - reference on the ggems manager
      \brief Avoid assignement of the class by reference
    */
    TransformCalculator& operator=(
      TransformCalculator const& transform_calculator) = delete;

    /*!
      \fn TransformCalculator(TransformCalculator const&& transform_calculator) = delete
      \param ggems_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    TransformCalculator(
      TransformCalculator const&& transform_calculator) = delete;

    /*!
      \fn TransformCalculator& operator=(TransformCalculator const&& transform_calculator) = delete
      \param ggems_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    TransformCalculator& operator=(
      TransformCalculator const&& transform_calculator) = delete;

  public:
    /*!
      \fn void SetTranslation(float const& tx, float const& ty, float const& tz)
      \param tx - Translation in X
      \param ty - Translation in Y
      \param tz - Translation in Z
      \brief Set the translation in X, Y and Z
    */
    void SetTranslation(float const& tx, float const& ty, float const& tz);

    /*!
      \fn void SetTranslation(cl_float3 const& txyz)
      \param txyz - Translation in X, Y and Z
      \brief Set the translation in X, Y and Z
    */
    void SetTranslation(cl_float3 const& txyz);

    /*!
      \fn inline Matrix::float4x4 GetMatrixTranslation(void) const
      \return the translation matrix
      \brief Return the translation matrix
    */
    inline Matrix::float4x4 GetMatrixTranslation(void) const
    {
      return matrix_translation_;
    };

    /*!
      \fn void SetRotation(float const& rx, float const& ry, float const& rz)
      \param rx - Rotation in X
      \param ry - Rotation in Y
      \param rz - Rotation in Z
      \brief Set the Rotation in X, Y and Z around global axis
    */
    void SetRotation(float const& rx, float const& ry, float const& rz);

    /*!
      \fn void SetRotation(cl_float3 const& rxyz)
      \param rxyz - Rotation around X, Y and Z global axis
      \biref Set the rotation around global axis
    */
    void SetRotation(cl_float3 const& rxyz);

    /*!
      \fn inline Matrix::float4x4 GetMatrixRotation(void) const
      \return the translation matrix
      \brief Return the translation matrix
    */
    inline Matrix::float4x4 GetMatrixRotation(void) const
    {
      return matrix_rotation_;
    };

    /*!
      \fn void SetAxisTransformation(Matrix::float3x3 const& axis)
      \param axis - Matrix (3x3) that contains the mapping of the coordinates (ex. x becomes y and vice-versa). Values are 0, 1 or -1.
      \brief Set the transformation of the frame, usefull for mirroring or convert 3D to 2D
    */
    void SetAxisTransformation(Matrix::float3x3 const& axis);

    /*!
      \fn void SetAxisTransformation(float const& m00, float const& m01, float const& m02, float const& m10, float const& m11, float const& m12, float const& m20, float const& m21, float const& m22)
      \param m00 - Element 0,0 in the matrix 3x3 for local axis
      \param m01 - Element 0,1 in the matrix 3x3 for local axis
      \param m02 - Element 0,2 in the matrix 3x3 for local axis
      \param m10 - Element 1,0 in the matrix 3x3 for local axis
      \param m11 - Element 1,1 in the matrix 3x3 for local axis
      \param m12 - Element 1,2 in the matrix 3x3 for local axis
      \param m20 - Element 2,0 in the matrix 3x3 for local axis
      \param m21 - Element 2,1 in the matrix 3x3 for local axis
      \param m22 - Element 2,2 in the matrix 3x3 for local axis
      \brief Set the transformation of the frame, usefull for mirroring or convert 3D to 2D
    */
    void SetAxisTransformation(
      float const& m00, float const& m01, float const& m02,
      float const& m10, float const& m11, float const& m12,
      float const& m20, float const& m21, float const& m22);

    /*!
      \fn inline Matrix::float4x4 GetMatrixOrthographicProjection(void) const
      \return the matrix of orthographic projection
      \brief return the matrix of orthographic projection
    */
    inline Matrix::float4x4 GetMatrixOrthographicProjection(void) const
    {
      return matrix_orthographic_projection_;
    };

    /*!
      \fn inline cl_float3 GetPosition(void) const
      \return The position of source/detector...
      \brief Return the current position
    */
    inline cl_float3 GetPosition(void) const {return position_;}

    /*!
      \fn inline cl_float3 GetRotation(void) const
      \return The rotation of source/detector...
      \brief Return the current rotation
    */
    inline cl_float3 GetRotation(void) const {return rotation_;}

    /*!
      \fn inline Matrix::float3x3 GetLocalAxis(void) const
      \return The local axis matrix
      \brief return the local axis matrix
    */
    inline Matrix::float3x3 GetLocalAxis(void) const {return local_axis_;}

    /*!
      \fn void UpdateTransformationMatrix(void)
      \brief update the transformation matrix
    */
    void UpdateTransformationMatrix(void);

    /*!
      \fn Matrix::float4x4 GetTransformationMatrix(void) const
      \return the transformation matrix
      \brief return the transformation matrix
    */
    inline Matrix::float4x4 GetTransformationMatrix(void)
    {
      // Check if we need to update
      if (need_updated_) UpdateTransformationMatrix();

      // Return the transformation matrix
      return matrix_transformation_;
    }

  private:
    bool need_updated_; /*!< Check if the transformation matrix need to be updated */
    cl_float3 position_; /*!< Position of the source/detector... */
    cl_float3 rotation_; /*! Rotation of the source/detector... */
    Matrix::float3x3 local_axis_; /*!< Matrix of local axis */
    Matrix::float4x4 matrix_translation_; /*!< Matrix of translation */
    Matrix::float4x4 matrix_rotation_; /*!< Matrix of rotation */
    Matrix::float4x4 matrix_orthographic_projection_; /*!< Matrix of orthographic projection */
    Matrix::float4x4 matrix_transformation_; /*!< Matrix of transformation */
};

#endif // End of GUARD_GGEMS_TOOLS_MATRIX_HH
