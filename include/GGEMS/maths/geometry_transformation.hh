#ifndef GUARD_GGEMS_MATHS_GEOMETRY_TRANSFORMATION_HH
#define GUARD_GGEMS_MATHS_GEOMETRY_TRANSFORMATION_HH

/*!
  \file geometry_transformation.hh

  \brief Class managing the geometry transformation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday November 13, 2019
*/

#include "GGEMS/opencl/types.hh"
#include "GGEMS/maths/matrix_types.hh"

/*!
  \class GeometryTransformation
  \brief This class handles everything about geometry transformation
*/
class GeometryTransformation
{
  public:
    /*!
      \brief GeometryTransformation constructor
    */
    GeometryTransformation(void);

    /*!
      \brief GeometryTransformation destructor
    */
    ~GeometryTransformation(void);

  public:
    /*!
      \fn GeometryTransformation(GeometryTransformation const& geometry_transformation) = delete
      \param geometry_transformation - reference on the geometry transformation
      \brief Avoid copy of the class by reference
    */
    GeometryTransformation(
      GeometryTransformation const& transform_calculator) = delete;

    /*!
      \fn GeometryTransformation& operator=(GeometryTransformation const& geometry_transformation) = delete
      \param geometry_transformation - reference on the geometry transformation
      \brief Avoid assignement of the class by reference
    */
    GeometryTransformation& operator=(
      GeometryTransformation const& geometry_transformation) = delete;

    /*!
      \fn TransfGeometryTransformationormCalculator(GeometryTransformation const&& geometry_transformation) = delete
      \param geometry_transformation - rvalue reference on the geometry transformation
      \brief Avoid copy of the class by rvalue reference
    */
    GeometryTransformation(
      GeometryTransformation const&& geometry_transformation) = delete;

    /*!
      \fn GeometryTransformation& operator=(GeometryTransformation const&& geometry_transformation) = delete
      \param geometry_transformation - rvalue reference on the geometry transformation
      \brief Avoid copy of the class by rvalue reference
    */
    GeometryTransformation& operator=(
      GeometryTransformation const&& geometry_transformation) = delete;

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
      \fn inline float4x4 GetMatrixTranslation(void) const
      \return the translation matrix
      \brief Return the translation matrix
    */
    inline float4x4 GetMatrixTranslation(void) const
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
    void SetRotation(f323cl_t const& rxyz);

    /*!
      \fn inline Matrix::float4x4 GetMatrixRotation(void) const
      \return the translation matrix
      \brief Return the translation matrix
    */
    inline float4x4 GetMatrixRotation(void) const
    {
      return matrix_rotation_;
    };

    /*!
      \fn void SetAxisTransformation(float3x3 const& axis)
      \param axis - Matrix (3x3) that contains the mapping of the coordinates (ex. x becomes y and vice-versa). Values are 0, 1 or -1.
      \brief Set the transformation of the frame, usefull for mirroring or convert 3D to 2D
    */
    void SetAxisTransformation(float3x3 const& axis);

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
      \fn inline float4x4 GetMatrixOrthographicProjection(void) const
      \return the matrix of orthographic projection
      \brief return the matrix of orthographic projection
    */
    inline float4x4 GetMatrixOrthographicProjection(void) const
    {
      return matrix_orthographic_projection_;
    };

    /*!
      \fn inline f323cl_t GetPosition(void) const
      \return The position of source/detector...
      \brief Return the current position
    */
    inline f323cl_t GetPosition(void) const {return position_;}

    /*!
      \fn inline f323cl_t GetRotation(void) const
      \return The rotation of source/detector...
      \brief Return the current rotation
    */
    inline f323cl_t GetRotation(void) const {return rotation_;}

    /*!
      \fn inline float3x3 GetLocalAxis(void) const
      \return The local axis matrix
      \brief return the local axis matrix
    */
    inline float3x3 GetLocalAxis(void) const {return local_axis_;}

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
    inline float4x4 GetTransformationMatrix(void)
    {
      // Check if we need to update
      if (is_need_updated_) UpdateTransformationMatrix();

      // Return the transformation matrix
      return matrix_transformation_;
    }

  private:
    bool is_need_updated_; /*!< Check if the transformation matrix need to be updated */
    f323cl_t position_; /*!< Position of the source/detector... */
    f323cl_t rotation_; /*! Rotation of the source/detector... */
    float3x3 local_axis_; /*!< Matrix of local axis */
    float4x4 matrix_translation_; /*!< Matrix of translation */
    float4x4 matrix_rotation_; /*!< Matrix of rotation */
    float4x4 matrix_orthographic_projection_; /*!< Matrix of orthographic projection */
    float4x4 matrix_transformation_; /*!< Matrix of transformation */
};

#endif // End of GUARD_GGEMS_MATHS_TRANSFORMATION_MATRIX_HH
