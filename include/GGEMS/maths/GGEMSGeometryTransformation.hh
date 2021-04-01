#ifndef GUARD_GGEMS_MATHS_GGEMSGEOMETRYTRANSFORMATION_HH
#define GUARD_GGEMS_MATHS_GGEMSGEOMETRYTRANSFORMATION_HH

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
  \file GGEMSGeometryTransformation.hh

  \brief Class managing the geometry transformation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday November 13, 2019
*/

#include "GGEMS/maths/GGEMSMatrixTypes.hh"
#include "GGEMS/global/GGEMSOpenCLManager.hh"

/*!
  \class GGEMSGeometryTransformation
  \brief This class handles everything about geometry transformation
*/
class GGEMS_EXPORT GGEMSGeometryTransformation
{
  public:
    /*!
      \brief GGEMSGeometryTransformation constructor
    */
    GGEMSGeometryTransformation(void);

    /*!
      \brief GGEMSGeometryTransformation destructor
    */
    ~GGEMSGeometryTransformation(void);

    /*!
      \fn GGEMSGeometryTransformation(GGEMSGeometryTransformation const& geometry_transformation) = delete
      \param geometry_transformation - reference on the geometry transformation
      \brief Avoid copy of the class by reference
    */
    GGEMSGeometryTransformation(GGEMSGeometryTransformation const& transform_calculator) = delete;

    /*!
      \fn GGEMSGeometryTransformation& operator=(GGEMSGeometryTransformation const& geometry_transformation) = delete
      \param geometry_transformation - reference on the geometry transformation
      \brief Avoid assignement of the class by reference
    */
    GGEMSGeometryTransformation& operator=(GGEMSGeometryTransformation const& geometry_transformation) = delete;

    /*!
      \fn GGEMSGeometryTransformation(GGEMSGeometryTransformation const&& geometry_transformation) = delete
      \param geometry_transformation - rvalue reference on the geometry transformation
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSGeometryTransformation(GGEMSGeometryTransformation const&& geometry_transformation) = delete;

    /*!
      \fn GGEMSGeometryTransformation& operator=(GGEMSGeometryTransformation const&& geometry_transformation) = delete
      \param geometry_transformation - rvalue reference on the geometry transformation
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSGeometryTransformation& operator=(GGEMSGeometryTransformation const&& geometry_transformation) = delete;

    /*!
      \fn void SetTranslation(GGfloat const& tx, GGfloat const& ty, GGfloat const& tz)
      \param tx - Translation in X
      \param ty - Translation in Y
      \param tz - Translation in Z
      \brief Set the translation in X, Y and Z
    */
    void SetTranslation(GGfloat const& tx, GGfloat const& ty, GGfloat const& tz);

    /*!
      \fn void SetTranslation(GGfloat3 const& txyz)
      \param txyz - Translation in X, Y and Z
      \brief Set the translation in X, Y and Z
    */
    void SetTranslation(GGfloat3 const& txyz);

    /*!
      \fn inline float4x4 GetMatrixTranslation(void) const
      \return the translation matrix
      \brief Return the translation matrix
    */
    inline GGfloat44 GetMatrixTranslation(void) const {return matrix_translation_;};

    /*!
      \fn void SetRotation(GGfloat const& rx, GGfloat const& ry, GGfloat const& rz)
      \param rx - Rotation in X
      \param ry - Rotation in Y
      \param rz - Rotation in Z
      \brief Set the Rotation in X, Y and Z around global axis
    */
    void SetRotation(GGfloat const& rx, GGfloat const& ry, GGfloat const& rz);

    /*!
      \fn void SetRotation(GGfloat3 const& rxyz)
      \param rxyz - Rotation around X, Y and Z global axis
      \brief Set the rotation around global axis
    */
    void SetRotation(GGfloat3 const& rxyz);

    /*!
      \fn inline GGfloat44 GetMatrixRotation(void) const
      \return the translation matrix
      \brief Return the translation matrix
    */
    inline GGfloat44 GetMatrixRotation(void) const {return matrix_rotation_;};

    /*!
      \fn void SetAxisTransformation(GGfloat33 const& axis)
      \param axis - Matrix (3x3) that contains the mapping of the coordinates (ex. x becomes y and vice-versa). Values are 0, 1 or -1.
      \brief Set the transformation of the frame, usefull for mirroring or convert 3D to 2D
    */
    void SetAxisTransformation(GGfloat33 const& axis);

    /*!
      \fn void SetAxisTransformation(GGfloat3 const& m0, GGfloat3 const& m1, GGfloat3 const& m2)
      \param m0 - Row 0 in the matrix 3x3 for local axis
      \param m1 - Row 1 in the matrix 3x3 for local axis
      \param m2 - Row 2 in the matrix 3x3 for local axis
      \brief Set the transformation of the frame, usefull for mirroring or convert 3D to 2D
    */
    void SetAxisTransformation(GGfloat3 const& m0, GGfloat3 const& m1, GGfloat3 const& m2);

    /*!
      \fn inline GGfloat44 GetMatrixOrthographicProjection(void) const
      \return the matrix of orthographic projection
      \brief return the matrix of orthographic projection
    */
    inline GGfloat44 GetMatrixOrthographicProjection(void) const {return matrix_orthographic_projection_;};

    /*!
      \fn inline GGfloat3 GetPosition(void) const
      \return The position of source/detector...
      \brief Return the current position
    */
    inline GGfloat3 GetPosition(void) const {return position_;}

    /*!
      \fn inline GGfloat3 GetRotation(void) const
      \return The rotation of source/detector...
      \brief Return the current rotation
    */
    inline GGfloat3 GetRotation(void) const {return rotation_;}

    /*!
      \fn inline GGfloat33 GetLocalAxis(void) const
      \return The local axis matrix
      \brief return the local axis matrix
    */
    inline GGfloat33 GetLocalAxis(void) const {return local_axis_;}

    /*!
      \fn Matrix::float4x4 GetTransformationMatrix(GGsize const& index) const
      \param index - index of device
      \return the transformation matrix
      \brief return the transformation matrix
    */
    inline cl::Buffer* GetTransformationMatrix(GGsize const& index) const {return matrix_transformation_[index];}

  private:
    GGfloat3 position_; /*!< Position of the source/detector */
    GGfloat3 rotation_; /*!< Rotation of the source/detector */
    GGfloat33 local_axis_; /*!< Matrix of local axis */
    GGfloat44 matrix_translation_; /*!< Matrix of translation */
    GGfloat44 matrix_rotation_; /*!< Matrix of rotation */
    GGfloat44 matrix_orthographic_projection_; /*!< Matrix of orthographic projection */
    cl::Buffer** matrix_transformation_; /*!< OpenCL buffer storing the matrix transformation */
    GGsize number_activated_devices_; /*!< Number of activated device */
};

#endif // End of GUARD_GGEMS_MATHS_TRANSFORMATION_MATRIX_HH
