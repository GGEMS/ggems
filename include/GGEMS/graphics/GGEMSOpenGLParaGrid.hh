#ifndef GUARD_GGEMS_GRAPHICS_GGEMSOPENGLPARA_HH
#define GUARD_GGEMS_GRAPHICS_GGEMSOPENGLPARA_HH

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
  \file GGEMSOpenGLParaGrid.hh

  \brief Parallelepiped grid volume for OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday November 23, 2021
*/

#ifdef OPENGL_VISUALIZATION

#include "GGEMS/graphics/GGEMSOpenGLVolume.hh"

/*!
  \class GGEMSOpenGLParaGrid
  \brief This class define a parallelepiped volume for OpenGL
*/
class GGEMS_EXPORT GGEMSOpenGLParaGrid : public GGEMSOpenGLVolume
{
  public:
    /*!
      \param elements_x - x elements
      \param elements_y - y elements
      \param elements_z - z elements
      \param element_size_x - element size along X
      \param element_size_y - element size along Y
      \param element_size_z - element size along Z
      \param is_midplanes - true if only midplanes are drawn
      \brief GGEMSOpenGLParaGrid constructor
    */
    GGEMSOpenGLParaGrid(GGsize const& elements_x, GGsize const& elements_y, GGsize const& elements_z, GLfloat const& element_size_x, GLfloat const& element_size_y, GLfloat const& element_size_z, bool const& is_midplanes);

    /*!
      \brief GGEMSOpenGLParaGrid destructor
    */
    ~GGEMSOpenGLParaGrid(void);

    /*!
      \fn GGEMSOpenGLParaGrid(GGEMSOpenGLParaGrid const& para) = delete
      \param para - reference on the OpenGL para volume
      \brief Avoid copy by reference
    */
    GGEMSOpenGLParaGrid(GGEMSOpenGLParaGrid const& para) = delete;

    /*!
      \fn GGEMSOpenGLParaGrid& operator=(GGEMSOpenGLParaGrid const& para) = delete
      \param para - reference on the OpenGL para volume
      \brief Avoid assignement by reference
    */
    GGEMSOpenGLParaGrid& operator=(GGEMSOpenGLParaGrid const& para) = delete;

    /*!
      \fn GGEMSOpenGLParaGrid(GGEMSOpenGLParaGrid const&& para) = delete
      \param para - rvalue reference on OpenGL para volume
      \brief Avoid copy by rvalue reference
    */
    GGEMSOpenGLParaGrid(GGEMSOpenGLParaGrid const&& para) = delete;

    /*!
      \fn GGEMSOpenGLParaGrid& operator=(GGEMSOpenGLParaGrid const&& para) = delete
      \param para - rvalue reference on OpenGL para volume
      \brief Avoid copy by rvalue reference
    */
    GGEMSOpenGLParaGrid& operator=(GGEMSOpenGLParaGrid const&& para) = delete;

    /*!
      \fn void Build(void)
      \brief method building OpenGL volume and storing VAO and VBO
    */
    void Build(void) override;

  private:
    /*!
      \fn void WriteShaders(void)
      \brief write shader source file for each volume
    */
    void WriteShaders(void) override;

    /*!
      \fn GGEMSRGBColor GetRGBColor(GGsize const& index) const
      \param index - index of voxel for label
      \return RGB color of a voxel
      \brief Getting the RGB color of voxels
    */
    GGEMSRGBColor GetRGBColor(GGsize const& index) const;

    /*!
      \fn bool IsMaterialVisible(GGsize const index) const
      \param index - index of voxel for label
      \return true if material is visible
      \brief check if material is visible or not
    */
    bool IsMaterialVisible(GGsize const index) const;

  private:
    GGsize number_of_elements_; /*!< Number of elements to draw */
    GGsize elements_x_; /*!< Number of elements in X */
    GGsize elements_y_; /*!< Number of elements in Y */
    GGsize elements_z_; /*!< Number of elements in Z */
    GLfloat element_size_x_; /*!< Size of elements in X */
    GLfloat element_size_y_; /*!< Size of elements in Y */
    GLfloat element_size_z_; /*!< Size of elements in Z */
    bool is_midplanes_; /*!< Flag drawing midplanes for big volume */
};

#endif

#endif // End of GGEMS_GRAPHICS_GGEMSOPENGLSPHERE_HH
