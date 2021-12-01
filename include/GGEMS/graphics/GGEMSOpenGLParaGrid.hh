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
      \brief GGEMSOpenGLParaGrid constructor
    */
    GGEMSOpenGLParaGrid(GLint const& elements_x, GLint const& elements_y, GLint const& elements_z, GLfloat const& element_size_x, GLfloat const& element_size_y, GLfloat const& element_size_z);

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
    void WriteShaders(void);

  private:
    GLint elements_x_; /*!< Number of elements in X */
    GLint elements_y_; /*!< Number of elements in Y */
    GLint elements_z_; /*!< Number of elements in Z */
    GLfloat element_size_x_; /*!< Size of elements in X */
    GLfloat element_size_y_; /*!< Size of elements in Y */
    GLfloat element_size_z_; /*!< Size of elements in Z */
};

#endif

#endif // End of GGEMS_GRAPHICS_GGEMSOPENGLSPHERE_HH
