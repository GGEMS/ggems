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
  \file GGEMSOpenGLPara.hh

  \brief Parallelepiped volume for OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday November 23, 2021
*/

#ifdef OPENGL_VISUALIZATION

#include "GGEMS/graphics/GGEMSOpenGLVolume.hh"

/*!
  \class GGEMSOpenGLPara
  \brief This class define a parallelepiped volume for OpenGL
*/
class GGEMS_EXPORT GGEMSOpenGLPara : public GGEMSOpenGLVolume
{
  public:
    /*!
      \param elements_x - x elements
      \param elements_y - y elements
      \param elements_z - z elements
      \param element_size_x - element size along X
      \param element_size_y - element size along Y
      \param element_size_z - element size along Z
      \brief GGEMSOpenGLPara constructor
    */
    GGEMSOpenGLPara(GLint const& elements_x, GLint const& elements_y, GLint const& elements_z, GLfloat const& element_size_x, GLfloat const& element_size_y, GLfloat const& element_size_z);

    /*!
      \brief GGEMSOpenGLPara destructor
    */
    ~GGEMSOpenGLPara(void);

    /*!
      \fn GGEMSOpenGLPara(GGEMSOpenGLPara const& para) = delete
      \param para - reference on the OpenGL para volume
      \brief Avoid copy by reference
    */
    GGEMSOpenGLPara(GGEMSOpenGLPara const& para) = delete;

    /*!
      \fn GGEMSOpenGLPara& operator=(GGEMSOpenGLPara const& para) = delete
      \param para - reference on the OpenGL para volume
      \brief Avoid assignement by reference
    */
    GGEMSOpenGLPara& operator=(GGEMSOpenGLPara const& para) = delete;

    /*!
      \fn GGEMSOpenGLPara(GGEMSOpenGLPara const&& para) = delete
      \param para - rvalue reference on OpenGL para volume
      \brief Avoid copy by rvalue reference
    */
    GGEMSOpenGLPara(GGEMSOpenGLPara const&& para) = delete;

    /*!
      \fn GGEMSOpenGLPara& operator=(GGEMSOpenGLPara const&& para) = delete
      \param para - rvalue reference on OpenGL para volume
      \brief Avoid copy by rvalue reference
    */
    GGEMSOpenGLPara& operator=(GGEMSOpenGLPara const&& para) = delete;

    /*!
      \fn void Build(void)
      \brief method building OpenGL volume and storing VAO and VBO
    */
    void Build(void) override;

  private:
};

#endif

#endif // End of GGEMS_GRAPHICS_GGEMSOPENGLSPHERE_HH
