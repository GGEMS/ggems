#ifndef GUARD_GGEMS_GRAPHICS_GGEMSOPENGLSPHERE_HH
#define GUARD_GGEMS_GRAPHICS_GGEMSOPENGLSPHERE_HH

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
  \file GGEMSOpenGLSphere.hh

  \brief Sphere volume for OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday November 2, 2021
*/

#ifdef OPENGL_VISUALIZATION

#include "GGEMS/graphics/GGEMSOpenGLVolume.hh"

/*!
  \class GGEMSOpenGLSphere
  \brief This class define a sphere volume for OpenGL
*/
class GGEMS_EXPORT GGEMSOpenGLSphere : public GGEMSOpenGLVolume
{
  public:
    /*!
      \param radius - radius of sphere
      \brief GGEMSOpenGLSphere constructor
    */
    explicit GGEMSOpenGLSphere(GLfloat const& radius);

    /*!
      \brief GGEMSOpenGLSphere destructor
    */
    ~GGEMSOpenGLSphere(void);

    /*!
      \fn GGEMSOpenGLSphere(GGEMSOpenGLSphere const& sphere) = delete
      \param sphere - reference on the OpenGL sphere volume
      \brief Avoid copy by reference
    */
    GGEMSOpenGLSphere(GGEMSOpenGLSphere const& sphere) = delete;

    /*!
      \fn GGEMSOpenGLSphere& operator=(GGEMSOpenGLSphere const& sphere) = delete
      \param sphere - reference on the OpenGL sphere volume
      \brief Avoid assignement by reference
    */
    GGEMSOpenGLSphere& operator=(GGEMSOpenGLSphere const& sphere) = delete;

    /*!
      \fn GGEMSOpenGLSphere(GGEMSOpenGLSphere const&& sphere) = delete
      \param sphere - rvalue reference on OpenGL sphere volume
      \brief Avoid copy by rvalue reference
    */
    GGEMSOpenGLSphere(GGEMSOpenGLSphere const&& sphere) = delete;

    /*!
      \fn GGEMSOpenGLSphere& operator=(GGEMSOpenGLSphere const&& sphere) = delete
      \param sphere - rvalue reference on OpenGL sphere volume
      \brief Avoid copy by rvalue reference
    */
    GGEMSOpenGLSphere& operator=(GGEMSOpenGLSphere const&& sphere) = delete;

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

  private:
    GLfloat radius_; /*!< Radius of sphere */
};

#endif

#endif // End of GGEMS_GRAPHICS_GGEMSOPENGLSPHERE_HH
