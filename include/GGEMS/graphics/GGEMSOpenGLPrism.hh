#ifndef GUARD_GGEMS_GRAPHICS_GGEMSOPENGLPRISM_HH
#define GUARD_GGEMS_GRAPHICS_GGEMSOPENGLPRISM_HH

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
  \file GGEMSOpenGLPrism.hh

  \brief Prism volume for OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday November 8, 2021
*/

#ifdef OPENGL_VISUALIZATION

#include "GGEMS/graphics/GGEMSOpenGLVolume.hh"

/*!
  \class GGEMSOpenGLPrism
  \brief This class define a prism volume for OpenGL
*/
class GGEMS_EXPORT GGEMSOpenGLPrism : public GGEMSOpenGLVolume
{
  public:
    /*!
      \param base_radius - radius of base circle
      \param top_radius - radius of top circle
      \param height - height of prism
      \param sectors - number of sector for prism (number of side, 3: triangular prism ...)
      \param stacks - number of stacks
      \brief GGEMSOpenGLPrism constructor
    */
    GGEMSOpenGLPrism(GLfloat const& base_radius, GLfloat const& top_radius, GLfloat const& height, GGsize const& sectors, GGsize const& stacks);

    /*!
      \brief GGEMSOpenGLPrism destructor
    */
    ~GGEMSOpenGLPrism(void);

    /*!
      \fn GGEMSOpenGLPrism(GGEMSOpenGLPrism const& prism) = delete
      \param prism - reference on the OpenGL prism volume
      \brief Avoid copy by reference
    */
    GGEMSOpenGLPrism(GGEMSOpenGLPrism const& prism) = delete;

    /*!
      \fn GGEMSOpenGLPrism& operator=(GGEMSOpenGLPrism const& prism) = delete
      \param prism - reference on the OpenGL prism volume
      \brief Avoid assignement by reference
    */
    GGEMSOpenGLPrism& operator=(GGEMSOpenGLPrism const& prism) = delete;

    /*!
      \fn GGEMSOpenGLPrism(GGEMSOpenGLPrism const&& prism) = delete
      \param prism - rvalue reference on OpenGL prism volume
      \brief Avoid copy by rvalue reference
    */
    GGEMSOpenGLPrism(GGEMSOpenGLPrism const&& prism) = delete;

    /*!
      \fn GGEMSOpenGLPrism& operator=(GGEMSOpenGLPrism const&& prism) = delete
      \param prism - rvalue reference on OpenGL prism volume
      \brief Avoid copy by rvalue reference
    */
    GGEMSOpenGLPrism& operator=(GGEMSOpenGLPrism const&& prism) = delete;

    /*!
      \fn void Build(void)
      \brief method building OpenGL volume and storing VAO and VBO
    */
    void Build(void) override;

  private:
    /*!
      \fn void BuildUnitCircleVertices(void)
      \brief Build unit circle vertices once
    */
    void BuildUnitCircleVertices(void);

    /*!
      \fn void WriteShaders(void)
      \brief write shader source file for each volume
    */
    void WriteShaders(void) override;

  private:
    GLfloat base_radius_; /*!< Radius of base circle */
    GLfloat top_radius_; /*!< Radius of base circle */
    GLfloat height_; /*!< Height of prism */

    GLfloat* unit_circle_vertices_; /*!< vertices of a unit circle on XY plane */
};

#endif

#endif // End of GGEMS_GRAPHICS_GGEMSOPENGLSPHERE_HH
