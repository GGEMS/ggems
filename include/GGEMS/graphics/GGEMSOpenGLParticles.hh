#ifndef GUARD_GGEMS_GRAPHICS_GGEMSOPENGLPARTICLES_HH
#define GUARD_GGEMS_GRAPHICS_GGEMSOPENGLPARTICLES_HH

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
  \file GGEMSOpenGLParticles.hh

  \brief GGEMS class storing particles infos for OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday December 13, 2021
*/

#ifdef OPENGL_VISUALIZATION

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include <vector>

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \class GGEMSOpenGLParticles
  \brief GGEMS class storing particles infos for OpenGL
*/
class GGEMS_EXPORT GGEMSOpenGLParticles
{
  public:
    /*!
      \brief GGEMSOpenGLParticles constructor
    */
    GGEMSOpenGLParticles(void);

    /*!
      \brief GGEMSOpenGLParticles destructor
    */
    virtual ~GGEMSOpenGLParticles(void);

    /*!
      \fn GGEMSOpenGLParticles(GGEMSOpenGLParticles const& particles) = delete
      \param particles - reference on the GGEMS OpenGL particles
      \brief Avoid copy by reference
    */
    GGEMSOpenGLParticles(GGEMSOpenGLParticles const& particles) = delete;

    /*!
      \fn GGEMSOpenGLParticles& operator=(GGEMSOpenGLParticles const& particles) = delete
      \param particles - reference on the GGEMS OpenGL particles
      \brief Avoid assignement by reference
    */
    GGEMSOpenGLParticles& operator=(GGEMSOpenGLParticles const& particles) = delete;

    /*!
      \fn GGEMSOpenGLParticles(GGEMSOpenGLParticles const&& particles) = delete
      \param particles - rvalue reference on the GGEMS OpenGL particles
      \brief Avoid copy by rvalue reference
    */
    GGEMSOpenGLParticles(GGEMSOpenGLParticles const&& particles) = delete;

    /*!
      \fn GGEMSOpenGLParticles& operator=(GGEMSOpenGLParticles const&& particles) = delete
      \param particles - rvalue reference on the GGEMS OpenGL particles
      \brief Avoid copy by rvalue reference
    */
    GGEMSOpenGLParticles& operator=(GGEMSOpenGLParticles const&& particles) = delete;

    /*!
      \fn void CopyParticlePosition(GGsize const& source_index)
      \param source_index - source index
      \brief Copy particle position from OpenCL kernel to OpenGL memory
    */
    void CopyParticlePosition(GGsize const& source_index);

  private:
    GLuint vao_; /*!< vertex array object for all sources */
    std::vector<GLuint> vbo_; /*!< vertex buffer object, index 0 -> source 0, index 1 -> source 1 etc... */
};

#endif

#endif // End of GUARD_GGEMS_GRAPHICS_GGEMSOPENGLPARTICLES_HH
