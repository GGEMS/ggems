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
      \fn void CopyParticlePosition(void)
      \brief Copy particle position from OpenCL kernel to OpenGL memory
    */
    void CopyParticlePosition(void);

    /*!
      \fn void Draw(void) const
      \brief Draw particles into the screen
    */
    void Draw(void) const;

    /*!
      \fn void SetNumberOfParticles(GGsize const& number_of_particles)
      \param number_of_particles - number of particles to draw
      \brief set number of particles to draw to OpenGL window
    */
    void SetNumberOfParticles(GGsize const& number_of_particles);

    /*!
      \fn void UploadParticleToOpenGL(void) const
      \brief Upload particles infos to OpenGL buffers
    */
    void UploadParticleToOpenGL(void);

  private:
    /*!
      \fn void WriteShaders(void)
      \brief write shader source file for each volume
    */
    void WriteShaders(void);

  private:
    GLint number_of_vertices_; /*!< Number of vertices for OpenGL particles */
    GLint number_of_indices_; /*!< Number of indices */

    bool is_buffer_full_; /*!< flag for buffer */
    GGuint number_of_registered_particles_; /*!< Number of registered particles */
    GGsize number_of_particles_; /*!< Number of primary particles to follow */

    GLuint vao_; /*!< vertex array object for all sources */
    GLuint vbo_[2]; /*!< vbo index for vertex and index */

    GLfloat* vertex_; /*!< Pointer storing vertex positions */
    GLuint* index_; /*!< Pointer storing index positions */
    GLint index_increment_; /*< Index increment, useful to store position index */

    GLuint program_shader_id_; /*!< program id for shader */
    std::string vertex_shader_source_; /*!< vertex shader source file */
    std::string fragment_shader_source_; /*!< fragment shader source file */
};

#endif

#endif // End of GUARD_GGEMS_GRAPHICS_GGEMSOPENGLPARTICLES_HH
