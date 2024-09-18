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
  \file GGEMSOpenGLParticles.cc

  \brief GGEMS class storing particles infos for OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday December 13, 2021
*/

#ifdef OPENGL_VISUALIZATION

#include "GGEMS/graphics/GGEMSOpenGLParticles.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/graphics/GGEMSOpenGLManager.hh"
#include "GGEMS/physics/GGEMSPrimaryParticles.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLParticles::GGEMSOpenGLParticles(void)
{
  GGcout("GGEMSOpenGLParticles", "GGEMSOpenGLParticles", 3) << "GGEMSOpenGLParticles creating..." << GGendl;

  number_of_vertices_ = MAXIMUM_DISPLAYED_PARTICLES * MAXIMUM_INTERACTIONS;
  number_of_indices_ = MAXIMUM_DISPLAYED_PARTICLES * MAXIMUM_INTERACTIONS + (MAXIMUM_INTERACTIONS);

  vao_ = 0;
  vbo_[0] = 0;
  vbo_[1] = 0;

  vertex_ = new GLfloat[3UL*number_of_vertices_];
  index_ = new GLuint[number_of_indices_];
  index_increment_ = 0;

  number_of_registered_particles_ = 0;
  number_of_particles_ = 0;

  // Defining shaders
  WriteShaders();

  // Initializing shaders
  GGEMSOpenGLManager& opengl_manager = GGEMSOpenGLManager::GetInstance();
  opengl_manager.InitShaders(vertex_shader_source_, fragment_shader_source_, program_shader_id_);

  // Creating a VAO and VBO
  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);

  // Creating VBO
  glGenBuffers(2, vbo_);

  // Vertex
  glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
  glBufferData(GL_ARRAY_BUFFER, 3 * number_of_vertices_ * sizeof(GLfloat), nullptr, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), nullptr);
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Index
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_[1]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, number_of_indices_ * sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  glBindVertexArray(0);

  GGcout("GGEMSOpenGLParticles", "GGEMSOpenGLParticles", 3) << "GGEMSOpenGLParticles created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLParticles::~GGEMSOpenGLParticles(void)
{
  GGcout("GGEMSOpenGLParticles", "~GGEMSOpenGLParticles", 3) << "GGEMSOpenGLParticles erasing..." << GGendl;

  // Destroying vao and vbo
  glDeleteBuffers(1, &vao_);
  glDeleteBuffers(2, &vbo_[0]);

  if (vertex_) {
    delete[] vertex_;
    vertex_ = nullptr;
  }

  if (index_) {
    delete[] index_;
    index_ = nullptr;
  }

  // Destroying program shader
  glDeleteProgram(program_shader_id_);

  GGcout("GGEMSOpenGLParticles", "~GGEMSOpenGLParticles", 3) << "GGEMSOpenGLParticles erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLParticles::WriteShaders(void)
{
  GGEMSOpenGLManager& opengl_manager = GGEMSOpenGLManager::GetInstance();

  // A global vertex shader
  vertex_shader_source_ = "#version " + opengl_manager.GetOpenGLSLVersion() + "\n"
    "\n"
    "layout(location = 0) in vec3 position;\n"
    "\n"
    "uniform mat4 mvp;\n"
    "uniform vec3 color;\n"
    "\n"
    "out vec4 color_rgba;\n"
    "\n"
    "void main(void) {\n"
    "  color_rgba = vec4(color, 1.0);\n"
    "  gl_Position = mvp * vec4(position, 1.0);\n"
    "}\n";

  // A global fragment shader
  fragment_shader_source_ = "#version " + opengl_manager.GetOpenGLSLVersion() + "\n"
    "\n"
    "layout(location = 0) out vec4 out_color;\n"
    "\n"
    "in vec4 color_rgba;\n"
    "\n"
    "void main(void) {\n"
    "  out_color = color_rgba;\n"
    "}\n";
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLParticles::SetNumberOfParticles(GGsize const& number_of_particles)
{
  number_of_particles_ = number_of_particles;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLParticles::CopyParticlePosition(void)
{
  GGcout("GGEMSOpenGLParticles", "CopyParticlePosition", 3) << "Copying particles to OpenGL buffer..." << GGendl;

  // Exit if buffer is full
  if (number_of_registered_particles_ == MAXIMUM_DISPLAYED_PARTICLES) return;

  // Getting singletons
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();

  // Getting primary particles from OpenCL
  cl::Buffer* primary_particles = source_manager.GetParticles()->GetPrimaryParticles(0);
  GGEMSPrimaryParticles* primary_particles_device = opencl_manager.GetDeviceBuffer<GGEMSPrimaryParticles>(primary_particles, CL_TRUE, CL_MAP_READ, sizeof(GGEMSPrimaryParticles), 0);

  // Loop over particles
  for (GGsize i = 0; i < number_of_particles_; ++i) {
    // Getting number of interactions for each primary particles
    GGsize stored_interactions = static_cast<GGsize>(primary_particles_device->stored_particles_gl_[i]);

    // Loop over interactions
    for (GGsize j = 0; j < stored_interactions; ++j) {
      vertex_[j*3+0+number_of_registered_particles_*MAXIMUM_INTERACTIONS*3] = primary_particles_device->px_gl_[j+i*MAXIMUM_INTERACTIONS];
      vertex_[j*3+1+number_of_registered_particles_*MAXIMUM_INTERACTIONS*3] = primary_particles_device->py_gl_[j+i*MAXIMUM_INTERACTIONS];
      vertex_[j*3+2+number_of_registered_particles_*MAXIMUM_INTERACTIONS*3] = primary_particles_device->pz_gl_[j+i*MAXIMUM_INTERACTIONS];

      index_[index_increment_++] = static_cast<GLuint>(j+number_of_registered_particles_*MAXIMUM_INTERACTIONS);
    }

    index_[index_increment_++] = 0xFFFFFFFF; // End of line

    number_of_registered_particles_++;

    // Exit if buffer is full
    if (number_of_registered_particles_ == MAXIMUM_DISPLAYED_PARTICLES) break;
  }

  // Release the pointers
  opencl_manager.ReleaseDeviceBuffer(primary_particles, primary_particles_device, 0);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLParticles::UploadParticleToOpenGL(void)
{
  GGcout("GGEMSOpenGLParticles", "CopyParticlePosition", 3) << "Uploading particles to OpenGL buffer..." << GGendl;

  glBindVertexArray(vao_);

  // Vertex
  glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);

  // Mapping OpenGL buffer
  GLfloat* vbo_vertex_ptr = static_cast<GLfloat*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));

  // Copying data
  ::memcpy(vbo_vertex_ptr, vertex_, 3*number_of_vertices_*sizeof(GLfloat));

  glUnmapBuffer(GL_ARRAY_BUFFER);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Index
  glBindBuffer(GL_ARRAY_BUFFER, vbo_[1]);

  // Mapping OpenGL buffer
  GLuint* vbo_index_ptr = static_cast<GLuint*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));

  // Copying data
  ::memcpy(vbo_index_ptr, index_, index_increment_*sizeof(GLuint)); // No need to upload all buffer, only registered index

  glUnmapBuffer(GL_ARRAY_BUFFER);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLParticles::Draw(void) const
{
  // Getting OpenCL pointer
  GGEMSOpenGLManager& opengl_manager = GGEMSOpenGLManager::GetInstance();

  // Getting color of gamma particle
  GGEMSRGBColor gamma_color = opengl_manager.GetGammaParticleColor();

  // Getting projection and camera view matrix
  glm::mat4 projection_matrix = opengl_manager.GetProjection();
  glm::mat4 view_matrix = opengl_manager.GetCameraView();

  // Enabling shader program
  glUseProgram(program_shader_id_);

  glBindVertexArray(vao_);

  glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_[1]);

  GLintptr offset_pointer = 0 * sizeof(GLfloat);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), reinterpret_cast<GLvoid*>(offset_pointer));
  glEnableVertexAttribArray(0);

  glm::mat4 mvp_matrix = projection_matrix*view_matrix;
  glUniformMatrix4fv(glGetUniformLocation(program_shader_id_, "mvp"), 1, GL_FALSE, &mvp_matrix[0][0]);

  // Setting point size
  glPointSize(8.0f);

  // Interaction points, always in yellow
  GLintptr index_pointer = 0 * sizeof(GLfloat);
  glUniform3f(glGetUniformLocation(program_shader_id_, "color"), 1.0f, 1.0f, 0.0f);
  glDrawElements(GL_POINTS, static_cast<GLsizei>(index_increment_), GL_UNSIGNED_INT, reinterpret_cast<GLvoid*>(index_pointer));

  glUniform3f(glGetUniformLocation(program_shader_id_, "color"), gamma_color.red_, gamma_color.green_, gamma_color.blue_);
  glDrawElements(GL_LINE_STRIP, static_cast<GLsizei>(index_increment_), GL_UNSIGNED_INT, reinterpret_cast<GLvoid*>(index_pointer));

  glDisableVertexAttribArray(0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  // Disable shader program
  glUseProgram(0);
}

#endif
