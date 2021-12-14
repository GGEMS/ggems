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

#include "GGEMS/graphics/GGEMSOpenGLParticles.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/graphics/GGEMSOpenGLManager.hh"
#include "GGEMS/physics/GGEMSPrimaryParticles.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLParticles::GGEMSOpenGLParticles(void)
{
  GGcout("GGEMSOpenGLParticles", "GGEMSOpenGLParticles", 3) << "GGEMSOpenGLParticles creating..." << GGendl;

  number_of_vertices_ = MAXIMUM_DISPLAYED_PARTICLES * MAXIMUM_INTERACTIONS * 3;
  number_of_indices_ = MAXIMUM_DISPLAYED_PARTICLES * MAXIMUM_INTERACTIONS + (MAXIMUM_INTERACTIONS-1);

  vao_ = 0;
  vbo_[0] = 0; // Vertex
  vbo_[1] = 0; // Index
  number_of_registered_particles_ = 0;
  number_of_particles_ = 0;

  // Defining shaders
  WriteShaders();

  // Initializing shaders
  GGEMSOpenGLManager& opengl_manager = GGEMSOpenGLManager::GetInstance();
  opengl_manager.InitShaders(vertex_shader_source_, fragment_shader_source_, program_shader_id_);

  // Creating a VAO
  glGenVertexArrays(1, &vao_);

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

void GGEMSOpenGLParticles::CreatingVBO(void)
{
  GGcout("GGEMSOpenGLParticles", "CreatingVBO", 3) << "Creating a new VBO buffer for particles..." << GGendl;

  glBindVertexArray(vao_);

  // Creating VBO
  glGenBuffers(2, &vbo_[0]);

  // Vertex
  glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
  glBufferData(GL_ARRAY_BUFFER, number_of_vertices_ * sizeof(GLfloat), nullptr, GL_DYNAMIC_DRAW); // Allocating memory on OpenGL device
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Index
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_[1]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, number_of_indices_ * sizeof(GLint), nullptr, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
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

  // Creating a new VBO if first copy
  if (number_of_registered_particles_ == 0) CreatingVBO();

  // Exit if buffer is full
  if (number_of_registered_particles_ == MAXIMUM_DISPLAYED_PARTICLES) return;

  // Getting singletons
  GGEMSOpenGLManager& opengl_manager = GGEMSOpenGLManager::GetInstance();
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();

  // Getting primary particles from OpenCL
 // cl::Buffer* primary_particles = source_manager.GetParticles()->GetPrimaryParticles(0);
 // GGEMSPrimaryParticles* primary_particles_device = opencl_manager.GetDeviceBuffer<GGEMSPrimaryParticles>(primary_particles, CL_TRUE, CL_MAP_READ, sizeof(GGEMSPrimaryParticles), 0);

  // Getting pointer to OpenGL vertex
  glBindVertexArray(vao_);

  glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
  GLfloat* vbo_ptr = static_cast<GLfloat*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
  if (!vbo_ptr) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Error mapping vbo buffer!!!" << std::endl;
    GGEMSMisc::ThrowException("GGEMSOpenGLParticles", "CopyParticlePosition", oss.str());
  }

  // Loop over particles
  for (GGint i = 0; i < number_of_particles_; ++i) {
    // Getting number of interactions for each primary particles
    //GGint stored_interactions = primary_particles_device->stored_particles_gl_[i];
    vbo_ptr[0] = 1.0f;
  //   // Loop over interactions
  //   for (GGint j = 0; j < stored_interactions; ++j) {
  //     // vbo_ptr[0] = primary_particles_device->px_gl_[j+i*MAXIMUM_INTERACTIONS];
  //     // vbo_ptr[0] = primary_particles_device->py_gl_[j+i*MAXIMUM_INTERACTIONS];
  //     // vbo_ptr[0] = primary_particles_device->pz_gl_[j+i*MAXIMUM_INTERACTIONS];
  //   }

    number_of_registered_particles_++;

    // Exit if buffer is full
    if (number_of_registered_particles_ == MAXIMUM_DISPLAYED_PARTICLES) return;
  }

  glUnmapBuffer(GL_ARRAY_BUFFER);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  //std::cout << number_of_registered_particles_ << std::endl;

  // for (GGint i = 0; i < number_of_displayed_particles; ++i) {
  //   std::cout << "*****" << std::endl;
  //   std::cout << primary_particles_device->stored_particles_gl_[i] << std::endl;
  //   GGint stored_particles = primary_particles_device->stored_particles_gl_[i];
  //   for (GGint j = 0; j < stored_particles; ++j) {
  //     std::cout << primary_particles_device->px_gl_[j+i*MAXIMUM_INTERACTIONS] << " " << primary_particles_device->py_gl_[j+i*MAXIMUM_INTERACTIONS] << " " << primary_particles_device->pz_gl_[j+i*MAXIMUM_INTERACTIONS] << std::endl;
  //   }
  //   number_of_registered_particles_++;
  // }

  // if (number_of_registered_particles_ == MAXIMUM_DISPLAYED_PARTICLES) {
  //   is_buffer_full_ = true;
  // }

/*
    float* vbo_ptr = static_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
    if (!vbo_ptr) {
      std::ostringstream oss(std::ostringstream::out);
      oss << "Error mapping vbo buffer!!!" << std::endl;
      throw std::runtime_error(oss.str());
    }

    // Loop over vertices
    for (int i = 0; i < number_of_latitude_vertices_ * number_of_longitude_vertices_; ++i) {
      int longitude_index = i / number_of_latitude_vertices_;
      int latitude_index = i % number_of_latitude_vertices_;

      float theta = static_cast<float>(M_PI) * latitude_index / number_of_latitude_vertices_;
      float phi = 2.0f * static_cast<float>(M_PI) * longitude_index / static_cast<float>(number_of_longitude_vertices_) + tick_;
      float sign = -2.0f * static_cast<float>(longitude_index % 2) + 1.0f;

      vbo_ptr[i*3 + 0] = 0.75f * sin(theta) * cos(phi);
      vbo_ptr[i*3 + 1] = 0.75f * sign * cos(theta);
      vbo_ptr[i*3 + 2] = 0.75f * sin(theta) * sin(phi);
    }

    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);*/

  // Release the pointers
 // opencl_manager.ReleaseDeviceBuffer(primary_particles, primary_particles_device, 0);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLParticles::Draw(void) const
{
/*  if (is_visible_) {
    // Getting OpenCL pointer
    GGEMSOpenGLManager& opengl_manager = GGEMSOpenGLManager::GetInstance();

    // Getting projection and camera view matrix
    glm::mat4 projection_matrix = opengl_manager.GetProjection();
    glm::mat4 view_matrix = opengl_manager.GetCameraView();

    // Translation matrix
    glm::mat4 translate_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(position_x_, position_y_, position_z_));

    // Creates an identity quaternion (no rotation)
    glm::quat quaternion;

    // Conversion from Euler angles (in radians) to Quaternion
    glm::vec3 euler_angle(angle_x_, angle_y_, angle_z_);
    quaternion = glm::quat(euler_angle);
    glm::mat4 rotation_matrix = glm::toMat4(quaternion);

    // Rotation after translation
    glm::vec3 euler_angle_after_translation(update_angle_x_, update_angle_y_, update_angle_z_);
    quaternion = glm::quat(euler_angle_after_translation);
    glm::mat4 rotation_matrix_after_translation = glm::toMat4(quaternion);

    // Enabling shader program
    glUseProgram(program_shader_id_);

    glBindVertexArray(vao_);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_[1]);

    // If only 1 color read in umap directly
    if (material_rgb_.size() == 1) {
      GGEMSRGBColor rgb_unique = material_rgb_.begin()->second;
      glUniform3f(glGetUniformLocation(program_shader_id_, "color"), rgb_unique.red_, rgb_unique.green_, rgb_unique.blue_);
    }

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(GLfloat), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(GLfloat), (void*)(3*sizeof(GLfloat)));
    glEnableVertexAttribArray(1);

    glm::mat4 mvp_matrix = projection_matrix*view_matrix*rotation_matrix_after_translation*translate_matrix*rotation_matrix;
    glUniformMatrix4fv(glGetUniformLocation(program_shader_id_, "mvp"), 1, GL_FALSE, &mvp_matrix[0][0]);

    // Draw volume using index
    glDrawElements(GL_TRIANGLES, number_of_indices_, GL_UNSIGNED_INT, (void*)0);

    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Disable shader program
    glUseProgram(0);
  }*/
}
