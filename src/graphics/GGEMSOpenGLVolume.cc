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
  \file GGEMSOpenGLVolume.cc

  \brief GGEMS mother class defining volume for OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday November 2, 2021
*/

#ifdef OPENGL_VISUALIZATION

#include "GGEMS/graphics/GGEMSOpenGLVolume.hh"
#include "GGEMS/graphics/GGEMSOpenGLManager.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLVolume::GGEMSOpenGLVolume()
{
  GGcout("GGEMSOpenGLVolume", "GGEMSOpenGLVolume", 3) << "GGEMSOpenGLVolume creating..." << GGendl;

  position_x_ = 0.0f;
  position_y_ = 0.0f;
  position_z_ = 0.0f;

  angle_x_ = 0.0f;
  angle_y_ = 0.0f;
  angle_z_ = 0.0f;

  color_[0] = 1.0f; // red
  color_[1] = 0.0f;
  color_[2] = 0.0f;

  vao_ = 0;
  vbo_[0] = 0; // Vertex
  vbo_[1] = 0; // Indice

  vertices_ = nullptr;
  number_of_vertices_ = 0;
  indices_ = nullptr;
  number_of_indices_ = 0;
  number_of_triangles_ = 0;

  is_visible_ = true;

  number_of_stacks_ = 0;
  number_of_sectors_ = 0;

  // Store the OpenGL volume
  GGEMSOpenGLManager::GetInstance().Store(this);

  GGcout("GGEMSOpenGLVolume", "GGEMSOpenGLVolume", 3) << "GGEMSOpenGLVolume created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLVolume::~GGEMSOpenGLVolume(void)
{
  GGcout("GGEMSOpenGLVolume", "~GGEMSOpenGLVolume", 3) << "GGEMSOpenGLVolume erasing..." << GGendl;


  // Destroying vao and vbo
  glDeleteBuffers(1, &vao_);
  glDeleteBuffers(2, &vbo_[0]);

  // Destroying buffers
  if (vertices_) {
    delete[] vertices_;
    vertices_ = nullptr;
  }

  if (indices_) {
    delete[] indices_;
    indices_ = nullptr;
  }

  GGcout("GGEMSOpenGLVolume", "~GGEMSOpenGLVolume", 3) << "GGEMSOpenGLVolume erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLVolume::SetPosition(GLfloat const& position_x, GLfloat const& position_y, GLfloat const& position_z)
{
  position_x_ = position_x;
  position_y_ = position_y;
  position_z_ = position_z;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLVolume::SetXAngle(GLfloat const& angle_x)
{
  angle_x_ = glm::radians(angle_x);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLVolume::SetYAngle(GLfloat const& angle_y)
{
  angle_y_ = glm::radians(angle_y);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLVolume::SetZAngle(GLfloat const& angle_z)
{
  angle_z_ = glm::radians(angle_z);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLVolume::SetColor(std::string const& color)
{
  // Getting color map from GGEMSOpenGLManager
  ColorUMap colors = GGEMSOpenGLManager::GetInstance().GetColorUMap();

  // Select color
  ColorUMap::iterator it = colors.find(color);
  if (it != colors.end()) {
    for (int i = 0; i < 3; ++i) {
      color_[i] = GGEMSOpenGLColor::color[it->second][i];
    }
  }
  else {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Warning!!! Color background not found in the list !!!" << std::endl;
    oss << "Available colors: " << std::endl;
    oss << "    * black" << std::endl;
    oss << "    * blue" << std::endl;
    oss << "    * cyan" << std::endl;
    oss << "    * red" << std::endl;
    oss << "    * magenta" << std::endl;
    oss << "    * yellow" << std::endl;
    oss << "    * white" << std::endl;
    oss << "    * gray" << std::endl;
    oss << "    * silver" << std::endl;
    oss << "    * maroon" << std::endl;
    oss << "    * olive" << std::endl;
    oss << "    * green" << std::endl;
    oss << "    * purple" << std::endl;
    oss << "    * teal" << std::endl;
    oss << "    * navy";
    GGEMSMisc::ThrowException("GGEMSOpenGLManager", "SetBackgroundColor", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLVolume::SetVisible(bool const& is_visible)
{
  is_visible_ = is_visible;
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLVolume::Draw(void) const
{
  if (is_visible_) {
    // Getting OpenCL pointer
    GGEMSOpenGLManager& opengl_manager = GGEMSOpenGLManager::GetInstance();

    // Get program shader from OpenGL manager
    GLuint program_shader_id = opengl_manager.GetProgramShaderID();

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

    // Enabling shader program
    glUseProgram(program_shader_id);

    glBindVertexArray(vao_);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_[1]);

    // Setting color
    glUniform3f(glGetUniformLocation(program_shader_id, "color"), color_[0], color_[1], color_[2]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    glm::mat4 mvp_matrix = projection_matrix*view_matrix*translate_matrix*rotation_matrix;
    glUniformMatrix4fv(glGetUniformLocation(program_shader_id, "mvp"), 1, GL_FALSE, &mvp_matrix[0][0]);

    // Draw volume using index
    glDrawElements(GL_TRIANGLES, number_of_indices_, GL_UNSIGNED_INT, (void*)0);

    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Disable shader program
    glUseProgram(0);
  }
}

#endif
