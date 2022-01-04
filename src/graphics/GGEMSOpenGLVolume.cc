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
#include "GGEMS/materials/GGEMSMaterials.hh"

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

  update_angle_x_ = 0.0f;
  update_angle_y_ = 0.0f;
  update_angle_z_ = 0.0f;

  // By defaut, default material and white color
  GGEMSRGBColor rgb;
  rgb.red_ = 1.0f;
  rgb.green_ = 1.0f;
  rgb.blue_ = 1.0f;
  material_rgb_.insert(std::make_pair("Default", rgb));
  label_ =  nullptr;
  material_names_.clear();
  material_visible_.clear();

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

  // Destroying program shader
  glDeleteProgram(program_shader_id_);

  // Destroying buffers
  if (vertices_) {
    delete[] vertices_;
    vertices_ = nullptr;
  }

  if (indices_) {
    delete[] indices_;
    indices_ = nullptr;
  }

  if (label_) {
    delete[] label_;
    label_ = nullptr;
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
  // In radians
  angle_x_ = angle_x;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLVolume::SetYAngle(GLfloat const& angle_y)
{
  // In radians
  angle_y_ = angle_y;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLVolume::SetZAngle(GLfloat const& angle_z)
{
  // In radians
  angle_z_ = angle_z;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLVolume::SetXUpdateAngle(GLfloat const& update_angle_x)
{
  // In radians
  update_angle_x_ = update_angle_x;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLVolume::SetYUpdateAngle(GLfloat const& update_angle_y)
{
  // In radians
  update_angle_y_ = update_angle_y;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLVolume::SetZUpdateAngle(GLfloat const& update_angle_z)
{
  // In radians
  update_angle_z_ = update_angle_z;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLVolume::SetColorName(std::string const& color)
{
  // Getting color map from GGEMSOpenGLManager
  ColorUMap colors = GGEMSOpenGLManager::GetInstance().GetColorUMap();

  // Select color
  ColorUMap::iterator it = colors.find(color);
  if (it != colors.end()) {
    material_rgb_["Default"].red_ = GGEMSOpenGLColor::color[it->second][0];
    material_rgb_["Default"].green_ = GGEMSOpenGLColor::color[it->second][1];
    material_rgb_["Default"].blue_ = GGEMSOpenGLColor::color[it->second][2];
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

void GGEMSOpenGLVolume::SetMaterial(std::string const& material_name)
{
  // Cleaning previous color and material
  material_rgb_.clear();

  GGEMSMaterialsDatabaseManager& material_manager = GGEMSMaterialsDatabaseManager::GetInstance();
  GGEMSRGBColor rgb = material_manager.GetMaterialRGBColor(material_name);
  material_rgb_.insert(std::make_pair(material_name, rgb));
  material_names_.push_back(material_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLVolume::SetMaterial(GGEMSMaterials const* materials, cl::Buffer* label, GGsize const& number_of_voxels)
{
  // Cleaning previous color and material
  material_rgb_.clear();

  // Getting material manager
  GGEMSMaterialsDatabaseManager& material_manager = GGEMSMaterialsDatabaseManager::GetInstance();
  for (GGsize i = 0; i < materials->GetNumberOfMaterials(); ++i) {
    std::string material_name = materials->GetMaterialName(i);
    GGEMSRGBColor rgb = material_manager.GetMaterialRGBColor(material_name);
    material_rgb_.insert(std::make_pair(material_name, rgb));
    material_names_.push_back(material_name);
  }

  // Storing label from OpenCL
  label_ = new GGuchar[number_of_voxels];

  // Get pointer on OpenCL device
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGuchar* label_data_device = opencl_manager.GetDeviceBuffer<GGuchar>(label, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, number_of_voxels * sizeof(GGuchar), 0);

  // Copy data
  for (GGsize i = 0; i < number_of_voxels; ++i) label_[i] = label_data_device[i];

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(label, label_data_device, 0);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLVolume::SetCustomMaterialColor(MaterialRGBColorUMap const& custom_material_rgb)
{
  // Loop over custom material
  for (auto&& i : custom_material_rgb) {
    // Finding custom material color in default container
    MaterialRGBColorUMap::iterator iter = material_rgb_.find(i.first);
    if (iter != material_rgb_.end()) {
      iter->second.red_ = i.second.red_;
      iter->second.green_ = i.second.green_;
      iter->second.blue_ = i.second.blue_;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLVolume::SetMaterialVisible(MaterialVisibleUMap const& material_visible)
{
  material_visible_ = material_visible;
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

    GLintptr vertex_position_offset = 0 * sizeof(GLfloat);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(GLfloat), reinterpret_cast<GLvoid*>(vertex_position_offset));
    glEnableVertexAttribArray(0);

    GLintptr vertex_color_offset = 3 * sizeof(GLfloat);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(GLfloat), reinterpret_cast<GLvoid*>(vertex_color_offset));
    glEnableVertexAttribArray(1);

    glm::mat4 mvp_matrix = projection_matrix*view_matrix*rotation_matrix_after_translation*translate_matrix*rotation_matrix;
    glUniformMatrix4fv(glGetUniformLocation(program_shader_id_, "mvp"), 1, GL_FALSE, &mvp_matrix[0][0]);

    // Draw volume using index
    GLintptr index_pointer = 0 * sizeof(GLfloat);
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(number_of_indices_), GL_UNSIGNED_INT, reinterpret_cast<GLvoid*>(index_pointer));

    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Disable shader program
    glUseProgram(0);
  }
}

#endif
