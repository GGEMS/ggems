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
  \file GGEMSOpenGLParaGrid.cc

  \brief Parallelepiped grid volume for OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday November 23, 2021
*/

#ifdef OPENGL_VISUALIZATION

#include "GGEMS/graphics/GGEMSOpenGLManager.hh"
#include "GGEMS/graphics/GGEMSOpenGLParaGrid.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLParaGrid::GGEMSOpenGLParaGrid(GGsize const& elements_x, GGsize const& elements_y, GGsize const& elements_z, GLfloat const& element_size_x, GLfloat const& element_size_y, GLfloat const& element_size_z, bool const& is_midplanes)
: GGEMSOpenGLVolume(),
  elements_x_(elements_x),
  elements_y_(elements_y),
  elements_z_(elements_z),
  element_size_x_(element_size_x),
  element_size_y_(element_size_y),
  element_size_z_(element_size_z),
  is_midplanes_(is_midplanes)
{
  GGcout("GGEMSOpenGLParaGrid", "GGEMSOpenGLParaGrid", 3) << "GGEMSOpenGLParaGrid creating..." << GGendl;

  // Number of elements to draw
  if (is_midplanes_) {
    GGsize z = 0;
    if (elements_z_ > 1) z = elements_z_ - 1;

    number_of_elements_ = elements_x_ * elements_y_ + elements_x_ * elements_z_ + elements_y_ * elements_z_ - elements_x_ - elements_y_ - z;
  }
  else {
    number_of_elements_ = elements_x_ * elements_y_ * elements_z_;
  }

  // For each parallelepiped there are 8 vertices, 3 positions for each vertex and a RGB color for each vertex
  number_of_vertices_ = number_of_elements_ * 8UL * 3UL * 2UL;
  vertices_ = new GLfloat[number_of_vertices_];

  // For each parallelepiped there are 12 triangles
  number_of_triangles_ = number_of_elements_ * 12UL;
  number_of_indices_ = number_of_triangles_ * 3UL;
  indices_ = new GLuint[number_of_indices_];

  // Defining shaders
  WriteShaders();

  // Initializing shaders
  GGEMSOpenGLManager& opengl_manager = GGEMSOpenGLManager::GetInstance();
  opengl_manager.InitShaders(vertex_shader_source_, fragment_shader_source_, program_shader_id_);

  is_color_in_vertex_buffer_ = true;

  GGcout("GGEMSOpenGLParaGrid", "GGEMSOpenGLParaGrid", 3) << "GGEMSOpenGLParaGrid created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLParaGrid::~GGEMSOpenGLParaGrid(void)
{
  GGcout("GGEMSOpenGLParaGrid", "~GGEMSOpenGLParaGrid", 3) << "GGEMSOpenGLParaGrid erasing..." << GGendl;

  GGcout("GGEMSOpenGLParaGrid", "~GGEMSOpenGLParaGrid", 3) << "GGEMSOpenGLParaGrid erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLParaGrid::WriteShaders(void)
{
  GGEMSOpenGLManager& opengl_manager = GGEMSOpenGLManager::GetInstance();

  // A global vertex shader
  vertex_shader_source_ = "#version " + opengl_manager.GetOpenGLSLVersion() + "\n"
    "\n"
    "layout(location = 0) in vec3 position;\n"
    "layout(location = 1) in vec3 color;\n"
    "\n"
    "uniform mat4 mvp;\n"
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

GGEMSRGBColor GGEMSOpenGLParaGrid::GetRGBColor(GGsize const& index) const
{
  // Read label and get material color
  GGuchar index_material = 0;
  if (material_rgb_.size() > 1) index_material = label_[index];

  // Getting material name and read rgb color
  std::string material_name = material_names_.at(index_material);
  GGEMSRGBColor rgb = material_rgb_.at(material_name);

  return rgb;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

bool GGEMSOpenGLParaGrid::IsMaterialVisible(GGsize const index) const
{
  // Read label and get material color
  GGuchar index_material = 0;
  if (material_rgb_.size() > 1) index_material = label_[index];

  // Getting material name and read visibility
  std::string material_name = material_names_.at(index_material);

  MaterialVisibleUMap::const_iterator iter = material_visible_.find(material_name);
  if (iter == material_visible_.end()) {
    return true;
  }
  else {
    return iter->second;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLParaGrid::Build(void)
{
  GGcout("GGEMSOpenGLParaGrid", "Build", 3) << "Building OpenGL para..." << GGendl;

  // Storing vertices
  GLfloat x_offset = 0.0f, y_offset = 0.0f, z_offset = 0.0f;
  GGsize index = 0;
  GGsize flag_index = 0;

  // Center of first parallelepiped
  GLfloat x_center = -(element_size_x_*static_cast<GLfloat>(elements_x_)*0.5f) + element_size_x_*0.5f;
  GLfloat y_center = -(element_size_y_*static_cast<GLfloat>(elements_y_)*0.5f) + element_size_y_*0.5f;
  GLfloat z_center = -(element_size_z_*static_cast<GLfloat>(elements_z_)*0.5f) + element_size_z_*0.5f;

  // if midplanes, computing index of midplane
  GGsize x_midplane = elements_x_ / 2UL;
  GGsize y_midplane = elements_y_ / 2UL;
  GGsize z_midplane = elements_z_ / 2UL;

  // Buffer storing visible voxel
  std::vector<bool> is_visible_voxel;
  is_visible_voxel.resize(number_of_elements_);

  // Loop over all parallelepipeds
  if (is_midplanes_) {

    for (GGsize k = 0; k < elements_z_; ++k) {
      z_offset = z_center + static_cast<GLfloat>(k)*element_size_z_;
      for (GGsize j = 0; j < elements_y_; ++j) {
        y_offset = y_center + static_cast<GLfloat>(j)*element_size_y_;
        for (GGsize i = 0; i < elements_x_; ++i) {
          x_offset = x_center + static_cast<GLfloat>(i)*element_size_x_;

          if (k == z_midplane || j == y_midplane || i == x_midplane) {

            // Getting color of voxel
            GGEMSRGBColor rgb = GetRGBColor(i + j*elements_x_ + k*elements_x_*elements_y_);
            is_visible_voxel[flag_index++] = IsMaterialVisible(i + j*elements_x_ + k*elements_x_*elements_y_);

            // 0
            vertices_[index++] = x_offset - element_size_x_*0.5f;
            vertices_[index++] = y_offset - element_size_y_*0.5f;
            vertices_[index++] = z_offset - element_size_z_*0.5f;
            vertices_[index++] = rgb.red_;
            vertices_[index++] = rgb.green_;
            vertices_[index++] = rgb.blue_;

            // 1
            vertices_[index++] = x_offset - element_size_x_*0.5f;
            vertices_[index++] = y_offset + element_size_y_*0.5f;
            vertices_[index++] = z_offset - element_size_z_*0.5f;
            vertices_[index++] = rgb.red_;
            vertices_[index++] = rgb.green_;
            vertices_[index++] = rgb.blue_;

            // 2
            vertices_[index++] = x_offset - element_size_x_*0.5f;
            vertices_[index++] = y_offset + element_size_y_*0.5f;
            vertices_[index++] = z_offset + element_size_z_*0.5f;
            vertices_[index++] = rgb.red_;
            vertices_[index++] = rgb.green_;
            vertices_[index++] = rgb.blue_;

            // 3
            vertices_[index++] = x_offset - element_size_x_*0.5f;
            vertices_[index++] = y_offset - element_size_y_*0.5f;
            vertices_[index++] = z_offset + element_size_z_*0.5f;
            vertices_[index++] = rgb.red_;
            vertices_[index++] = rgb.green_;
            vertices_[index++] = rgb.blue_;

            // 4
            vertices_[index++] = x_offset + element_size_x_*0.5f;
            vertices_[index++] = y_offset - element_size_y_*0.5f;
            vertices_[index++] = z_offset - element_size_z_*0.5f;
            vertices_[index++] = rgb.red_;
            vertices_[index++] = rgb.green_;
            vertices_[index++] = rgb.blue_;

            // 5
            vertices_[index++] = x_offset + element_size_x_*0.5f;
            vertices_[index++] = y_offset + element_size_y_*0.5f;
            vertices_[index++] = z_offset - element_size_z_*0.5f;
            vertices_[index++] = rgb.red_;
            vertices_[index++] = rgb.green_;
            vertices_[index++] = rgb.blue_;

            // 6
            vertices_[index++] = x_offset + element_size_x_*0.5f;
            vertices_[index++] = y_offset + element_size_y_*0.5f;
            vertices_[index++] = z_offset + element_size_z_*0.5f;
            vertices_[index++] = rgb.red_;
            vertices_[index++] = rgb.green_;
            vertices_[index++] = rgb.blue_;

            // 7
            vertices_[index++] = x_offset + element_size_x_*0.5f;
            vertices_[index++] = y_offset - element_size_y_*0.5f;
            vertices_[index++] = z_offset + element_size_z_*0.5f;
            vertices_[index++] = rgb.red_;
            vertices_[index++] = rgb.green_;
            vertices_[index++] = rgb.blue_;
          }
        }
      }
    }
  }
  else {
    for (GGsize k = 0; k < elements_z_; ++k) {
      z_offset = z_center + static_cast<GLfloat>(k)*element_size_z_;
      for (GGsize j = 0; j < elements_y_; ++j) {
        y_offset = y_center + static_cast<GLfloat>(j)*element_size_y_;
        for (GGsize i = 0; i < elements_x_; ++i) {
          x_offset = x_center + static_cast<GLfloat>(i)*element_size_x_;

          // Getting color of voxel
          GGEMSRGBColor rgb = GetRGBColor(i + j*elements_x_ + k*elements_x_*elements_y_);
          is_visible_voxel[flag_index++] = IsMaterialVisible(i + j*elements_x_ + k*elements_x_*elements_y_);

          // 0
          vertices_[index++] = x_offset - element_size_x_*0.5f;
          vertices_[index++] = y_offset - element_size_y_*0.5f;
          vertices_[index++] = z_offset - element_size_z_*0.5f;
          vertices_[index++] = rgb.red_;
          vertices_[index++] = rgb.green_;
          vertices_[index++] = rgb.blue_;

          // 1
          vertices_[index++] = x_offset - element_size_x_*0.5f;
          vertices_[index++] = y_offset + element_size_y_*0.5f;
          vertices_[index++] = z_offset - element_size_z_*0.5f;
          vertices_[index++] = rgb.red_;
          vertices_[index++] = rgb.green_;
          vertices_[index++] = rgb.blue_;

          // 2
          vertices_[index++] = x_offset - element_size_x_*0.5f;
          vertices_[index++] = y_offset + element_size_y_*0.5f;
          vertices_[index++] = z_offset + element_size_z_*0.5f;
          vertices_[index++] = rgb.red_;
          vertices_[index++] = rgb.green_;
          vertices_[index++] = rgb.blue_;

          // 3
          vertices_[index++] = x_offset - element_size_x_*0.5f;
          vertices_[index++] = y_offset - element_size_y_*0.5f;
          vertices_[index++] = z_offset + element_size_z_*0.5f;
          vertices_[index++] = rgb.red_;
          vertices_[index++] = rgb.green_;
          vertices_[index++] = rgb.blue_;

          // 4
          vertices_[index++] = x_offset + element_size_x_*0.5f;
          vertices_[index++] = y_offset - element_size_y_*0.5f;
          vertices_[index++] = z_offset - element_size_z_*0.5f;
          vertices_[index++] = rgb.red_;
          vertices_[index++] = rgb.green_;
          vertices_[index++] = rgb.blue_;

          // 5
          vertices_[index++] = x_offset + element_size_x_*0.5f;
          vertices_[index++] = y_offset + element_size_y_*0.5f;
          vertices_[index++] = z_offset - element_size_z_*0.5f;
          vertices_[index++] = rgb.red_;
          vertices_[index++] = rgb.green_;
          vertices_[index++] = rgb.blue_;

          // 6
          vertices_[index++] = x_offset + element_size_x_*0.5f;
          vertices_[index++] = y_offset + element_size_y_*0.5f;
          vertices_[index++] = z_offset + element_size_z_*0.5f;
          vertices_[index++] = rgb.red_;
          vertices_[index++] = rgb.green_;
          vertices_[index++] = rgb.blue_;

          // 7
          vertices_[index++] = x_offset + element_size_x_*0.5f;
          vertices_[index++] = y_offset - element_size_y_*0.5f;
          vertices_[index++] = z_offset + element_size_z_*0.5f;
          vertices_[index++] = rgb.red_;
          vertices_[index++] = rgb.green_;
          vertices_[index++] = rgb.blue_;
        }
      }
    }
  }

  index = 0;
  for (GGsize i = 0; i < number_of_elements_; ++i) {
    if (is_visible_voxel.at(i)) {
      // Triangle 0
      indices_[index++] = 2+static_cast<GLuint>(i)*8;
      indices_[index++] = 3+static_cast<GLuint>(i)*8;
      indices_[index++] = 7+static_cast<GLuint>(i)*8;

      // 1
      indices_[index++] = 2+static_cast<GLuint>(i)*8;
      indices_[index++] = 7+static_cast<GLuint>(i)*8;
      indices_[index++] = 6+static_cast<GLuint>(i)*8;

      // 2
      indices_[index++] = 6+static_cast<GLuint>(i)*8;
      indices_[index++] = 7+static_cast<GLuint>(i)*8;
      indices_[index++] = 4+static_cast<GLuint>(i)*8;

      // 3
      indices_[index++] = 6+static_cast<GLuint>(i)*8;
      indices_[index++] = 4+static_cast<GLuint>(i)*8;
      indices_[index++] = 5+static_cast<GLuint>(i)*8;

      // 4
      indices_[index++] = 5+static_cast<GLuint>(i)*8;
      indices_[index++] = 4+static_cast<GLuint>(i)*8;
      indices_[index++] = 0+static_cast<GLuint>(i)*8;

      // 5
      indices_[index++] = 5+static_cast<GLuint>(i)*8;
      indices_[index++] = 0+static_cast<GLuint>(i)*8;
      indices_[index++] = 1+static_cast<GLuint>(i)*8;

      // 6
      indices_[index++] = 1+static_cast<GLuint>(i)*8;
      indices_[index++] = 0+static_cast<GLuint>(i)*8;
      indices_[index++] = 3+static_cast<GLuint>(i)*8;

      // 7
      indices_[index++] = 1+static_cast<GLuint>(i)*8;
      indices_[index++] = 3+static_cast<GLuint>(i)*8;
      indices_[index++] = 2+static_cast<GLuint>(i)*8;

      // 8
      indices_[index++] = 1+static_cast<GLuint>(i)*8;
      indices_[index++] = 2+static_cast<GLuint>(i)*8;
      indices_[index++] = 6+static_cast<GLuint>(i)*8;

      // 9
      indices_[index++] = 1+static_cast<GLuint>(i)*8;
      indices_[index++] = 6+static_cast<GLuint>(i)*8;
      indices_[index++] = 5+static_cast<GLuint>(i)*8;

      // 10
      indices_[index++] = 0+static_cast<GLuint>(i)*8;
      indices_[index++] = 3+static_cast<GLuint>(i)*8;
      indices_[index++] = 7+static_cast<GLuint>(i)*8;

      // 11
      indices_[index++] = 0+static_cast<GLuint>(i)*8;
      indices_[index++] = 7+static_cast<GLuint>(i)*8;
      indices_[index++] = 4+static_cast<GLuint>(i)*8;
    }
  }

  number_of_indices_ = index;

  // Creating a VAO
  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);

  // Creating 2 VBOs
  glGenBuffers(2, vbo_);

  // Vertex
  glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
  glBufferData(GL_ARRAY_BUFFER, number_of_vertices_ * sizeof(GLfloat), vertices_, GL_STATIC_DRAW); // Allocating memory on OpenGL device
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Indices
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_[1]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, number_of_indices_ * sizeof(GLint), indices_, GL_STATIC_DRAW); // Allocating memory on OpenGL device
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
}

#endif
