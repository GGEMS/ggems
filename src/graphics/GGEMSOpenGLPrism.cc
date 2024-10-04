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
  \file GGEMSOpenGLPrism.cc

  \brief Prism volume for OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday November 2, 2021
*/

#ifdef OPENGL_VISUALIZATION

#include "GGEMS/graphics/GGEMSOpenGLPrism.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/global/GGEMSConstants.hh"
#include "GGEMS/graphics/GGEMSOpenGLManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLPrism::GGEMSOpenGLPrism(GLfloat const& base_radius, GLfloat const& top_radius, GLfloat const& height, GGsize const& sectors, GGsize const& stacks)
: GGEMSOpenGLVolume()
{
  GGcout("GGEMSOpenGLPrism", "GGEMSOpenGLPrism", 3) << "GGEMSOpenGLPrism creating..." << GGendl;

  base_radius_ = base_radius;
  top_radius_ = top_radius;
  height_ = height;

  number_of_sectors_ = sectors;
  number_of_stacks_ = stacks;

  if (number_of_stacks_ < 1) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Minimum number of stacks for prism is 1";
    GGEMSMisc::ThrowException("GGEMSOpenGLPrism", "GGEMSOpenGLPrism", oss.str());
  }

  if (number_of_sectors_ < 3) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Minimum number of sectors for prism is 3";
    GGEMSMisc::ThrowException("GGEMSOpenGLPrism", "GGEMSOpenGLPrism", oss.str());
  }

  // Allocating memory for prism vertices
  // extreme point postions are taken (+1 stack and sector)
  number_of_vertices_ = 2*3*(number_of_sectors_+1) + (number_of_stacks_+1)*(number_of_sectors_+1)*3;
  vertices_ = new GLfloat[number_of_vertices_];

  // Compute number of (triangulated) indices.
  // In each sector there are 2 triangles
  number_of_triangles_ = number_of_stacks_ * number_of_sectors_*2 + 2*number_of_sectors_;
  number_of_indices_ = number_of_triangles_*3;
  indices_ = new GLuint[number_of_indices_];

  // Defining shaders
  WriteShaders();

  // Initializing shaders
  GGEMSOpenGLManager& opengl_manager = GGEMSOpenGLManager::GetInstance();
  opengl_manager.InitShaders(vertex_shader_source_, fragment_shader_source_, program_shader_id_);

  GGcout("GGEMSOpenGLPrism", "GGEMSOpenGLPrism", 3) << "GGEMSOpenGLPrism created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLPrism::~GGEMSOpenGLPrism(void)
{
  GGcout("GGEMSOpenGLPrism", "~GGEMSOpenGLPrism", 3) << "GGEMSOpenGLPrism erasing..." << GGendl;

  if (unit_circle_vertices_) {
    delete[] unit_circle_vertices_;
    unit_circle_vertices_ = nullptr;
  }

  GGcout("GGEMSOpenGLPrism", "~GGEMSOpenGLPrism", 3) << "GGEMSOpenGLPrism erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLPrism::WriteShaders(void)
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

void GGEMSOpenGLPrism::BuildUnitCircleVertices(void)
{
  GLfloat sector_step = 2.0f * PI / static_cast<GGfloat>(number_of_sectors_);
  GLfloat sector_angle = 0.0f;

  unit_circle_vertices_ = new GLfloat[(number_of_sectors_+1)*3];

  for (GGsize i = 0; i <= number_of_sectors_; ++i) {
    sector_angle = static_cast<GLfloat>(i) * sector_step;
    unit_circle_vertices_[i*3+0] = std::cos(sector_angle); // x
    unit_circle_vertices_[i*3+1] = std::sin(sector_angle); // y
    unit_circle_vertices_[i*3+2] = 0.0f; // z
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLPrism::Build(void)
{
  GGcout("GGEMSOpenGLPrism", "Build", 3) << "Building OpenGL prism..." << GGendl;

  // We compute the vertices of a unit circle on XY plane only once
  BuildUnitCircleVertices();

  // Put vertices of side cylinder to array by scaling unit circle
  GLfloat x = 0.0f, y = 0.0f, z = 0.0f;
  GGsize added_vertices = 0;
  GGsize index = 0;
  for (GGsize i = 0; i <= number_of_stacks_; ++i) {
    // Vertex position z
    z = -(height_ * 0.5f) + static_cast<GLfloat>(i) / static_cast<GLfloat>(number_of_stacks_) * height_;
    GLfloat radius = base_radius_ + static_cast<GLfloat>(i) / static_cast<GLfloat>(number_of_stacks_) * (top_radius_-base_radius_);

    // Loop over sectors
    for (GGsize j = 0, k = 0; j <= number_of_sectors_; ++j, k += 3) {
      x = unit_circle_vertices_[k]*radius;
      y = unit_circle_vertices_[k+1]*radius;

      vertices_[index++] = x;
      vertices_[index++] = y;
      vertices_[index++] = z;

      added_vertices += 1;
    }
  }

  // remember where the base vertices start
  GGsize base_vertex_index = added_vertices;

  // put vertices of base of cylinder
  z = -height_ * 0.5f;
  vertices_[index++] = 0;
  vertices_[index++] = 0;
  vertices_[index++] = z;
  added_vertices += 1;

  for (GGsize i = 0, j = 0; i < number_of_sectors_; ++i, j += 3) {
    x = unit_circle_vertices_[j]*base_radius_;
    y = unit_circle_vertices_[j+1]*base_radius_;

    vertices_[index++] = x;
    vertices_[index++] = y;
    vertices_[index++] = z;

    added_vertices += 1;
  }

  // remember where the top vertices start
  GGsize top_vertex_index = added_vertices;

  // put vertices of top of cylinder
  z = height_ * 0.5f;
  vertices_[index++] = 0;
  vertices_[index++] = 0;
  vertices_[index++] = z;
  added_vertices += 1;

  for (GGsize i = 0, j = 0; i < number_of_sectors_; ++i, j += 3) {
    x = unit_circle_vertices_[j]*top_radius_;
    y = unit_circle_vertices_[j+1]*top_radius_;

    vertices_[index++] = x;
    vertices_[index++] = y;
    vertices_[index++] = z;

    added_vertices += 1;
  }

  // Put indices for sides
  GGsize k1 = 0, k2 = 0;
  index = 0;
  for (GGsize i = 0; i < number_of_stacks_; ++i) {
    k1 = i * (number_of_sectors_+1); // beginning of current stack
    k2 = k1 + number_of_sectors_ + 1; // beginning of next stack

    for (GGsize j = 0; j < number_of_sectors_; ++j, ++k1, ++k2) {
      // first triangle
      indices_[index++] = static_cast<GLuint>(k1);
      indices_[index++] = static_cast<GLuint>(k1+1);
      indices_[index++] = static_cast<GLuint>(k2);
      // second triangle
      indices_[index++] = static_cast<GLuint>(k2);
      indices_[index++] = static_cast<GLuint>(k1+1);
      indices_[index++] = static_cast<GLuint>(k2+1);
    }
  }

  // Put indices for base
  for (GGsize i = 0, k = base_vertex_index + 1; i < number_of_sectors_; ++i, ++k) {
    if(i < (number_of_sectors_ - 1)) {
      indices_[index++] = static_cast<GLuint>(base_vertex_index);
      indices_[index++] = static_cast<GLuint>(k+1);
      indices_[index++] = static_cast<GLuint>(k);
    }
    else { // last triangle
      indices_[index++] = static_cast<GLuint>(base_vertex_index);
      indices_[index++] = static_cast<GLuint>(base_vertex_index+1);
      indices_[index++] = static_cast<GLuint>(k);
    }
  }

  // Put indices for top
  for (GGsize i = 0, k = top_vertex_index + 1; i < number_of_sectors_; ++i, ++k) {
    if (i < (number_of_sectors_ - 1)) {
      indices_[index++] = static_cast<GLuint>(top_vertex_index);
      indices_[index++] = static_cast<GLuint>(k);
      indices_[index++] = static_cast<GLuint>(k+1);
    }
    else {
      indices_[index++] = static_cast<GLuint>(top_vertex_index);
      indices_[index++] = static_cast<GLuint>(k);
      indices_[index++] = static_cast<GLuint>(top_vertex_index+1);
    }
  }

  // Creating a VAO
  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);

  // Creating 2 VBOs
  glGenBuffers(2, vbo_);

  // Vertex
  glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
  glBufferData(GL_ARRAY_BUFFER, number_of_vertices_ * sizeof(GLfloat), vertices_, GL_STATIC_DRAW); // Allocating memory on OpenGL device
  GLintptr offset_pointer = 0 * sizeof(GLfloat);
  //glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), reinterpret_cast<GLvoid*>(offset_pointer));
  //glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Indices
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_[1]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, number_of_indices_ * sizeof(GLint), indices_, GL_STATIC_DRAW); // Allocating memory on OpenGL device
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
}

#endif
