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
  \file GGEMSOpenGLSphere.cc

  \brief Sphere volume for OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday November 2, 2021
*/

#ifdef OPENGL_VISUALIZATION

#include "GGEMS/graphics/GGEMSOpenGLManager.hh"
#include "GGEMS/graphics/GGEMSOpenGLSphere.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/global/GGEMSConstants.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLSphere::GGEMSOpenGLSphere(GLfloat const& radius)
: GGEMSOpenGLVolume(),
  radius_(radius)
{
  GGcout("GGEMSOpenGLSphere", "GGEMSOpenGLSphere", 3) << "GGEMSOpenGLSphere creating..." << GGendl;

  number_of_sectors_ = 64; // value by default (longitude)
  number_of_stacks_ = 32; // value by default (latitude)

  // Allocating memory for sphere vertices
  // extreme point postions are taken (+1 stack and sector)
  number_of_vertices_ = (number_of_stacks_+1)*(number_of_sectors_+1)*3;
  vertices_ = new GLfloat[number_of_vertices_];

  // Compute number of (triangulated) indices.
  // In each sector/stack there are 2 triangles
  // For first and last stack there is 1 triangle, 3 indices for each triangle
  number_of_triangles_ = ((number_of_stacks_*2)-2)*number_of_sectors_;
  number_of_indices_ = number_of_triangles_*3;
  indices_ = new GLuint[number_of_indices_];

  // Defining shaders
  WriteShaders();

  // Initializing shaders
  GGEMSOpenGLManager& opengl_manager = GGEMSOpenGLManager::GetInstance();
  opengl_manager.InitShaders(vertex_shader_source_, fragment_shader_source_, program_shader_id_);

  GGcout("GGEMSOpenGLSphere", "GGEMSOpenGLSphere", 3) << "GGEMSOpenGLSphere created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLSphere::~GGEMSOpenGLSphere(void)
{
  GGcout("GGEMSOpenGLSphere", "~GGEMSOpenGLSphere", 3) << "GGEMSOpenGLSphere erasing..." << GGendl;

  GGcout("GGEMSOpenGLSphere", "~GGEMSOpenGLSphere", 3) << "GGEMSOpenGLSphere erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLSphere::WriteShaders(void)
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

void GGEMSOpenGLSphere::Build(void)
{
  GGcout("GGEMSOpenGLSphere", "Build", 3) << "Building OpenGL sphere..." << GGendl;

  // Compute x, y, z with
  // xy = radius * cos(stack_angle)
  // x = xy * cos(sector_angle)
  // y = xy * sin(sector_angle)
  // z = radius * sin(stack_angle)

  GLfloat sector_step = 2.0f * PI / static_cast<GGfloat>(number_of_sectors_);
  GLfloat sector_angle = 0.0f;
  GLfloat stack_step = PI / static_cast<GGfloat>(number_of_stacks_);
  GLfloat stack_angle = 0.0f;

  GLfloat x = 0.0f, y = 0.0f, xy = 0.0f, z = 0.0f;
  GLint index = 0;
  // Loop over the stacks
  for (GGsize i = 0; i <= number_of_stacks_; ++i) {
    // Stack angle
    stack_angle = HALF_PI - static_cast<GLfloat>(i) * stack_step; // from pi/2 to -pi/2

    xy = radius_ * std::cos(stack_angle);
    z = radius_ * std::sin(stack_angle);

    // Loop over the sectors
    for (GGsize j = 0; j <= number_of_sectors_; ++j) {
      sector_angle = static_cast<GLfloat>(j) * sector_step; // from 0 to 2pi

      x = xy * std::cos(sector_angle);
      y = xy * std::sin(sector_angle);

      vertices_[index++] = x;
      vertices_[index++] = y;
      vertices_[index++] = z;
    }
  }

  // There are 2 triangles in each stack/sector
  // At top and bottom stack, there is 1 triangle per sector
  // Indices inside stack/sector
  //  k1--k1+1
  //  |  / |
  //  | /  |
  //  k2--k2+1
  index = 0;
  GGsize k1 = 0, k2 = 0;
  for (GGsize i = 0; i < number_of_stacks_; ++i) {
    k1 = i * (number_of_sectors_ + 1);
    k2 = k1 + number_of_sectors_ + 1;
    for (GGsize j = 0; j < number_of_sectors_; ++j, ++k1, ++k2) {
      // 2 triangles per sector excluding 1st and last stacks
      if (i != 0) { // Triangle k1, k2, k1+1
        indices_[index++] = static_cast<GLuint>(k1);
        indices_[index++] = static_cast<GLuint>(k2);
        indices_[index++] = static_cast<GLuint>(k1+1);
      }

      if (i != number_of_stacks_-1) { // Triangle k1+1, k2, k2+1
        indices_[index++] = static_cast<GLuint>(k1+1);
        indices_[index++] = static_cast<GLuint>(k2);
        indices_[index++] = static_cast<GLuint>(k2+1);
      }
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
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), reinterpret_cast<GLvoid*>(offset_pointer));
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Indices
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_[1]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, number_of_indices_ * sizeof(GLint), indices_, GL_STATIC_DRAW); // Allocating memory on OpenGL device
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
}

#endif
