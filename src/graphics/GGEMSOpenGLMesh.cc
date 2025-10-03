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
  \file GGEMSOpenGLMesh.cc

  \brief Mesh volume for OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday September 5, 2024
*/

#ifdef OPENGL_VISUALIZATION

#include "GGEMS/graphics/GGEMSOpenGLMesh.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLMesh::GGEMSOpenGLMesh(GGEMSTriangle3* triangles, GGuint const& number_of_triangles)
: GGEMSOpenGLVolume(),
  triangles_(triangles)
{
  GGcout("GGEMSOpenGLMesh", "GGEMSOpenGLMesh", 3) << "GGEMSOpenGLMesh creating..." << GGendl;

  number_of_triangles_ = number_of_triangles;

  number_of_vertices_ = number_of_triangles_ * 3 * 3;
  vertices_ = new GLfloat[number_of_vertices_];

  number_of_indices_ = number_of_triangles_ * 3;
  indices_ = new GLuint[number_of_indices_];

  // Defining shaders
  WriteShaders();

  is_color_in_vertex_buffer_ = false;

  // Initializing shaders
  GGEMSOpenGLManager& opengl_manager = GGEMSOpenGLManager::GetInstance();
  opengl_manager.InitShaders(vertex_shader_source_, fragment_shader_source_, program_shader_id_);

  GGcout("GGEMSOpenGLMesh", "GGEMSOpenGLMesh", 3) << "GGEMSOpenGLMesh created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLMesh::~GGEMSOpenGLMesh(void)
{
  GGcout("GGEMSOpenGLMesh", "~GGEMSOpenGLMesh", 3) << "GGEMSOpenGLMesh erasing..." << GGendl;

  GGcout("GGEMSOpenGLMesh", "~GGEMSOpenGLMesh", 3) << "GGEMSOpenGLMesh erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLMesh::WriteShaders(void)
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

void GGEMSOpenGLMesh::Build(void)
{
  GGcout("GGEMSOpenGLMesh", "Build", 3) << "Building OpenGL mesh..." << GGendl;

  GGsize index_vertex = 0;
  for (GGuint i = 0; i < number_of_triangles_; ++i) {
    for (GGuint j = 0; j < 3; ++j) {
      vertices_[index_vertex++] = triangles_[i].pts_[j].x_;
      vertices_[index_vertex++] = triangles_[i].pts_[j].y_;
      vertices_[index_vertex++] = triangles_[i].pts_[j].z_;
    }
  }

  for (GGsize i = 0; i < number_of_indices_; ++i) {
    indices_[i] = static_cast<GLuint>(i);
  }

  // Creating a VAO
  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);

  // Creating 2 VBOs
  glGenBuffers(2, vbo_);

  // Vertex
  glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
  glBufferData(GL_ARRAY_BUFFER, number_of_vertices_ * sizeof(GLfloat), vertices_, GL_STATIC_DRAW); // Allocating memory on OpenGL device
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), nullptr);
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Indices
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_[1]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, number_of_indices_ * sizeof(GLuint), indices_, GL_STATIC_DRAW); // Allocating memory on OpenGL device
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
}

#endif
