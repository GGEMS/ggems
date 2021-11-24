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

#include "GGEMS/graphics/GGEMSOpenGLParaGrid.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/global/GGEMSConstants.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLParaGrid::GGEMSOpenGLParaGrid(GLint const& elements_x, GLint const& elements_y, GLint const& elements_z, GLfloat const& element_size_x, GLfloat const& element_size_y, GLfloat const& element_size_z)
: GGEMSOpenGLVolume(),
  elements_x_(elements_x),
  elements_y_(elements_y),
  elements_z_(elements_z),
  element_size_x_(element_size_x),
  element_size_y_(element_size_y),
  element_size_z_(element_size_z)
{
  GGcout("GGEMSOpenGLParaGrid", "GGEMSOpenGLParaGrid", 3) << "GGEMSOpenGLParaGrid creating..." << GGendl;

  // For each parallelepiped there are 8 vertices
  number_of_vertices_ = elements_x_ * elements_y_ * elements_z_ * 8 * 3;
  vertices_ = new GLfloat[number_of_vertices_];

  // For each parallelepiped there are 12 triangles
  number_of_triangles_ = elements_x_ * elements_y_ * elements_z_ * 12;
  number_of_indices_ = number_of_triangles_ * 3;
  indices_ = new GLuint[number_of_indices_];

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

void GGEMSOpenGLParaGrid::Build(void)
{
  GGcout("GGEMSOpenGLParaGrid", "Build", 3) << "Building OpenGL para..." << GGendl;

  // Storing vertices
  GLfloat x_offset = 0.0f, y_offset = 0.0f, z_offset = 0.0f;
  GLint index = 0;

  // Center of first parallelepiped
  GLfloat x_center = -(element_size_x_*elements_x_*0.5f) + element_size_x_*0.5f;
  GLfloat y_center = -(element_size_y_*elements_y_*0.5f) + element_size_y_*0.5f;
  GLfloat z_center = -(element_size_z_*elements_z_*0.5f) + element_size_z_*0.5f;

  // Loop over all parallelepipeds
  for (GLint k = 0; k < elements_z_; ++k) {
    z_offset = z_center + k*element_size_z_;

    for (GLint j = 0; j < elements_y_; ++j) {
      y_offset = y_center + j*element_size_y_;

      for (GLint i = 0; i < elements_x_; ++i) {
        x_offset = x_center + i*element_size_x_;

        // 0
        vertices_[index++] = x_offset - element_size_x_*0.5f;
        vertices_[index++] = y_offset - element_size_y_*0.5f;
        vertices_[index++] = z_offset - element_size_z_*0.5f;

        // 1
        vertices_[index++] = x_offset - element_size_x_*0.5f;
        vertices_[index++] = y_offset + element_size_y_*0.5f;
        vertices_[index++] = z_offset - element_size_z_*0.5f;

        // 2
        vertices_[index++] = x_offset - element_size_x_*0.5f;
        vertices_[index++] = y_offset + element_size_y_*0.5f;
        vertices_[index++] = z_offset + element_size_z_*0.5f;

        // 3
        vertices_[index++] = x_offset - element_size_x_*0.5f;
        vertices_[index++] = y_offset - element_size_y_*0.5f;
        vertices_[index++] = z_offset + element_size_z_*0.5f;

        // 4
        vertices_[index++] = x_offset + element_size_x_*0.5f;
        vertices_[index++] = y_offset - element_size_y_*0.5f;
        vertices_[index++] = z_offset - element_size_z_*0.5f;

        // 5
        vertices_[index++] = x_offset + element_size_x_*0.5f;
        vertices_[index++] = y_offset + element_size_y_*0.5f;
        vertices_[index++] = z_offset - element_size_z_*0.5f;

        // 6
        vertices_[index++] = x_offset + element_size_x_*0.5f;
        vertices_[index++] = y_offset + element_size_y_*0.5f;
        vertices_[index++] = z_offset + element_size_z_*0.5f;

        // 7
        vertices_[index++] = x_offset + element_size_x_*0.5f;
        vertices_[index++] = y_offset - element_size_y_*0.5f;
        vertices_[index++] = z_offset + element_size_z_*0.5f;
      }
    }
  }

  // Storing indices
  index = 0;
  for (GLint i = 0; i < elements_x_*elements_y_*elements_z_; ++i) {
    // Triangle 0
    indices_[index++] = 2+i*8;
    indices_[index++] = 3+i*8;
    indices_[index++] = 7+i*8;

    // 1
    indices_[index++] = 2+i*8;
    indices_[index++] = 7+i*8;
    indices_[index++] = 6+i*8;

    // 2
    indices_[index++] = 6+i*8;
    indices_[index++] = 7+i*8;
    indices_[index++] = 4+i*8;

    // 3
    indices_[index++] = 6+i*8;
    indices_[index++] = 4+i*8;
    indices_[index++] = 5+i*8;

    // 4
    indices_[index++] = 5+i*8;
    indices_[index++] = 4+i*8;
    indices_[index++] = 0+i*8;

    // 5
    indices_[index++] = 5+i*8;
    indices_[index++] = 0+i*8;
    indices_[index++] = 1+i*8;

    // 6
    indices_[index++] = 1+i*8;
    indices_[index++] = 0+i*8;
    indices_[index++] = 3+i*8;

    // 7
    indices_[index++] = 1+i*8;
    indices_[index++] = 3+i*8;
    indices_[index++] = 2+i*8;

    // 8
    indices_[index++] = 1+i*8;
    indices_[index++] = 2+i*8;
    indices_[index++] = 6+i*8;

    // 9
    indices_[index++] = 1+i*8;
    indices_[index++] = 6+i*8;
    indices_[index++] = 5+i*8;

    // 10
    indices_[index++] = 0+i*8;
    indices_[index++] = 3+i*8;
    indices_[index++] = 7+i*8;

    // 11
    indices_[index++] = 0+i*8;
    indices_[index++] = 7+i*8;
    indices_[index++] = 4+i*8;
  }

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
