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

#include "GGEMS/graphics/GGEMSOpenGLPrism.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/global/GGEMSConstants.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLPrism::GGEMSOpenGLPrism(GLfloat const& base_radius, GLfloat const& top_radius, GLfloat const& height, GLint const& sectors, GLint const& stacks)
: GGEMSOpenGLVolume()
{
  GGcout("GGEMSOpenGLPrism", "GGEMSOpenGLPrism", 3) << "GGEMSOpenGLPrism creating..." << GGendl;

  base_radius_ = base_radius;
  top_radius_ = top_radius;
  height_ = height;

  number_of_sectors_ = sectors;
  number_of_stacks_ = stacks;

  // Allocating memory for sphere vertices
  // extreme point postions are taken (+1 stack and sector)
  //number_of_vertices_ = (number_of_stacks_+1)*(number_of_sectors_+1)*3;
  //vertices_ = new GLfloat[number_of_vertices_];

  // Compute number of (triangulated) indices.
  // In each sector/stack there are 2 triangles
  // For first and last stack there is 1 triangle, 3 indices for each triangle
  //number_of_triangles_ = ((number_of_stacks_*2)-2)*number_of_sectors_;
  //number_of_indices_ = number_of_triangles_*3;
  //indices_ = new GLuint[number_of_indices_];

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

void GGEMSOpenGLPrism::BuildUnitCircleVertices(void)
{
  GLfloat sector_step = 2.0f * PI / static_cast<GGfloat>(number_of_sectors_);
  GLfloat sector_angle = 0.0f;

  unit_circle_vertices_ = new GLfloat[(number_of_sectors_+1)*3];

  for (GLint i = 0; i <= number_of_sectors_; ++i) {
    sector_angle = i * sector_step;
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

/*

    float x, y, z;                                  // vertex position
    float radius;                                   // radius for each stack

    // get normals for cylinder sides
    std::vector<float> sideNormals = getSideNormals();

    // put vertices of side cylinder to array by scaling unit circle
    for(int i = 0; i <= stackCount; ++i)
    {
        z = -(height * 0.5f) + (float)i / stackCount * height;      // vertex position z
        radius = baseRadius + (float)i / stackCount * (topRadius - baseRadius);     // lerp
        float t = 1.0f - (float)i / stackCount;   // top-to-bottom

        for(int j = 0, k = 0; j <= sectorCount; ++j, k += 3)
        {
            x = unitCircleVertices[k];
            y = unitCircleVertices[k+1];
            addVertex(x * radius, y * radius, z);   // position
        }
    }

    // remember where the base.top vertices start
    unsigned int baseVertexIndex = (unsigned int)vertices.size() / 3;

    // put vertices of base of cylinder
    z = -height * 0.5f;
    addVertex(0, 0, z);
    addNormal(0, 0, -1);
    addTexCoord(0.5f, 0.5f);
    for(int i = 0, j = 0; i < sectorCount; ++i, j += 3)
    {
        x = unitCircleVertices[j];
        y = unitCircleVertices[j+1];
        addVertex(x * baseRadius, y * baseRadius, z);
        addNormal(0, 0, -1);
        addTexCoord(-x * 0.5f + 0.5f, -y * 0.5f + 0.5f);    // flip horizontal
    }

    // remember where the base vertices start
    unsigned int topVertexIndex = (unsigned int)vertices.size() / 3;

    // put vertices of top of cylinder
    z = height * 0.5f;
    addVertex(0, 0, z);
    addNormal(0, 0, 1);
    addTexCoord(0.5f, 0.5f);
    for(int i = 0, j = 0; i < sectorCount; ++i, j += 3)
    {
        x = unitCircleVertices[j];
        y = unitCircleVertices[j+1];
        addVertex(x * topRadius, y * topRadius, z);
        addNormal(0, 0, 1);
        addTexCoord(x * 0.5f + 0.5f, -y * 0.5f + 0.5f);
    }

    // put indices for sides
    unsigned int k1, k2;
    for(int i = 0; i < stackCount; ++i)
    {
        k1 = i * (sectorCount + 1);     // bebinning of current stack
        k2 = k1 + sectorCount + 1;      // beginning of next stack

        for(int j = 0; j < sectorCount; ++j, ++k1, ++k2)
        {
            // 2 trianles per sector
            addIndices(k1, k1 + 1, k2);
            addIndices(k2, k1 + 1, k2 + 1);

            // vertical lines for all stacks
            lineIndices.push_back(k1);
            lineIndices.push_back(k2);
            // horizontal lines
            lineIndices.push_back(k2);
            lineIndices.push_back(k2 + 1);
            if(i == 0)
            {
                lineIndices.push_back(k1);
                lineIndices.push_back(k1 + 1);
            }
        }
    }

    // remember where the base indices start
    baseIndex = (unsigned int)indices.size();

    // put indices for base
    for(int i = 0, k = baseVertexIndex + 1; i < sectorCount; ++i, ++k)
    {
        if(i < (sectorCount - 1))
            addIndices(baseVertexIndex, k + 1, k);
        else    // last triangle
            addIndices(baseVertexIndex, baseVertexIndex + 1, k);
    }

    // remember where the base indices start
    topIndex = (unsigned int)indices.size();

    for(int i = 0, k = topVertexIndex + 1; i < sectorCount; ++i, ++k)
    {
        if(i < (sectorCount - 1))
            addIndices(topVertexIndex, k, k + 1);
        else
            addIndices(topVertexIndex, k, topVertexIndex + 1);
    }

    // generate interleaved vertex array as well
    buildInterleavedVertices();
    */

/*
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
*/
}

#endif
