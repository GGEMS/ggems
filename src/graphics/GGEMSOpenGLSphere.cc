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

  \brief Sphere volume for for OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday November 2, 2021
*/

#include "GGEMS/graphics/GGEMSOpenGLSphere.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/global/GGEMSConstants.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLSphere::GGEMSOpenGLSphere(GGfloat const& radius)
: GGEMSOpenGLVolume(),
  radius_(radius)
{
  GGcout("GGEMSOpenGLSphere", "GGEMSOpenGLSphere", 3) << "GGEMSOpenGLSphere creating..." << GGendl;

  number_of_sectors_ = 24; // longitude
  number_of_stacks_ = 10; // latitude

  // Allocating memory for sphere vertices
  // Final point is taken (+1 stack and sector)
  vertices_ = new GGfloat[(number_of_stacks_+1)*(number_of_sectors_+1)*3];
  number_of_vertices_ = (number_of_stacks_+1)*(number_of_sectors_+1)*3;

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

void GGEMSOpenGLSphere::Build(void)
{
  GGcout("GGEMSOpenGLSphere", "~GGEMSOpenGLSphere", 3) << "Building OpenGL sphere..." << GGendl;

  // Compute x, y, z with
  // xy = radius * cos(stack_angle)
  // x = xy * cos(sector_angle)
  // y = xy * sin(sector_angle)
  // z = radius * sin(stack_angle)

  GGfloat sector_step = 2.0f * PI / static_cast<GGfloat>(number_of_sectors_);
  GGfloat sector_angle = 0.0f;
  GGfloat stack_step = PI / static_cast<GGfloat>(number_of_stacks_);
  GGfloat stack_angle = 0.0f;

  GGfloat x = 0.0f, y = 0.0f, xy = 0.0f, z = 0.0f;
  GGint index = 0;
  // Loop over the stacks
  for (GGsize i = 0; i <= number_of_stacks_; ++i) {
    // Stack angle
    stack_angle = HALF_PI - i * stack_step; // from pi/2 to -pi/2

    xy = radius_ * std::cos(stack_angle);
    z = radius_ * std::sin(stack_angle);

    // Loop over the sectors
    for (GGsize j = 0; j <= number_of_sectors_; ++j) {
      sector_angle = j * sector_step; // from 0 to 2pi

      x = xy * std::cos(sector_angle);
      y = xy * std::sin(sector_angle);

      vertices_[index++] = x;
      vertices_[index++] = y;
      vertices_[index++] = z;
    }
  }

/*
   // indices
    //  k1--k1+1
    //  |  / |
    //  | /  |
    //  k2--k2+1
    unsigned int k1, k2;
    for(int i = 0; i < stackCount; ++i)
    {
        k1 = i * (sectorCount + 1);     // beginning of current stack
        k2 = k1 + sectorCount + 1;      // beginning of next stack

        for(int j = 0; j < sectorCount; ++j, ++k1, ++k2)
        {
            // 2 triangles per sector excluding 1st and last stacks
            if(i != 0)
            {
                addIndices(k1, k2, k1+1);   // k1---k2---k1+1
            }

            if(i != (stackCount-1))
            {
                addIndices(k1+1, k2, k2+1); // k1+1---k2---k2+1
            }

            // vertical lines for all stacks
            lineIndices.push_back(k1);
            lineIndices.push_back(k2);
            if(i != 0)  // horizontal lines except 1st stack
            {
                lineIndices.push_back(k1);
                lineIndices.push_back(k1 + 1);
            }
        }
    }

    // generate interleaved vertex array as well
    buildInterleavedVertices();
*/
}
