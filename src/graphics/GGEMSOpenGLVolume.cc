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

#include "GGEMS/graphics/GGEMSOpenGLVolume.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLVolume::GGEMSOpenGLVolume()
: position_x_(0.0f),
  position_y_(0.0f),
  position_z_(0.0f),
  color_("red")
{
  GGcout("GGEMSOpenGLVolume", "GGEMSOpenGLVolume", 3) << "GGEMSOpenGLVolume creating..." << GGendl;

  GGcout("GGEMSOpenGLVolume", "GGEMSOpenGLVolume", 3) << "GGEMSOpenGLVolume created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLVolume::~GGEMSOpenGLVolume(void)
{
  GGcout("GGEMSOpenGLVolume", "~GGEMSOpenGLVolume", 3) << "GGEMSOpenGLVolume erasing..." << GGendl;

  // Destroying vertex
  if (vertices_) {
    delete[] vertices_;
    vertices_ = nullptr;
  }

  GGcout("GGEMSOpenGLVolume", "~GGEMSOpenGLVolume", 3) << "GGEMSOpenGLVolume erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLVolume::SetPosition(GGfloat const& position_x, GGfloat const& position_y, GGfloat const& position_z)
{
  position_x_ = position_x;
  position_y_ = position_y;
  position_z_ = position_z;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLVolume::SetColor(std::string const& color)
{
  color_ = color;
}
