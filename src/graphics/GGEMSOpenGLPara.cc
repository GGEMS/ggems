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
  \file GGEMSOpenGLPara.cc

  \brief Parallelepiped volume for OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday November 23, 2021
*/

#ifdef OPENGL_VISUALIZATION

#include "GGEMS/graphics/GGEMSOpenGLPara.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/global/GGEMSConstants.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLPara::GGEMSOpenGLPara(GLint const& elements_x, GLint const& elements_y, GLint const& elements_z, GLfloat const& element_size_x, GLfloat const& element_size_y, GLfloat const& element_size_z)
: GGEMSOpenGLVolume()
{
  GGcout("GGEMSOpenGLPara", "GGEMSOpenGLPara", 3) << "GGEMSOpenGLPara creating..." << GGendl;

  GGcout("GGEMSOpenGLPara", "GGEMSOpenGLPara", 3) << "GGEMSOpenGLPara created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLPara::~GGEMSOpenGLPara(void)
{
  GGcout("GGEMSOpenGLPara", "~GGEMSOpenGLPara", 3) << "GGEMSOpenGLPara erasing..." << GGendl;

  GGcout("GGEMSOpenGLPara", "~GGEMSOpenGLPara", 3) << "GGEMSOpenGLPara erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLPara::Build(void)
{
  GGcout("GGEMSOpenGLPara", "Build", 3) << "Building OpenGL para..." << GGendl;
}

#endif
