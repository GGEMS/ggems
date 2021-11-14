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
  \file GGEMSOpenGLAxis.cc

  \brief Axis volume for OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday November 10, 2021
*/

#ifdef OPENGL_VISUALIZATION

#include "GGEMS/graphics/GGEMSOpenGLAxis.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLAxis::GGEMSOpenGLAxis(void)
{
  GGcout("GGEMSOpenGLAxis", "GGEMSOpenGLAxis", 3) << "GGEMSOpenGLAxis creating..." << GGendl;

  // The axis system is composed by:
  //     * a sphere
  //     * a tube + cone for X axis
  //     * a tube + cone for Y axis
  //     * a tube + cone for Z axis

  // 0.6 mm Sphere in (0, 0, 0)
  // sphere_test = new GGEMSOpenGLSphere(0.2f*mm);
  // sphere_test->SetVisible(true);
  // sphere_test->SetColor("yellow");
  // sphere_test->Build();

  // prism_test = new GGEMSOpenGLPrism(1.0f*mm, 0.0001f*mm, 3.0f*mm, 36, 12);
  // prism_test->SetVisible(true);
  // prism_test->SetColor("purple");
  // prism_test->SetPosition(0.0f, 0.0f, 0.0f);
  // prism_test->SetXAngle(15.0f);
  // prism_test->SetYAngle(45.0f);
  // prism_test->SetZAngle(10.0f);
  // prism_test->Build();

  GGcout("GGEMSOpenGLAxis", "GGEMSOpenGLAxis", 3) << "GGEMSOpenGLAxis created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLAxis::~GGEMSOpenGLAxis(void)
{
  GGcout("GGEMSOpenGLAxis", "~GGEMSOpenGLAxis", 3) << "GGEMSOpenGLPrism erasing..." << GGendl;

  GGcout("GGEMSOpenGLAxis", "~GGEMSOpenGLAxis", 3) << "GGEMSOpenGLPrism erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLAxis::Draw(void) const
{
//   // opengl_volumes_[0]->Draw();
//   // opengl_volumes_[1]->Draw();
}

#endif
