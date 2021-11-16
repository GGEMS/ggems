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
#include "GGEMS/graphics/GGEMSOpenGLSphere.hh"
#include "GGEMS/graphics/GGEMSOpenGLPrism.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"
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
  // All volume pointer are destroyed by GGEMSOpenGLManager

  ////////////////////////
  // X axis
  GGEMSOpenGLPrism* cylinder = new GGEMSOpenGLPrism(3.5f*mm, 3.5f*mm, 100.0f*mm, 48, 24);
  cylinder->SetVisible(true);
  cylinder->SetColor("green");
  cylinder->SetPosition(50.0f, 0.0f, 0.0f*mm);
  cylinder->SetYAngle(90.0f);
  cylinder->Build();

  GGEMSOpenGLPrism* arrow = new GGEMSOpenGLPrism(5.0f*mm, 0.0001f*mm, 7.5f*mm, 48, 24);
  arrow->SetVisible(true);
  arrow->SetColor("green");
  arrow->SetPosition(102.5f*mm, 0.0f, 0.0f*mm);
  arrow->SetYAngle(90.0f);
  arrow->Build();

  ////////////////////////
  // Y axis
  cylinder = new GGEMSOpenGLPrism(3.5f*mm, 3.5f*mm, 100.0f*mm, 48, 24);
  cylinder->SetVisible(true);
  cylinder->SetColor("red");
  cylinder->SetPosition(0.0f, -50.0f, 0.0f*mm);
  cylinder->SetXAngle(90.0f);
  cylinder->Build();

  arrow = new GGEMSOpenGLPrism(5.0f*mm, 0.0001f*mm, 7.5f*mm, 48, 24);
  arrow->SetVisible(true);
  arrow->SetColor("red");
  arrow->SetPosition(0.0f*mm, -102.5f*mm, 0.0f*mm);
  arrow->SetXAngle(90.0f);
  arrow->Build();

  ////////////////////////
  // Z axis
  cylinder = new GGEMSOpenGLPrism(3.5f*mm, 3.5f*mm, 100.0f*mm, 48, 24);
  cylinder->SetVisible(true);
  cylinder->SetColor("blue");
  cylinder->SetPosition(0.0f, 0.0f, -50.0f*mm);
  cylinder->Build();

  arrow = new GGEMSOpenGLPrism(5.0f*mm, 0.0001f*mm, 7.5f*mm, 48, 24);
  arrow->SetVisible(true);
  arrow->SetColor("blue");
  arrow->SetPosition(0.0f*mm, 0.0f*mm, -102.5f*mm);
  arrow->SetYAngle(180.0f);
  arrow->Build();

  GGEMSOpenGLSphere* sphere = new GGEMSOpenGLSphere(5.0f*mm);
  sphere->SetVisible(true);
  sphere->SetColor("yellow");
  sphere->Build();

  GGcout("GGEMSOpenGLAxis", "GGEMSOpenGLAxis", 3) << "GGEMSOpenGLAxis created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLAxis::~GGEMSOpenGLAxis(void)
{
  GGcout("GGEMSOpenGLAxis", "~GGEMSOpenGLAxis", 3) << "GGEMSOpenGLAxis erasing..." << GGendl;

  GGcout("GGEMSOpenGLAxis", "~GGEMSOpenGLAxis", 3) << "GGEMSOpenGLAxis erased!!!" << GGendl;
}

#endif
