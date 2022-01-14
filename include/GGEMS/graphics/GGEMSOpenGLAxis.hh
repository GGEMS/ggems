#ifndef GUARD_GGEMS_GRAPHICS_GGEMSOPENGLAXIS_HH
#define GUARD_GGEMS_GRAPHICS_GGEMSOPENGLAXIS_HH

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
  \file GGEMSOpenGLAxis.hh

  \brief Axis volume for OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday November 10, 2021
*/

#ifdef OPENGL_VISUALIZATION

#include "GGEMS/global/GGEMSExport.hh"

/*!
  \class GGEMSOpenGLAxis
  \brief This class define an axis volume for OpenGL
*/
class GGEMS_EXPORT GGEMSOpenGLAxis
{
  public:
    /*!
      \brief GGEMSOpenGLAxis constructor
    */
    GGEMSOpenGLAxis(void);

    /*!
      \brief GGEMSOpenGLAxis destructor
    */
    ~GGEMSOpenGLAxis(void);

    /*!
      \fn GGEMSOpenGLAxis(GGEMSOpenGLAxis const& axis_volume) = delete
      \param axis_volume - reference on the GGEMS OpenGL axis_volume
      \brief Avoid copy by reference
    */
    GGEMSOpenGLAxis(GGEMSOpenGLAxis const& axis_volume) = delete;

    /*!
      \fn GGEMSOpenGLAxis& operator=(GGEMSOpenGLAxis const& axis_volume) = delete
      \param volume - reference on the GGEMS OpenGL volume
      \brief Avoid assignement by reference
    */
    GGEMSOpenGLAxis& operator=(GGEMSOpenGLAxis const& axis_volume) = delete;

    /*!
      \fn GGEMSOpenGLAxis(GGEMSOpenGLAxis const&& axis_volume) = delete
      \param axis_volume - rvalue reference on the GGEMS OpenGL axis_volume
      \brief Avoid copy by rvalue reference
    */
    GGEMSOpenGLAxis(GGEMSOpenGLAxis const&& axis_volume) = delete;

    /*!
      \fn GGEMSOpenGLAxis& operator=(GGEMSOpenGLAxis const&& axis_volume) = delete
      \param volume - rvalue reference on the GGEMS OpenGL axis_volume
      \brief Avoid copy by rvalue reference
    */
    GGEMSOpenGLAxis& operator=(GGEMSOpenGLAxis const&& axis_volume) = delete;
};

#endif

#endif // End of GUARD_GGEMS_GRAPHICS_GGEMSOPENGLAXIS
