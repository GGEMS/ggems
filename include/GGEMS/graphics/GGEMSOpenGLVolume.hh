#ifndef GUARD_GGEMS_GRAPHICS_GGEMSOPENGLVOLUME_HH
#define GUARD_GGEMS_GRAPHICS_GGEMSOPENGLVOLUME_HH

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
  \file GGEMSOpenGLVolume.hh

  \brief GGEMS mother class defining volume for OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday November 2, 2021
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \class GGEMSOpenGLVolume
  \brief GGEMS mother class defining volume for OpenGL
*/
class GGEMS_EXPORT GGEMSOpenGLVolume
{
  public:
    /*!
      \brief GGEMSOpenGLVolume constructor
    */
    GGEMSOpenGLVolume(void);

    /*!
      \brief GGEMSOpenGLVolume destructor
    */
    virtual ~GGEMSOpenGLVolume(void);

    /*!
      \fn GGEMSOpenGLVolume(GGEMSOpenGLVolume const& volume) = delete
      \param volume - reference on the GGEMS OpenGL volume
      \brief Avoid copy by reference
    */
    GGEMSOpenGLVolume(GGEMSOpenGLVolume const& volume) = delete;

    /*!
      \fn GGEMSOpenGLVolume& operator=(GGEMSOpenGLVolume const& volume) = delete
      \param volume - reference on the GGEMS OpenGL volume
      \brief Avoid assignement by reference
    */
    GGEMSOpenGLVolume& operator=(GGEMSOpenGLVolume const& volume) = delete;

    /*!
      \fn GGEMSOpenGLVolume(GGEMSOpenGLVolume const&& volume) = delete
      \param volume - rvalue reference on the GGEMS OpenGL volume
      \brief Avoid copy by rvalue reference
    */
    GGEMSOpenGLVolume(GGEMSOpenGLVolume const&& volume) = delete;

    /*!
      \fn GGEMSOpenGLVolume& operator=(GGEMSOpenGLVolume const&& volume) = delete
      \param volume - rvalue reference on the GGEMS OpenGL volume
      \brief Avoid copy by rvalue reference
    */
    GGEMSOpenGLVolume& operator=(GGEMSOpenGLVolume const&& volume) = delete;

    /*!
      \fn void SetPosition(GGfloat const& position_x, GGfloat const& position_y, GGfloat const& position_z)
      \param position_x - position in x
      \param position_y - position in y
      \param position_z - position in z
      \brief set position of sphere
    */
    void SetPosition(GGfloat const& position_x, GGfloat const& position_y, GGfloat const& position_z);

    /*!
      \fn void SetColor(std::string const& color)
      \param color - Color of OpenGL sphere
      \brief setting color of OpenGL sphere
    */
    void SetColor(std::string const& color);

    /*!
      \fn void Build(void) = 0
      \brief method building OpenGL volume and storing VAO and VBO
    */
    virtual void Build(void) = 0;

  protected:
    GGfloat position_x_; /*!< Position in X of OpenGL volume */
    GGfloat position_y_; /*!< Position in Y of OpenGL volume */
    GGfloat position_z_; /*!< Position in Z of OpenGL volume */
    std::string color_; /*!< Color of volume */

    GGfloat* vertices_; /*!< Pointer storing position vertex for OpenGL volume */
    GGsize number_of_vertices_; /*!< Number of vertices for OpenGL volume */
};

#endif // End of GUARD_GGEMS_GRAPHICS_GGEMSOPENGLVOLUME_HH
