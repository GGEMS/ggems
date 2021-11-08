#ifndef GUARD_GGEMS_GRAPHICS_GGEMSOPENGLCYLINDER_HH
#define GUARD_GGEMS_GRAPHICS_GGEMSOPENGLCYLINDER_HH

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
  \file GGEMSOpenGLCylinder.hh

  \brief Cylinder volume for for OpenGL, including cone, prism etc...

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday November 8, 2021
*/

#ifdef OPENGL_VISUALIZATION

#include "GGEMS/graphics/GGEMSOpenGLVolume.hh"

/*!
  \class GGEMSOpenGLCylinder
  \brief This class define a cylinder volume for OpenGL, including cone, prism etc...
*/
class GGEMS_EXPORT GGEMSOpenGLCylinder : public GGEMSOpenGLVolume
{
  public:
    /*!
      \brief GGEMSOpenGLCylinder constructor
    */
    explicit GGEMSOpenGLCylinder(void);

    /*!
      \brief GGEMSOpenGLCylinder destructor
    */
    ~GGEMSOpenGLCylinder(void);

    /*!
      \fn GGEMSOpenGLCylinder(GGEMSOpenGLCylinder const& cylinder) = delete
      \param cylinder - reference on the OpenGL cylinder volume
      \brief Avoid copy by reference
    */
    GGEMSOpenGLCylinder(GGEMSOpenGLCylinder const& cylinder) = delete;

    /*!
      \fn GGEMSOpenGLCylinder& operator=(GGEMSOpenGLCylinder const& cylinder) = delete
      \param cylinder - reference on the OpenGL cylinder volume
      \brief Avoid assignement by reference
    */
    GGEMSOpenGLCylinder& operator=(GGEMSOpenGLCylinder const& cylinder) = delete;

    /*!
      \fn GGEMSOpenGLCylinder(GGEMSOpenGLCylinder const&& cylinder) = delete
      \param cylinder - rvalue reference on OpenGL cylinder volume
      \brief Avoid copy by rvalue reference
    */
    GGEMSOpenGLCylinder(GGEMSOpenGLCylinder const&& cylinder) = delete;

    /*!
      \fn GGEMSOpenGLCylinder& operator=(GGEMSOpenGLCylinder const&& cylinder) = delete
      \param cylinder - rvalue reference on OpenGL cylinder volume
      \brief Avoid copy by rvalue reference
    */
    GGEMSOpenGLCylinder& operator=(GGEMSOpenGLCylinder const&& cylinder) = delete;

    /*!
      \fn void Build(void)
      \brief method building OpenGL volume and storing VAO and VBO
    */
    void Build(void) override;

  private:
    //GGfloat radius_; /*!< Radius of sphere */
    //GGint number_of_stacks_; /*!< Number of stacks (latitude) on sphere */
    //GGint number_of_sectors_; /*!< Number of sectors (longitude) on sphere */
};

#endif

#endif // End of GUARD_GGEMS_GRAPHICS_GGEMSOPENGLCYLINDER_HH
