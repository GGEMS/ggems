#ifndef GUARD_GGEMS_GRAPHICS_GGEMSOPENGLMESH_HH
#define GUARD_GGEMS_GRAPHICS_GGEMSOPENGLMESH_HH

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
  \file GGEMSOpenGLMesh.hh

  \brief Mesh volume for OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday September 5, 2024
*/

#ifdef OPENGL_VISUALIZATION

#include "GGEMS/graphics/GGEMSOpenGLVolume.hh"
#include "GGEMS/geometries/GGEMSPrimitiveGeometries.hh"

/*!
  \class GGEMSOpenGLMesh
  \brief This class define a mesh volume for OpenGL
*/
class GGEMS_EXPORT GGEMSOpenGLMesh : public GGEMSOpenGLVolume
{
  public:
    /*!
      \param triangles - list of mesh triangles
      \param number_of_triangles - number of mesh triangles
      \brief GGEMSOpenGLSphere constructor
    */
    GGEMSOpenGLMesh(GGEMSTriangle3* triangles, GGuint const& number_of_triangles);

    /*!
      \brief GGEMSOpenGLMesh destructor
    */
    ~GGEMSOpenGLMesh(void) override;

    /*!
      \fn GGEMSOpenGLMesh(GGEMSOpenGLMesh const& mesh) = delete
      \param mesh - reference on the OpenGL mesh volume
      \brief Avoid copy by reference
    */
    GGEMSOpenGLMesh(GGEMSOpenGLMesh const& mesh) = delete;

    /*!
      \fn GGEMSOpenGLMesh& operator=(GGEMSOpenGLMesh const& mesh) = delete
      \param mesh - reference on the OpenGL mesh volume
      \brief Avoid assignement by reference
    */
    GGEMSOpenGLMesh& operator=(GGEMSOpenGLMesh const& mesh) = delete;

    /*!
      \fn GGEMSOpenGLMesh(GGEMSOpenGLMesh const&& mesh) = delete
      \param mesh - rvalue reference on OpenGL mesh volume
      \brief Avoid copy by rvalue reference
    */
    GGEMSOpenGLMesh(GGEMSOpenGLMesh const&& mesh) = delete;

    /*!
      \fn GGEMSOpenGLMesh& operator=(GGEMSOpenGLMesh const&& mesh) = delete
      \param mesh - rvalue reference on OpenGL mesh volume
      \brief Avoid copy by rvalue reference
    */
    GGEMSOpenGLMesh& operator=(GGEMSOpenGLMesh const&& mesh) = delete;

    /*!
      \fn void Build(void)
      \brief method building OpenGL volume and storing VAO and VBO
    */
    void Build(void) override;

  private:
    /*!
      \fn void WriteShaders(void)
      \brief write shader source file for each volume
    */
    void WriteShaders(void) override;

  private:
    GGEMSTriangle3* triangles_;
};

#endif

#endif // End of GUARD_GGEMS_GRAPHICS_GGEMSOPENGLMESH_HH
