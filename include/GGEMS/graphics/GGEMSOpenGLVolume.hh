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

#ifdef OPENGL_VISUALIZATION

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/materials/GGEMSMaterialsDatabaseManager.hh"

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
      \fn void SetPosition(GLfloat const& position_x, GLfloat const& position_y, GLfloat const& position_z)
      \param position_x - position in x
      \param position_y - position in y
      \param position_z - position in z
      \brief set position of sphere
    */
    void SetPosition(GLfloat const& position_x, GLfloat const& position_y, GLfloat const& position_z);

    /*!
      \fn void SetXAngle(GLfloat const& angle_x)
      \param angle_x - angle in X
      \brief set angle of rotation in X
    */
    void SetXAngle(GLfloat const& angle_x);

    /*!
      \fn void SetYAngle(GLfloat const& angle_y)
      \param angle_y - angle in Y
      \brief set angle of rotation in Y
    */
    void SetYAngle(GLfloat const& angle_y);

    /*!
      \fn void SetZAngle(GLfloat const& angle_z)
      \param angle_z - angle in Z
      \brief set angle of rotation in Z
    */
    void SetZAngle(GLfloat const& angle_z);

    /*!
      \fn void SetXUpdateAngle(GLfloat const& update_angle_x)
      \param update_angle_x - angle in X
      \brief set angle of rotation in X (after translation)
    */
    void SetXUpdateAngle(GLfloat const& update_angle_x);

    /*!
      \fn void SetYUpdateAngle(GLfloat const& update_angle_y)
      \param update_angle_y - angle in Y
      \brief set angle of rotation in Y (after translation)
    */
    void SetYUpdateAngle(GLfloat const& update_angle_y);

    /*!
      \fn void SetZUpdateAngle(GLfloat const& update_angle_z)
      \param update_angle_z - angle in Z
      \brief set angle of rotation in Z (after translation)
    */
    void SetZUpdateAngle(GLfloat const& update_angle_z);

    /*!
      \fn void SetColorName(std::string const& color)
      \param color - Color of OpenGL sphere
      \brief setting color of OpenGL sphere
    */
    void SetColorName(std::string const& color);

    /*!
      \fn void SetColorRGB(GGEMSRGBColor const& rgb)
      \param rgb - rgb parameters
      \brief set rgb colors
    */
    void SetColorRGB(GGEMSRGBColor const& rgb);

    /*!
      \fn void SetVisible(bool const& is_visible)
      \param is_visible - flag to diplay or not volume
    */
    void SetVisible(bool const& is_visible);

    /*!
      \fn void Build(void) = 0
      \brief method building OpenGL volume and storing VAO and VBO
    */
    virtual void Build(void) = 0;

    /*!
      \fn void Draw(void) const
      \brief Draw volume into the screen
    */
    void Draw(void) const;

  protected:
    /*!
      \fn void InitShaders(void)
      \brief compile shader and store it
    */
    void InitShaders(void);

    /*!
      \fn void WriteShaders(void)
      \brief write shader source file for each volume
    */
    virtual void WriteShaders(void) = 0;

    /*!
      \fn std::string GetOpenGLSLVersion(void) const
      \brief Get version of GLSL
      \return GLSL version in string
    */
    std::string GetOpenGLSLVersion(void) const;

  private:
    /*!
      \fn void CompileShader(GLuint const& shader) const
      \param shader - index of shader from glCreateShader
      \brief compiling shader
    */
    void CompileShader(GLuint const& shader) const;

  protected:
    GLint number_of_stacks_; /*!< Number of stacks (latitude) */
    GLint number_of_sectors_; /*!< Number of sectors (longitude) */

    GLfloat position_x_; /*!< Position in X of OpenGL volume */
    GLfloat position_y_; /*!< Position in Y of OpenGL volume */
    GLfloat position_z_; /*!< Position in Z of OpenGL volume */

    GLfloat angle_x_; /*!< Angle of rotation around center of volume in X in radian */
    GLfloat angle_y_; /*!< Angle of rotation around center of volume in Y in radian */
    GLfloat angle_z_; /*!< Angle of rotation around center of volume in Z in radian */

    GLfloat update_angle_x_; /*!< Angle after translation, volume rotate around isocenter */
    GLfloat update_angle_y_; /*!< Angle after translation, volume rotate around isocenter */
    GLfloat update_angle_z_; /*!< Angle after translation, volume rotate around isocenter */

    GLfloat color_[3]; /*!< Color of volume */

    GLuint vao_; /*!< vertex array object, 1 for each object */
    GLuint vbo_[2]; /*!< vertex buffer object, index 0 -> vertex, index 1 -> indice */
    GLuint program_shader_id_; /*!< program id for shader, specific to a volume */
    std::string vertex_shader_source_; /*!< vertex shader source file */
    std::string fragment_shader_source_; /*!< fragment shader source file */

    GLfloat* vertices_; /*!< Pointer storing position vertex for OpenGL volume */
    GLint number_of_vertices_; /*!< Number of vertices for OpenGL volume */

    GLuint* indices_; /*!< Indices of vertex (triangulated) */
    GLint number_of_indices_; /*!< Number of indices */

    GLint number_of_triangles_; /*!< Number of triangles for a volume */

    bool is_visible_; /*!< true: volume display */
};

#endif

#endif // End of GUARD_GGEMS_GRAPHICS_GGEMSOPENGLVOLUME_HH
