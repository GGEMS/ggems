#ifndef GUARD_GGEMS_GLOBAL_GGEMSOPENGLMANAGER_HH
#define GUARD_GGEMS_GLOBAL_GGEMSOPENGLMANAGER_HH

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
  \file GGEMSOpenGLManager.hh

  \brief Singleton class storing all informations about OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday October 25, 2021
*/

#include <unordered_map>

#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

typedef std::unordered_map<std::string, int> ColorUMap; /*!< Unordered map with key : name of the color, index of color */

/*!
  \class GGEMSOpenGLManager
  \brief Singleton class storing all informations about OpenGL
*/
class GGEMS_EXPORT GGEMSOpenGLManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGEMSOpenGLManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGEMSOpenGLManager(void);

  public:
    /*!
      \fn static GGEMSOpenGLManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSOpenGLManager
    */
    static GGEMSOpenGLManager& GetInstance(void)
    {
      static GGEMSOpenGLManager instance;
      return instance;
    }

    /*!
      \fn GGEMSOpenGLManager(GGEMSOpenGLManager const& opengl_manager) = delete
      \param opengl_manager - reference on the singleton
      \brief Avoid copy of the singleton by reference
    */
    GGEMSOpenGLManager(GGEMSOpenGLManager const& opengl_manager) = delete;

    /*!
      \fn GGEMSOpenGLManager& operator=(GGEMSOpenGLManager const& opengl_manager) = delete
      \param opengl_manager - reference on the singleton
      \brief Avoid assignement of the singleton by reference
    */
    GGEMSOpenGLManager& operator=(GGEMSOpenGLManager const& opengl_manager) = delete;

    /*!
      \fn GGEMSOpenGLManager(GGEMSOpenGLManager const&& opengl_manager) = delete
      \param opengl_manager - rvalue reference on the singleton
      \brief Avoid copy of the singleton by rvalue reference
    */
    GGEMSOpenGLManager(GGEMSOpenGLManager const&& opengl_manager) = delete;

    /*!
      \fn GGEMSOpenGLManager& operator=(GGEMSOpenGLManager const&& opengl_manager) = delete
      \param opengl_manager - rvalue reference on the singleton
      \brief Avoid copy of the singleton by rvalue reference
    */
    GGEMSOpenGLManager& operator=(GGEMSOpenGLManager const&& opengl_manager) = delete;

    /*!
      \fn void Initialize(void)
      \brief Initialize OpenGL for visualization
    */
    void Initialize(void);

    /*!
      \fn void SetMSAA(int const& msaa_factor)
      \param msaa_factor - MSAA factor
      \brief set msaa factor for OpenGL
    */
    void SetMSAA(int const& msaa_factor);

    /*!
      \fn void SetWindowDimensions(int const& width, int const& height)
      \param width - window width
      \param height - window height
      \brief set the window dimension for OpenGL
    */
    void SetWindowDimensions(int const& width, int const& height);

    /*!
      \fn void SetBackgroundColor(std::string const& color)
      \param color - name of color for background
      \brief set background color
    */
    void SetBackgroundColor(std::string const& color);

    /*!
      \fn void PrintKeys(void) const
      \brief print key help to screen
    */
    void PrintKeys(void) const;



    void Draw(void);

  private:
    /*!
      \fn void InitGL(void)
      \brief Initialization of OpenGL params, window and shaders
    */
    void InitGL(void);

    /*!
      \fn void UpdateFPSCounter(void)
      \brief Compute and display FPS in GLFW window
    */
    void UpdateFPSCounter(void);

    /*!
      \fn void GLFWErrorCallback(int error_code, char const* description)
      \param error_code - error code returned by OpenGL
      \param description - description of the error
      \brief callback for errors returned by OpenGL
    */
    static void GLFWErrorCallback(int error_code, char const* description);

    /*!
      \fn void GLFWWindowSizeCallback(GLFWwindow* window, int width, int height)
      \param window - pointer to GLFW window
      \param width - width of window
      \param height - height of window
      \brief callback for window size
    */
    static void GLFWWindowSizeCallback(GLFWwindow* window, int width, int height);

    /*!
      \fn void GLFWKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
      \param window - pointer to GLFW window
      \param key - keyboard key that was pressed or released
      \param scancode - system-specific scancode of the key
      \param action - GLFW_PRESS, GLFW_RELEASE or GLFW_REPEAT
      \param mods - bit field describing which modifier keys were held down
      \brief callback for keyboard key
    */
    static void GLFWKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

  private:
    // Interactive members using callback
    GLFWwindow* window_; /*!< Pointer storing GLFW window */
    static int window_width_; /*!< GLFW window width */
    static int window_height_; /*!< GLFW window height */
    int msaa_; /*!< MSAA: Multi sample anti-aliasing factor */
    ColorUMap colors_; /*!< List of colors */
    float background_color_[3]; /*!< window background color */
};

/*!
  \namespace GGEMSOpenGLColor
  \brief Namespace storing RGB color for OpenGL
*/
namespace GGEMSOpenGLColor
{
  __constant float color[][3] = {
    // black, blue, lime              
    {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f},
    // cyan, red, magenta
    {0.0f, 1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f},
    // yellow, white, gray
    {1.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.502f, 0.502f, 0.502f},
    // silver, maroon, olive
    {0.753f, 0.753f, 0.753f}, {0.502f, 0.0f, 0.0f}, {0.502f, 0.502f, 0.0f},
    // green, purple, teal
    {0.0f, 0.502f, 0.0f}, {0.502f, 1.0f, 0.502f}, {0.0f, 0.502f, 0.502f},
    // navy
    {0.0f, 1.0f, 0.502f}
  };
}

/*!
  \fn GGEMSOpenGLManager* get_instance_ggems_opengl_manager(void)
  \return the pointer on the singleton
  \brief Get the GGEMSOpenGLManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSOpenGLManager* get_instance_ggems_opengl_manager(void);

/*!
  \fn void set_window_dimensions_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, int width, int height)
  \param opengl_manager - pointer on the singleton
  \param width - glfw window width
  \param height - glfw window height
  \brief Set GLFW window dimension
*/
extern "C" GGEMS_EXPORT void set_window_dimensions_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, int width, int height);

/*!
  \fn void set_msaa_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, int msaa_factor)
  \param opengl_manager - pointer on the singleton
  \param msaa_factor - msaa factor
  \brief Set MSAA (multi sample anti-aliasing factor) factor
*/
extern "C" GGEMS_EXPORT void set_msaa_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, int msaa_factor);

/*!
  \fn void set_background_color_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, char const* color)
  \param opengl_manager - pointer on the singleton
  \param color - background color
  \brief Set background color in GLFW window
*/
extern "C" GGEMS_EXPORT void set_background_color_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, char const* color = "");

#endif // GUARD_GGEMS_GLOBAL_GGEMSOPENGLMANAGER_HH
