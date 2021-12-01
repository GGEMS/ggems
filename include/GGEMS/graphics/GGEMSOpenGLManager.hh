#ifndef GUARD_GGEMS_GRAPHICS_GGEMSOPENGLMANAGER_HH
#define GUARD_GGEMS_GRAPHICS_GGEMSOPENGLMANAGER_HH

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

#ifdef OPENGL_VISUALIZATION

#include <unordered_map>

#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"
#include "GGEMS/maths/GGEMSMatrixTypes.hh"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>

class GGEMSOpenGLVolume;
class GGEMSOpenGLAxis;

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
      \fn void SetDrawAxis(bool const& is_draw_axis)
      \param is_draw_axis - flag for axis drawing activation
      \brief set flag for axis drawing
    */
    void SetDrawAxis(bool const& is_draw_axis);

    /*!
      \fn void PrintKeys(void) const
      \brief print key help to screen
    */
    void PrintKeys(void) const;

    /*!
      \fn void Display(void)
      \brief display window
    */
    void Display(void);

    /*!
      \fn void Store(GGEMSOpenGLVolume* opengl_volume)
      \param opengl_volume - pointer to opengl volume
      \brief store opengl volume in singleton
    */
    void Store(GGEMSOpenGLVolume* opengl_volume);

    /*!
      \fn void SetWorldSize(float const& x_size, float const& y_size, float const& z_size)
      \param x_size - size in X of world scene
      \param y_size - size in Y of world scene
      \param z_size - size in Z of world scene
      \brief Set size of world scene for opengl
    */
    void SetWorldSize(float const& x_size, float const& y_size, float const& z_size);

    /*!
      \fn ColorUMap GetColorUMap(void) const
      \brief Getting color map
      \return the list of colors
    */
    inline ColorUMap GetColorUMap(void) const {return colors_;}

    /*!
      \fn int GetWindowWidth(void) const
      \brief get the window width of GLFW window
      \return the window width of GLFW window
    */
    inline int GetWindowWidth(void) const {return window_width_;}

    /*!
      \fn int GetWindowHeight(void) const
      \brief get the window height of GLFW window
      \return the window height of GLFW window
    */
    inline int GetWindowHeight(void) const {return window_height_;}

    /*!
      \fn glm::mat4 GetProjection(void) const
      \brief getting the projection matrix
      \return the projection matrix: perspective or ortho
    */
    inline glm::mat4 GetProjection(void) const {return projection_;}

    /*!
      \fn glm::mat4 GetCameraView(void) const
      \brief getting the camera view
      \return matrix of the camera view
    */
    inline glm::mat4 GetCameraView(void) const {return camera_view_;}

    /*!
      \fn void SetImageOutput(std::string const& image_output_basename)
      \param image_output_basename - name of basename for output
      \brief store visu scene in a png file
    */
    void SetImageOutput(std::string const& image_output_basename);

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
      \fn void UpdateProjectionAndView(void)
      \brief Check and update parameters for projection matrix and camera view
    */
    void UpdateProjectionAndView(void);

    /*
      \fn void SaveWindow(GLFWwindow* w)
      \param w - pointer to GLFW window
      \brief save window scene to a png file
    */
    void SaveWindow(GLFWwindow* w);

    /*!
      \fn void Draw(void) const
      \brief Draw all volumes
    */
    void Draw(void) const;

    /*!
      \fn float GetCurrentZoom(void) const
      \brief Computing the current zoom of OpenGL window
      \return Current zoom in a float type
    */
    inline float GetCurrentZoom(void) const
    {
      if (zoom_ >= 0.0f) {
        return 1.0f + zoom_ / 5.0f;
      }
      else {
        float current_zoom = 0.05f * zoom_ + 1.0f;
        if (current_zoom < 0.0f) {
          current_zoom = 0.01f;
        }
        return current_zoom;
      }
    }

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

    /*!
      \fn void GLFWScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
      \param window - pointer to GLFW window
      \param xoffset - offset in X
      \param yoffset - offset in Y
      \brief callback for mouse scroll
    */
    static void GLFWScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    /*!
      \fn void GLFWMouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
      \param window - pointer to GLFW window
      \param button - mouse key
      \param action - GLFW_PRESS, GLFW_RELEASE or GLFW_REPEAT
      \param mods - bit field describing which modifier keys were held down
      \brief callback for mouse button
    */
    static void GLFWMouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

    /*!
      \fn void GLFWCursorPosCallback(GLFWwindow* window, double x, double y);
      \param window - pointer to GLFW window
      \param xoffset - offset in X
      \param yoffset - offset in Y
      \brief cursor position of mouse
    */
    static void GLFWCursorPosCallback(GLFWwindow* window, double x, double y);

  private:
    GLFWwindow* window_; /*!< Pointer storing GLFW window */
    static int window_width_; /*!< GLFW window width */
    static int window_height_; /*!< GLFW window height */
    static double x_mouse_cursor_; /*!< Mouse cursor in X */
    static double y_mouse_cursor_; /*!< Mouse cursor in Y */
    static float pitch_angle_; /*!< Pitch angle for mouse */
    static float yaw_angle_; /*!< Yaw angle for mouse */
    static bool is_first_mouse_; /*!< First use of mouse */
    static double last_mouse_x_position_; /*!< Last position of mouse in x */
    static double last_mouse_y_position_; /*!< Last position of mouse in y */
    static bool is_left_button_; /*!< If left button is used */
    static bool is_middle_button_; /*!< If middle button is used */
    int msaa_; /*!< MSAA: Multi sample anti-aliasing factor */
    ColorUMap colors_; /*!< List of colors */
    float background_color_[3]; /*!< window background color */
    bool is_draw_axis_; /*!< Flag for axis drawing activation */
    float x_world_size_; /*!< World size along x axis */
    float y_world_size_; /*!< World size along y axis */
    float z_world_size_; /*!< World size along z axis */

    GGEMSOpenGLVolume** opengl_volumes_; /*!< OpenGL volume to display or not */
    GGsize number_of_opengl_volumes_; /*!< Number of OpenGL volumes */

    // OpenGL matrices
    glm::mat4 camera_view_; /*!< Camera view */
    static glm::vec3 camera_position_; /*!< Position of the camera */
    static glm::vec3 camera_target_; /*!< Target of the camera */
    static glm::vec3 camera_up_; /*!< Vector to the top */
    glm::mat4 projection_; /*!< Projection matrix (ortho or perspective), perspective by defaut */
    static bool is_wireframe_; /*!< Line mode: wireframe or solid (if bool is false) */
    static float zoom_; /*!< Value of zoom */
    static bool is_save_image_; /*!< Save window to png image file */
    std::string image_output_basename_; /*!< Image output basename */
    int image_output_index_; /*!< Image output index */
    GGEMSOpenGLAxis* axis_; /*!< pointer to axis volume */

    // OpenGL timing
    static double delta_time_; /*! Time between 2 frames */
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
    {0.0f, 0.502f, 0.0f}, {0.502f, 0.0f, 0.502f}, {0.0f, 0.502f, 0.502f},
    // navy
    {0.0f, 0.0f, 0.502f}
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

/*!
  \fn void set_draw_axis_opengl_manager(GGEMSOpenGLManager* opengl_manager, bool const is_draw_axis)
  \param opengl_manager - pointer on the singleton
  \param is_draw_axis - flag on axis drawing
  \brief activate axis drawing
*/
extern "C" GGEMS_EXPORT void set_draw_axis_opengl_manager(GGEMSOpenGLManager* opengl_manager, bool const is_draw_axis);

#endif // End of OPENGL_VISUALIZATION

#endif // GUARD_GGEMS_GLOBAL_GGEMSOPENGLMANAGER_HH
