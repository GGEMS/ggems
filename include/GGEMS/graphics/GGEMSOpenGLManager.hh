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
class GGEMSOpenGLParticles;

typedef std::unordered_map<std::string, GGint> ColorUMap; /*!< Unordered map with key : name of the color, index of color */

/*!
  \namespace GGEMSOpenGLColor
  \brief Namespace storing RGB color for OpenGL
*/
namespace GGEMSOpenGLColor
{
  __constant GGfloat color[][3] = {
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
      \fn void SetMSAA(GGint const& msaa_factor)
      \param msaa_factor - MSAA factor
      \brief set msaa factor for OpenGL
    */
    void SetMSAA(GGint const& msaa_factor);

    /*!
      \fn void SetWindowDimensions(GGint const& width, GGint const& height)
      \param width - window width
      \param height - window height
      \brief set the window dimension for OpenGL
    */
    void SetWindowDimensions(GGint const& width, GGint const& height);

    /*!
      \fn void SetBackgroundColor(std::string const& color)
      \param color - name of color for background
      \brief set background color
    */
    void SetBackgroundColor(std::string const& color);

    /*!
      \fn void SetDrawAxis(GGbool const& is_draw_axis)
      \param is_draw_axis - flag for axis drawing activation
      \brief set flag for axis drawing
    */
    void SetDrawAxis(GGbool const& is_draw_axis);

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
      \fn void SetWorldSize(GGfloat const& x_size, GGfloat const& y_size, GGfloat const& z_size, std::string const& unit = "mm")
      \param x_size - size in X of world scene
      \param y_size - size in Y of world scene
      \param z_size - size in Z of world scene
      \param unit - length unit
      \brief Set size of world scene for opengl
    */
    void SetWorldSize(GGfloat const& x_size, GGfloat const& y_size, GGfloat const& z_size, std::string const& unit = "mm");

    /*!
      \fn ColorUMap GetColorUMap(void) const
      \brief Getting color map
      \return the list of colors
    */
    inline ColorUMap GetColorUMap(void) const {return colors_;}

    /*!
      \fn void GetRGBColor(std::string const& color_name, GGfloat* rgb) const
      \param color_name - name of color
      \param rgb - array of RGB returned by method
      \brief RGB color a registered color
    */
    inline void GetRGBColor(std::string const& color_name, GGfloat* rgb) const
    {
      ColorUMap::const_iterator it = colors_.find(color_name);
      if (it != colors_.end()) {
        for (GGint i = 0; i < 3; ++i) {
          rgb[i] = GGEMSOpenGLColor::color[it->second][i];
        }
      }
      else {
        GGwarn("GGEMSOpenGLManager", "GetRGBColor", 0) << "Color: " << color_name << " not found!!! White is set by default." << GGendl;
        rgb[0] = 1.0f;
        rgb[1] = 1.0f;
        rgb[2] = 1.0f;
      }
    }

    /*!
      \fn GGint GetWindowWidth(void) const
      \brief get the window width of GLFW window
      \return the window width of GLFW window
    */
    inline GGint GetWindowWidth(void) const {return window_width_;}

    /*!
      \fn GGint GetWindowHeight(void) const
      \brief get the window height of GLFW window
      \return the window height of GLFW window
    */
    inline GGint GetWindowHeight(void) const {return window_height_;}

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

    /*!
      \fn static GGbool IsOpenGLActivated(void)
      \brief check if OpenGL is activated
      \return true if OpenGL is activated
    */
    static GGbool IsOpenGLActivated(void) {return is_opengl_activated_;};

    /*!
      \fn void SetDisplayedParticles(GGint const& number_of_displayed_particles)
      \param number_of_displayed_particles - Number of particles to display on OpenGL screen
      \brief Set the number of particles to display on screen
    */
    void SetDisplayedParticles(GGint const& number_of_displayed_particles);

    /*!
      \fn GGint GetNumberOfDisplayedParticles(void) const
      \brief Getting the number of displayed particles
      \return the number of displayed particles
    */
    inline GGint GetNumberOfDisplayedParticles(void) const {return number_of_displayed_particles_;}

    /*!
      \fn void CopyParticlePositionToOpenGL(GGsize const& source_index)
      \param source_index - source index
      \brief Copy particle position from OpenCL kernel to OpenGL memory
    */
    void CopyParticlePositionToOpenGL(GGsize const& source_index);

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
      \fn GGfloat GetCurrentZoom(void) const
      \brief Computing the current zoom of OpenGL window
      \return Current zoom in a float type
    */
    inline GGfloat GetCurrentZoom(void) const
    {
      if (zoom_ >= 0.0f) {
        return 1.0f + zoom_ / 5.0f;
      }
      else {
        GGfloat current_zoom = 0.05f * zoom_ + 1.0f;
        if (current_zoom < 0.0f) {
          current_zoom = 0.01f;
        }
        return current_zoom;
      }
    }

    /*!
      \fn void GLFWErrorCallback(GGint error_code, char const* description)
      \param error_code - error code returned by OpenGL
      \param description - description of the error
      \brief callback for errors returned by OpenGL
    */
    static void GLFWErrorCallback(GGint error_code, char const* description);

    /*!
      \fn void GLFWWindowSizeCallback(GLFWwindow* window, GGint width, GGint height)
      \param window - pointer to GLFW window
      \param width - width of window
      \param height - height of window
      \brief callback for window size
    */
    static void GLFWWindowSizeCallback(GLFWwindow* window, GGint width, GGint height);

    /*!
      \fn void GLFWKeyCallback(GLFWwindow* window, GGint key, GGint scancode, GGint action, GGint mods)
      \param window - pointer to GLFW window
      \param key - keyboard key that was pressed or released
      \param scancode - system-specific scancode of the key
      \param action - GLFW_PRESS, GLFW_RELEASE or GLFW_REPEAT
      \param mods - bit field describing which modifier keys were held down
      \brief callback for keyboard key
    */
    static void GLFWKeyCallback(GLFWwindow* window, GGint key, GGint scancode, GGint action, GGint mods);

    /*!
      \fn void GLFWScrollCallback(GLFWwindow* window, GGdouble xoffset, GGdouble yoffset)
      \param window - pointer to GLFW window
      \param xoffset - offset in X
      \param yoffset - offset in Y
      \brief callback for mouse scroll
    */
    static void GLFWScrollCallback(GLFWwindow* window, GGdouble xoffset, GGdouble yoffset);

    /*!
      \fn void GLFWMouseButtonCallback(GLFWwindow* window, GGint button, GGint action, GGint mods);
      \param window - pointer to GLFW window
      \param button - mouse key
      \param action - GLFW_PRESS, GLFW_RELEASE or GLFW_REPEAT
      \param mods - bit field describing which modifier keys were held down
      \brief callback for mouse button
    */
    static void GLFWMouseButtonCallback(GLFWwindow* window, GGint button, GGint action, GGint mods);

    /*!
      \fn void GLFWCursorPosCallback(GLFWwindow* window, GGdouble x, GGdouble y);
      \param window - pointer to GLFW window
      \param xoffset - offset in X
      \param yoffset - offset in Y
      \brief cursor position of mouse
    */
    static void GLFWCursorPosCallback(GLFWwindow* window, GGdouble x, GGdouble y);

  private:
    GLFWwindow* window_; /*!< Pointer storing GLFW window */
    static GGbool is_opengl_activated_; /*!< flag for OpenGL activation */
    static GGint window_width_; /*!< GLFW window width */
    static GGint window_height_; /*!< GLFW window height */
    static GGdouble x_mouse_cursor_; /*!< Mouse cursor in X */
    static GGdouble y_mouse_cursor_; /*!< Mouse cursor in Y */
    static GGfloat pitch_angle_; /*!< Pitch angle for mouse */
    static GGfloat yaw_angle_; /*!< Yaw angle for mouse */
    static GGbool is_first_mouse_; /*!< First use of mouse */
    static GGdouble last_mouse_x_position_; /*!< Last position of mouse in x */
    static GGdouble last_mouse_y_position_; /*!< Last position of mouse in y */
    static GGbool is_left_button_; /*!< If left button is used */
    static GGbool is_middle_button_; /*!< If middle button is used */
    static GGbool is_wireframe_; /*!< Line mode: wireframe or solid (if GGbool is false) */
    static GGfloat zoom_; /*!< Value of zoom */
    static GGbool is_save_image_; /*!< Save window to png image file */
    std::string image_output_basename_; /*!< Image output basename */
    GGint image_output_index_; /*!< Image output index */
    GGint msaa_; /*!< MSAA: Multi sample anti-aliasing factor */
    ColorUMap colors_; /*!< List of colors */
    GGfloat background_color_[3]; /*!< window background color */
    GGbool is_draw_axis_; /*!< Flag for axis drawing activation */
    GGfloat x_world_size_; /*!< World size along x axis */
    GGfloat y_world_size_; /*!< World size along y axis */
    GGfloat z_world_size_; /*!< World size along z axis */

    // OpenGL matrices
    glm::mat4 camera_view_; /*!< Camera view */
    static glm::vec3 camera_position_; /*!< Position of the camera */
    static glm::vec3 camera_target_; /*!< Target of the camera */
    static glm::vec3 camera_up_; /*!< Vector to the top */
    glm::mat4 projection_; /*!< Projection matrix (ortho or perspective), perspective by defaut */

    // OpenGL timing
    static GGdouble delta_time_; /*! Time between 2 frames */

    // OpenGL object to draw
    GGEMSOpenGLVolume** opengl_volumes_; /*!< OpenGL volume to display or not */
    GGEMSOpenGLAxis* axis_; /*!< pointer to axis volume */
    GGEMSOpenGLParticles* particles_; /*!< pointer to particles infos for OpenGL */
    GGsize number_of_opengl_volumes_; /*!< Number of OpenGL volumes */
    GGint number_of_displayed_particles_; /*!< Number of displayed particles */
};

/*!
  \fn GGEMSOpenGLManager* get_instance_ggems_opengl_manager(void)
  \return the pointer on the singleton
  \brief Get the GGEMSOpenGLManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSOpenGLManager* get_instance_ggems_opengl_manager(void);

/*!
  \fn void set_window_dimensions_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, GGint width, GGint height)
  \param opengl_manager - pointer on the singleton
  \param width - glfw window width
  \param height - glfw window height
  \brief Set GLFW window dimension
*/
extern "C" GGEMS_EXPORT void set_window_dimensions_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, GGint width, GGint height);

/*!
  \fn void set_msaa_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, GGint msaa_factor)
  \param opengl_manager - pointer on the singleton
  \param msaa_factor - msaa factor
  \brief Set MSAA (multi sample anti-aliasing factor) factor
*/
extern "C" GGEMS_EXPORT void set_msaa_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, GGint msaa_factor);

/*!
  \fn void set_background_color_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, char const* color)
  \param opengl_manager - pointer on the singleton
  \param color - background color
  \brief Set background color in GLFW window
*/
extern "C" GGEMS_EXPORT void set_background_color_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, char const* color = "");

/*!
  \fn void set_draw_axis_opengl_manager(GGEMSOpenGLManager* opengl_manager, GGbool const is_draw_axis)
  \param opengl_manager - pointer on the singleton
  \param is_draw_axis - flag on axis drawing
  \brief activate axis drawing
*/
extern "C" GGEMS_EXPORT void set_draw_axis_opengl_manager(GGEMSOpenGLManager* opengl_manager, GGbool const is_draw_axis);

/*!
  \fn void set_world_size_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, GGfloat const& x_size, GGfloat const& y_size, GGfloat const& z_size, char const* unit)
  \param opengl_manager - pointer on the singleton
  \param x_size - x world size
  \param y_size - y world size
  \param z_size - z world size
  \param unit - length unit
  \brief set world size
*/
extern "C" GGEMS_EXPORT void set_world_size_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, GGfloat const x_size, GGfloat const y_size, GGfloat const z_size, char const* unit);

/*!
  \fn void set_image_output_opengl_manager(GGEMSOpenGLManager* opengl_manager, char const* output_path)
  \param opengl_manager - pointer on the singleton
  \brief Initializing GGEMS OpenGL
*/
extern "C" GGEMS_EXPORT void set_image_output_opengl_manager(GGEMSOpenGLManager* opengl_manager, char const* output_path);

/*!
  \fn void initialize_opengl_manager(GGEMSOpenGLManager* opengl_manager)
  \param opengl_manager - pointer on the singleton
  \brief Initializing GGEMS OpenGL
*/
extern "C" GGEMS_EXPORT void initialize_opengl_manager(GGEMSOpenGLManager* opengl_manager);

/*!
  \fn void display_opengl_manager(GGEMSOpenGLManager* opengl_manager)
  \param opengl_manager - pointer on the singleton
  \brief Displaying GGEMS OpenGL Window
*/
extern "C" GGEMS_EXPORT void display_opengl_manager(GGEMSOpenGLManager* opengl_manager);

/*!
  \fn void set_displayed_particles_opengl_manager(GGEMSOpenGLManager* opengl_manager, GGint const number_of_displayed_particles)
  \param opengl_manager - pointer on the singleton
  \param number_of_displayed_particles - number of displayed particles
  \brief Displaying GGEMS OpenGL Window
*/
extern "C" GGEMS_EXPORT void set_displayed_particles_opengl_manager(GGEMSOpenGLManager* opengl_manager, GGint const number_of_displayed_particles);

#endif // End of OPENGL_VISUALIZATION

#endif // GUARD_GGEMS_GLOBAL_GGEMSOPENGLMANAGER_HH
