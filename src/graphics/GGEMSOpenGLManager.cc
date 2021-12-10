/*!
  \file GGEMSOpenGLManager.cc

  \brief Singleton class storing all informations about OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday October 25, 2021
*/

#ifdef OPENGL_VISUALIZATION

#include <sstream>
#include <filesystem>

#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/graphics/GGEMSOpenGLManager.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/graphics/GGEMSOpenGLVolume.hh"
#include "GGEMS/graphics/GGEMSOpenGLAxis.hh"

#define STB_IMAGE_IMPLEMENTATION
#include "GGEMS/externs/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "GGEMS/externs/stb_image_write.h"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Definition of static members
GGbool GGEMSOpenGLManager::is_opengl_activated_ = false;
GGint GGEMSOpenGLManager::window_width_ = 800;
GGint GGEMSOpenGLManager::window_height_ = 600;
GGbool GGEMSOpenGLManager::is_wireframe_ = true;
GGfloat GGEMSOpenGLManager::zoom_ = 0.0f;
glm::vec3 GGEMSOpenGLManager::camera_position_ = glm::vec3(0.0f, 0.0f, -3.0f);
glm::vec3 GGEMSOpenGLManager::camera_target_ = glm::vec3(0.0, 0.0, 1.0f);
glm::vec3 GGEMSOpenGLManager::camera_up_ = glm::vec3(0.0, -1.0, 0.0f);
GGdouble GGEMSOpenGLManager::delta_time_ = 0.0f;
GGdouble GGEMSOpenGLManager::x_mouse_cursor_ = 0.0;
GGdouble GGEMSOpenGLManager::y_mouse_cursor_ = 0.0;
GGfloat GGEMSOpenGLManager::pitch_angle_ = 0.0;
GGfloat GGEMSOpenGLManager::yaw_angle_ = 90.0; // yaw is initialized to -90.0 degrees since a yaw of 0.0 results in a direction vector pointing to the right so we initially rotate a bit to the left
GGbool GGEMSOpenGLManager::is_first_mouse_ = true;
GGdouble GGEMSOpenGLManager::last_mouse_x_position_ = 0.0;
GGdouble GGEMSOpenGLManager::last_mouse_y_position_ = 0.0;
GGbool GGEMSOpenGLManager::is_save_image_ = false;
GGbool GGEMSOpenGLManager::is_left_button_ = false;
GGbool GGEMSOpenGLManager::is_middle_button_ = false;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLManager::GGEMSOpenGLManager(void)
{
  GGcout("GGEMSOpenGLManager", "GGEMSOpenGLManager", 3) << "GGEMSOpenGLManager creating..." << GGendl;

  window_ = nullptr;
  msaa_ = 8;
  is_draw_axis_ = false;

  // Initializing background color (black by default)
  background_color_[0] = 0.0f;
  background_color_[1] = 0.0f;
  background_color_[2] = 0.0f;

  // Initializing list of colors
  colors_["black"]   = 0;
  colors_["blue"]    = 1;
  colors_["lime"]    = 2;
  colors_["cyan"]    = 3;
  colors_["red"]     = 4;
  colors_["magenta"] = 5;
  colors_["yellow"]  = 6;
  colors_["white"]   = 7;
  colors_["gray"]    = 8;
  colors_["silver"]  = 9;
  colors_["maroon"]  = 10;
  colors_["olive"]   = 11;
  colors_["green"]   = 12;
  colors_["purple"]  = 13;
  colors_["teal"]    = 14;
  colors_["navy"]    = 15;

  number_of_displayed_particles_ = 4096;

  number_of_opengl_volumes_ = 0;
  opengl_volumes_ = nullptr;

  camera_view_ = glm::mat4(1.0f);
  projection_ = glm::mat4(1.0f);
  image_output_basename_ = "";
  image_output_index_ = 0;

  // By default, world is a square of 2 meters
  x_world_size_ = 2.0*m;
  y_world_size_ = 2.0*m;
  z_world_size_ = 2.0*m;

  GGcout("GGEMSOpenGLManager", "GGEMSOpenGLManager", 3) << "GGEMSOpenGLManager created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLManager::~GGEMSOpenGLManager(void)
{
  GGcout("GGEMSOpenGLManager", "~GGEMSOpenGLManager", 3) << "GGEMSOpenGLManager erasing..." << GGendl;

  // Destroying GLFW window
  if (window_) {
    glfwDestroyWindow(window_);
    window_ = nullptr;
    // Closing GLFW
    glfwTerminate();
  }

  if (axis_) {
    delete axis_;
    axis_ = nullptr;
  }

  // Destroying volumes
  if (opengl_volumes_) {
    delete[] opengl_volumes_;
    opengl_volumes_ = nullptr;
  }

  GGcout("GGEMSOpenGLManager", "~GGEMSOpenGLManager", 3) << "GGEMSOpenGLManager erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::Store(GGEMSOpenGLVolume* opengl_volume)
{
  GGcout("GGEMSOpenGLManager", "Store", 3) << "Storing new OpenGL volume in GGEMS OpenGL manager..." << GGendl;

  if (number_of_opengl_volumes_ == 0) {
    opengl_volumes_ = new GGEMSOpenGLVolume*[1];
    opengl_volumes_[0] = opengl_volume;
  }
  else {
    GGEMSOpenGLVolume** tmp = new GGEMSOpenGLVolume*[number_of_opengl_volumes_+1];
    for (GGsize i = 0; i < number_of_opengl_volumes_; ++i) {
      tmp[i] = opengl_volumes_[i];
    }

    tmp[number_of_opengl_volumes_] = opengl_volume;

    delete[] opengl_volumes_;
    opengl_volumes_ = tmp;
  }

  number_of_opengl_volumes_++;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::SetWorldSize(GGfloat const& x_size, GGfloat const& y_size, GGfloat const& z_size, std::string const& unit)
{
  x_world_size_ = DistanceUnit(x_size, unit);
  y_world_size_ = DistanceUnit(y_size, unit);
  z_world_size_ = DistanceUnit(z_size, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::SetMSAA(GGint const& msaa_factor)
{
  // Checking msaa factor value, should be 1, 2, 4 or 8
  if (msaa_factor != 1 && msaa_factor != 2 && msaa_factor != 4 && msaa_factor != 8) {
    GGwarn("GGEMSOpenGLManager", "SetMSAA", 0) << "Warning!!! MSAA factor should be 1x, 2x, 4x or 8x !!! MSAA factor set to 1x" << GGendl;
    msaa_ = 1;
  }
  else {
    msaa_ = msaa_factor;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::SetBackgroundColor(std::string const& color)
{
  // Select color
  ColorUMap::iterator it = colors_.find(color);
  if (it != colors_.end()) {
    for (GGint i = 0; i < 3; ++i) {
      background_color_[i] = GGEMSOpenGLColor::color[it->second][i];
    }
  }
  else {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Warning!!! Color background not found in the list !!!" << std::endl;
    oss << "Available colors: " << std::endl;
    oss << "    * black" << std::endl;
    oss << "    * blue" << std::endl;
    oss << "    * cyan" << std::endl;
    oss << "    * red" << std::endl;
    oss << "    * magenta" << std::endl;
    oss << "    * yellow" << std::endl;
    oss << "    * white" << std::endl;
    oss << "    * gray" << std::endl;
    oss << "    * silver" << std::endl;
    oss << "    * maroon" << std::endl;
    oss << "    * olive" << std::endl;
    oss << "    * green" << std::endl;
    oss << "    * purple" << std::endl;
    oss << "    * teal" << std::endl;
    oss << "    * navy";
    GGEMSMisc::ThrowException("GGEMSOpenGLManager", "SetBackgroundColor", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::SetWindowDimensions(GGint const& width, GGint const& height)
{
  window_width_ = width;
  window_height_ = height;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::SetDrawAxis(GGbool const& is_draw_axis)
{
  is_draw_axis_ = is_draw_axis;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::SetDisplayedParticles(GGint const& number_of_displayed_particles)
{
  if (number_of_displayed_particles > 4096) { // 4096 is the max number of displayed particles
    GGwarn("GGEMSOpenGLManager", "SetDisplayedParticles", 0) << "Your number of displayed particles: " << number_of_displayed_particles
      << " is > 4096 which is the limit. So, the number of displayed particles is set to 4096." << GGendl;
    number_of_displayed_particles_ = 4096;
  }
  else {
    number_of_displayed_particles_ = number_of_displayed_particles;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::Initialize(void)
{
  GGcout("GGEMSOpenGLManager", "Initialize", 3) << "Initializing the OpenGL manager..." << GGendl;

  InitGL(); // Initializing GLFW, GL and GLEW
  if (is_draw_axis_) axis_ = new GGEMSOpenGLAxis();

  is_opengl_activated_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::InitGL(void)
{
  GGcout("GGEMSOpenGLManager", "InitGL", 3) << "Initializing OpenGL..." << GGendl;

  // Initializing the callback error function
  glfwSetErrorCallback(GGEMSOpenGLManager::GLFWErrorCallback);

  // Initializing GLFW
  if (!glfwInit()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Error initializing GLFW!!!";
    GGEMSMisc::ThrowException("GGEMSOpenGLManager", "InitGL", oss.str());
  }

  // Selection GL version
  #ifdef __APPLE__
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  #endif

  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
  glEnable(GL_MULTISAMPLE);
  glfwWindowHint(GLFW_SAMPLES, msaa_);

  // Creating window
  window_ = glfwCreateWindow(window_width_, window_height_, "GGEMS OpenGL", nullptr, nullptr);
  if (!window_) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Error creating a GLFW window!!!";
    GGEMSMisc::ThrowException("GGEMSOpenGLManager", "InitGL", oss.str());
  }

  // Loading logo image, setting image and freeing memory
  GLFWimage logo[1];
  std::string logo_path = LOGO_PATH;
  std::string logo_filename = logo_path + "/ggems_logo_256_256.png";
  logo[0].pixels = stbi_load(logo_filename.c_str(), &logo[0].width, &logo[0].height, nullptr, 4);
  glfwSetWindowIcon(window_, 1, logo);
  stbi_image_free(logo[0].pixels);

  // Selecting current context
  glfwMakeContextCurrent(window_);

  // Initializing GLEW
  glewExperimental = true;
  if (glewInit() != GLEW_OK) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Error initializing a GLEW!!!";
    GGEMSMisc::ThrowException("GGEMSOpenGLManager", "InitGL", oss.str());
  }

  // Give callback functions to GLFW
  glfwSetWindowSizeCallback(window_, GGEMSOpenGLManager::GLFWWindowSizeCallback);
  glfwSetKeyCallback(window_, GGEMSOpenGLManager::GLFWKeyCallback);
  glfwSetScrollCallback(window_, GGEMSOpenGLManager::GLFWScrollCallback);
  glfwSetMouseButtonCallback(window_, GGEMSOpenGLManager::GLFWMouseButtonCallback);
  glfwSetCursorPosCallback(window_, GGEMSOpenGLManager::GLFWCursorPosCallback);

  GGcout("GGEMSOpenGLManager", "InitGL", 1) << "OpenGL infos:" << GGendl;
  GGcout("GGEMSOpenGLManager", "InitGL", 1) << "-------------" << GGendl;
  GGcout("GGEMSOpenGLManager", "InitGL", 1) << "    * Vendor: " << glGetString(GL_VENDOR) << GGendl;
  GGcout("GGEMSOpenGLManager", "InitGL", 1) << "    * Renderer: " << glGetString(GL_RENDERER) << GGendl;
  GGcout("GGEMSOpenGLManager", "InitGL", 1) << "    * OpenGL Version: " << glGetString(GL_VERSION) << GGendl;
  GGcout("GGEMSOpenGLManager", "InitGL", 1) << "    * GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << GGendl;
  GGcout("GGEMSOpenGLManager", "InitGL", 1) << "    * GLFW Version: " << glfwGetVersionString() << GGendl;
  GGcout("GGEMSOpenGLManager", "InitGL", 1) << "    * GLEW Version: " << glewGetString(GLEW_VERSION) << GGendl;
  GGcout("GGEMSOpenGLManager", "InitGL", 1) << "    * GLFW window dimensions: " << window_width_ << "x" << window_height_ << GGendl;
  GGcout("GGEMSOpenGLManager", "InitGL", 1) << "    * MSAA factor: " << msaa_ << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::PrintKeys(void) const
{
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "Keys:" << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "    * [Esc/X]              Quit application" << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "    * [C]                  Wireframe view" << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "    * [V]                  Solid view" << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "    * [R]                  Reset view" << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "    * [K]                  Save current window to a PNG file" << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "    * [+/-]                Zoom in/out" << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "    * [Up/Down]            Move forward/back" << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "    * [W/S]                " << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "    * [Left/Right]         Move left/right" << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "    * [A/D]                " << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "Mouse:" << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "    * [Scroll Up/Down]     Zoom in/out" << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "    * [Left button]        Rotation" << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "    * [Middle button]      Translation" << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::Display(void)
{
  glfwSwapInterval(1); // Control frame rate
  glClearColor(background_color_[0], background_color_[1], background_color_[2], 1.0f); // Setting background colors

  PrintKeys();

  GGdouble last_frame_time = 0.0;
  while (!glfwWindowShouldClose(window_)) {
    // Printing FPS at top of window
    UpdateFPSCounter();

    // Computing delta time
    GGdouble current_frame_time = glfwGetTime();
    delta_time_ = current_frame_time - last_frame_time;
    last_frame_time = current_frame_time;

    // Render here
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    if (is_wireframe_) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    else glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // Rescale window and adapt projection matrix
    glViewport(0, 0, window_width_, window_height_); 

    // Check camera parameters (projection mode + view)
    UpdateProjectionAndView();

    // Draw all registered volumes
    Draw();

    if (is_save_image_) SaveWindow(window_);

    // Swap front and back buffers
    glfwSwapBuffers(window_);
    // Poll for and process events
    glfwPollEvents();
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::Draw(void) const
{
  // Loop over the volumes and draw them
  for (GGint i = 0; i < number_of_opengl_volumes_; ++i) {
    opengl_volumes_[i]->Draw();
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::UpdateFPSCounter(void)
{
  static GGdouble previous_seconds = glfwGetTime();
  static GGint frame_count = 0;
  GGdouble current_seconds = glfwGetTime();
  GGdouble elapsed_seconds = current_seconds - previous_seconds;

  if (elapsed_seconds > 0.25) {
    previous_seconds = current_seconds;
    GGdouble fps = static_cast<GGdouble>(frame_count) / elapsed_seconds;
    std::ostringstream oss(std::ostringstream::out);
    oss << "GGEMS OpenGL @ fps: " << fps;
    glfwSetWindowTitle(window_, oss.str().c_str());
    frame_count = 0;
  }

  frame_count++;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::UpdateProjectionAndView(void)
{
  projection_ = glm::ortho(
    (-x_world_size_/2.0f), (x_world_size_/2.0f),
    (-y_world_size_/2.0f), (y_world_size_/2.0f),
    (-z_world_size_/2.0f), (z_world_size_/2.0f)
  );

  GGfloat current_zoom = GetCurrentZoom();
  projection_ = glm::scale(projection_, glm::vec3(current_zoom, current_zoom, 1.0f));

  camera_view_ = glm::lookAt(
    camera_position_, // Position of the camera in world Space
    camera_position_ + camera_target_, // Direction of camera (origin here)
    camera_up_ // Which vector is up
  );
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::SetImageOutput(std::string const& image_output_basename)
{
  // Get last "/"
  std::size_t n = image_output_basename.find_last_of("/");

  if (n != std::string::npos) { // "/" found, check if directory exists
    std::filesystem::path path_name(image_output_basename.substr(0, n));

    // Check if path exists
    if (std::filesystem::is_directory(path_name)) { // Path exists
      image_output_basename_ = image_output_basename;
    }
    else {
      // Creating path
      GGwarn("GGEMSOpenGLManager", "SetImageOutput", 0) << "Directory (" << image_output_basename << ") storing OpenGL scene does not exits!!!" << GGendl;
      GGwarn("GGEMSOpenGLManager", "SetImageOutput", 0) << "This directory is now created." << GGendl;
      std::filesystem::create_directories(path_name);
    }
  }
  else { // store file basename
    image_output_basename_ = image_output_basename;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::SaveWindow(GLFWwindow* w)
{
  // Get image output index in 000 format
  std::ostringstream index_stream(std::ostringstream::out);
  index_stream << std::setw(3) << std::setfill('0') << image_output_index_;

  // Get filename
  std::string filename = image_output_basename_ + "_" + index_stream.str() + ".png";

  // Get size of frame buffer and compute stride
  GGint frame_width = 0, frame_height = 0;
  glfwGetFramebufferSize(w, &frame_width, &frame_height);

  GGint number_of_channels = 3;

  GGint stride = number_of_channels * frame_width;
  stride += (stride % 4) ? (4 - stride % 4) : 0;

  GGint buffer_size = stride * frame_height;

  // Storage mode
  glPixelStorei(GL_PACK_ALIGNMENT, 4);
  glReadBuffer(GL_FRONT);

  // Read buffer in memory
  char* buffer = new char[buffer_size];
  glReadPixels(0, 0, frame_width, frame_height, GL_RGB, GL_UNSIGNED_BYTE, buffer);

  // Storing buffer in png file
  stbi_set_flip_vertically_on_load(true);
  stbi_write_png(filename.c_str(), frame_width, frame_height, number_of_channels, buffer, stride);

  GGcout("GGEMSOpenGLManager", "SaveWindow", 0) << "OpenGL scene saved in " << filename << GGendl;

  image_output_index_ += 1; // For next image
  is_save_image_ = false;
  delete[] buffer;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::GLFWKeyCallback(GLFWwindow* window, GGint key, GGint, GGint action, GGint)
{
  GGfloat camera_speed = 100.0f * static_cast<GGfloat>(delta_time_); // Defining a camera speed depending on the delta time

  switch (key) {
    case GLFW_KEY_ESCAPE: {
      glfwSetWindowShouldClose(window, true);
      break;
    }
    case GLFW_KEY_X: {
      glfwSetWindowShouldClose(window, true);
      break;
    }
    // case GLFW_KEY_SPACE: {
    //   if (action == GLFW_PRESS) {
    //     if (pause_mode_ == 0) pause_mode_ = 1;
    //     else pause_mode_ = 0;
    //   }
    //   break;
    // }
    case GLFW_KEY_KP_SUBTRACT: {
      zoom_ -= 1.0f;
      break;
    }
    case GLFW_KEY_KP_ADD: {
      zoom_ += 1.0f;
      break;
    }
    case GLFW_KEY_UP: {
      camera_position_ += glm::normalize(camera_up_) * camera_speed;
      break;
    }
    case GLFW_KEY_KP_8: {
      camera_position_ += glm::normalize(camera_up_) * camera_speed;
      break;
    }
    case GLFW_KEY_W: {
      camera_position_ += glm::normalize(camera_up_) * camera_speed;
      break;
    }
    case GLFW_KEY_DOWN: {
      camera_position_ -= glm::normalize(camera_up_) * camera_speed;
      break;
    }
    case GLFW_KEY_S: {
      camera_position_ -= glm::normalize(camera_up_) * camera_speed;
      break;
    }
    case GLFW_KEY_KP_2: {
      camera_position_ -= glm::normalize(camera_up_) * camera_speed;
      break;
    }
    case GLFW_KEY_LEFT: {
      camera_position_ -= glm::normalize(glm::cross(camera_target_, camera_up_)) * camera_speed;
      break;
    }
    case GLFW_KEY_KP_4: {
      camera_position_ -= glm::normalize(glm::cross(camera_target_, camera_up_)) * camera_speed;
      break;
    }
    case GLFW_KEY_A: {
      camera_position_ -= glm::normalize(glm::cross(camera_target_, camera_up_)) * camera_speed;
      break;
    }
    case GLFW_KEY_RIGHT: {
      camera_position_ += glm::normalize(glm::cross(camera_target_, camera_up_)) * camera_speed;
      break;
    }
    case GLFW_KEY_KP_6: {
      camera_position_ += glm::normalize(glm::cross(camera_target_, camera_up_)) * camera_speed;
      break;
    }
    case GLFW_KEY_D: {
      camera_position_ += glm::normalize(glm::cross(camera_target_, camera_up_)) * camera_speed;
      break;
    }
    case GLFW_KEY_R: {
      camera_position_ = glm::vec3(0.0f, 0.0f, -3.0f);
      camera_target_ = glm::vec3(0.0, 0.0, 1.0f);
      camera_up_ = glm::vec3(0.0, -1.0, 0.0f);
      is_wireframe_ = true;
      zoom_ = 0.0f;
      pitch_angle_ = 0.0;
      yaw_angle_ = 90.0;
      is_first_mouse_ = true;
      break;
    }
    case GLFW_KEY_K : {
      if (action == GLFW_PRESS) {
        is_save_image_ = true;
      }
      break;
    }
    case GLFW_KEY_C : {
      is_wireframe_ = true;
      break;
    }
    case GLFW_KEY_V : {
      is_wireframe_ = false;
      break;
    }
    default: {
      break;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::GLFWScrollCallback(GLFWwindow*, GGdouble, GGdouble yoffset)
{
  zoom_ += static_cast<GGfloat>(yoffset);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::GLFWWindowSizeCallback(GLFWwindow*, GGint width, GGint height)
{
  window_width_ = width;
  window_height_ = height;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::GLFWMouseButtonCallback(GLFWwindow* window, GGint button, GGint action, GGint)
{
  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
    is_left_button_ = true;
    is_middle_button_ = false;
  }
  else if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS) {
    is_left_button_ = false;
    is_middle_button_ = true;
  }
  else {
    is_left_button_ = false;
    is_middle_button_ = false;
  }

  if (action == GLFW_PRESS) {
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwGetCursorPos(window, &x_mouse_cursor_, &y_mouse_cursor_);
  }
  else {
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::GLFWCursorPosCallback(GLFWwindow* window, GGdouble x, GGdouble y)
{
  if (is_left_button_) {
    if (is_first_mouse_) {
      is_first_mouse_ = false;
      last_mouse_x_position_ = x_mouse_cursor_;
      last_mouse_y_position_ = y_mouse_cursor_;
    }

    // Offset between cursor position and current position
    GGdouble x_cursor_offset = x - last_mouse_x_position_;
    GGdouble y_cursor_offset = y - last_mouse_y_position_;
    last_mouse_x_position_ = x;
    last_mouse_y_position_ = y;

    GGdouble mouse_sensitivity = 0.05;

    x_cursor_offset *= mouse_sensitivity;
    y_cursor_offset *= mouse_sensitivity;

    yaw_angle_ -= static_cast<GGfloat>(x_cursor_offset);
    pitch_angle_ += static_cast<GGfloat>(y_cursor_offset);

    if (pitch_angle_ > 89.0) pitch_angle_ = 89.0;
    if (pitch_angle_ < -89.0) pitch_angle_ = -89.0;

    glm::vec3 target;
    target.x = cos(glm::radians(yaw_angle_)) * cos(glm::radians(pitch_angle_));
    target.y = sin(glm::radians(pitch_angle_));
    target.z = sin(glm::radians(yaw_angle_)) * cos(glm::radians(pitch_angle_));
    camera_target_ = glm::normalize(target);
  }
  else if (is_middle_button_) {
    if (is_first_mouse_) {
      is_first_mouse_ = false;
      last_mouse_x_position_ = x_mouse_cursor_;
      last_mouse_y_position_ = y_mouse_cursor_;
    }

    // Offset between cursor position and current position
    GGdouble x_cursor_offset = -x + last_mouse_x_position_;
    GGdouble y_cursor_offset = -y + last_mouse_y_position_;
    last_mouse_x_position_ = x;
    last_mouse_y_position_ = y;

    glm::vec3 position(0.5*x_cursor_offset, 0.5*y_cursor_offset, 0.0f);
    camera_position_ -= position;
  }
  else {
    is_first_mouse_ = true;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::GLFWErrorCallback(GGint error_code, char const* description)
{
  std::ostringstream oss(std::ostringstream::out);
  oss << "!!!!!!!!" << std::endl;
  oss << "GLFW error code: " << error_code << std::endl;
  oss << description << std::endl;
  oss << "!!!!!!!!";
  GGEMSMisc::ThrowException("GGEMSOpenGLManager", "GLFWErrorCallback", oss.str());
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLManager* get_instance_ggems_opengl_manager(void)
{
  return &GGEMSOpenGLManager::GetInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_window_dimensions_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, GGint width, GGint height)
{
  opengl_manager->SetWindowDimensions(width, height);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_msaa_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, GGint msaa_factor)
{
  opengl_manager->SetMSAA(msaa_factor);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_background_color_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, char const* color)
{
  opengl_manager->SetBackgroundColor(color);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_draw_axis_opengl_manager(GGEMSOpenGLManager* opengl_manager, GGbool const is_draw_axis)
{
  opengl_manager->SetDrawAxis(is_draw_axis);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_world_size_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, GGfloat const x_size, GGfloat const y_size, GGfloat const z_size, char const* unit)
{
  opengl_manager->SetWorldSize(x_size, y_size, z_size, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_image_output_opengl_manager(GGEMSOpenGLManager* opengl_manager, char const* output_path)
{
  opengl_manager->SetImageOutput(output_path);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void initialize_opengl_manager(GGEMSOpenGLManager* opengl_manager)
{
  opengl_manager->Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void display_opengl_manager(GGEMSOpenGLManager* opengl_manager)
{
  opengl_manager->Display();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_displayed_particles_opengl_manager(GGEMSOpenGLManager* opengl_manager, GGint const number_of_displayed_particles)
{
  opengl_manager->SetDisplayedParticles(number_of_displayed_particles);
}

#endif // End of OPENGL_VISUALIZATION
