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
#include "GGEMS/graphics/GGEMSOpenGLSphere.hh"
#include "GGEMS/graphics/GGEMSOpenGLPrism.hh"

#define STB_IMAGE_IMPLEMENTATION
#include "GGEMS/externs/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "GGEMS/externs/stb_image_write.h"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Definition of static members
int GGEMSOpenGLManager::window_width_ = 800;
int GGEMSOpenGLManager::window_height_ = 600;
bool GGEMSOpenGLManager::is_perspective_ = true;
bool GGEMSOpenGLManager::is_wireframe_ = true;
float GGEMSOpenGLManager::zoom_ = 0.0f;
glm::vec3 GGEMSOpenGLManager::camera_position_ = glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 GGEMSOpenGLManager::camera_target_ = glm::vec3(0.0, 0.0, -1.0f);
glm::vec3 GGEMSOpenGLManager::camera_up_ = glm::vec3(0.0, 1.0, 0.0f);
double GGEMSOpenGLManager::delta_time_ = 0.0f;
double GGEMSOpenGLManager::x_mouse_cursor_ = 0.0;
double GGEMSOpenGLManager::y_mouse_cursor_ = 0.0;
float GGEMSOpenGLManager::pitch_angle_ = 0.0;
float GGEMSOpenGLManager::yaw_angle_ = -90.0; // yaw is initialized to -90.0 degrees since a yaw of 0.0 results in a direction vector pointing to the right so we initially rotate a bit to the left
bool GGEMSOpenGLManager::is_first_mouse_ = true;
double GGEMSOpenGLManager::last_mouse_x_position_ = 0.0;
double GGEMSOpenGLManager::last_mouse_y_position_ = 0.0;
bool GGEMSOpenGLManager::is_save_image_ = false;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLManager::GGEMSOpenGLManager(void)
{
  GGcout("GGEMSOpenGLManager", "GGEMSOpenGLManager", 3) << "GGEMSOpenGLManager creating..." << GGendl;

  window_ = nullptr;
  msaa_ = 1;
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

  number_of_opengl_volumes_ = 0;
  opengl_volumes_ = nullptr;

  camera_view_ = glm::mat4(1.0f);
  projection_ = glm::mat4(1.0f);
  image_output_basename_ = "";
  image_output_index_ = 0;

  GGcout("GGEMSOpenGLManager", "GGEMSOpenGLManager", 3) << "GGEMSOpenGLManager created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLManager::~GGEMSOpenGLManager(void)
{
  GGcout("GGEMSOpenGLManager", "~GGEMSOpenGLManager", 3) << "GGEMSOpenGLManager erasing..." << GGendl;

  // Destroying GLFW window
  glfwDestroyWindow(window_); // destroying GLFW window
  window_ = nullptr;

  if (sphere_test) delete sphere_test;
  if (prism_test) delete prism_test;

  glDeleteProgram(program_shader_id_);

  // Closing GLFW
  glfwTerminate();

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

void GGEMSOpenGLManager::SetMSAA(int const& msaa_factor)
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
    for (int i = 0; i < 3; ++i) {
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

void GGEMSOpenGLManager::SetWindowDimensions(int const& width, int const& height)
{
  window_width_ = width;
  window_height_ = height;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::SetDrawAxis(bool const& is_draw_axis)
{
  is_draw_axis_ = is_draw_axis;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::SetProjectionMode(std::string const& projection_mode)
{
  std::string mode = projection_mode;
  std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);

  if (mode == "perspective") {
    is_perspective_ = 1;
  }
  else if (mode == "ortho") {
    is_perspective_ = 0;
  }
  else {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Available projection mode are: 'perspective' or 'ortho' only!!!";
    GGEMSMisc::ThrowException("GGEMSOpenGLManager", "SetProjectionMode", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::Initialize(void)
{
  GGcout("GGEMSOpenGLManager", "Initialize", 3) << "Initializing the OpenGL manager..." << GGendl;

  InitGL(); // Initializing GLFW, GL and GLEW
  InitShader(); // Compile and store shader
  if (is_draw_axis_) InitAxisVolume(); // Initialization of axis
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
  glEnable(GL_DEPTH_TEST); // Enable depth buffering
  glDepthFunc(GL_LEQUAL); // Accept fragment if it closer to the camera than the former one or GL_LESS
  glEnable(GL_MULTISAMPLE); // Activating anti-aliasing
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

void GGEMSOpenGLManager::InitShader(void)
{
  // Creating shaders
  GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
  GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);

  // A global vertex shader
  std::string vertex_shader_source_str = "#version " + GetOpenGLSLVersion() + "\n"
    "\n"
    "layout(location = 0) in vec3 position;\n"
    "\n"
    "uniform mat4 mvp;\n"
    "uniform vec3 color;\n"
    "\n"
    "out vec4 color_rgba;\n"
    "\n"
    "void main(void) {\n"
    "  color_rgba = vec4(color, 1.0);\n"
    "  gl_Position = mvp * vec4(position, 1.0);\n"
    "}\n";

  // A global fragment shader
  std::string fragment_shader_source_str = "#version " + GetOpenGLSLVersion() + "\n"
    "\n"
    "layout(location = 0) out vec4 out_color;\n"
    "\n"
    "in vec4 color_rgba;\n"
    "\n"
    "void main(void) {\n"
    "  out_color = color_rgba;\n"
    "}\n";

  // Setting the source code
  char const* vertex_shader_source = vertex_shader_source_str.c_str();
  char const* fragment_shader_source = fragment_shader_source_str.c_str();
  glShaderSource(vert_shader, 1, &vertex_shader_source, nullptr);
  glShaderSource(frag_shader, 1, &fragment_shader_source, nullptr);

  // Compiling shaders
  CompileShader(vert_shader);
  CompileShader(frag_shader);

  // Linking the program
  program_shader_id_ = glCreateProgram();
  glAttachShader(program_shader_id_, vert_shader);
  glAttachShader(program_shader_id_, frag_shader);
  glLinkProgram(program_shader_id_);

  // Deleting shaders
  glDeleteShader(vert_shader);
  glDeleteShader(frag_shader);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::InitAxisVolume(void)
{
  // The axis system is composed by:
  //     * a sphere
  //     * a tube + cone for X axis
  //     * a tube + cone for Y axis
  //     * a tube + cone for Z axis

  // 0.6 mm Sphere in (0, 0, 0)
  sphere_test = new GGEMSOpenGLSphere(2.0f*mm);

  sphere_test->SetColor("yellow");
  sphere_test->SetVisible(false);
  sphere_test->Build();

  prism_test = new GGEMSOpenGLPrism(1.0f*mm, 0.5f*mm, 2.0f*mm, 36, 10);
  prism_test->Build();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::CompileShader(GLuint const& shader) const
{
  GLint sucess = 0;
  glCompileShader(shader);
  glGetShaderiv(shader, GL_COMPILE_STATUS, &sucess);
  if(sucess == GL_FALSE) {
    GLint max_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &max_length);

    // The max_length includes the NULL character
    std::vector<GLchar> error_log(max_length);
    glGetShaderInfoLog(shader, max_length, &max_length, &error_log[0]);

    std::ostringstream oss(std::ostringstream::out);
    oss << "Error compiling shader!!!" << std::endl;
    for (std::size_t i = 0; i < error_log.size(); ++i) oss << error_log[i];

    glDeleteShader(shader); // Don't leak the shader.
    throw std::runtime_error(oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

std::string GGEMSOpenGLManager::GetOpenGLSLVersion(void) const
{
  std::string glsl_version(reinterpret_cast<char const*>(glGetString(GL_SHADING_LANGUAGE_VERSION)));
  std::string digits("0123456789");

  std::size_t n = glsl_version.find_first_of(digits);
  if (n != std::string::npos)
  {
    std::size_t m = glsl_version.find_first_not_of(digits+".", n);
    std::string tmp = glsl_version.substr(n, m != std::string::npos ? m-n : m);
    // Deleting '.'
    tmp.erase(std::remove(tmp.begin(), tmp.end(), '.'), tmp.end());
    return tmp;
  }
  else {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Impossible to get GLSL version!!!";
    GGEMSMisc::ThrowException("GGEMSOpenGLManager", "GetOpenGLSLVersion", oss.str());
  }
  return std::string();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::PrintKeys(void) const
{
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "Keys:" << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "    * [Esc/X]              Quit application" << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "    * [P]                  Perspective projection" << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "    * [O]                  Ortho projection" << GGendl;
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
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::DrawAxis(void)
{
  opengl_volumes_[0]->Draw();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::Display(void)
{
  glfwSwapInterval(1); // Control frame rate
  glClearColor(background_color_[0], background_color_[1], background_color_[2], 1.0f); // Setting background colors

  PrintKeys();

  double last_frame_time = 0.0;
  while (!glfwWindowShouldClose(window_)) {
    // Printing FPS at top of window
    UpdateFPSCounter();

    // Computing delta time
    double current_frame_time = glfwGetTime();
    delta_time_ = current_frame_time - last_frame_time;
    last_frame_time = current_frame_time;

    // Render here
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (is_wireframe_) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    else glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // Rescale window and adapt projection matrix
    glViewport(0, 0, window_width_, window_height_); 

    // Check camera parameters (projection mode + view)
    UpdateProjectionAndView();

    if (is_draw_axis_) DrawAxis();

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

void GGEMSOpenGLManager::UpdateFPSCounter(void)
{
  static double previous_seconds = glfwGetTime();
  static int frame_count = 0;
  double current_seconds = glfwGetTime();
  double elapsed_seconds = current_seconds - previous_seconds;

  if (elapsed_seconds > 0.25) {
    previous_seconds = current_seconds;
    double fps = static_cast<double>(frame_count) / elapsed_seconds;
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
  if (is_perspective_) {
    // Computing fov depending on zoom
    float fov = 45.0f + zoom_;
    if (fov <= 1.0f) {
      fov = 1.0f;
      zoom_ = -44.0f;
    }

    projection_ = glm::perspective(glm::radians(fov), static_cast<float>(window_width_) / static_cast<float>(window_height_), 0.1f, 100.0f);
  }
  else {
    float current_zoom = 1.0f + zoom_ /10.0f;
    if (current_zoom <= 0.0f)
    {
      current_zoom = 0.1f;
      zoom_ = -10.0f;
    }
    projection_ = glm::ortho(-10.0f/current_zoom,10.0f/current_zoom,-10.0f/current_zoom,10.0f/current_zoom,-10.0f,10.0f);
  }

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
  int frame_width = 0, frame_height = 0;
  glfwGetFramebufferSize(w, &frame_width, &frame_height);

  int number_of_channels = 3;

  int stride = number_of_channels * frame_width;
  stride += (stride % 4) ? (4 - stride % 4) : 0;

  int buffer_size = stride * frame_height;

  // Storage mode
  glPixelStorei(GL_PACK_ALIGNMENT, 4);
  glReadBuffer(GL_FRONT);

  // Read buffer in memory
  char* buffer = new char[buffer_size];
  glReadPixels(0, 0, frame_width, frame_height, GL_RGB, GL_UNSIGNED_BYTE, buffer);

  // Storing buffer in png file
  stbi_set_flip_vertically_on_load(true);
  stbi_write_png(filename.c_str(), frame_width, frame_height, number_of_channels, buffer, stride);

  image_output_index_ += 1; // For next image
  is_save_image_ = false;
  delete[] buffer;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::GLFWKeyCallback(GLFWwindow* window, int key, int, int action, int)
{
  float camera_speed = 2.5f * static_cast<float>(delta_time_); // Defining a camera speed depending on the delta time

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
      zoom_ += 1.0f;
      break;
    }
    case GLFW_KEY_KP_ADD: {
      zoom_ -= 1.0f;
      break;
    }
    case GLFW_KEY_UP: {
      camera_position_ += camera_speed * camera_target_;
      break;
    }
    case GLFW_KEY_KP_8: {
      camera_position_ += camera_speed * camera_target_;
      break;
    }
    case GLFW_KEY_W: {
      camera_position_ += camera_speed * camera_target_;
      break;
    }
    case GLFW_KEY_DOWN: {
      camera_position_ -= camera_speed * camera_target_;
      break;
    }
    case GLFW_KEY_S: {
      camera_position_ -= camera_speed * camera_target_;
      break;
    }
    case GLFW_KEY_KP_5: {
      camera_position_ -= camera_speed * camera_target_;
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
      camera_position_ = glm::vec3(0.0f, 0.0f, 5.0f);
      camera_target_ = glm::vec3(0.0, 0.0, -1.0f);
      camera_up_ = glm::vec3(0.0, 1.0, 0.0f);
      is_perspective_ = true;
      is_wireframe_ = true;
      zoom_ = 0.0f;
      pitch_angle_ = 0.0;
      yaw_angle_ = -90.0;
      is_first_mouse_ = true;
      break;
    }
    case GLFW_KEY_K : {
      if (action == GLFW_PRESS) {
        is_save_image_ = true;
      }
      break;
    }
    case GLFW_KEY_P : {
      is_perspective_ = true;
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
    case GLFW_KEY_O : {
      is_perspective_ = false;
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

void GGEMSOpenGLManager::GLFWScrollCallback(GLFWwindow*, double, double yoffset)
{
  zoom_ -= static_cast<float>(yoffset);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::GLFWWindowSizeCallback(GLFWwindow*, int width, int height)
{
  window_width_ = width;
  window_height_ = height;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::GLFWMouseButtonCallback(GLFWwindow* window, int button, int action, int)
{
  if (button != GLFW_MOUSE_BUTTON_LEFT) return;

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

void GGEMSOpenGLManager::GLFWCursorPosCallback(GLFWwindow* window, double x, double y)
{
  if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) {

    if (is_first_mouse_) {
      is_first_mouse_ = false;
      last_mouse_x_position_ = x_mouse_cursor_;
      last_mouse_y_position_ = y_mouse_cursor_;
    }

    // Offset between cursor position and current position
    double x_cursor_offset = x - last_mouse_x_position_;
    double y_cursor_offset = last_mouse_y_position_ - y;
    last_mouse_x_position_ = x;
    last_mouse_y_position_ = y;

    double mouse_sensitivity = 0.05;

    x_cursor_offset *= mouse_sensitivity;
    y_cursor_offset *= mouse_sensitivity;

    yaw_angle_ += static_cast<float>(x_cursor_offset);
    pitch_angle_ += static_cast<float>(y_cursor_offset);

    if (pitch_angle_ > 89.0) pitch_angle_ = 89.0;
    if (pitch_angle_ < -89.0) pitch_angle_ = -89.0;

    glm::vec3 target;
    target.x = cos(glm::radians(yaw_angle_)) * cos(glm::radians(pitch_angle_));
    target.y = sin(glm::radians(pitch_angle_));
    target.z = sin(glm::radians(yaw_angle_)) * cos(glm::radians(pitch_angle_));
    camera_target_ = glm::normalize(target);
  }
  else {
    is_first_mouse_ = true;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::GLFWErrorCallback(int error_code, char const* description)
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

void set_window_dimensions_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, int width, int height)
{
  opengl_manager->SetWindowDimensions(width, height);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_msaa_ggems_opengl_manager(GGEMSOpenGLManager* opengl_manager, int msaa_factor)
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

void set_draw_axis_opengl_manager(GGEMSOpenGLManager* opengl_manager, bool const is_draw_axis)
{
  opengl_manager->SetDrawAxis(is_draw_axis);
}

#endif // End of OPENGL_VISUALIZATION
