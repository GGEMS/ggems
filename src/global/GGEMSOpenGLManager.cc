/*!
  \file GGEMSOpenGLManager.cc

  \brief Singleton class storing all informations about OpenGL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday October 25, 2021
*/

#include <sstream>

#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/global/GGEMSOpenGLManager.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

#define STB_IMAGE_IMPLEMENTATION
#include "GGEMS/externs/stb_image.h"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Definition of static members
int GGEMSOpenGLManager::window_width_ = 800;
int GGEMSOpenGLManager::window_height_ = 600;

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

  // Closing GLFW
  glfwTerminate();

  GGcout("GGEMSOpenGLManager", "~GGEMSOpenGLManager", 3) << "GGEMSOpenGLManager erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::SetMSAA(int const& msaa_factor)
{
  msaa_ = msaa_factor;
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
    oss << "Warning!!! Color background not found in the list !!!";
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

void GGEMSOpenGLManager::Initialize(void)
{
  GGcout("GGEMSOpenGLManager", "Initialize", 3) << "Initializing the OpenGL manager..." << GGendl;

  // Initializing GLFW, GL and GLEW
  InitGL();
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

  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
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
  // glfwSetScrollCallback(window_, Sphere::GLFWScrollCallback);
  // glfwSetMouseButtonCallback(window_, Sphere::GLFWMouseButtonCallback);
  // glfwSetCursorPosCallback(window_, Sphere::GLFWCursorPosCallback);

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

  std::string test = GetOpenGLSLVersion();
  std::cout << test << std::endl;
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
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "    * [Esc/X]           Quit application" << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << "Mouse:" << GGendl;
  GGcout("GGEMSOpenGLManager", "PrintKeys", 0) << GGendl;

/*
  std::cout << std::endl;
  std::cout << "Keys:" << std::endl;
  std::cout << "    * [R]                          Reset view" << std::endl;
  std::cout << "    * [P]                          Perspective projection" << std::endl;
  std::cout << "    * [O]                          Ortho projection" << std::endl;
  std::cout << "    * [Esc] / [X]                  Quit application" << std::endl;
  std::cout << "    * [Space]                      Stop / Restart application" << std::endl;
  std::cout << "    * [+/-]                        Zoom in/out" << std::endl;
  std::cout << "    * [Up/Down]                    Translation up/dowm" << std::endl;
  std::cout << "    * [Left/Right]                 Translation left/right" << std::endl;
  std::cout << std::endl;
  std::cout << "Mouse:" << std::endl;
  std::cout << "    * [Scroll Up/Down]             Zoom in/out" << std::endl;
  std::cout << std::endl;
*/
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::DrawAxis(void)
{
  //glPushMatrix();
  //glMatrixMode(GL_MODELVIEW);
  // glDisable( GL_CULL_FACE );

  // Draw central point in (0 0 0)
  //glPointSize(10.0);
  //glLineWidth(10.0f);
  // glEnable(GL_PROGRAM_POINT_SIZE);
  // glEnable(GL_POINT_SMOOTH);
  // glEnable(GL_BLEND);

  // glTranslatef( 0.0, 0.0, 5.0 );
  // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  // glPointSize(10);
  // glLineWidth(2.5); 
  // glColor3f(1.0, 0.0, 0.0);

  // glBegin(GL_LINES);
  //glColor3f(1.0f, 0.0f, 0.0f); // Red
  //glColor3f(1.0,0.0,0.0);
  // glVertex3f(0.0f, 0.0f, 0.0f);
  // glVertex3f(100.0f, 100.0f, 0.0f);
  // glEnd();

  // glDisable(GL_BLEND);
  // glDisable(GL_POINT_SMOOTH);
  // glDisable(GL_PROGRAM_POINT_SIZE);
/*
   glTranslatef( 0.0, 0.0, DIST_BALL );


   for ( col = 0; col <= colTotal; col++ )
   {
 
      xl = -GRID_SIZE / 2 + col * sizeCell;
      xr = xl + widthLine;

      yt =  GRID_SIZE / 2;
      yb = -GRID_SIZE / 2 - widthLine;

      glBegin( GL_POLYGON );

      glColor3f( 0.6f, 0.1f, 0.6f );              

      glVertex3f( xr, yt, z_offset );    
      glVertex3f( xl, yt, z_offset );       
      glVertex3f( xl, yb, z_offset );     
      glVertex3f( xr, yb, z_offset );  

      glEnd();
   }
  */
/*
glPointSize(10.0);
            glEnable(GL_PROGRAM_POINT_SIZE);
            glEnable(GL_POINT_SMOOTH);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glBegin(GL_POINTS);
            glEnd();
            */
 // glPopMatrix();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::Display(void)
{
  glfwSwapInterval(1); // Control frame rate
  glClearColor(background_color_[0], background_color_[1], background_color_[2], 1.0f); // Setting background colors
  // glMatrixMode(GL_PROJECTION);
  // glLoadIdentity();
  // glOrtho(0.0,GGEMSOpenGLManager::window_width_,GGEMSOpenGLManager::window_height_,0.0,0.0,1.0);

  while (!glfwWindowShouldClose(window_)) {
    // Printing FPS at top of window
    UpdateFPSCounter();

    // Render here
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //glPushMatrix();

    // Rescale window
    glViewport(0, 0, GGEMSOpenGLManager::window_width_, GGEMSOpenGLManager::window_height_);

    // glMatrixMode(GL_MODELVIEW);
    // glLoadIdentity();

    // glPointSize(10.0f);
    // glLineWidth(2.5f);
    // glColor3f(1.0f, 0.0f, 0.0f);

    // glBegin(GL_LINES);
    // glVertex2f(0.0f, 0.0f);
    // glVertex2f(100.0f, 100.0f);
    // glEnd();

    if (is_draw_axis_) DrawAxis();

    //glPopMatrix();
    //glFlush();

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

void GGEMSOpenGLManager::GLFWKeyCallback(GLFWwindow* window, int key, int, int action, int)
{
  if (action != GLFW_PRESS) return;

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
    // case GLFW_KEY_KP_ADD: {
    //   zoom_ += 0.125f;
    //   break;
    // }
    // case GLFW_KEY_KP_SUBTRACT: {
    //   zoom_ -= 0.125f;
    //   if (zoom_ < 0.0f) zoom_ = 0.0f;
    //   break;
    // }
    // case GLFW_KEY_UP: {
    //   y_translate_ += 1.0f;
    //   break;
    // }
    // case GLFW_KEY_DOWN: {
    //   y_translate_ -= 1.0f;
    //   break;
    // }
    // case GLFW_KEY_LEFT: {
    //   x_translate_ -= 1.0f;
    //   break;
    // }
    // case GLFW_KEY_RIGHT: {
    //   x_translate_ += 1.0f;
    //   break;
    // }
    // case GLFW_KEY_R: {
    //   x_translate_ = 0.0f;
    //   y_translate_ = 0.0f;
    //   zoom_ = 1.0f;
    //   break;
    // }
    // case GLFW_KEY_P : {
    //   is_perpective_mode_ = 1;
    //   break;
    // }
    // case GLFW_KEY_O : {
    //   is_perpective_mode_ = 0;
    //   break;
    // }
    default: {
      break;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::GLFWWindowSizeCallback(GLFWwindow*, int width, int height)
{
  GGEMSOpenGLManager::window_width_ = width;
  GGEMSOpenGLManager::window_height_ = height;
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
