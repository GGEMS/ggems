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

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>

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

  // Initialization of matrices
  // mvp_ = {
  //   {1.0f, 0.0f, 0.0f, 0.0f},
  //   {0.0f, 1.0f, 0.0f, 0.0f},
  //   {0.0f, 0.0f, 1.0f, 0.0f},
  //   {0.0f, 0.0f, 0.0f, 1.0f}
  // };

  // ortho_projection_ = {
  //   {1.0f, 0.0f, 0.0f, 0.0f},
  //   {0.0f, 1.0f, 0.0f, 0.0f},
  //   {0.0f, 0.0f, 1.0f, 0.0f},
  //   {0.0f, 0.0f, 0.0f, 1.0f}
  // };

  // perspective_projection_ = {
  //   {1.0f, 0.0f, 0.0f, 0.0f},
  //   {0.0f, 1.0f, 0.0f, 0.0f},
  //   {0.0f, 0.0f, 1.0f, 0.0f},
  //   {0.0f, 0.0f, 0.0f, 1.0f}
  // };

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

  glDeleteBuffers(1, &vao_axis_);
  glDeleteBuffers(1, &vbo_axis_);
  glDeleteProgram(program_shader_id_);

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

  InitGL(); // Initializing GLFW, GL and GLEW
  InitShaders(); // Compile and store shaders
  InitBuffers(); // Initialization of OpenGL buffers, for axis
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

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);

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
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLManager::InitShaders(void)
{
  // Creating shaders
  GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
  GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);

  // A global vertex shader
  std::string vertex_shader_source_str = "#version " + GetOpenGLSLVersion() + "\n"
    "\n"
    "layout(location = 0) in vec3 position;\n"
    "\n"
    "uniform vec3 color;\n"
    "\n"
    "out vec4 color_rgba;\n"
    "\n"
    "void main(void) {\n"
    "  color_rgba = vec4(color, 1.0);\n"
    "  gl_Position = vec4(position, 1.0);\n"
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

void GGEMSOpenGLManager::InitBuffers(void)
{
    float vertex_buffer_axis[] = {0.0f,  0.5f,  0.0f, 0.5f, -0.5f,  0.0f, -0.5f, -0.5f,  0.0f};
  // Creating a vao for each axis and a vbo
  //CheckOpenGLError(glGetError(), "GGEMSOpenGLManager", "TOTO");
  glGenVertexArrays(1, &vao_axis_);
  //CheckOpenGLError(glGetError(), "GGEMSOpenGLManager", "InitBuffers");

  // GLenum err = glGetError();
   // std::cout << "Error mapping vbo buffer!!!: " << err << std::endl;
   glBindVertexArray(vao_axis_); // Lock current vao

//   // An array representing 6 vertices

    // -0.5f, -0.5f, 0.0f, 0.0f, 0.5f, 0.0f, 0.5f, -0.5f, 0.0f};//, // X
//     /*0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, // Y
//     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f  // Z
//   };*/

   
//   //for (int i = 0; i < 1; ++i) {


//     glBindBuffer(GL_ARRAY_BUFFER, vbo_axis_); // Lock vbo for position

//     // Reading data
     glGenBuffers(1, &vbo_axis_);
     glBufferData(GL_ARRAY_BUFFER, 9*sizeof(float), vertex_buffer_axis, GL_STATIC_DRAW);
     glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
     glEnableVertexAttribArray(0);

     //glBindBuffer(GL_ARRAY_BUFFER, 0); // Unlock vbo
     //glBindVertexArray(0);// Unlock current vao
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
  //glFinish();
  // glm::mat4 projection_matrix = glm::ortho(-10.0f,10.0f,-20.0f,20.0f,-30.0f,30.0f);
  //glm::mat4 projection_matrix = glm::ortho(-2.0f, 2.0f, -2.0f, 2.0f, -2.0f, 2.0f);
  //glm::mat4 mvp = glm::mat4(1.0f);
//   // ortho_projection_.m0_[0] = 2.0f / (10.0f - (-10.0f));
//   // ortho_projection_.m1_[1] = 2.0f / (20.0f - (-20.0f));
//   // ortho_projection_.m2_[2] = 2.0f / (30.0f - (-30.0f));
//   // ortho_projection_.m3_[0] = - (10.0f + (-10.0f)) / (10.0f - (-10.0f));
//   // ortho_projection_.m3_[1] = - (20.0f + (-20.0f)) / (20.0f - (-20.0f));
//   // ortho_projection_.m3_[2] = - (30.0f + (-30.0f)) / (30.0f - (-30.0f));

//   // Enabling shader program
   glUseProgram(program_shader_id_);

   glBindVertexArray(vao_axis_);
   //glBindBuffer(GL_ARRAY_BUFFER, vbo_axis_);
  //  glBindBuffer(GL_ARRAY_BUFFER, vbo_axis_);
  //  float* vbo_ptr = static_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
  //  vbo_ptr[0] = 0.5f;
  //  vbo_ptr[1] = -0.5f;
  //  vbo_ptr[2] = 0.0f;

  //   glUnmapBuffer(GL_ARRAY_BUFFER);
  //   glBindBuffer(GL_ARRAY_BUFFER, 0);

//   glBindBuffer(GL_ARRAY_BUFFER, vbo_axis_);
//   //float* vbo_ptr = static_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY));
// //   if (!vbo_ptr) {
// //     GLenum err = glGetError();
// //     std::cout << "Error mapping vbo buffer!!!: " << err << std::endl;
// // }
//   glBindBuffer(GL_ARRAY_BUFFER, 0);

//   // Set color and MVP matrix to shader
  //glPointSize(10.0);
   glUniform3f(glGetUniformLocation(program_shader_id_,"color"), 1.0f, 1.0f, 0.0f);
   //glUniformMatrix4fv(glGetUniformLocation(program_shader_id_, "mvp"), 1, GL_FALSE, &mvp[0][0]);

  glDrawArrays(GL_TRIANGLES, 0, 3);

  //glBindBuffer(GL_ARRAY_BUFFER, 0);

  // glBindVertexArray(0);

//   // Disabling shader program
 //  glUseProgram(0);

  // std::cout << "MY 4x4 MATRIX:" << std::endl;
  // std::cout << mvp_.m0_[0] << " " << mvp_.m0_[1] << " " << mvp_.m0_[2] << " " << mvp_.m0_[3] << std::endl;
  // std::cout << mvp_.m1_[0] << " " << mvp_.m1_[1] << " " << mvp_.m1_[2] << " " << mvp_.m1_[3] << std::endl;
  // std::cout << mvp_.m2_[0] << " " << mvp_.m2_[1] << " " << mvp_.m2_[2] << " " << mvp_.m2_[3] << std::endl;
  // std::cout << mvp_.m3_[0] << " " << mvp_.m3_[1] << " " << mvp_.m3_[2] << " " << mvp_.m3_[3] << std::endl;

  // glm::mat4 projection_matrix = glm::ortho(-10.0f,10.0f,-20.0f,20.0f,-30.0f,30.0f);

  // std::cout << "GLM" << std::endl;
  // std::cout << glm::to_string(projection_matrix) << std::endl;

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
    //glViewport(0, 0, GGEMSOpenGLManager::window_width_, GGEMSOpenGLManager::window_height_);

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

void GGEMSOpenGLManager::CheckOpenGLError(GLenum const& error, std::string const& class_name, std::string const& method_name) const
{
  if (error != GL_NO_ERROR) GGEMSMisc::ThrowException(class_name, method_name, ErrorType(error));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

std::string GGEMSOpenGLManager::ErrorType(GLenum const& error) const
{
  // Error description storing in a ostringstream
  std::ostringstream oss(std::ostringstream::out);
  oss << std::endl;

  switch (error) {
    case 1280: {
      oss << "GL_INVALID_ENUM:" << std::endl;
      oss << "    * if an enumeration parameter is not legal." << std::endl;
      return oss.str();
    }
    case 1281: {
      oss << "GL_INVALID_VALUE:" << std::endl;
      oss << "    * if a value parameter is not legal." << std::endl;
      return oss.str();
    }
    case 1282: {
      oss << "GL_INVALID_OPERATION:" << std::endl;
      oss << "    * if the state for a command is not legal for its given parameters." << std::endl;
      return oss.str();
    }
    case 1283: {
      oss << "GL_STACK_OVERFLOW:" << std::endl;
      oss << "    * if a stack pushing operation causes a stack overflow." << std::endl;
      return oss.str();
    }
    case 1284: {
      oss << "GL_STACK_UNDERFLOW:" << std::endl;
      oss << "    * if a stack popping operation occurs while the stack is at its lowest point." << std::endl;
      return oss.str();
    }
    case 1285: {
      oss << "GL_OUT_OF_MEMORY:" << std::endl;
      oss << "    * if a memory allocation operation cannot allocate (enough) memory." << std::endl;
      return oss.str();
    }
    case 1286: {
      oss << "GL_INVALID_FRAMEBUFFER_OPERATION:" << std::endl;
      oss << "    * if reading or writing to a framebuffer that is not complete." << std::endl;
      return oss.str();
    }
    default: {
      oss << "Unknown OpenGL error" << std::endl;
      oss << "    * if an enumeration parameter is not legal." << std::endl;
      return oss.str();
    }
  }
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
