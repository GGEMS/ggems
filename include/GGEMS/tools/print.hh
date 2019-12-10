#ifndef GUARD_GGEMS_TOOLS_PRINT_HH
#define GUARD_GGEMS_TOOLS_PRINT_HH

/*!
  \file print.hh

  \brief Print a custom std::cout end std::cerr handling verbosity

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Monday September 23, 2019
*/

#include <iostream>
#include <ostream>
#include <iomanip>
#include "GGEMS/global/ggems_export.hh"

#ifdef _WIN32
#ifdef _MSC_VER
#define NOMINMAX
#endif
#include <windows.h>
#endif

// Simple redefinition of std::endl
#define GGEMSendl (static_cast<std::ostream& (*)(std::ostream&)>(std::endl))
#define GGEMScin (std::cin)

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

/*!
  \enum ConsoleColor
  \brief define a color for the console terminal
*/
enum ConsoleColor : unsigned char
{
  black = 0, blue, green, aqua, red, purple, yellow, white, gray
};

#ifdef _WIN32
namespace
{
  WORD const color [] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08
  };
}
#else
namespace
{
  std::string const color[] = {
    "\033[30m", "\033[34m", "\033[32m", "\033[36m", "\033[31m",
    "\033[35m", "\033[33m", "\033[97m", "\033[37m"
  };

  std::string const color_default("\033[0m");
}
#endif

/*!
  \class GGEMSStream
  \brief Generic class redefining standard output
*/
class GGEMS_EXPORT GGEMSStream
{
  public:
    /*!
      \brief GGEMSStream constructor
      \param stream - an output stream
      \param color - define a color on the screen
    */
    GGEMSStream(std::ostream& stream, ConsoleColor const& color);

    /*!
      \brief GGEMSStream destructor
    */
    ~GGEMSStream(){};

  public:
    /*!
      \fn GGEMSStream& operator()(std::string const& class_name, std::string const& method_name, int const& verbosity_level)
      \param class_name - Name of the class to print
      \param method_name - Name of the method to print
      \param verbosity_level - Verbosity level to display
      \brief setting private members to display them to standart output
    */
    GGEMSStream& operator()(std::string const& class_name,
      std::string const& method_name, int const& verbosity_level);

    /*!
      \fn GGEMSStream& operator<<(std::string const& message)
      \param message - message displayed to screen
      \brief overloading << to print a message in standard output
    */
    template<typename T>
    GGEMSStream& operator<<(T const& message);

    /*!
      \fn void SetVerbosity(int const& verbosity_limit)
      \param verbosity_limit - give the verbosity limit applied during execution
      \brief Set the global verbosity
    */
    void SetVerbosity(int const& verbosity_limit);

  private:
    std::string class_name_; /*!< Name of the class to print */
    std::string method_name_; /*!< Name of the method to print */
    int verbosity_limit_; /*! Verbosity limit fixed by user */
    int verbosity_level_; /*!< Verbosity level of the print */
    int stream_counter_; /*!< Counter printing multiple stream */

  protected: // Avoid warnings using clang on Windows system
    std::ostream& stream_; /*!< Stream handling std::cout or std::endl */
    ConsoleColor color_index_; /*!< Color to print on screen */
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
GGEMSStream& GGEMSStream::operator<<(T const& message)
{
  #ifdef _WIN32
  // Get current color of terminal
  CONSOLE_SCREEN_BUFFER_INFO info;
  GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info);
  HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
  FlushConsoleInputBuffer(hConsole);
  #endif

  if (stream_counter_ == 0) {
    if (verbosity_level_ <= verbosity_limit_) {
      stream_ << std::scientific << "[";
      if (!class_name_.empty() && !method_name_.empty()) {
        #ifdef _WIN32
        SetConsoleTextAttribute(hConsole, color[color_index_]);
        stream_ << "GGEMS " << class_name_ << "::" << method_name_;
        SetConsoleTextAttribute(hConsole, info.wAttributes);
        #else
        stream_ << color[color_index_] << "GGEMS " << class_name_ << "::"
          << method_name_ << color_default;
        #endif
      }
      else {
        #ifdef _WIN32
        SetConsoleTextAttribute(hConsole, color[color_index_]);
        stream_ << "GGEMS";
        SetConsoleTextAttribute(hConsole, info.wAttributes);
        #else
        stream_ << color[color_index_] << "GGEMS" << color_default;
        #endif
      }
      stream_ << "](" << verbosity_level_ << ") " << message;
      stream_counter_++;
    }
  }
  else {
    if (verbosity_level_ <= verbosity_limit_) {
      stream_ << message;
    }
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

extern GGEMS_EXPORT GGEMSStream GGEMScout;
extern GGEMS_EXPORT GGEMSStream GGEMScerr;
extern GGEMS_EXPORT GGEMSStream GGEMSwarn;

/*!
  \fn void set_verbose(int verbosity)
  \brief Set the verbosity of output stream
*/
extern "C" GGEMS_EXPORT void set_verbose(int verbosity);

#endif // GUARD_GGEMS_TOOLS_PRINT_HH
