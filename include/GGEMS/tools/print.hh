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
#include <Windows.h>
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
enum class ConsoleColor {red, yellow, green};

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
    std::ostream& stream_; /*!< Stream handling std::cout or std::endl */
    ConsoleColor color_; /*!< Color to print on screen */
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
      if (!class_name_.empty() && !method_name_.empty()) {
        stream_ << std::scientific << "[";
          if (color_ == ConsoleColor::red) {
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, 4);
            stream_ << "GGEMS " << class_name_ << "::" << method_name_;
            SetConsoleTextAttribute(hConsole, info.wAttributes);
            #else
            stream_ << "\033[31mGGEMS " << class_name_ << "::" << method_name_
              << "\033[0m";
            #endif
          }
          else if (color_ == ConsoleColor::green) {
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, 2);
            stream_ << "GGEMS " << class_name_ << "::" << method_name_;
            SetConsoleTextAttribute(hConsole, info.wAttributes);
            #else
            stream_ << "\033[32mGGEMS " << class_name_ << "::" << method_name_
              << "\033[0m";
            #endif
          }
          else if (color_ == ConsoleColor::yellow) {
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, 6);
            stream_ << "GGEMS " << class_name_ << "::" << method_name_;
            SetConsoleTextAttribute(hConsole, info.wAttributes);
            #else
            stream_ << "\033[33mGGEMS " << class_name_ << "::" << method_name_;
              << "\033[0m";
            #endif
          }
          stream_ << "](" << verbosity_level_ << ") " << message;
      }
      else {
        stream_ << "[";
          if (color_ == ConsoleColor::red) {
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, 4);
            stream_ << "GGEMS";
            SetConsoleTextAttribute(hConsole, info.wAttributes);
            #else
            stream_ << "\033[31mGGEMS\033[0m";
            #endif
          }
          else if (color_ == ConsoleColor::green) {
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, 2);
            stream_ << "GGEMS";
            SetConsoleTextAttribute(hConsole, info.wAttributes);
            #else
            stream_ << "\033[32mGGEMS\033[0m";
            #endif
          }
          else if (color_ == ConsoleColor::yellow) {
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, 6);
            stream_ << "GGEMS";
            SetConsoleTextAttribute(hConsole, info.wAttributes);
            #else
            stream_ << "\033[33mGGEMS\033[0m";
            #endif
          }
        stream_ << "](" << verbosity_level_ << ") " << message;
      }
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
