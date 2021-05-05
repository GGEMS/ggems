#ifndef GUARD_GGEMS_TOOLS_GGEMSPRINT_HH
#define GUARD_GGEMS_TOOLS_GGEMSPRINT_HH

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
  \file GGEMSPrint.hh

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
#include <mutex>

namespace {
  std::mutex mutex;
}

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

#ifdef _WIN32
#ifdef _MSC_VER
#define NOMINMAX
#endif
#include <windows.h>
#endif

/*!
  \def GGendl
  \brief overload C++ std::endl
*/
#define GGendl (static_cast<std::ostream& (*)(std::ostream&)>(std::endl))

/*!
  \def GGcin
  \brief overload C++ std::cin
*/
#define GGcin (std::cin)

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

/*!
  \enum GGEMSConsoleColor
  \brief define a color for the console terminal
*/
enum GGEMSConsoleColor : GGuchar
{
  black = 0,
  blue,
  green,
  aqua,
  red,
  purple,
  yellow,
  white,
  gray
};

/*!
  \brief namespace storing color code
*/
namespace
{
  #ifdef _WIN32
  WORD constexpr kColor [] = {
    0x00,
    0x01,
    0x02,
    0x03,
    0x04,
    0x05,
    0x06,
    0x07,
    0x08
  }; /*!< List of color for Windows */
  #else
  std::string const kColor[] = {
    "\033[30m",
    "\033[34m",
    "\033[32m",
    "\033[36m",
    "\033[31m",
    "\033[35m",
    "\033[33m",
    "\033[97m",
    "\033[37m"
  }; /*!< List of color for Unix */
  std::string const kDefaultColor("\033[0m"); /*!< Default color for Unix */
  #endif
}

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
    GGEMSStream(std::ostream& stream, GGEMSConsoleColor const& color);

    /*!
      \brief GGStream destructor
    */
    ~GGEMSStream(){};

  public:
    /*!
      \fn GGEMSStream& operator()(std::string const& class_name, std::string const& method_name, GGint const& verbosity_level)
      \param class_name - Name of the class to print
      \param method_name - Name of the method to print
      \param verbosity_level - Verbosity level to display
      \return sstream of message
      \brief setting private members to display them to standart output
    */
    GGEMSStream& operator()(std::string const& class_name, std::string const& method_name, GGint const& verbosity_level);

    /*!
      \fn template<typename T> GGEMSStream& operator<<(T const& message)
      \tparam T - type of data
      \param message - message displayed to screen
      \return stream of message
      \brief overloading << to print a message in standard output
    */
    template<typename T>
    GGEMSStream& operator<<(T const& message);

    /*!
      \fn void SetVerbosity(GGint const& verbosity_limit)
      \param verbosity_limit - give the verbosity limit applied during execution
      \brief Set the global verbosity
    */
    void SetVerbosity(GGint const& verbosity_limit);

  private:
    std::string class_name_; /*!< Name of the class to print */
    std::string method_name_; /*!< Name of the method to print */
    GGint verbosity_limit_; /*!< Verbosity limit fixed by user */
    GGint verbosity_level_; /*!< Verbosity level of the print */
    GGint stream_counter_; /*!< Counter printing multiple stream */

  protected: // Avoid warnings using clang on Windows system
    std::ostream& stream_; /*!< Stream handling std::cout or std::endl */
    GGEMSConsoleColor color_index_; /*!< Color to print on screen */
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
        SetConsoleTextAttribute(hConsole, ::kColor[color_index_]);
        stream_ << "GGEMS " << class_name_ << "::" << method_name_;
        SetConsoleTextAttribute(hConsole, info.wAttributes);
        #else
        stream_ << kColor[color_index_] << "GGEMS " << class_name_ << "::" << method_name_ << kDefaultColor;
        #endif
      }
      else {
        #ifdef _WIN32
        SetConsoleTextAttribute(hConsole, ::kColor[color_index_]);
        stream_ << "GGEMS";
        SetConsoleTextAttribute(hConsole, info.wAttributes);
        #else
        stream_ << kColor[color_index_] << "GGEMS" << kDefaultColor;
        #endif
      }
      stream_ << "](" << verbosity_level_ << ") " << message;
      stream_counter_++;
    }
  }
  else {
    if (verbosity_level_ <= verbosity_limit_) stream_ << message;
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

extern GGEMS_EXPORT GGEMSStream GGcout; /*!< Define a new std::cout with green color */
extern GGEMS_EXPORT GGEMSStream GGcerr; /*!< Define a new std::cerr with green red */
extern GGEMS_EXPORT GGEMSStream GGwarn; /*!< Define a new std::cout for warning with orange color */

/*!
  \fn void set_ggems_verbose(GGint verbosity)
  \param verbosity - level of verbosity
  \brief Set the verbosity of output stream
*/
extern "C" GGEMS_EXPORT void set_ggems_verbose(GGint verbosity);

#endif // GUARD_GGEMS_TOOLS_GGPRINT_HH
