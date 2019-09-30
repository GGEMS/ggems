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

// Simple redefinition of std::endl
#define GGEMSendl (static_cast<std::ostream& (*)(std::ostream&)>(std::endl))
#define GGEMScin (std::cin)

#ifdef _WIN32
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
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
    */
    explicit GGEMSStream(std::ostream& stream);

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
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
GGEMSStream& GGEMSStream::operator<<(T const& message)
{
  if (stream_counter_ == 0) {
    if (verbosity_level_ <= verbosity_limit_) {
      if (!class_name_.empty() && !method_name_.empty()) {
        stream_ << std::fixed << std::setprecision(20) << "[GGEMS "
          << class_name_ << "::" << method_name_ << "](" << verbosity_level_
          << ") " << message;
      }
      else {
        stream_ << "[GGEMS](" << verbosity_level_ << ") " << message;
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

/*!
  \fn void set_verbose(int verbosity)
  \brief Set the verbosity of output stream
*/
extern "C" GGEMS_EXPORT void set_verbose(int verbosity);

#endif // GUARD_GGEMS_TOOLS_PRINT_HH
