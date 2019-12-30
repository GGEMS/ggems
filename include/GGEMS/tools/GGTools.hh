#ifndef GUARD_GGEMS_TOOLS_GGTOOLS_HH
#define GUARD_GGEMS_TOOLS_GGTOOLS_HH

/*!
  \file GGTools.hh

  \brief Namespaces for different useful fonctions

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Monday September 30, 2019
*/

#include <fstream>

#include "GGEMS/global/GGConfiguration.hh"
#include "GGEMS/tools/GGTypes.hh"

/*!
  \namespace GGStream
  \brief namespace checking file stream in input and output
*/
namespace GGStream
{
  /*!
    \fn void CheckInputStream(std::ifstream const& input_stream, std::string const& filename)
    \param input_stream - the input stream
    \param filename - filename associated to the input stream
    \brief check the input stream during the opening
  */
  void CheckInputStream(std::ifstream const& input_stream,
    std::string const& filename);
}

/*!
  \namespace GGMisc
  \brief namespace storing miscellaneous functions
*/
namespace GGMisc
{
  /*!
    \fn bool IsEqual(T const& a, T const& b)
    \tparam T - float or double number
    \param a - first value
    \param b - second value
    \brief Check if 2 floats/doubles are equal (or almost equal)
  */
  template<typename T>
  GGbool IsEqual(T const& a, T const& b);

  /*!
    \fn void ThrowException(std::string const& class_name, std::string const& method_name, std::string const& message)
    \param class_name - Name of the class
    \param method_name - Name of the methode or function
    \param message - Message to print for the exception
    \brief Throw a C++ exception
  */
  void ThrowException(std::string const& class_name, std::string const&
    method_name, std::string const& message);
}

/*!
  \namespace GGEMSTools
  \brief namespace storing GGEMS tool functions
*/
namespace GGEMSTools
{
  /*!
    \fn void PrintBanner()
    \brief Print GGEMS banner
  */
  void PrintBanner();
}

#endif // End of GUARD_GGEMS_TOOLS_GGTOOLS_HH