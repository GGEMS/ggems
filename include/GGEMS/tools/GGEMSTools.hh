#ifndef GUARD_GGEMS_TOOLS_GGEMSTOOLS_HH
#define GUARD_GGEMS_TOOLS_GGEMSTOOLS_HH

/*!
  \file GGEMSTools.hh

  \brief Namespaces for different useful fonctions

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Monday September 30, 2019
*/

#include <fstream>

#include "GGEMS/global/GGEMSConfiguration.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \namespace GGEMSFileStream
  \brief namespace checking file stream in input and output
*/
namespace GGEMSFileStream
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
  \namespace GGEMSMisc
  \brief namespace storing miscellaneous functions
*/
namespace GGEMSMisc
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

#endif // End of GUARD_GGEMS_TOOLS_GGEMSTOOLS_HH
