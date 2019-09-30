#ifndef GUARD_GGEMS_TOOLS_FUNCTIONS_HH
#define GUARD_GGEMS_TOOLS_FUNCTIONS_HH

/*!
  \file functions.hh

  \brief Namespaces for different useful fonctions

  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Wednesday October 10, 2018
*/

#include <fstream>
#include "GGEMS/global/opencl_manager.hh"
#include "GGEMS/global/ggems_configuration.hh"

/*!
  \namespace Stream
  \brief namespace checking file stream in input and output
*/
namespace Stream
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
  \namespace Misc
  \brief namespace storing miscellaneous functions
*/
namespace Misc
{
  /*!
    \fn bool IsEqual(T const& a, T const& b)
    \tparam T - float or double number
    \param a - first value
    \param b - second value
    \brief Check if 2 floats/doubles are equal (or almost equal)
  */
  template<typename T>
  bool IsEqual(T const& a, T const& b);

  /*!
    \fn void ThrowException(std::string const& class_name, std::string const&method_name, std::string const& message)
  */
  void ThrowException(std::string const& class_name, std::string const&
    method_name, std::string const& message);
}

#endif // End of GUARD_GGEMS_TOOLS_FUNCTIONS_HH