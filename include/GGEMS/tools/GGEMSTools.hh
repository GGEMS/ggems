#ifndef GUARD_GGEMS_TOOLS_GGEMSTOOLS_HH
#define GUARD_GGEMS_TOOLS_GGEMSTOOLS_HH

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
  \file GGEMSTools.hh

  \brief Namespaces for different useful fonctions

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Monday September 30, 2019
*/

#include <fstream>
#include <cmath>

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
  void CheckInputStream(std::ifstream const& input_stream, std::string const& filename);
}

/*!
  \namespace GGEMSMisc
  \brief namespace storing miscellaneous functions
*/
namespace GGEMSMisc
{
  /*!
    \fn void ThrowException(std::string const& class_name, std::string const& method_name, std::string const& message)
    \param class_name - Name of the class
    \param method_name - Name of the methode or function
    \param message - Message to print for the exception
    \brief Throw a C++ exception
  */
  [[noreturn]] void ThrowException(std::string const& class_name, std::string const& method_name, std::string const& message);
}

#endif // End of GUARD_GGEMS_TOOLS_GGEMSTOOLS_HH
