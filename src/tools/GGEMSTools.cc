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
  \file GGEMSTools.cc

  \brief Namespaces for different useful fonctions

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Wednesday October 10, 2018
*/

#include <sstream>
#include <cerrno>
#include <cstring>

#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

#include "GGEMS/global/GGEMSManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSFileStream::CheckInputStream(std::ifstream const& input_stream, std::string const& filename)
{
  if (!input_stream) {
    std::ostringstream oss(std::ostringstream::out);
    #ifdef _WIN32
    char buffer_error[ 256 ];
    oss << "Problem reading filename '" << filename << "': " << strerror_s(buffer_error, 256, errno);
    #else
    oss << "Problem reading filename '" << filename << "': " << strerror(errno);
    #endif
    throw std::runtime_error(oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMisc::ThrowException(std::string const& class_name, std::string const& method_name, std::string const& message)
{
  std::ostringstream oss(std::ostringstream::out);
  oss << message;
  GGcerr(class_name, method_name, 0) << oss.str() << GGendl;
  throw std::runtime_error("");
}
