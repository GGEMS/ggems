#ifndef GUARD_GGEMS_IO_GGEMSTEXTREADER_HH
#define GUARD_GGEMS_IO_GGEMSTEXTREADER_HH

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
  \file GGEMSTextReader.hh

  \brief Namespaces for different useful fonctions reading input text file.
  Namespaces for material database file, MHD file and other files

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Tuesday February 4, 2020
*/

/// \cond
#include <fstream>
#include <sstream>
/// \endcond

#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \namespace GGEMSMaterialTextReader
  \brief namespace reading material database file
*/
namespace GGEMSMaterialReader
{
  /*!
    \fn std::string ReadMaterialName(std::string const& line)
    \param line - string to analyze
    \return name of the material
    \brief return the name of the material
  */
  std::string ReadMaterialName(std::string const& line);

  /*!
    \fn GGfloat ReadMaterialDensity(std::string const& line)
    \param line - string to analyze
    \return density of material
    \brief return the density of material
  */
  GGfloat ReadMaterialDensity(std::string const& line);

  /*!
    \fn GGsize ReadMaterialNumberOfElements(std::string const& line)
    \param line - string to analyze
    \return number of elements in material
    \brief return the number of elements in material
  */
  GGsize ReadMaterialNumberOfElements(std::string const& line);

  /*!
    \fn std::string ReadMaterialElementName(std::string const& line)
    \param line - string to analyze
    \return name of the material element
    \brief return the name of material element name
  */
  std::string ReadMaterialElementName(std::string const& line);

  /*!
    \fn GGfloat ReadMaterialElementFraction(std::string const& line)
    \param line - string to analyze
    \return number of elements fraction
    \brief return the number of element fraction
  */
  GGfloat ReadMaterialElementFraction(std::string const& line);
}

/*!
  \namespace GGEMSTextReader
  \brief namespace reading common text file
*/
namespace GGEMSTextReader
{
  /*!
    \fn void SkipComment(std::ifstream& stream, std::string& line, char const comment = '#')
    \param stream - stream of a file
    \param line - line containing the string of the line
    \param comment - special comment caracter to skip
    \brief skip a special line beginning by a comment caracter
  */
  void SkipComment(std::ifstream& stream, std::string& line, char const comment = '#');

  /*!
    \fn bool IsBlankLine(std::string const& line)
    \param line - string to analyze
    \return true is blank line otherwize false
    \brief check if the line is blank or not
  */
  bool IsBlankLine(std::string const& line);

  /*!
    \fn void RemoveSpace(std::string& line)
    \param line - string to analyze
    \brief remove all spaces and tab from a string
  */
  void RemoveSpace(std::string& line);
}

/*!
  \namespace GGEMSMHDReader
  \brief namespace reading mhd header
*/
namespace GGEMSMHDReader
{
  /*!
    \fn std::string ReadKey(std::string& line)
    \param line - string to analyze
    \return the key of the MHD reader
    \brief get the key of MHD header
  */
  std::string ReadKey(std::string& line);

  /*!
    \fn std::istringstream ReadValue(std::string& line)
    \param line - string to analyze
    \return line in stream
    \brief get string stream of value
  */
  std::istringstream ReadValue(std::string& line);
}

/*!
  \namespace GGEMSRangeReader
  \brief namespace reading range material in text file
*/
namespace GGEMSRangeReader
{
  /*!
    \fn std::istringstream ReadRangeMaterial(std::string& line)
    \param line - string to analyze
    \return the string stream of value for material range
    \brief get string stream of value for material range
  */
  std::istringstream ReadRangeMaterial(std::string& line);
}

#endif // End of GUARD_GGEMS_IO_GGEMSTEXTREADER_HH
