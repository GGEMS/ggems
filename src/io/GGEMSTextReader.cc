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
  \file GGEMSTextReader.cc

  \brief Namespaces for different useful fonctions reading input text file.
  Namespaces for material database file, MHD file and other files

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Tuesday February 4, 2020
*/

#include <string>

#include "GGEMS/io/GGEMSTextReader.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

std::string GGEMSMaterialReader::ReadMaterialName(std::string const& line)
{
  return line.substr(line.find_first_not_of("\t "), line.find(":"));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSMaterialReader::ReadMaterialDensity(std::string const& line)
{
  // Get the position of the first and last number of density
  GGsize first_pos = line.find_first_of("0123456789", line.find("d="));
  GGsize last_pos = line.find_first_not_of("0123456789.", first_pos);
  std::string density_str = line.substr(first_pos, last_pos != std::string::npos ? last_pos - first_pos : last_pos);

  // Convert string to float
  GGfloat density = 0.0f;
  std::stringstream(density_str) >> density;

  // Check units of density and convert
  first_pos = last_pos;
  last_pos = line.find_first_of(";");
  std::string unit_str = line.substr(first_pos, last_pos != std::string::npos ? last_pos - first_pos : last_pos);

  if (unit_str == "g/cm3") {
    density *= g/cm3;
  }
  else if (unit_str == "mg/cm3") {
    density *= mg/cm3;
  }
  else {
    GGEMSMisc::ThrowException("GGEMSMaterialReader", "ReadMaterialDensity", "Unknown density unit in material database file!!!");
  }

  return density;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGsize GGEMSMaterialReader::ReadMaterialNumberOfElements(std::string const& line)
{
  // Get the position of the first and last number of density
  GGsize first_pos = line.find_first_of("0123456789", line.find("n="));
  GGsize last_pos = line.find_last_of(";");
  std::string element_str = line.substr(first_pos,last_pos != std::string::npos ? last_pos - first_pos : last_pos);

  // Convert string to unsigned char
  GGsize number_elements = 0;
  std::stringstream(element_str) >> number_elements;

  return number_elements;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

std::string GGEMSMaterialReader::ReadMaterialElementName(std::string const& line)
{
  GGsize first_pos = line.find("name=")+5;
  GGsize last_pos = line.find_first_of(";");
  std::string element_name_str = line.substr(first_pos, last_pos != std::string::npos ? last_pos - first_pos : last_pos);

  return element_name_str;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSMaterialReader::ReadMaterialElementFraction(std::string const& line)
{
  GGsize first_pos = line.find_first_of("0123456789", line.find("f="));
  GGsize last_pos = line.find_last_of(";");
  std::string fraction_str = line.substr(first_pos, last_pos != std::string::npos ? last_pos - first_pos : last_pos);

  // Convert string to float
  GGfloat fraction = 0.0f;
  std::stringstream(fraction_str) >> fraction;

  return fraction;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSTextReader::SkipComment(std::ifstream& stream, std::string& line, char const comment)
{
  // If first caracter = comment -> it's a comment and get the next line
  while (1) {
    std::size_t pos = line.find_first_not_of("\t ");
    if (pos == std::string::npos) break;
    else if (line[pos] == comment) std::getline(stream, line);
    else break;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

bool GGEMSTextReader::IsBlankLine(std::string const& line)
{
  if (line.find_first_not_of("\t\n ") == std::string::npos) return true;
  else return false;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSTextReader::RemoveSpace(std::string& line)
{
  // Erasing tab
  line.erase(std::remove(line.begin(), line.end(), '\t'), line.end());
  // Erasing space
  line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
  // Erasing eof
  line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
  // Erasing carriage return
  line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
  // Erasing form feed
  line.erase(std::remove(line.begin(), line.end(), '\f'), line.end());
  // Erasing vertical tab
  line.erase(std::remove(line.begin(), line.end(), '\v'), line.end());
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

std::string GGEMSMHDReader::ReadKey(std::string& line)
{
  // Getting the key
  std::string key = line.substr(line.find_first_not_of("\t "), line.find('=')-1);
  
  // Remove space/tab
  GGEMSTextReader::RemoveSpace(key);

  // Return the key
  return key;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

std::istringstream GGEMSMHDReader::ReadValue(std::string& line)
{
  std::istringstream iss(line.substr(line.find("=")+1), std::istringstream::in);
  return iss;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

std::istringstream GGEMSRangeReader::ReadRangeMaterial(std::string& line)
{
  std::istringstream iss(line, std::istringstream::in);
  return iss;
}
