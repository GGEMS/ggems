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
#include <iostream>
#include "GGEMS/io/GGEMSTextReader.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSTextReader::SkipComment(std::ifstream& stream, std::string& line,
  char const comment)
{
  // If first caracter = comment -> it's a comment and get the next line
  if (line[line.find_first_not_of(' ')] == comment) std::getline(stream, line);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*void GGEMSTextReader::SkipBlankLine(std::ifstream& stream, std::string& line)
{
  // Skip blank or empty line
  if (line.find_first_not_of("\t\n ") == std::string::npos)
    std::getline(stream, line);
}*/
