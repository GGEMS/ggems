#ifndef GUARD_GGEMS_IO_GGEMSTEXTREADER_HH
#define GUARD_GGEMS_IO_GGEMSTEXTREADER_HH

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

#include <fstream>

/*!
  \namespace GGEMSMaterialTextReader
  \brief namespace reading material database file
*/
/*namespace GGEMSMaterialTextReader
{
  
}*/

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
  void SkipComment(std::ifstream& stream, std::string& line,
    char const comment = '#');

  /*!
    \fn void SkipBlankLine
  */
  //void SkipBlankLine(std::ifstream& stream, std::string& line);
}

#endif // End of GUARD_GGEMS_IO_GGEMSTEXTREADER_HH
