/*!
  \file GGEMSMaterial.cc

  \brief GGEMS class managing the complete material database and a specific
  material

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday February 3, 2020
*/

#include "GGEMS/io/GGEMSTextReader.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/physics/GGEMSMaterials.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSingleMaterial::GGEMSSingleMaterial(void)
{
  GGcout("GGEMSSingleMaterial", "GGEMSSingleMaterial", 3)
    << "Allocation of GGEMS Material..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSingleMaterial::~GGEMSSingleMaterial(void)
{
  GGcout("GGEMSSingleMaterial", "GGEMSSingleMaterial", 3)
    << "Deallocation of GGEMS Material..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMaterialsDatabase::GGEMSMaterialsDatabase(void)
{
  GGcout("GGEMSMaterialsDatabase", "GGEMSMaterialsDatabase", 3)
    << "Allocation of GGEMS Materials database..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMaterialsDatabase::~GGEMSMaterialsDatabase(void)
{
  GGcout("GGEMSMaterialsDatabase", "~GGEMSMaterialsDatabase", 3)
    << "Deallocation of GGEMS Materials database..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterialsDatabase::LoadMaterialsDatabase(std::string const& filename)
{
  GGcout("GGEMSMaterialsDatabase", "LoadMaterialsDatabase", 0)
    << "Loading materials database in GGEMS..." << GGendl;

  // Opening the input file containing materials
  std::ifstream database_stream(filename, std::ios::in);
  GGEMSFileStream::CheckInputStream(database_stream, filename);

  // Reading database file
  std::string line;
  while (std::getline(database_stream, line)) {
    // Skip comment
    GGEMSTextReader::SkipComment(database_stream, line);
    // Check if blank line
    if (GGEMSTextReader::IsBlankLine(line)) continue;

    // Remove space/tab from line
    GGEMSTextReader::RemoveSpace(line);

    // Store infos about materials
    std::cout << GGEMSMaterialReader::ReadMaterialName(line) << std::endl;
    std::cout << GGEMSMaterialReader::ReadMaterialDensity(line) << std::endl;

    // Loop over number of elements
    GGushort const kNumberOfElements =
      GGEMSMaterialReader::ReadMaterialNumberOfElements(line);
    for (GGushort i = 0; i < kNumberOfElements; ++i) {
      // Get next line element by element
      std::getline(database_stream, line);
      // Remove space/tab from line
      GGEMSTextReader::RemoveSpace(line);
      std::cout << GGEMSMaterialReader::ReadMaterialElementName(line)
        << " " << GGEMSMaterialReader::ReadMaterialElementFraction(line)
        << std::endl;
    }
  }

  // Closing file stream
  database_stream.close();
}
