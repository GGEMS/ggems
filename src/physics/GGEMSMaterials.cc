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
    GGEMSTextReader::SkipComment(database_stream, line);
    //GGEMSTextReader::SkipBlankLine(database_stream, line);
    std::cout << line << std::endl;
  }

  // Closing file stream
  database_stream.close();
}
