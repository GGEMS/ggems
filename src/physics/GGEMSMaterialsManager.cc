/*!
  \file GGEMSMaterialsManager.cc

  \brief GGEMS singleton class managing the material database

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday January 23, 2020
*/

#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/physics/GGEMSMaterialsManager.hh"
#include "GGEMS/physics/GGEMSMaterials.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMaterialsManager::GGEMSMaterialsManager(void)
: is_database_loaded_(false)
{
  GGcout("GGEMSMaterialsManager", "GGEMSMaterialsManager", 3)
    << "Allocation of GGEMS Materials singleton..." << GGendl;

  // Creating a material database
  p_material_database_ = new GGEMSMaterialsDatabase;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMaterialsManager::~GGEMSMaterialsManager(void)
{
  // Deleting the material database
  if (p_material_database_) {
    delete p_material_database_;
    p_material_database_ = nullptr;
  }

  GGcout("GGEMSMaterialsManager", "~GGEMSMaterialsManager", 3)
    << "Deallocation of GGEMS Materials singleton..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterialsManager::SetMaterialsDatabase(char const* filename)
{
  // Converting char* to string
  std::string filename_str(filename);

  // Loading materials and elements in database
  if (is_database_loaded_) {
    GGwarn("GGEMSMaterialsManager", "SetMaterialsDatabase", 0)
      << "Material database if already loaded!!!" << GGendl;
  }
  else {
    // Materials
    p_material_database_->LoadMaterialsDatabase(filename_str);
    // Chemical Elements
    p_material_database_->LoadChemicalElements();
  }

  // Database if loaded
  is_database_loaded_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMaterialsManager* get_instance_materials_manager(void)
{
  return &GGEMSMaterialsManager::GetInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_materials_database_ggems_materials_manager(
  GGEMSMaterialsManager* p_ggems_materials_manager, char const* filename)
{
  p_ggems_materials_manager->SetMaterialsDatabase(filename);
}
