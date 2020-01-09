/*!
  \file GGEMSPhantomCreatorManager.cc

  \brief Singleton class generating voxelized phantom from analytical volume

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thursday January 9, 2020
*/

#include "GGEMS/geometry/GGEMSPhantomCreatorManager.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhantomCreatorManager::GGEMSPhantomCreatorManager(void)
{
  GGcout("GGEMSPhantomCreatorManager", "GGEMSPhantomCreatorManager", 3)
    << "Allocation of Phantom Creator Manager singleton..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhantomCreatorManager::~GGEMSPhantomCreatorManager(void)
{
  GGcout("GGEMSPhantomCreatorManager", "~GGEMSPhantomCreatorManager", 3)
    << "Deallocation of Phantom Creator Manager singleton..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantomCreatorManager::SetElementSizes(GGdouble const& voxel_width,
  GGdouble const& voxel_height, GGdouble const& voxel_depth)
{
  ;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantomCreatorManager::SetPhantomDimensions(GGuint const& phantom_width,
  GGuint const& phantom_height, GGuint const& phantom_depth)
{
  ;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantomCreatorManager::SetOutputBasename(
  std::string const& output_MHD_basename)
{
  ;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhantomCreatorManager* get_instance_phantom_creator_manager(void)
{
  return &GGEMSPhantomCreatorManager::GetInstance();
}
