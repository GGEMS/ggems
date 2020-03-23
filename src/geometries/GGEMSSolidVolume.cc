/*!
  \file GGEMSSolidVolume.cc

  \brief Mother class handle solid volume

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday January 13, 2020
*/

#include "GGEMS/geometries/GGEMSSolidVolume.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolidVolume::GGEMSSolidVolume(void)
: label_value_(1.0),
  positions_(GGfloat3{{0.0, 0.0, 0.0}}),
  kernel_draw_solid_(nullptr),
  opencl_manager_(GGEMSOpenCLManager::GetInstance()),
  phantom_creator_manager_(GGEMSPhantomCreatorManager::GetInstance())
{
  GGcout("GGEMSSolidVolume", "GGEMSSolidVolume", 3) << "Allocation of GGEMSSolidVolume..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolidVolume::~GGEMSSolidVolume(void)
{
  GGcout("GGEMSSolidVolume", "~GGEMSSolidVolume", 3) << "Deallocation of GGEMSSolidVolume..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolidVolume::SetLabelValue(GGfloat const& label_value)
{
  label_value_ = label_value;
}

void GGEMSSolidVolume::SetMaterial(char const* material)
{
  // Adding the material to phantom creator manager
  phantom_creator_manager_.AddLabelAndMaterial(label_value_, material);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolidVolume::SetPosition(GGfloat const& pos_x, GGfloat const& pos_y, GGfloat const& pos_z, char const* unit)
{
  positions_.s[0] = GGEMSUnits::DistanceUnit(pos_x, unit);
  positions_.s[1] = GGEMSUnits::DistanceUnit(pos_y, unit);
  positions_.s[2] = GGEMSUnits::DistanceUnit(pos_z, unit);
}
