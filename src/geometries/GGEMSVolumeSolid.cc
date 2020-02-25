/*!
  \file GGEMSVolumeSolid.cc

  \brief Mother class handle solid volume

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday January 13, 2020
*/

#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/geometries/GGEMSVolumeSolid.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVolumeSolid::GGEMSVolumeSolid(void)
: label_value_(1.0),
  positions_(GGdouble3{{0.0, 0.0, 0.0}}),
  kernel_draw_solid_(nullptr),
  opencl_manager_(GGEMSOpenCLManager::GetInstance()),
  phantom_creator_manager_(GGEMSPhantomCreatorManager::GetInstance())
{
  GGcout("GGEMSVolumeSolid", "GGEMSVolumeSolid", 3) << "Allocation of GGEMSVolumeSolid..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVolumeSolid::~GGEMSVolumeSolid(void)
{
  GGcout("GGEMSVolumeSolid", "~GGEMSVolumeSolid", 3) << "Deallocation of GGEMSVolumeSolid..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVolumeSolid::SetLabelValue(GGfloat const& label_value)
{
  label_value_ = label_value;
}

void GGEMSVolumeSolid::SetMaterial(char const* material)
{
  // Adding the material to phantom creator manager
  phantom_creator_manager_.AddLabelAndMaterial(label_value_, material);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVolumeSolid::SetPosition(GGdouble const& pos_x, GGdouble const& pos_y, GGdouble const& pos_z, char const* unit)
{
  positions_.s[0] = GGEMSUnits::BestDistanceUnit(pos_x, unit);
  positions_.s[1] = GGEMSUnits::BestDistanceUnit(pos_y, unit);
  positions_.s[2] = GGEMSUnits::BestDistanceUnit(pos_z, unit);
}
