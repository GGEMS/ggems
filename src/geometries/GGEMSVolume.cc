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
  \file GGEMSVolume.cc

  \brief Mother class handle volume

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday January 13, 2020
*/

#include "GGEMS/geometries/GGEMSVolume.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVolume::GGEMSVolume(void)
: label_value_(1.0f),
  positions_(GGfloat3{{0.0f, 0.0f, 0.0f}})
{
  GGcout("GGEMSVolume", "GGEMSVolume", 3) << "GGEMSVolume creating..." << GGendl;

  // Getting number of activated device. But only 1 device is used to create volume
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGsize number_activated_devices = opencl_manager.GetNumberOfActivatedDevice();

  // Storing a kernel for 1 device
  kernel_draw_volume_ = new cl::Kernel*[number_activated_devices];

  GGcout("GGEMSVolume", "GGEMSVolume", 3) << "GGEMSVolume created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVolume::~GGEMSVolume(void)
{
  GGcout("GGEMSVolume", "~GGEMSVolume", 3) << "GGEMSVolume erasing..." << GGendl;

  if (kernel_draw_volume_) {
    delete[] kernel_draw_volume_;
    kernel_draw_volume_ = nullptr;
  }

  GGcout("GGEMSVolume", "~GGEMSVolume", 3) << "GGEMSVolume erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVolume::SetLabelValue(GGfloat const& label_value)
{
  label_value_ = label_value;
}

void GGEMSVolume::SetMaterial(std::string const& material)
{
  // Get the volume creator manager
  GGEMSVolumeCreatorManager& volume_creator_manager = GGEMSVolumeCreatorManager::GetInstance();

  // Adding the material to phantom creator manager
  volume_creator_manager.AddLabelAndMaterial(label_value_, material);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVolume::SetPosition(GGfloat const& pos_x, GGfloat const& pos_y, GGfloat const& pos_z, std::string const& unit)
{
  positions_.s[0] = DistanceUnit(pos_x, unit);
  positions_.s[1] = DistanceUnit(pos_y, unit);
  positions_.s[2] = DistanceUnit(pos_z, unit);
}
