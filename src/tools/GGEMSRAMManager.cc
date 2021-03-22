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
  \file GGEMSRAMManager.cc

  \brief GGEMS class handling RAM memory

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday May 5, 2020
*/

#include "GGEMS/global/GGEMSOpenCLManager.hh"

#include "GGEMS/tools/GGEMSRAMManager.hh"
#include "GGEMS/tools/GGEMSTools.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSRAMManager::GGEMSRAMManager(void)
: allocated_ram_(0)
{
  GGcout("GGEMSRAMManager", "GGEMSRAMManager", 3) << "Allocation of GGEMS RAM Manager..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  max_available_ram_ = opencl_manager.GetMaxRAMMemoryOnActivatedContext();
  max_buffer_size_ = opencl_manager.GetMaxBufferAllocationSize();

  allocated_memories_.clear();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSRAMManager::~GGEMSRAMManager(void)
{
  GGcout("GGEMSRAMManager", "~GGEMSRAMManager", 3) << "Deallocation of GGEMS RAM Manager..." << GGendl;

  allocated_memories_.clear();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRAMManager::IncrementRAMMemory(std::string const& class_name, GGsize const& size)
{
  // Checking if class has already allocated memory, if not, creating one
  if (allocated_memories_.find(class_name) == allocated_memories_.end()) {
    allocated_memories_.insert(std::make_pair(class_name, size));
  }
  else {
    allocated_memories_[class_name] += size;
  }

  // Increment size
  allocated_ram_ += size;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRAMManager::DecrementRAMMemory(GGsize const& size)
{
  // Increment size
  allocated_ram_ -= size;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRAMManager::PrintRAMStatus(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get the name of the activated context
  std::string device_name = opencl_manager.GetNameOfActivatedDevice();

  // Compute allocated percent RAM
  GGfloat percent_allocated_RAM = static_cast<GGfloat>(allocated_ram_) * 100.0f / static_cast<GGfloat>(max_available_ram_);

  GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "Device: " << device_name << GGendl;
  GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "Total RAM memory allocated on OpenCL device: " << allocated_ram_ << " / " << max_available_ram_ << " bytes (" << percent_allocated_RAM << "%)" << GGendl;
  GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "Details: " << GGendl;
  for (auto&& i : allocated_memories_) {
    GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "    + In '" << i.first << "': " << i.second << " bytes allocated (" << static_cast<GGfloat>(i.second) * 100.0f /  static_cast<GGfloat>(allocated_ram_) << "%)" << GGendl;
  }
}
