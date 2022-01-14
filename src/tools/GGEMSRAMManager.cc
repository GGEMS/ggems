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
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSRAMManager::GGEMSRAMManager(void)
{
  GGcout("GGEMSRAMManager", "GGEMSRAMManager", 3) << "GGEMSRAMManager creating..." << GGendl;

  // Get the OpenCL manager and number of detected device
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  number_detected_devices_ = opencl_manager.GetNumberOfDetectedDevice();

  // Creating buffer for each detected device
  allocated_ram_ = new GGsize[number_detected_devices_];
  std::fill(allocated_ram_, allocated_ram_+number_detected_devices_, 0);

  max_available_ram_ = new GGsize[number_detected_devices_];
  max_buffer_size_ = new GGsize[number_detected_devices_];
  allocated_memories_ = new AllocatedMemoryUMap[number_detected_devices_];

  for (GGsize i = 0; i < number_detected_devices_; ++i) {
    max_available_ram_[i] = opencl_manager.GetRAMMemory(i);
    max_buffer_size_[i] = opencl_manager.GetMaxBufferAllocationSize(i);
    allocated_memories_[i].clear();
  }

  GGcout("GGEMSRAMManager", "GGEMSRAMManager", 3) << "GGEMSRAMManager created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSRAMManager::~GGEMSRAMManager(void)
{
  GGcout("GGEMSRAMManager", "~GGEMSRAMManager", 3) << "GGEMSRAMManager erasing..." << GGendl;

  if (allocated_ram_) {
    delete allocated_ram_;
    allocated_ram_ = nullptr;
  }

  if (max_available_ram_) {
    delete max_available_ram_;
    max_available_ram_ = nullptr;
  }

  if (max_buffer_size_) {
    delete max_buffer_size_;
    max_buffer_size_ = nullptr;
  }

  if (allocated_memories_) {
    for (GGsize i = 0; i < number_detected_devices_; ++i) allocated_memories_[i].clear();
  }

  GGcout("GGEMSRAMManager", "~GGEMSRAMManager", 3) << "GGEMSRAMManager erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRAMManager::Clean(void)
{
  GGcout("GGEMSRAMManager", "Clean", 3) << "GGEMSRAMManager cleaning..." << GGendl;

  GGcout("GGEMSRAMManager", "Clean", 3) << "GGEMSRAMManager cleaned!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRAMManager::IncrementRAMMemory(std::string const& class_name, GGsize const& index, GGsize const& size)
{
  // Getting OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get index of the device
  GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(index);

  // Checking if class has already allocated memory, if not, creating one
  if (allocated_memories_[device_index].find(class_name) == allocated_memories_[device_index].end()) {
    allocated_memories_[device_index].insert(std::make_pair(class_name, size));
  }
  else {
    allocated_memories_[device_index][class_name] += size;
  }

  // Increment size
  allocated_ram_[device_index] += size;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRAMManager::DecrementRAMMemory(std::string const& class_name, GGsize const& index, GGsize const& size)
{
  // Getting OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get index of the device
  GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(index);

  // decrement size
  allocated_ram_[device_index] -= size;
  allocated_memories_[device_index][class_name] -= size;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRAMManager::PrintRAMStatus(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGsize number_activated_device = opencl_manager.GetNumberOfActivatedDevice();

  GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << GGendl;

  // Loop over activated device
  for (GGsize i = 0; i < number_activated_device; ++i) {
    GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(i);

    // Compute allocated percent RAM
    GGfloat percent_allocated_RAM = static_cast<GGfloat>(allocated_ram_[device_index]) * 100.0f / static_cast<GGfloat>(max_available_ram_[device_index]);

    GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "Device: " << opencl_manager.GetDeviceName(device_index) << GGendl;
    GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "-------" << GGendl;
    GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "Total RAM memory allocated: " << BestDigitalUnit(allocated_ram_[device_index]) << " / " << BestDigitalUnit(max_available_ram_[device_index]) << " (" << percent_allocated_RAM << "%)" << GGendl;
    GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "Details: " << GGendl;
    for (auto&& j : allocated_memories_[device_index]) {
      GGfloat usage = static_cast<GGfloat>(j.second) * 100.0f / static_cast<GGfloat>(allocated_ram_[device_index]);
      if (allocated_ram_[device_index] == 0) usage = 0.0f;
      GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "    + In '" << j.first << "': " << BestDigitalUnit(j.second) << " allocated (" << usage << "%)" << GGendl;
    }
  }

  GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSRAMManager* get_instance_ggems_ram_manager(void)
{
  return &GGEMSRAMManager::GetInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_infos_ram_manager(GGEMSRAMManager* ram_manager)
{
  ram_manager->PrintRAMStatus();
}
