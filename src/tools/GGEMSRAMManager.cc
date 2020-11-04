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
#include "GGEMS/tools/GGEMSPrint.hh"
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
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSRAMManager::~GGEMSRAMManager(void)
{
  GGcout("GGEMSRAMManager", "~GGEMSRAMManager", 3) << "Deallocation of GGEMS RAM Manager..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRAMManager::IncrementRAMMemory(GGulong const& size)
{
  // Increment size
  allocated_ram_ += size;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRAMManager::DecrementRAMMemory(GGulong const& size)
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
  std::string const kContextName = opencl_manager.GetNameOfActivatedContext();

  // Compute allocated percent RAM
  GGfloat const kPercentAllocatedRAM = static_cast<GGfloat>(allocated_ram_) * 100.0f / static_cast<GGfloat>(max_available_ram_);

  GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "Device: " << kContextName << GGendl;
  GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "RAM memory usage: " << allocated_ram_ << " / " << max_available_ram_ << " bytes (" << kPercentAllocatedRAM << "%)" << GGendl;
}
