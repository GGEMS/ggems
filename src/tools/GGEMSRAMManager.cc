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
{
  GGcout("GGEMSRAMManager", "GGEMSRAMManager", 3) << "Allocation of GGEMS RAM Manager..." << GGendl;

  allocated_ram_.resize(7);
  for (auto&& i : allocated_ram_) i = 0;

  // Initialize name of allocated memory type
  name_of_allocated_memory_.resize(7);
  name_of_allocated_memory_[GGEMSRAMType::total] = "total";
  name_of_allocated_memory_[GGEMSRAMType::material] = "material";
  name_of_allocated_memory_[GGEMSRAMType::geometry] = "geometry";
  name_of_allocated_memory_[GGEMSRAMType::process] = "process";
  name_of_allocated_memory_[GGEMSRAMType::particle] = "particle";
  name_of_allocated_memory_[GGEMSRAMType::random] = "random";
  name_of_allocated_memory_[GGEMSRAMType::source] = "source";
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

void GGEMSRAMManager::AddMaterialRAMMemory(GGulong const& size)
{
  GGcout("GGEMSRAMManager","AddMaterialRAMMemory", 3) << "Adding allocated RAM memory for material on activated OpenCL context..." << GGendl;

  // Increment size
  allocated_ram_[GGEMSRAMType::material] += size;
  AddTotalRAMMemory(size);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRAMManager::AddGeometryRAMMemory(GGulong const& size)
{
  GGcout("GGEMSRAMManager","AddGeometryRAMMemory", 3) << "Adding allocated RAM memory for geometry on activated OpenCL context..." << GGendl;

  // Increment size
  allocated_ram_[GGEMSRAMType::geometry] += size;
  AddTotalRAMMemory(size);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRAMManager::AddProcessRAMMemory(GGulong const& size)
{
  GGcout("GGEMSRAMManager","AddProcessRAMMemory", 3) << "Adding allocated RAM memory for process on activated OpenCL context..." << GGendl;

  // Increment size
  allocated_ram_[GGEMSRAMType::process] += size;
  AddTotalRAMMemory(size);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRAMManager::AddRandomRAMMemory(GGulong const& size)
{
  GGcout("GGEMSRAMManager","AddRandomRAMMemory", 3) << "Adding allocated RAM memory for random on activated OpenCL context..." << GGendl;

  // Increment size
  allocated_ram_[GGEMSRAMType::random] += size;
  AddTotalRAMMemory(size);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRAMManager::AddParticleRAMMemory(GGulong const& size)
{
  GGcout("GGEMSRAMManager","AddParticleRAMMemory", 3) << "Adding allocated RAM memory for particle on activated OpenCL context..." << GGendl;

  // Increment size
  allocated_ram_[GGEMSRAMType::particle] += size;
  AddTotalRAMMemory(size);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRAMManager::AddSourceRAMMemory(GGulong const& size)
{
  GGcout("GGEMSRAMManager","AddSourceRAMMemory", 3) << "Adding allocated RAM memory for source on activated OpenCL context..." << GGendl;

  // Increment size
  allocated_ram_[GGEMSRAMType::source] += size;
  AddTotalRAMMemory(size);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRAMManager::AddTotalRAMMemory(GGulong const& size)
{
  GGcout("GGEMSRAMManager","AddGlobalRAMMemory", 3) << "Adding global allocated RAM memory on activated OpenCL context..." << GGendl;

  // Increment size
  allocated_ram_[GGEMSRAMType::total] += size;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRAMManager::PrintRAMStatus(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get the maximum memory on activated context
  GGulong const kMaxRAM = opencl_manager.GetMaxRAMMemoryOnActivatedContext();

  // Get the name of the activated context
  std::string const kContextName = opencl_manager.GetNameOfActivatedContext();

  // Get the max. size of type of memory
  std::size_t max_length_type_of_memory = 0;
  for (auto&& i : name_of_allocated_memory_) {
    if (max_length_type_of_memory < i.length()) max_length_type_of_memory = i.length();
  }

  // Get the max. size of number
  std::string const kTotalRAMString = std::to_string(allocated_ram_[0]) + " / " + std::to_string(kMaxRAM);
  std::size_t const kTotalStringSize = kTotalRAMString.length();

  // Get total line size of table
  std::size_t const kTotalSizeTable = 2 + max_length_type_of_memory + 3 + kTotalStringSize + 8;

  GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "Device: " << kContextName << GGendl;

  GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "";
  for (std::size_t i = 0; i < kTotalSizeTable; ++i) std::cout << "*";
  std::cout << std::endl;

  for (int i = 1; i < name_of_allocated_memory_.size(); ++i) {
    GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "* " << std::setw(max_length_type_of_memory) << name_of_allocated_memory_[i] << " | " << std::setw(kTotalStringSize) << allocated_ram_[i] << " bytes *" << GGendl;
  }

  GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "*";
  for (std::size_t i = 0; i < kTotalSizeTable-2; ++i) std::cout << "-";
  std::cout << "*" << std::endl;
  GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "*";
  for (std::size_t i = 0; i < kTotalSizeTable-2; ++i) std::cout << "-";
  std::cout << "*" << std::endl;

  GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "* " << std::setw(max_length_type_of_memory) << name_of_allocated_memory_[0] << " | " << std::setw(kTotalStringSize) << kTotalRAMString << " bytes *" << GGendl;

  // Compute allocated percent RAM
  GGfloat const kPercentRAM = static_cast<GGfloat>(allocated_ram_[0]) * 100.0f / static_cast<GGfloat>(kMaxRAM);
  std::string kPercentString = std::to_string(kPercentRAM);
  kPercentString.resize(5);
  GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "* " << std::setw(max_length_type_of_memory) << "" <<  " | " << std::setw(kTotalStringSize) << kPercentString << " %     *" << GGendl;

  GGcout("GGEMSRAMManager", "PrintRAMStatus", 0) << "";
  for (std::size_t i = 0; i < kTotalSizeTable; ++i) std::cout << "*";
  std::cout << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRAMManager::CheckRAMMemory(std::size_t const& size)
{
  GGcout("GGEMSRAMManager","CheckRAMMemory", 3) << "Checking allocated RAM memory..." << GGendl;

  // Get OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Getting memory infos
  GGulong const kMaxRAM = opencl_manager.GetMaxRAMMemoryOnActivatedContext();
  GGdouble const kPercentRAM = static_cast<GGdouble>(allocated_ram_[GGEMSRAMType::total] + size) * 100.0 / static_cast<GGdouble>(kMaxRAM);

  if (kPercentRAM >= 80.0f && kPercentRAM < 95.0f) {
    GGwarn("GGEMSRAMManager", "CheckRAMMemory", 0) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << GGendl;
    GGwarn("GGEMSRAMManager", "CheckRAMMemory", 0) << "!!!             MEMORY WARNING             !!!" << GGendl;
    GGwarn("GGEMSRAMManager", "CheckRAMMemory", 0) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << GGendl;
    GGwarn("GGEMSRAMManager", "CheckRAMMemory", 0) << "Allocated RAM (" << kPercentRAM << "%) is superior to 80%, the simulation will be automatically killed if RAM allocation is superior to 95%" << GGendl;
  }
  else if (kPercentRAM >= 95.0f) {
    GGcerr("GGEMSRAMManager", "CheckRAMMemory", 0) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << GGendl;
    GGcerr("GGEMSRAMManager", "CheckRAMMemory", 0) << "!!!             MEMORY ERROR             !!!" << GGendl;
    GGcerr("GGEMSRAMManager", "CheckRAMMemory", 0) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << GGendl;
    GGcerr("GGEMSRAMManager", "CheckRAMMemory", 0) << "RAM allocation (" << kPercentRAM << "%) is superior to 95%, the simulation is killed!!!" << GGendl;
    GGEMSMisc::ThrowException("GGEMSRAMManager", "CheckRAMMemory", "Not enough RAM memory on OpenCL device!!!");
  }
}
