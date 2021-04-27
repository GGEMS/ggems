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
  \file GGEMSPseudoRandomGenerator.cc

  \brief Class managing the random number in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday December 16, 2019
*/

#include <random>

#include "GGEMS/randoms/GGEMSPseudoRandomGenerator.hh"
#include "GGEMS/randoms/GGEMSRandom.hh"

#include "GGEMS/tools/GGEMSRAMManager.hh"

#include "GGEMS/sources/GGEMSSourceManager.hh"

#ifdef __linux__
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPseudoRandomGenerator::GGEMSPseudoRandomGenerator(void)
: pseudo_random_numbers_(nullptr),
  seed_(0)
{
  GGcout("GGEMSPseudoRandomGenerator", "GGEMSPseudoRandomGenerator", 3) << "GGEMSPseudoRandomGenerator creating..." << GGendl;

  GGcout("GGEMSPseudoRandomGenerator", "GGEMSPseudoRandomGenerator", 3) << "GGEMSPseudoRandomGenerator created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPseudoRandomGenerator::~GGEMSPseudoRandomGenerator(void)
{
  GGcout("GGEMSPseudoRandomGenerator", "~GGEMSPseudoRandomGenerator", 3) << "GGEMSPseudoRandomGenerator erasing..." << GGendl;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  if (pseudo_random_numbers_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(pseudo_random_numbers_[i], sizeof(GGEMSRandom), i);
    }
    delete[] pseudo_random_numbers_;
    pseudo_random_numbers_ = nullptr;
  }

  GGcout("GGEMSPseudoRandomGenerator", "~GGEMSPseudoRandomGenerator", 3) << "GGEMSPseudoRandomGenerator erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGuint GGEMSPseudoRandomGenerator::GenerateSeed(void) const
{
  #ifdef _WIN32
  HCRYPTPROV seedWin32;
  if (CryptAcquireContext(&seedWin32, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT ) == FALSE) {
    std::ostringstream oss(std::ostringstream::out);
    char buffer_error[256];
    oss << "Error finding a seed: " << strerror_s(buffer_error, 256, errno) << std::endl;
    GGEMSMisc::ThrowException("GGEMSManager", "GenerateSeed", oss.str());
  }
  return static_cast<uint32_t>(seedWin32);
  #else
  // Open a system random file
  GGint file_descriptor = ::open("/dev/urandom", O_RDONLY | O_NONBLOCK);
  if (file_descriptor < 0) {
    std::ostringstream oss( std::ostringstream::out );
    oss << "Error opening the file '/dev/urandom': " << strerror(errno) << std::endl;
    GGEMSMisc::ThrowException("GGEMSManager", "GenerateSeed", oss.str());
  }

  // Buffer storing 8 characters
  char seedArray[sizeof(GGuint)];
  ssize_t bytes_read = ::read(file_descriptor, reinterpret_cast<GGuint*>(seedArray), sizeof(GGuint));
  if (bytes_read == -1) {
    std::ostringstream oss( std::ostringstream::out );
    oss << "Error reading the file '/dev/urandom': " << strerror(errno) << std::endl;
    GGEMSMisc::ThrowException("GGEMSManager", "GenerateSeed", oss.str());
  }
  ::close(file_descriptor);
  GGuint *seedUInt = reinterpret_cast<GGuint*>(seedArray);
  return *seedUInt;
  #endif
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPseudoRandomGenerator::Initialize(GGuint const& seed)
{
  GGcout("GGEMSPseudoRandomGenerator", "Initialize", 1) << "Initialization of GGEMSPseudoRandomGenerator..." << GGendl;

  seed_ = seed == 0 ? GenerateSeed() : seed;

  // Allocation of the Random structure
  AllocateRandom();

  // Generate seeds for each particle
  InitializeSeeds();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPseudoRandomGenerator::InitializeSeeds(void)
{
  GGcout("GGEMSPseudoRandomGenerator", "InitializeSeeds", 1) << "Initialization of seeds for each particles..." << GGendl;

  // Initialize the Mersenne Twister engine
  std::mt19937 mt_gen(seed_);

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Loop over activated device
  for (GGsize i = 0; i < number_activated_devices_; ++i) {
    // Get the pointer on device
    GGEMSRandom* random_device = opencl_manager.GetDeviceBuffer<GGEMSRandom>(pseudo_random_numbers_[i], sizeof(GGEMSRandom), i);

    // For each particle a seed is generated
    for (GGsize i = 0; i < MAXIMUM_PARTICLES; ++i) {
      random_device->prng_state_1_[i] = static_cast<GGuint>(mt_gen());
      random_device->prng_state_2_[i] = static_cast<GGuint>(mt_gen());
      random_device->prng_state_3_[i] = static_cast<GGuint>(mt_gen());
      random_device->prng_state_4_[i] = static_cast<GGuint>(mt_gen());
      random_device->prng_state_5_[i] = 0;
    }

    // Release the pointer, mandatory step!!!
    opencl_manager.ReleaseDeviceBuffer(pseudo_random_numbers_[i], random_device, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPseudoRandomGenerator::AllocateRandom(void)
{
  GGcout("GGEMSPseudoRandomGenerator", "AllocateRandom", 1) << "Allocation of random numbers..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Getting number of activated device
  number_activated_devices_ = opencl_manager.GetNumberOfActivatedDevice();

  // Allocation of memory on OpenCL device
  pseudo_random_numbers_ = new cl::Buffer*[number_activated_devices_];
  for (GGsize i = 0; i < number_activated_devices_; ++i) {
    pseudo_random_numbers_[i] = opencl_manager.Allocate(nullptr, sizeof(GGEMSRandom), i, CL_MEM_READ_WRITE, "GGEMSPseudoRandomGenerator");
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPseudoRandomGenerator::PrintInfos(void) const
{
  GGcout("GGEMSPseudoRandomGenerator", "PrintInfos", 0) << "Printing infos about random" << GGendl;
  GGcout("GGEMSPseudoRandomGenerator", "PrintInfos", 0) << "Seed: " << seed_ << GGendl;

  // Getting OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Loop over the activated devices
  for (GGsize i = 0; i < number_activated_devices_; ++i) {
    GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(i);

    GGEMSRandom* random_device = opencl_manager.GetDeviceBuffer<GGEMSRandom>(pseudo_random_numbers_[i], sizeof(GGEMSRandom), i);

    GGuint state[2][5] = {
      {
        random_device->prng_state_1_[0],
        random_device->prng_state_2_[0],
        random_device->prng_state_3_[0],
        random_device->prng_state_4_[0],
        random_device->prng_state_5_[0]
      },
      {
        random_device->prng_state_1_[1],
        random_device->prng_state_2_[1],
        random_device->prng_state_3_[1],
        random_device->prng_state_4_[1],
        random_device->prng_state_5_[1]
      }
    };

    // Release the pointer, mandatory step!!!
    opencl_manager.ReleaseDeviceBuffer(pseudo_random_numbers_[i], random_device, i);

    GGcout("GGEMSPseudoRandomGenerator", "PrintInfos", 0) << "Device: " << opencl_manager.GetDeviceName(device_index) << GGendl;
    GGcout("GGEMSPseudoRandomGenerator", "PrintInfos", 0) << "-------" << GGendl;
    GGcout("GGEMSPseudoRandomGenerator", "PrintInfos", 0) << "Random state of two first particles for JKISS engine:" << GGendl;
    GGcout("GGEMSPseudoRandomGenerator", "PrintInfos", 0) << "    * state 0: " << state[0][0] << " " << state[1][0] << GGendl;
    GGcout("GGEMSPseudoRandomGenerator", "PrintInfos", 0) << "    * state 1: " << state[0][1] << " " << state[1][1] << GGendl;
    GGcout("GGEMSPseudoRandomGenerator", "PrintInfos", 0) << "    * state 2: " << state[0][2] << " " << state[1][2] << GGendl;
    GGcout("GGEMSPseudoRandomGenerator", "PrintInfos", 0) << "    * state 3: " << state[0][3] << " " << state[1][3] << GGendl;
    GGcout("GGEMSPseudoRandomGenerator", "PrintInfos", 0) << "    * state 4: " << state[0][4] << " " << state[1][4] << GGendl;
  }
}
