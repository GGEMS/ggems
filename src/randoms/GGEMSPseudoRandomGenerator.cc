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

#include "GGEMS/randoms/GGEMSPseudoRandomGenerator.hh"
#include "GGEMS/randoms/GGEMSRandom.hh"
#include "GGEMS/tools/GGEMSRAMManager.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPseudoRandomGenerator::GGEMSPseudoRandomGenerator(void)
: pseudo_random_numbers_cl_(nullptr)
{
  GGcout("GGEMSPseudoRandomGenerator", "GGEMSPseudoRandomGenerator", 3) << "Allocation of GGEMSPseudoRandomGenerator..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPseudoRandomGenerator::~GGEMSPseudoRandomGenerator(void)
{
  GGcout("GGEMSPseudoRandomGenerator", "~GGEMSPseudoRandomGenerator", 3) << "Deallocation of GGEMSPseudoRandomGenerator..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPseudoRandomGenerator::Initialize(void)
{
  GGcout("GGEMSPseudoRandomGenerator", "Initialize", 1) << "Initialization of GGEMSPseudoRandomGenerator..." << GGendl;

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

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get the pointer on device
  GGEMSRandom* random_device = opencl_manager.GetDeviceBuffer<GGEMSRandom>(pseudo_random_numbers_cl_.get(), sizeof(GGEMSRandom));

  // For each particle a seed is generated
  for (std::size_t i = 0; i < MAXIMUM_PARTICLES; ++i) {
    random_device->prng_state_1_[i] = static_cast<GGuint>(rand());
    random_device->prng_state_2_[i] = static_cast<GGuint>(rand());
    random_device->prng_state_3_[i] = static_cast<GGuint>(rand());
    random_device->prng_state_4_[i] = static_cast<GGuint>(rand());
    random_device->prng_state_5_[i] = static_cast<GGuint>(0);
  }

  // Release the pointer, mandatory step!!!
  opencl_manager.ReleaseDeviceBuffer(pseudo_random_numbers_cl_.get(), random_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPseudoRandomGenerator::AllocateRandom(void)
{
  GGcout("GGEMSPseudoRandomGenerator", "AllocateRandom", 1) << "Allocation of random numbers..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Allocation of memory on OpenCL device
  pseudo_random_numbers_cl_ = opencl_manager.Allocate(nullptr, sizeof(GGEMSRandom), CL_MEM_READ_WRITE);
}
