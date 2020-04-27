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
#include "GGEMS/randoms/GGEMSRandomStack.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPseudoRandomGenerator::GGEMSPseudoRandomGenerator(void)
: pseudo_random_numbers_(nullptr),
  opencl_manager_(GGEMSOpenCLManager::GetInstance())
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

  // Get the pointer on device
  GGEMSRandom* random_device = opencl_manager_.GetDeviceBuffer<GGEMSRandom>(pseudo_random_numbers_, sizeof(GGEMSRandom));

  // For each particle a seed is generated
  for (std::size_t i = 0; i < MAXIMUM_PARTICLES; ++i) {
    random_device->prng_state_1_[i] = static_cast<GGuint>(rand());
    random_device->prng_state_2_[i] = static_cast<GGuint>(rand());
    random_device->prng_state_3_[i] = static_cast<GGuint>(rand());
    random_device->prng_state_4_[i] = static_cast<GGuint>(rand());
    random_device->prng_state_5_[i] = static_cast<GGuint>(0);
  }

  // Release the pointer, mandatory step!!!
  opencl_manager_.ReleaseDeviceBuffer(pseudo_random_numbers_, random_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPseudoRandomGenerator::AllocateRandom(void)
{
  GGcout("GGEMSPseudoRandomGenerator", "AllocateRandom", 1) << "Allocation of random numbers..." << GGendl;

  // Allocation of memory on OpenCL device
  pseudo_random_numbers_ = opencl_manager_.Allocate(nullptr, sizeof(GGEMSRandom), CL_MEM_READ_WRITE);
  opencl_manager_.AddRAMMemory(sizeof(GGEMSRandom));
  GGEMSSourceManager::GetInstance().AddSourceRAM(sizeof(GGEMSRandom));
}
