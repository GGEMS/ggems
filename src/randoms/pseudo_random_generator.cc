/*!
  \file pseudo_random_generator.cc

  \brief Class managing the random number in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday December 16, 2019
*/

//#include "GGEMS/tools/functions.hh"
#include "GGEMS/randoms/pseudo_random_generator.hh"
#include "GGEMS/randoms/random.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

RandomGenerator::RandomGenerator(void)
: p_random_numbers_(nullptr),
  opencl_manager_(OpenCLManager::GetInstance())
{
  GGEMScout("RandomGenerator", "RandomGenerator", 3)
    << "Allocation of RandomGenerator..." << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

RandomGenerator::~RandomGenerator(void)
{
  // Freeing the device buffer
  if (p_random_numbers_) {
    //opencl_manager_.Deallocate(p_primary_particles_, sizeof(PrimaryParticles));
    p_random_numbers_ = nullptr;
  }

  GGEMScout("RandomGenerator", "~RandomGenerator", 3)
    << "Deallocation of RandomGenerator..." << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void RandomGenerator::Initialize(void)
{
  GGEMScout("RandomGenerator", "Initialize", 1)
    << "Initialization of RandomGenerator..." << GGEMSendl;

  // Allocation of the Random structure
  AllocateRandom();

  // Generate seeds for each particle
  InitializeSeeds();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void RandomGenerator::InitializeSeeds(void)
{
  GGEMScout("RandomGenerator", "InitializeSeeds", 1)
    << "Initialization of seeds for each particles..." << GGEMSendl;

  // Get the pointer on device
  Random* p_random =
    opencl_manager_.GetDeviceBuffer<Random>(p_random_numbers_,sizeof(Random));

  // For each particle a seed is generated
  for (std::size_t i = 0; i < MAXIMUM_PARTICLES; ++i) {
    p_random->p_prng_state_1_[i] = static_cast<cl_uint>(rand());
    p_random->p_prng_state_2_[i] = static_cast<cl_uint>(rand());
    p_random->p_prng_state_3_[i] = static_cast<cl_uint>(rand());
    p_random->p_prng_state_4_[i] = static_cast<cl_uint>(rand());
    p_random->p_prng_state_5_[i] = static_cast<cl_uint>(0);
  }

  // Release the pointer, mandatory step!!!
  opencl_manager_.ReleaseDeviceBuffer(p_random_numbers_, p_random);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void RandomGenerator::AllocateRandom(void)
{
  GGEMScout("RandomGenerator", "AllocateRandom", 1)
    << "Allocation of random numbers..." << GGEMSendl;

  // Allocation of memory on OpenCL device
  p_random_numbers_ = opencl_manager_.Allocate(nullptr,
    sizeof(Random), CL_MEM_READ_WRITE);
}
