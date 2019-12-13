/*!
  \file particles.cc

  \brief Class managing the particles in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday October 3, 2019
*/

#include <numeric>

#include "GGEMS/global/ggems_manager.hh"
#include "GGEMS/processes/particles.hh"
#include "GGEMS/tools/functions.hh"
#include "GGEMS/tools/print.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

Particle::Particle(void)
: p_primary_particles_(nullptr),
  opencl_manager_(OpenCLManager::GetInstance())
{
  GGEMScout("Particle", "Particle", 1)
    << "Allocation of Particle..." << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

Particle::~Particle(void)
{
  // Freeing the device buffer
  if (p_primary_particles_) {
    opencl_manager_.Deallocate(p_primary_particles_, sizeof(PrimaryParticles));
    p_primary_particles_ = nullptr;
  }

  GGEMScout("Particle", "~Particle", 1)
    << "Deallocation of Particle..." << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void Particle::Initialize(void)
{
  GGEMScout("Particle", "Initialize", 1)
    << "Initialization of Particle..." << GGEMSendl;

  // Allocation of the PrimaryParticle structure
  AllocatePrimaryParticles();

  // Generate seeds for each particle
  InitializeSeeds();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void Particle::InitializeSeeds(void)
{
  GGEMScout("Particle", "InitializeSeeds", 1)
    << "Initialization of seeds for each particles..." << GGEMSendl;

  // Get the pointer on device
  PrimaryParticles* p_primary_particles =
    opencl_manager_.GetDeviceBuffer<PrimaryParticles>(p_primary_particles_,
      1*sizeof(PrimaryParticles));

  // For each particle a seed is generated
  for (uint64_t i = 0; i < MAXIMUM_PARTICLES; ++i) {
    p_primary_particles->p_prng_state_1_[i] = static_cast<cl_uint>(rand());
    p_primary_particles->p_prng_state_2_[i] = static_cast<cl_uint>(rand());
    p_primary_particles->p_prng_state_3_[i] = static_cast<cl_uint>(rand());
    p_primary_particles->p_prng_state_4_[i] = static_cast<cl_uint>(rand());
    p_primary_particles->p_prng_state_5_[i] = static_cast<cl_uint>(0);
  }

  // Release the pointer, mandatory step!!!
  opencl_manager_.ReleaseDeviceBuffer(p_primary_particles_,
    p_primary_particles);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void Particle::AllocatePrimaryParticles(void)
{
  GGEMScout("Particle", "AllocatePrimaryParticles", 1)
    << "Allocation of primary particles..." << GGEMSendl;

  // Allocation of memory on OpenCL device
  p_primary_particles_ = opencl_manager_.Allocate(nullptr,
    sizeof(PrimaryParticles), CL_MEM_READ_WRITE);
}
