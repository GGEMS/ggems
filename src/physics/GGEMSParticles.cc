/*!
  \file GGEMSParticles.cc

  \brief Class managing the particles in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday October 3, 2019
*/

#include "GGEMS/physics/GGEMSParticles.hh"
#include "GGEMS/physics/GGEMSPrimaryParticlesStack.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSParticles::GGEMSParticles(void)
: p_primary_particles_(nullptr),
  opencl_manager_(GGEMSOpenCLManager::GetInstance())
{
  GGcout("GGEMSParticles", "GGEMSParticles", 3)
    << "Allocation of GGEMSParticles..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSParticles::~GGEMSParticles(void)
{
  // Freeing the device buffer
  if (p_primary_particles_) {
    opencl_manager_.Deallocate(p_primary_particles_,
      sizeof(GGEMSPrimaryParticles));
    p_primary_particles_ = nullptr;
  }

  GGcout("GGEMSParticles", "~GGEMSParticles", 3)
    << "Deallocation of GGEMSParticles..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSParticles::Initialize(void)
{
  GGcout("GGEMSParticles", "Initialize", 1)
    << "Initialization of GGEMSParticles..." << GGendl;

  // Allocation of the PrimaryParticle structure
  AllocatePrimaryParticles();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSParticles::AllocatePrimaryParticles(void)
{
  GGcout("GGEMSParticles", "AllocatePrimaryParticles", 1)
    << "Allocation of primary particles..." << GGendl;

  // Allocation of memory on OpenCL device
  p_primary_particles_ = opencl_manager_.Allocate(nullptr,
    sizeof(GGEMSPrimaryParticles), CL_MEM_READ_WRITE);
}
