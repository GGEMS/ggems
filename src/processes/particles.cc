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
: number_of_particles_(0),
  p_primary_particles_(nullptr)
{
  GGEMScout("Particle", "Particle", 1)
    << "Allocation of Particle..." << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

Particle::~Particle(void)
{
  // Get the pointer on OpenCL singleton
  OpenCLManager& opencl_manager = OpenCLManager::GetInstance();

  // Freeing the device buffer
  if (p_primary_particles_) {
    opencl_manager.Deallocate(p_primary_particles_, sizeof(PrimaryParticles));
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

  // Get the GGEMS singleton
  GGEMSManager& ggems_manager = GGEMSManager::GetInstance();

  // Get the number of the particles in the first batch
  number_of_particles_ = ggems_manager.GetNumberOfParticlesInFirstBatch();
  if (number_of_particles_ == 0) {
    Misc::ThrowException("Particle", "Initialize",
      "Number of particle is 0 in the first batch!!!");
  }

  // Allocation of the PrimaryParticle structure
  AllocatePrimaryParticles();

  // Generate seeds for each particle
  InitializeSeeds();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void Particle::SetNumberOfParticlesInBatch(
  uint64_t const& number_of_particles_in_batch)
{
  // Updating the number of particle variable
  number_of_particles_ = number_of_particles_in_batch;

  // Copy this data on device memory
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void Particle::InitializeSeeds(void)
{
  GGEMScout("Particle", "InitializeSeeds", 1)
    << "Initialization of seeds for each particles..." << GGEMSendl;

  // Get the pointer on the OpenCL Manager singleton
  OpenCLManager& opencl_manager = OpenCLManager::GetInstance();

  // Get the pointer on device
  PrimaryParticles* p_primary_particles =
    opencl_manager.GetDeviceBufferWrite<PrimaryParticles>(p_primary_particles_);

  // For each particle a seed is generated
  for (uint64_t i = 0; i < number_of_particles_; ++i) {
    p_primary_particles->p_prng_state_1_[i] = static_cast<cl_uint>(rand());
    p_primary_particles->p_prng_state_2_[i] = static_cast<cl_uint>(rand());
    p_primary_particles->p_prng_state_3_[i] = static_cast<cl_uint>(rand());
    p_primary_particles->p_prng_state_4_[i] = static_cast<cl_uint>(rand());
    p_primary_particles->p_prng_state_5_[i] = static_cast<cl_uint>(0);
  }

  // To Delete!!!!!! Gestion du nombre de particules
  p_primary_particles->number_of_primaries_ = number_of_particles_;

  // Release the pointer, mandatory step!!!
  opencl_manager.ReleaseDeviceBuffer(p_primary_particles_, p_primary_particles);

  // Auxiliary function test
  cl::CommandQueue* p_queue = opencl_manager.GetCommandQueue();
  std::string const kOpenCLKernelPath = OPENCL_KERNEL_PATH;
  std::string const kFilename = kOpenCLKernelPath
    + "/print_primary_particle.cl";
  cl::Kernel* p_kernel = opencl_manager.CompileKernel(kFilename,
    "print_primary_particle");

  // Get the event to print time elapsed in kernel
  cl::Event* p_event = opencl_manager.GetEvent();

  p_kernel->setArg(0, *p_primary_particles_);

  // Define the number of work-item to launch
  cl::NDRange global(number_of_particles_);
  cl::NDRange offset(0);

  // Launching kernel
  cl_int kernel_status = p_queue->enqueueNDRangeKernel(*p_kernel, offset,
    global, cl::NullRange, nullptr, p_event);
  opencl_manager.CheckOpenCLError(kernel_status);
  p_queue->finish(); // Wait until the kernel status is finish

  // Displaying time in kernel
  opencl_manager.DisplayElapsedTimeInKernel("print_primary_particle");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void Particle::AllocatePrimaryParticles(void)
{
  GGEMScout("Particle", "AllocatePrimaryParticles", 1)
    << "Allocation of primary particles..." << GGEMSendl;

  // Get the pointer on OpenCL manager
  OpenCLManager& opencl_manager = OpenCLManager::GetInstance();

  // Allocation of memory on OpenCL device
  p_primary_particles_ = opencl_manager.Allocate(nullptr,
    sizeof(PrimaryParticles), CL_MEM_READ_WRITE);
}
