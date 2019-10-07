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
#include "GGEMS/global/ggems_configuration.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

Particle::Particle()
: number_of_particles_(0),
  p_primary_particles_(nullptr)
{
  GGEMScout("Particle", "Particle", 1)
    << "Allocation of Particle..." << GGEMSendl;

  // Allocating memory for primary particles
  p_primary_particles_ = new PrimaryParticles;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

Particle::~Particle()
{
  // Get the pointer on OpenCL singleton
  OpenCLManager& opencl_manager = OpenCLManager::GetInstance();

  // Freeing the device buffer
  if (p_primary_particles_->p_E_) {
    opencl_manager.Deallocate(p_primary_particles_->p_E_,
      number_of_particles_ * sizeof(cl_float));
  }

  if (p_primary_particles_) {
    delete p_primary_particles_;
    p_primary_particles_ = nullptr;
  }

  GGEMScout("Particle", "~Particle", 1)
    << "Deallocation of Particle..." << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void Particle::Initialize()
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
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void Particle::AllocatePrimaryParticles()
{
  GGEMScout("Particle", "AllocatePrimaryParticles", 1)
    << "Allocation of primary particles..." << GGEMSendl;

  // Get the pointer on OpenCL manager
  OpenCLManager& opencl_manager = OpenCLManager::GetInstance();

  // Allocation of memory on OpenCL device
  p_primary_particles_->p_E_ = opencl_manager.Allocate(nullptr,
    sizeof(cl_float)*number_of_particles_, CL_MEM_READ_WRITE);

  // Allocation of device memory for the particle
  //p_primary_particles_->p_E_ = new cl::Buffer();

  //p_primary_particles_->number_of_primaries_ = number_of_particles_;

  // Test with OpenCL
  /*OpenCLManager& opencl_manager = OpenCLManager::GetInstance();
  std::string const kOpenCLKernelPath = OPENCL_KERNEL_PATH;
  std::string const kFilename = kOpenCLKernelPath + "/print_primary_particle.cl";
  cl::Kernel* p_kernel = opencl_manager.CompileKernel(kFilename, "print_primary_particle");

  cl::CommandQueue* p_queue = opencl_manager.GetCommandQueue();
  cl::Event* p_event = opencl_manager.GetEvent();
  cl::Context* p_context = opencl_manager.GetContext();

  p_primary_particles_cl_ = new PrimaryParticles;

  p_primary_particles_cl_->p_E_ = new cl::Buffer(*p_context,
    CL_MEM_READ_WRITE, sizeof(cl_float)*1000000, nullptr, nullptr);

  float* p_pp = static_cast<float*>(
    p_queue->enqueueMapBuffer(*p_primary_particles_cl_->p_E_, CL_TRUE,
    CL_MAP_WRITE, 0, sizeof(cl_float)*1000000, nullptr, nullptr, nullptr));

  std::iota(p_pp, p_pp + 1000000, 0);

  // Unmapping the buffer
  p_queue->enqueueUnmapMemObject(*p_primary_particles_cl_->p_E_, p_pp);

  p_kernel->setArg(0, *p_primary_particles_cl_->p_E_);
  p_kernel->setArg(1, 15);

  // Define the number of work-item to launch
  cl::NDRange global(1000000);
  cl::NDRange offset(0);

  // Launching kernel
  cl_int kernel_status = p_queue->enqueueNDRangeKernel(*p_kernel, offset, global, cl::NullRange, nullptr, p_event);
  opencl_manager.CheckOpenCLError(kernel_status);
  p_queue->finish();

  delete p_primary_particles_cl_->p_E_;
  delete p_primary_particles_cl_;

  opencl_manager.DisplayElapsedTimeInKernel("print_primary_particle");*/
}
