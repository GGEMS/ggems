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
  \file GGEMSDosimetryCalculator.cc

  \brief Class providing tools storing and computing dose in phantom

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Wednesday January 13, 2021
*/

#include <sstream>

#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/navigators/GGEMSDosimetryCalculator.hh"
#include "GGEMS/navigators/GGEMSDoseParams.hh"
#include "GGEMS/navigators/GGEMSNavigatorManager.hh"
#include "GGEMS/navigators/GGEMSNavigator.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolid.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/io/GGEMSMHDImage.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSDosimetryCalculator::GGEMSDosimetryCalculator(void)
: dosel_sizes_({-1.0f, -1.0f, -1.0f}),
  dosimetry_output_filename("dosi"),
  navigator_(nullptr),
  kernel_compute_dose_timer_(GGEMSChrono::Zero())
{
  GGcout("GGEMSDosimetryCalculator", "GGEMSDosimetryCalculator", 3) << "Allocation of GGEMSDosimetryCalculator..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSDosimetryCalculator::~GGEMSDosimetryCalculator(void)
{
  GGcout("GGEMSDosimetryCalculator", "~GGEMSDosimetryCalculator", 3) << "Deallocation of GGEMSDosimetryCalculator..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SetDoselSizes(GGfloat3 const& dosel_sizes)
{
  dosel_sizes_ = dosel_sizes;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SetOutputDosimetryFilename(std::string const& output_filename)
{
  dosimetry_output_filename = output_filename;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::CheckParameters(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  if (!navigator_) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "A navigator has to be associated to GGEMSDosimetryCalculator!!!";
    GGEMSMisc::ThrowException("GGEMSDosimetryCalculator", "CheckParameters", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SetNavigator(std::string const& navigator_name)
{
  GGEMSNavigatorManager& navigator_manager = GGEMSNavigatorManager::GetInstance();
  navigator_ = navigator_manager.GetNavigator(navigator_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::InitializeKernel(void)
{
  GGcout("GGEMSDosimetryCalculator", "InitializeKernel", 3) << "Initializing kernel for dose computation..." << GGendl;

  // Getting OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Getting the path to kernel
  std::string openCL_kernel_path = OPENCL_KERNEL_PATH;
  std::string compute_dose_filename = openCL_kernel_path + "/ComputeDoseGGEMSVoxelizedSolid.cl";

  // Compiling the kernels
  kernel_compute_dose_ = opencl_manager.CompileKernel(compute_dose_filename, "compute_dose_ggems_voxelized_solid", nullptr, nullptr);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::ComputeDose(void)
{
  // Getting the OpenCL manager and infos for work-item launching
  /*GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue_cl = opencl_manager.GetCommandQueue();
  cl::Event* event_cl = opencl_manager.GetEvent();

  // Pointer to primary particles, and number to particles in buffer
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  cl::Buffer* primary_particles_cl = source_manager.GetParticles()->GetPrimaryParticles();
  GGlong number_of_particles = source_manager.GetParticles()->GetNumberOfParticles();

  // Getting work group size, and work-item number
  std::size_t work_group_size = opencl_manager.GetWorkGroupSize();
  std::size_t number_of_work_items = number_of_particles + (work_group_size - number_of_particles%work_group_size);

  // Parameters for work-item in kernel
  cl::NDRange global_wi(number_of_work_items);
  cl::NDRange local_wi(work_group_size);

  // Loop over all the solids
  for (auto&& s : solids_) {
    // Getting solid data infos
    cl::Buffer* solid_data_cl = s->GetSolidData();

    // Getting kernel, and setting parameters
    std::shared_ptr<cl::Kernel> kernel_cl = s->GetKernelParticleSolidDistance().lock();
    kernel_cl->setArg(0, number_of_particles);
    kernel_cl->setArg(1, *primary_particles_cl);
    kernel_cl->setArg(2, *solid_data_cl);

    // Launching kernel
    GGint kernel_status = queue_cl->enqueueNDRangeKernel(*kernel_cl, 0, global_wi, local_wi, nullptr, event_cl);
    opencl_manager.CheckOpenCLError(kernel_status, "GGEMSNavigator", "ParticleSolidDistance");
    queue_cl->finish(); // Wait until the kernel status is finish

    // Incrementing elapsed time in kernel
    kernel_particle_solid_distance_timer_ += opencl_manager.GetElapsedTimeInKernel();
  }*/
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::Initialize(void)
{
  GGcout("GGEMSDosimetryCalculator", "Initialize", 3) << "Initializing dosimetry calculator..." << GGendl;

  CheckParameters();

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Allocate dosemetry params on OpenCL device
  dose_params_ = opencl_manager.Allocate(nullptr, sizeof(GGEMSDoseParams), CL_MEM_READ_WRITE);

  // Get pointer on OpenCL device for dose parameters
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_.get(), sizeof(GGEMSDoseParams));

  // Get the voxels size
  GGfloat3 voxel_sizes = dosel_sizes_;
  if (dosel_sizes_.x < 0.0f && dosel_sizes_.y < 0.0f && dosel_sizes_.z < 0.0f) { // Custom dosel size
    voxel_sizes = dynamic_cast<GGEMSVoxelizedSolid*>(navigator_->GetSolids().at(0).get())->GetVoxelSizes();
  }

  // Storing voxel size
  dose_params_device->size_of_dosels_ = voxel_sizes;

  // Take inverse of size
  dose_params_device->inv_size_of_dosels_ = {
    1.0f / voxel_sizes.x,
    1.0f / voxel_sizes.y,
    1.0f / voxel_sizes.z
  };

  // Get border of volumes from phantom
  GGEMSOBB obb_geometry = dynamic_cast<GGEMSVoxelizedSolid*>(navigator_->GetSolids().at(0).get())->GetOBBGeometry();
  dose_params_device->border_min_xyz_ = {
    obb_geometry.border_min_xyz_[0],
    obb_geometry.border_min_xyz_[1],
    obb_geometry.border_min_xyz_[2]
  };
  dose_params_device->border_max_xyz_ = {
    obb_geometry.border_max_xyz_[0],
    obb_geometry.border_max_xyz_[1],
    obb_geometry.border_max_xyz_[2]
  };

  // Get the size of the dose map
  GGfloat3 dosemap_size = {
    obb_geometry.border_max_xyz_[0] - obb_geometry.border_min_xyz_[0],
    obb_geometry.border_max_xyz_[1] - obb_geometry.border_min_xyz_[1],
    obb_geometry.border_max_xyz_[2] - obb_geometry.border_min_xyz_[2],
  };

  // Get the number of voxels
  GGint3 number_of_dosels = {
    static_cast<int>(floor(dosemap_size.x / voxel_sizes.x)),
    static_cast<int>(floor(dosemap_size.y / voxel_sizes.y)),
    static_cast<int>(floor(dosemap_size.z / voxel_sizes.z))
  };

  dose_params_device->number_of_dosels_ = number_of_dosels;
  dose_params_device->slice_number_of_dosels_ = number_of_dosels.x * number_of_dosels.y;
  GGint total_number_of_dosels = number_of_dosels.x * number_of_dosels.y * number_of_dosels.z;
  dose_params_device->total_number_of_dosels_ = number_of_dosels.x * number_of_dosels.y * number_of_dosels.z;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_.get(), dose_params_device);

  // Allocated buffers storing dose on OpenCL device
  dose_recording_.edep_ = opencl_manager.Allocate(nullptr, total_number_of_dosels*sizeof(GGDosiType), CL_MEM_READ_WRITE);
  dose_recording_.edep_squared_ = opencl_manager.Allocate(nullptr, total_number_of_dosels*sizeof(GGDosiType), CL_MEM_READ_WRITE);
  dose_recording_.hit_ = opencl_manager.Allocate(nullptr, total_number_of_dosels*sizeof(GGint), CL_MEM_READ_WRITE);
  dose_recording_.photon_tracking_ = opencl_manager.Allocate(nullptr, total_number_of_dosels*sizeof(GGint), CL_MEM_READ_WRITE);
  dose_recording_.dose_ = opencl_manager.Allocate(nullptr, total_number_of_dosels*sizeof(GGfloat), CL_MEM_READ_WRITE);
  dose_recording_.uncertainty_dose_ = opencl_manager.Allocate(nullptr, total_number_of_dosels*sizeof(GGfloat), CL_MEM_READ_WRITE);

  // Set buffer to zero
  opencl_manager.Clean(dose_recording_.edep_, total_number_of_dosels*sizeof(GGDosiType));
  opencl_manager.Clean(dose_recording_.edep_squared_, total_number_of_dosels*sizeof(GGDosiType));
  opencl_manager.Clean(dose_recording_.hit_, total_number_of_dosels*sizeof(GGint));
  opencl_manager.Clean(dose_recording_.photon_tracking_, total_number_of_dosels*sizeof(GGint));
  opencl_manager.Clean(dose_recording_.dose_, total_number_of_dosels*sizeof(GGfloat));
  opencl_manager.Clean(dose_recording_.uncertainty_dose_, total_number_of_dosels*sizeof(GGfloat));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SavePhotonTracking(std::string const& basename) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device for dose parameters
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_.get(), sizeof(GGEMSDoseParams));

  GGint* photon_tracking = new GGint[dose_params_device->total_number_of_dosels_];
  std::memset(photon_tracking, 0, dose_params_device->total_number_of_dosels_*sizeof(GGint));

  GGEMSMHDImage mhdImage;
  mhdImage.SetBaseName(basename + "_photon_tracking");
  mhdImage.SetDataType("MET_INT");
  mhdImage.SetDimensions(dose_params_device->number_of_dosels_);
  mhdImage.SetElementSizes(dose_params_device->size_of_dosels_);

  GGint* photon_tracking_device = opencl_manager.GetDeviceBuffer<GGint>(dose_recording_.photon_tracking_.get(), dose_params_device->total_number_of_dosels_*sizeof(GGint));

  for (GGint i = 0; i < dose_params_device->total_number_of_dosels_; ++i) photon_tracking[i] = photon_tracking_device[i];

  // Writing data
  mhdImage.Write<GGint>(photon_tracking, dose_params_device->total_number_of_dosels_);
  opencl_manager.ReleaseDeviceBuffer(dose_recording_.photon_tracking_.get(), photon_tracking_device);
  delete[] photon_tracking;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_.get(), dose_params_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SaveHit(std::string const& basename) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device for dose parameters
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_.get(), sizeof(GGEMSDoseParams));

  GGint* hit_tracking = new GGint[dose_params_device->total_number_of_dosels_];
  std::memset(hit_tracking, 0, dose_params_device->total_number_of_dosels_*sizeof(GGint));

  GGEMSMHDImage mhdImage;
  mhdImage.SetBaseName(basename + "_hit");
  mhdImage.SetDataType("MET_INT");
  mhdImage.SetDimensions(dose_params_device->number_of_dosels_);
  mhdImage.SetElementSizes(dose_params_device->size_of_dosels_);

  GGint* hit_device = opencl_manager.GetDeviceBuffer<GGint>(dose_recording_.hit_.get(), dose_params_device->total_number_of_dosels_*sizeof(GGint));

  for (GGint i = 0; i < dose_params_device->total_number_of_dosels_; ++i) hit_tracking[i] = hit_device[i];

  // Writing data
  mhdImage.Write<GGint>(hit_tracking, dose_params_device->total_number_of_dosels_);
  opencl_manager.ReleaseDeviceBuffer(dose_recording_.hit_.get(), hit_device);
  delete[] hit_tracking;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_.get(), dose_params_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SaveEdep(std::string const& basename) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device for dose parameters
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_.get(), sizeof(GGEMSDoseParams));

  GGDosiType* edep_tracking = new GGDosiType[dose_params_device->total_number_of_dosels_];
  std::memset(edep_tracking, 0, dose_params_device->total_number_of_dosels_*sizeof(GGDosiType));

  GGEMSMHDImage mhdImage;
  mhdImage.SetBaseName(basename + "_edep");
  if (sizeof(GGDosiType) == 4) mhdImage.SetDataType("MET_FLOAT");
  else if (sizeof(GGDosiType) == 8) mhdImage.SetDataType("MET_DOUBLE");
  mhdImage.SetDimensions(dose_params_device->number_of_dosels_);
  mhdImage.SetElementSizes(dose_params_device->size_of_dosels_);

  GGDosiType* edep_device = opencl_manager.GetDeviceBuffer<GGDosiType>(dose_recording_.edep_.get(), dose_params_device->total_number_of_dosels_*sizeof(GGDosiType));

  for (GGint i = 0; i < dose_params_device->total_number_of_dosels_; ++i) edep_tracking[i] = edep_device[i];

  // Writing data
  mhdImage.Write<GGDosiType>(edep_tracking, dose_params_device->total_number_of_dosels_);
  opencl_manager.ReleaseDeviceBuffer(dose_recording_.edep_.get(), edep_device);
  delete[] edep_tracking;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_.get(), dose_params_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SaveEdepSquared(std::string const& basename) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device for dose parameters
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_.get(), sizeof(GGEMSDoseParams));

  GGDosiType* edep_squared_tracking = new GGDosiType[dose_params_device->total_number_of_dosels_];
  std::memset(edep_squared_tracking, 0, dose_params_device->total_number_of_dosels_*sizeof(GGDosiType));

  GGEMSMHDImage mhdImage;
  mhdImage.SetBaseName(basename + "_edep_squared");
  if (sizeof(GGDosiType) == 4) mhdImage.SetDataType("MET_FLOAT");
  else if (sizeof(GGDosiType) == 8) mhdImage.SetDataType("MET_DOUBLE");
  mhdImage.SetDimensions(dose_params_device->number_of_dosels_);
  mhdImage.SetElementSizes(dose_params_device->size_of_dosels_);

  GGDosiType* edep_squared_device = opencl_manager.GetDeviceBuffer<GGDosiType>(dose_recording_.edep_squared_.get(), dose_params_device->total_number_of_dosels_*sizeof(GGDosiType));

  for (GGint i = 0; i < dose_params_device->total_number_of_dosels_; ++i) edep_squared_tracking[i] = edep_squared_device[i];

  // Writing data
  mhdImage.Write<GGDosiType>(edep_squared_tracking, dose_params_device->total_number_of_dosels_);
  opencl_manager.ReleaseDeviceBuffer(dose_recording_.edep_squared_.get(), edep_squared_device);
  delete[] edep_squared_tracking;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_.get(), dose_params_device);
}
