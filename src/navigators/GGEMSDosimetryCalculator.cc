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

#include "GGEMS/global/GGEMSManager.hh"
#include "GGEMS/navigators/GGEMSDosimetryCalculator.hh"
#include "GGEMS/navigators/GGEMSDoseParams.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolid.hh"
#include "GGEMS/io/GGEMSMHDImage.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSDosimetryCalculator::GGEMSDosimetryCalculator(void)
: dosimetry_output_filename_("dosi"),
  navigator_(nullptr),
  is_photon_tracking_(false),
  is_edep_(false),
  is_hit_tracking_(false),
  is_edep_squared_(false),
  is_uncertainty_(false),
  scale_factor_(1.0f),
  is_water_reference_(FALSE),
  minimum_density_(0.0f)
{
  GGcout("GGEMSDosimetryCalculator", "GGEMSDosimetryCalculator", 3) << "Allocation of GGEMSDosimetryCalculator..." << GGendl;

  dosel_sizes_.x = -1.0f;
  dosel_sizes_.y = -1.0f;
  dosel_sizes_.z = -1.0f;

  // Checking double precision computation
  #ifdef DOSIMETRY_DOUBLE_PRECISION
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  if (!opencl_manager.IsDoublePrecisionAtomicAddition()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Your OpenCL device does not support double precision for atomic operation!!!" << std::endl;
    oss << "Please, recompile with DOSIMETRY_DOUBLE_PRECISION to OFF. Precision will be lost only for dosimetry application" << std::endl;
    GGEMSMisc::ThrowException("GGEMSDosimetryCalculator", "GGEMSDosimetryCalculator", oss.str());
  }
  #endif
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

void GGEMSDosimetryCalculator::AttachToNavigator(std::string const& navigator_name)
{
  GGEMSNavigatorManager& navigator_manager = GGEMSNavigatorManager::GetInstance();
  navigator_ = navigator_manager.GetNavigator(navigator_name);

  // Activate dosimetry mode in navigator
  navigator_->SetDosimetryCalculator(this);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SetDoselSizes(GGfloat const& dosel_x, GGfloat const& dosel_y, GGfloat const& dosel_z, std::string const& unit)
{
  dosel_sizes_.x = DistanceUnit(dosel_x, unit);
  dosel_sizes_.y = DistanceUnit(dosel_y, unit);
  dosel_sizes_.z = DistanceUnit(dosel_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SetOutputDosimetryBasename(std::string const& output_filename)
{
  dosimetry_output_filename_ = output_filename;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SetScaleFactor(GGfloat const& scale_factor)
{
  scale_factor_ = scale_factor;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SetPhotonTracking(bool const& is_activated)
{
  is_photon_tracking_ = is_activated;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SetEdep(bool const& is_activated)
{
  is_edep_ = is_activated;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SetHitTracking(bool const& is_activated)
{
  is_hit_tracking_ = is_activated;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SetEdepSquared(bool const& is_activated)
{
  is_edep_squared_ = is_activated;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SetUncertainty(bool const& is_activated)
{
  is_uncertainty_ = is_activated;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SetWaterReference(bool const& is_activated)
{
  if (is_activated) is_water_reference_ = TRUE;
  else is_water_reference_ = FALSE;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SetMinimumDensity(float const& minimum_density, std::string const& unit)
{
  minimum_density_ = DensityUnit(minimum_density, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::CheckParameters(void) const
{
  if (!navigator_) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "A navigator has to be associated to GGEMSDosimetryCalculator!!!";
    GGEMSMisc::ThrowException("GGEMSDosimetryCalculator", "CheckParameters", oss.str());
  }
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

void GGEMSDosimetryCalculator::ComputeDoseAndSaveResults(void)
{
  // Getting the OpenCL manager and infos for work-item launching
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue = opencl_manager.GetCommandQueue();
  cl::Event* event = opencl_manager.GetEvent();

  // Get pointer on OpenCL device for dose parameters
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_.get(), sizeof(GGEMSDoseParams));

  GGsize number_of_dosels = static_cast<GGsize>(dose_params_device->total_number_of_dosels_);

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_.get(), dose_params_device);

  // Getting work group size, and work-item number
  GGsize work_group_size = opencl_manager.GetWorkGroupSize();
  GGsize number_of_work_items = opencl_manager.GetBestWorkItem(number_of_dosels);

  // Parameters for work-item in kernel
  cl::NDRange global_wi(number_of_work_items);
  cl::NDRange local_wi(work_group_size);

  // Getting kernel, and setting parameters
  std::shared_ptr<cl::Kernel> kernel = kernel_compute_dose_.lock();
  kernel->setArg(0, number_of_dosels);
  kernel->setArg(1, *dose_params_.get());
  kernel->setArg(2, *dose_recording_.edep_.get());
  kernel->setArg(3, *dose_recording_.hit_.get());
  kernel->setArg(4, *dose_recording_.edep_squared_.get());
  kernel->setArg(5, *navigator_->GetSolids().at(0)->GetSolidData()); // 1 solid in voxelized phantom
  kernel->setArg(6, *navigator_->GetSolids().at(0)->GetLabelData());
  kernel->setArg(7, *navigator_->GetMaterials().lock()->GetMaterialTables().lock().get());
  kernel->setArg(8, *dose_recording_.dose_.get());
  kernel->setArg(9, *dose_recording_.uncertainty_dose_.get());
  kernel->setArg(10, scale_factor_);
  kernel->setArg(11, is_water_reference_);
  kernel->setArg(12, minimum_density_);

  // Launching kernel
  GGint kernel_status = queue->enqueueNDRangeKernel(*kernel, 0, global_wi, local_wi, nullptr, event);
  opencl_manager.CheckOpenCLError(kernel_status, "GGEMSDosimetryCalculator", "ComputeDose");
  queue->finish(); // Wait until the kernel status is finish

  // Get GGEMS Manager
  GGEMSManager& ggems_manager = GGEMSManager::GetInstance();
  if (ggems_manager.IsKernelVerbose()) {
    GGEMSChrono::DisplayTime(opencl_manager.GetElapsedTimeInKernel(), "Dose Computation");
  }

  // Saving results
  SaveDose();
  if (is_photon_tracking_) SavePhotonTracking();
  if (is_edep_) SaveEdep();
  if (is_hit_tracking_) SaveHit();
  if (is_edep_squared_) SaveEdepSquared();
  if (is_uncertainty_) SaveUncertainty();
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

  // If photon tracking activated, voxel sizes of phantom and dosels size should be the same, otherwise artefacts!!!
  if (is_photon_tracking_) {
    if (voxel_sizes.x != dynamic_cast<GGEMSVoxelizedSolid*>(navigator_->GetSolids().at(0).get())->GetVoxelSizes().x ||
        voxel_sizes.y != dynamic_cast<GGEMSVoxelizedSolid*>(navigator_->GetSolids().at(0).get())->GetVoxelSizes().y ||
        voxel_sizes.z != dynamic_cast<GGEMSVoxelizedSolid*>(navigator_->GetSolids().at(0).get())->GetVoxelSizes().z) {
      std::ostringstream oss(std::ostringstream::out);
      oss << "Dosel size and voxel size in voxelized phantom have to be the same when photon tracking is activated!!!";
      GGEMSMisc::ThrowException("GGEMSDosimetryCalculator", "Initialize", oss.str());
    }
  }

  // Storing voxel size
  dose_params_device->size_of_dosels_ = voxel_sizes;

  // Take inverse of size
  dose_params_device->inv_size_of_dosels_.x = 1.0f / voxel_sizes.x;
  dose_params_device->inv_size_of_dosels_.y = 1.0f / voxel_sizes.y;
  dose_params_device->inv_size_of_dosels_.z = 1.0f / voxel_sizes.z;

  // Get border of volumes from phantom
  GGEMSOBB obb_geometry = dynamic_cast<GGEMSVoxelizedSolid*>(navigator_->GetSolids().at(0).get())->GetOBBGeometry();
  dose_params_device->border_min_xyz_ = obb_geometry.border_min_xyz_;
  dose_params_device->border_max_xyz_ = obb_geometry.border_max_xyz_;

  // Get the size of the dose map
  GGfloat3 dosemap_size;
  for (GGsize i = 0; i < 3; ++i) {
    dosemap_size.s[i] = obb_geometry.border_max_xyz_.s[i] - obb_geometry.border_min_xyz_.s[i];
  }

  // Get the number of voxels
  GGsize3 number_of_dosels;
  number_of_dosels.x_ = static_cast<GGsize>(dosemap_size.x / voxel_sizes.x);
  number_of_dosels.y_ = static_cast<GGsize>(dosemap_size.y / voxel_sizes.y);
  number_of_dosels.z_ = static_cast<GGsize>(dosemap_size.z / voxel_sizes.z);

  dose_params_device->number_of_dosels_.x = static_cast<GGint>(number_of_dosels.x_);
  dose_params_device->number_of_dosels_.y = static_cast<GGint>(number_of_dosels.y_);
  dose_params_device->number_of_dosels_.z = static_cast<GGint>(number_of_dosels.z_);

  dose_params_device->slice_number_of_dosels_ = static_cast<GGint>(number_of_dosels.x_ * number_of_dosels.y_);
  GGsize total_number_of_dosels = number_of_dosels.x_ * number_of_dosels.y_ * number_of_dosels.z_;
  dose_params_device->total_number_of_dosels_ = static_cast<GGint>(total_number_of_dosels);

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_.get(), dose_params_device);

  // Allocated buffers storing dose on OpenCL device
  dose_recording_.edep_ = opencl_manager.Allocate(nullptr, total_number_of_dosels*sizeof(GGDosiType), CL_MEM_READ_WRITE);
  dose_recording_.dose_ = opencl_manager.Allocate(nullptr, total_number_of_dosels*sizeof(GGfloat), CL_MEM_READ_WRITE);

  dose_recording_.uncertainty_dose_ = is_uncertainty_ ? opencl_manager.Allocate(nullptr, total_number_of_dosels*sizeof(GGfloat), CL_MEM_READ_WRITE) : nullptr;
  dose_recording_.edep_squared_ = (is_edep_squared_||is_uncertainty_) ? opencl_manager.Allocate(nullptr, total_number_of_dosels*sizeof(GGDosiType), CL_MEM_READ_WRITE) : nullptr;
  dose_recording_.hit_ = (is_hit_tracking_||is_uncertainty_) ? opencl_manager.Allocate(nullptr, total_number_of_dosels*sizeof(GGint), CL_MEM_READ_WRITE) : nullptr;

  dose_recording_.photon_tracking_ = is_photon_tracking_ ? opencl_manager.Allocate(nullptr, total_number_of_dosels*sizeof(GGint), CL_MEM_READ_WRITE) : nullptr;

  // Set buffer to zero
  opencl_manager.CleanBuffer(dose_recording_.edep_, total_number_of_dosels*sizeof(GGDosiType));
  opencl_manager.CleanBuffer(dose_recording_.dose_, total_number_of_dosels*sizeof(GGfloat));

  if (is_uncertainty_) opencl_manager.CleanBuffer(dose_recording_.uncertainty_dose_, total_number_of_dosels*sizeof(GGfloat));
  if (is_edep_squared_||is_uncertainty_) opencl_manager.CleanBuffer(dose_recording_.edep_squared_, total_number_of_dosels*sizeof(GGDosiType));
  if (is_hit_tracking_||is_uncertainty_) opencl_manager.CleanBuffer(dose_recording_.hit_, total_number_of_dosels*sizeof(GGint));

  if (is_photon_tracking_) opencl_manager.CleanBuffer(dose_recording_.photon_tracking_, total_number_of_dosels*sizeof(GGint));

  InitializeKernel();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SavePhotonTracking(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device for dose parameters
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_.get(), sizeof(GGEMSDoseParams));

  GGsize total_number_of_dosels = static_cast<GGsize>(dose_params_device->total_number_of_dosels_);
  GGint* photon_tracking = new GGint[total_number_of_dosels];
  std::memset(photon_tracking, 0, total_number_of_dosels*sizeof(GGint));

  GGsize3 dimensions;
  dimensions.x_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.x);
  dimensions.y_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.y);
  dimensions.z_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.z);

  GGEMSMHDImage mhdImage;
  mhdImage.SetOutputFileName(dosimetry_output_filename_ + "_photon_tracking.mhd");
  mhdImage.SetDataType("MET_INT");
  mhdImage.SetDimensions(dimensions);
  mhdImage.SetElementSizes(dose_params_device->size_of_dosels_);

  GGint* photon_tracking_device = opencl_manager.GetDeviceBuffer<GGint>(dose_recording_.photon_tracking_.get(), total_number_of_dosels*sizeof(GGint));

  for (GGsize i = 0; i < total_number_of_dosels; ++i) photon_tracking[i] = photon_tracking_device[i];

  // Writing data
  mhdImage.Write<GGint>(photon_tracking, total_number_of_dosels);
  opencl_manager.ReleaseDeviceBuffer(dose_recording_.photon_tracking_.get(), photon_tracking_device);
  delete[] photon_tracking;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_.get(), dose_params_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SaveHit(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device for dose parameters
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_.get(), sizeof(GGEMSDoseParams));

  GGsize total_number_of_dosels = static_cast<GGsize>(dose_params_device->total_number_of_dosels_);
  GGint* hit_tracking = new GGint[total_number_of_dosels];
  std::memset(hit_tracking, 0, total_number_of_dosels*sizeof(GGint));

  GGsize3 dimensions;
  dimensions.x_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.x);
  dimensions.y_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.y);
  dimensions.z_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.z);

  GGEMSMHDImage mhdImage;
  mhdImage.SetOutputFileName(dosimetry_output_filename_ + "_hit.mhd");
  mhdImage.SetDataType("MET_INT");
  mhdImage.SetDimensions(dimensions);
  mhdImage.SetElementSizes(dose_params_device->size_of_dosels_);

  GGint* hit_device = opencl_manager.GetDeviceBuffer<GGint>(dose_recording_.hit_.get(), total_number_of_dosels*sizeof(GGint));

  for (GGsize i = 0; i < total_number_of_dosels; ++i) hit_tracking[i] = hit_device[i];

  // Writing data
  mhdImage.Write<GGint>(hit_tracking, total_number_of_dosels);
  opencl_manager.ReleaseDeviceBuffer(dose_recording_.hit_.get(), hit_device);
  delete[] hit_tracking;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_.get(), dose_params_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SaveEdep(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device for dose parameters
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_.get(), sizeof(GGEMSDoseParams));

  GGsize total_number_of_dosels = static_cast<GGsize>(dose_params_device->total_number_of_dosels_);
  GGDosiType* edep_tracking = new GGDosiType[total_number_of_dosels];
  std::memset(edep_tracking, 0, total_number_of_dosels*sizeof(GGDosiType));

  GGsize3 dimensions;
  dimensions.x_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.x);
  dimensions.y_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.y);
  dimensions.z_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.z);

  GGEMSMHDImage mhdImage;
  mhdImage.SetOutputFileName(dosimetry_output_filename_ + "_edep.mhd");
  if (sizeof(GGDosiType) == 4) mhdImage.SetDataType("MET_FLOAT");
  else if (sizeof(GGDosiType) == 8) mhdImage.SetDataType("MET_DOUBLE");
  mhdImage.SetDimensions(dimensions);
  mhdImage.SetElementSizes(dose_params_device->size_of_dosels_);

  GGDosiType* edep_device = opencl_manager.GetDeviceBuffer<GGDosiType>(dose_recording_.edep_.get(), total_number_of_dosels*sizeof(GGDosiType));

  for (GGsize i = 0; i < total_number_of_dosels; ++i) edep_tracking[i] = edep_device[i];

  // Writing data
  mhdImage.Write<GGDosiType>(edep_tracking, total_number_of_dosels);
  opencl_manager.ReleaseDeviceBuffer(dose_recording_.edep_.get(), edep_device);
  delete[] edep_tracking;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_.get(), dose_params_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SaveEdepSquared(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device for dose parameters
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_.get(), sizeof(GGEMSDoseParams));

  GGsize total_number_of_dosels = static_cast<GGsize>(dose_params_device->total_number_of_dosels_);
  GGDosiType* edep_squared_tracking = new GGDosiType[total_number_of_dosels];
  std::memset(edep_squared_tracking, 0, total_number_of_dosels*sizeof(GGDosiType));

  GGsize3 dimensions;
  dimensions.x_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.x);
  dimensions.y_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.y);
  dimensions.z_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.z);

  GGEMSMHDImage mhdImage;
  mhdImage.SetOutputFileName(dosimetry_output_filename_ + "_edep_squared.mhd");
  if (sizeof(GGDosiType) == 4) mhdImage.SetDataType("MET_FLOAT");
  else if (sizeof(GGDosiType) == 8) mhdImage.SetDataType("MET_DOUBLE");
  mhdImage.SetDimensions(dimensions);
  mhdImage.SetElementSizes(dose_params_device->size_of_dosels_);

  GGDosiType* edep_squared_device = opencl_manager.GetDeviceBuffer<GGDosiType>(dose_recording_.edep_squared_.get(), total_number_of_dosels*sizeof(GGDosiType));

  for (GGsize i = 0; i < total_number_of_dosels; ++i) edep_squared_tracking[i] = edep_squared_device[i];

  // Writing data
  mhdImage.Write<GGDosiType>(edep_squared_tracking, total_number_of_dosels);
  opencl_manager.ReleaseDeviceBuffer(dose_recording_.edep_squared_.get(), edep_squared_device);
  delete[] edep_squared_tracking;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_.get(), dose_params_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SaveDose(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device for dose parameters
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_.get(), sizeof(GGEMSDoseParams));

  GGsize total_number_of_dosels = static_cast<GGsize>(dose_params_device->total_number_of_dosels_);
  GGfloat* dose = new GGfloat[total_number_of_dosels];
  std::memset(dose, 0, total_number_of_dosels*sizeof(GGfloat));

  GGsize3 dimensions;
  dimensions.x_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.x);
  dimensions.y_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.y);
  dimensions.z_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.z);

  GGEMSMHDImage mhdImage;
  mhdImage.SetOutputFileName(dosimetry_output_filename_ + "_dose.mhd");
  mhdImage.SetDataType("MET_FLOAT");
  mhdImage.SetDimensions(dimensions);
  mhdImage.SetElementSizes(dose_params_device->size_of_dosels_);

  GGfloat* dose_device = opencl_manager.GetDeviceBuffer<GGfloat>(dose_recording_.dose_.get(), total_number_of_dosels*sizeof(GGfloat));

  for (GGsize i = 0; i < total_number_of_dosels; ++i) dose[i] = dose_device[i];

  // Writing data
  mhdImage.Write<GGfloat>(dose, total_number_of_dosels);
  opencl_manager.ReleaseDeviceBuffer(dose_recording_.dose_.get(), dose_device);
  delete[] dose;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_.get(), dose_params_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SaveUncertainty(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device for dose parameters
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_.get(), sizeof(GGEMSDoseParams));

  GGsize total_number_of_dosels = static_cast<GGsize>(dose_params_device->total_number_of_dosels_);
  GGfloat* uncertainty = new GGfloat[total_number_of_dosels];
  std::memset(uncertainty, 0, total_number_of_dosels*sizeof(GGfloat));

  GGsize3 dimensions;
  dimensions.x_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.x);
  dimensions.y_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.y);
  dimensions.z_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.z);

  GGEMSMHDImage mhdImage;
  mhdImage.SetOutputFileName(dosimetry_output_filename_ + "_uncertainty.mhd");
  mhdImage.SetDataType("MET_FLOAT");
  mhdImage.SetDimensions(dimensions);
  mhdImage.SetElementSizes(dose_params_device->size_of_dosels_);

  GGfloat* uncertainty_device = opencl_manager.GetDeviceBuffer<GGfloat>(dose_recording_.uncertainty_dose_.get(), total_number_of_dosels*sizeof(GGfloat));

  for (GGsize i = 0; i < total_number_of_dosels; ++i) uncertainty[i] = uncertainty_device[i];

  // Writing data
  mhdImage.Write<GGfloat>(uncertainty, total_number_of_dosels);
  opencl_manager.ReleaseDeviceBuffer(dose_recording_.uncertainty_dose_.get(), uncertainty_device);
  delete[] uncertainty;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_.get(), dose_params_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSDosimetryCalculator* create_ggems_dosimetry_calculator(void)
{
  return new(std::nothrow) GGEMSDosimetryCalculator();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void delete_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator)
{
  if (dose_calculator) {
    delete dose_calculator;
    dose_calculator = nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void scale_factor_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, GGfloat const scale_factor)
{
  dose_calculator->SetScaleFactor(scale_factor);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_dosel_size_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, GGfloat const dose_x, GGfloat const dose_y, GGfloat const dose_z, char const* unit)
{
  dose_calculator->SetDoselSizes(dose_x, dose_y, dose_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_dose_output_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, char const* dose_output_filename)
{
  dose_calculator->SetOutputDosimetryBasename(dose_output_filename);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void dose_photon_tracking_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated)
{
  dose_calculator->SetPhotonTracking(is_activated);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void dose_edep_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated)
{
  dose_calculator->SetEdep(is_activated);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void dose_hit_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated)
{
  dose_calculator->SetHitTracking(is_activated);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void dose_edep_squared_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated)
{
  dose_calculator->SetEdepSquared(is_activated);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void dose_uncertainty_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated)
{
  dose_calculator->SetUncertainty(is_activated);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void water_reference_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated)
{
  dose_calculator->SetWaterReference(is_activated);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void minimum_density_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, GGfloat const minimum_density, char const* unit)
{
  dose_calculator->SetMinimumDensity(minimum_density, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void attach_to_navigator_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, char const* navigator)
{
  dose_calculator->AttachToNavigator(navigator);
}
