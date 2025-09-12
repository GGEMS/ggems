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

#include "GGEMS/navigators/GGEMSDosimetryCalculator.hh"
#include "GGEMS/navigators/GGEMSDoseParams.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolid.hh"
#include "GGEMS/io/GGEMSMHDImage.hh"
#include "GGEMS/tools/GGEMSProfilerManager.hh"

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
  minimum_density_(0.0f),
  kernel_compute_dose_(nullptr)
{
  GGcout("GGEMSDosimetryCalculator", "GGEMSDosimetryCalculator", 3) << "GGEMSDosimetryCalculator creating..." << GGendl;

  dosel_sizes_.s[0] = -1.0f;
  dosel_sizes_.s[1] = -1.0f;
  dosel_sizes_.s[2] = -1.0f;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  // Get the number of activated device
  number_activated_devices_ = opencl_manager.GetNumberOfActivatedDevice();

  // Checking double precision computation
  #ifdef DOSIMETRY_DOUBLE_PRECISION
  // Get device index
  for (GGsize i = 0; i < number_activated_devices_; ++i) {
    GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(i);
    if (!opencl_manager.IsDoublePrecisionAtomicAddition(device_index)) {
      std::ostringstream oss(std::ostringstream::out);
      oss << "Your OpenCL device: " << opencl_manager.GetDeviceName(device_index) << ", does not support double precision for atomic operation!!!" << std::endl;
      oss << "Please, recompile with DOSIMETRY_DOUBLE_PRECISION to OFF. Precision will be lost only for dosimetry application" << std::endl;
      GGEMSMisc::ThrowException("GGEMSDosimetryCalculator", "GGEMSDosimetryCalculator", oss.str());
    }
  }
  #endif

  // Allocating buffer on each OpenCL device for dose params
  dose_params_ = new cl::Buffer*[number_activated_devices_];
  dose_recording_.edep_ = new cl::Buffer*[number_activated_devices_];
  dose_recording_.dose_ = new cl::Buffer*[number_activated_devices_];
  dose_recording_.uncertainty_dose_ = new cl::Buffer*[number_activated_devices_];
  dose_recording_.edep_squared_ = new cl::Buffer*[number_activated_devices_];
  dose_recording_.hit_ = new cl::Buffer*[number_activated_devices_];
  dose_recording_.photon_tracking_ = new cl::Buffer*[number_activated_devices_];

  GGcout("GGEMSDosimetryCalculator", "GGEMSDosimetryCalculator", 3) << "GGEMSDosimetryCalculator created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSDosimetryCalculator::~GGEMSDosimetryCalculator(void)
{
  GGcout("GGEMSDosimetryCalculator", "~GGEMSDosimetryCalculator", 3) << "GGEMSSourceManager erasing..." << GGendl;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  if (dose_params_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(dose_params_[i], sizeof(GGEMSDoseParams ), i);
    }
    delete[] dose_params_;
    dose_params_ = nullptr;
  }

  if (dose_recording_.edep_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(dose_recording_.edep_[i], total_number_of_dosels_*sizeof(GGDosiType), i);
    }
    delete[] dose_recording_.edep_;
    dose_recording_.edep_ = nullptr;
  }

  if (dose_recording_.dose_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(dose_recording_.dose_[i], total_number_of_dosels_*sizeof(GGfloat), i);
    }
    delete[] dose_recording_.dose_;
    dose_recording_.dose_ = nullptr;
  }

  if (dose_recording_.uncertainty_dose_) {
    if (is_uncertainty_) {
      for (GGsize i = 0; i < number_activated_devices_; ++i) {
        opencl_manager.Deallocate(dose_recording_.uncertainty_dose_[i], total_number_of_dosels_*sizeof(GGfloat), i);
      }
    }
    delete[] dose_recording_.uncertainty_dose_;
    dose_recording_.uncertainty_dose_ = nullptr;
  }

  if (dose_recording_.edep_squared_) {
    if (is_edep_squared_||is_uncertainty_) {
      for (GGsize i = 0; i < number_activated_devices_; ++i) {
        opencl_manager.Deallocate(dose_recording_.edep_squared_[i], total_number_of_dosels_*sizeof(GGDosiType), i);
      }
    }
    delete[] dose_recording_.edep_squared_;
    dose_recording_.edep_squared_ = nullptr;
  }

  if (dose_recording_.hit_) {
    if (is_hit_tracking_||is_uncertainty_) {
      for (GGsize i = 0; i < number_activated_devices_; ++i) {
        opencl_manager.Deallocate(dose_recording_.hit_[i], total_number_of_dosels_*sizeof(GGint), i);
      }
    }
    delete[] dose_recording_.hit_;
    dose_recording_.hit_ = nullptr;
  }

  if (dose_recording_.photon_tracking_) {
    if (is_photon_tracking_) {
      for (GGsize i = 0; i < number_activated_devices_; ++i) {
        opencl_manager.Deallocate(dose_recording_.photon_tracking_[i], total_number_of_dosels_*sizeof(GGint), i);
      }
    }
    delete[] dose_recording_.photon_tracking_;
    dose_recording_.photon_tracking_ = nullptr;
  }

  if (kernel_compute_dose_) {
    delete[] kernel_compute_dose_;
    kernel_compute_dose_ = nullptr;
  }

  GGcout("GGEMSDosimetryCalculator", "~GGEMSDosimetryCalculator", 3) << "GGEMSSourceManager erased!!!" << GGendl;
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
  dosel_sizes_.s[0] = DistanceUnit(dosel_x, unit);
  dosel_sizes_.s[1] = DistanceUnit(dosel_y, unit);
  dosel_sizes_.s[2] = DistanceUnit(dosel_z, unit);
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

void GGEMSDosimetryCalculator::SetTLE(bool const& is_activated)
{
  navigator_->EnableTLE(is_activated);
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

void GGEMSDosimetryCalculator::InitializeVoxelizedKernel(void)
{
  GGcout("GGEMSDosimetryCalculator", "InitializeVoxelizedKernel", 3) << "Initializing kernel for dose computation in voxelized volume..." << GGendl;

  // Getting OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Getting the path to kernel
  std::string openCL_kernel_path = OPENCL_KERNEL_PATH;
  std::string compute_dose_filename = openCL_kernel_path + "/ComputeDoseGGEMSVoxelizedSolid.cl";

  // Storing a kernel for each device
  kernel_compute_dose_ = new cl::Kernel*[number_activated_devices_];

  // Compiling the kernels
  opencl_manager.CompileKernel(compute_dose_filename, "compute_dose_ggems_voxelized_solid", kernel_compute_dose_, nullptr, nullptr);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::InitializeMeshedKernel(void)
{
  GGcout("GGEMSDosimetryCalculator", "InitializeMeshedKernel", 3) << "Initializing kernel for dose computation in meshed volume..." << GGendl;

  // Getting OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Getting the path to kernel
  std::string openCL_kernel_path = OPENCL_KERNEL_PATH;
  std::string compute_dose_filename = openCL_kernel_path + "/ComputeDoseGGEMSMeshSolid.cl";

  // Storing a kernel for each device
  kernel_compute_dose_ = new cl::Kernel*[number_activated_devices_];

  // Compiling the kernels
  opencl_manager.CompileKernel(compute_dose_filename, "compute_dose_ggems_meshed_solid", kernel_compute_dose_, nullptr, nullptr);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::ComputeDose(GGsize const& thread_index)
{
  // Getting the OpenCL manager and infos for work-item launching
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue = opencl_manager.GetCommandQueue(thread_index);

  // Get Device name and storing methode name + device
  GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(thread_index);
  std::string device_name = opencl_manager.GetDeviceName(device_index);
  std::ostringstream oss(std::ostringstream::out);
  oss << "GGEMSDosimetryCalculator::ComputeDose in " << device_name << ", index " << device_index;

  // Get pointer on OpenCL device for dose parameters
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_[thread_index], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSDoseParams), thread_index);

  GGsize number_of_dosels = static_cast<GGsize>(dose_params_device->total_number_of_dosels_);

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_[thread_index], dose_params_device, thread_index);

  // Getting kernel, and setting parameters
  GGuint index_arg = 0;
  kernel_compute_dose_[thread_index]->setArg(index_arg++, number_of_dosels);
  kernel_compute_dose_[thread_index]->setArg(index_arg++, *dose_params_[thread_index]);
  kernel_compute_dose_[thread_index]->setArg(index_arg++, *dose_recording_.edep_[thread_index]);

  if (!dose_recording_.hit_[thread_index])
    kernel_compute_dose_[thread_index]->setArg(index_arg++, sizeof(cl_mem), nullptr);
  else
    kernel_compute_dose_[thread_index]->setArg(index_arg++, *dose_recording_.hit_[thread_index]);

  if (!dose_recording_.edep_squared_[thread_index])
    kernel_compute_dose_[thread_index]->setArg(index_arg++, sizeof(cl_mem), nullptr);
  else
    kernel_compute_dose_[thread_index]->setArg(index_arg++, *dose_recording_.edep_squared_[thread_index]);

  kernel_compute_dose_[thread_index]->setArg(index_arg++, *navigator_->GetSolids(0)->GetSolidData(thread_index)); // 1 solid in voxelized phantom
  kernel_compute_dose_[thread_index]->setArg(index_arg++, *navigator_->GetSolids(0)->GetLabelData(thread_index));
  kernel_compute_dose_[thread_index]->setArg(index_arg++, *navigator_->GetMaterials()->GetMaterialTables(thread_index));
  kernel_compute_dose_[thread_index]->setArg(index_arg++, *dose_recording_.dose_[thread_index]);

  if (!dose_recording_.uncertainty_dose_[thread_index])
    kernel_compute_dose_[thread_index]->setArg(index_arg++, sizeof(cl_mem), nullptr);
  else
    kernel_compute_dose_[thread_index]->setArg(index_arg++, *dose_recording_.uncertainty_dose_[thread_index]);

  kernel_compute_dose_[thread_index]->setArg(index_arg++, scale_factor_);
  kernel_compute_dose_[thread_index]->setArg(index_arg++, is_water_reference_);
  kernel_compute_dose_[thread_index]->setArg(index_arg++, minimum_density_);

  GGsize work_group_size = opencl_manager.GetKernelWorkGroupSize(kernel_compute_dose_[thread_index]);
  GGsize max_work_item = opencl_manager.GetDeviceMaxWorkItemSize(device_index, 0) * work_group_size;

  // Number total of work items
  GGsize number_of_work_item = (((number_of_dosels - 1) / work_group_size) + 1) * work_group_size; // Multiple of work group size

  // Organize work item in batch
  bool is_last_batch = false;
  GGsize number_of_work_item_in_batch = max_work_item;
  GGsize number_batchs_work_item = (number_of_work_item / (max_work_item + 1)) + 1;

  for (GGsize i = 0; i < number_batchs_work_item; ++i) {
    if (i == number_batchs_work_item - 1) is_last_batch = true;

    if (is_last_batch) {
      number_of_work_item_in_batch = (number_of_work_item % max_work_item) == 0 ?
        max_work_item :
        number_of_work_item % max_work_item;
    }

    cl::NDRange global_wi(number_of_work_item_in_batch);
    cl::NDRange offset_wi(i * max_work_item);
    cl::NDRange local_wi(number_of_work_item_in_batch / work_group_size);

    cl::Event event;
    GGint kernel_status = queue->enqueueNDRangeKernel(*kernel_compute_dose_[thread_index], offset_wi, global_wi, local_wi, nullptr, &event);
    opencl_manager.CheckOpenCLError(kernel_status, "GGEMSDosimetryCalculator", "ComputeDose");

    // GGEMS Profiling
    GGEMSProfilerManager::GetInstance().HandleEvent(event, oss.str());

    queue->finish();
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::Initialize(std::string const nav_type)
{
  GGcout("GGEMSDosimetryCalculator", "Initialize", 3) << "Initializing dosimetry calculator..." << GGendl;

  CheckParameters();

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Allocating dosimetry parameters on each device
  for (GGsize j = 0; j < number_activated_devices_; ++j) {
    // Allocate dosemetry params on OpenCL device
    dose_params_[j] = opencl_manager.Allocate(nullptr, sizeof(GGEMSDoseParams), j, CL_MEM_READ_WRITE, "GGEMSDosimetryCalculator");

    // Get pointer on OpenCL device for dose parameters
    GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_[j], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSDoseParams), j);

    // Get the voxels size
    GGfloat3 voxel_sizes = dosel_sizes_;
    if (dosel_sizes_.s[0] < 0.0f && dosel_sizes_.s[1] < 0.0f && dosel_sizes_.s[2] < 0.0f) { // Custom dosel size
      voxel_sizes = navigator_->GetSolids(0)->GetVoxelSizes(j);
    }

    // If photon tracking activated, voxel sizes of phantom and dosels size should be the same, otherwise artefacts!!!
    if (is_photon_tracking_) {
      if (voxel_sizes.s[0] != navigator_->GetSolids(0)->GetVoxelSizes(j).s[0] ||
          voxel_sizes.s[1] != navigator_->GetSolids(0)->GetVoxelSizes(j).s[1] ||
          voxel_sizes.s[2] != navigator_->GetSolids(0)->GetVoxelSizes(j).s[2]) {
        std::ostringstream oss(std::ostringstream::out);
        oss << "Dosel size and voxel size in voxelized phantom have to be the same when photon tracking is activated!!!";
        GGEMSMisc::ThrowException("GGEMSDosimetryCalculator", "Initialize", oss.str());
      }
    }

    // Storing voxel size
    dose_params_device->size_of_dosels_ = voxel_sizes;

    // Take inverse of size
    dose_params_device->inv_size_of_dosels_.s[0] = 1.0f / voxel_sizes.s[0];
    dose_params_device->inv_size_of_dosels_.s[1] = 1.0f / voxel_sizes.s[1];
    dose_params_device->inv_size_of_dosels_.s[2] = 1.0f / voxel_sizes.s[2];

    // Get border of volumes from phantom
    GGEMSOBB obb_geometry = navigator_->GetSolids(0)->GetOBBGeometry(j);
    dose_params_device->border_min_xyz_ = obb_geometry.border_min_xyz_;
    dose_params_device->border_max_xyz_ = obb_geometry.border_max_xyz_;

    // Get the size of the dose map
    GGfloat3 dosemap_size;
    for (GGsize i = 0; i < 3; ++i) {
      dosemap_size.s[i] = obb_geometry.border_max_xyz_.s[i] - obb_geometry.border_min_xyz_.s[i];
    }

    // Get the number of voxels
    GGsize3 number_of_dosels;
    number_of_dosels.x_ = static_cast<GGsize>(dosemap_size.s[0] / voxel_sizes.s[0]);
    number_of_dosels.y_ = static_cast<GGsize>(dosemap_size.s[1] / voxel_sizes.s[1]);
    number_of_dosels.z_ = static_cast<GGsize>(dosemap_size.s[2] / voxel_sizes.s[2]);

    dose_params_device->number_of_dosels_.s[0] = static_cast<GGint>(number_of_dosels.x_);
    dose_params_device->number_of_dosels_.s[1] = static_cast<GGint>(number_of_dosels.y_);
    dose_params_device->number_of_dosels_.s[2] = static_cast<GGint>(number_of_dosels.z_);

    dose_params_device->slice_number_of_dosels_ = static_cast<GGint>(number_of_dosels.x_ * number_of_dosels.y_);
    total_number_of_dosels_ = number_of_dosels.x_ * number_of_dosels.y_ * number_of_dosels.z_;
    dose_params_device->total_number_of_dosels_ = static_cast<GGint>(total_number_of_dosels_);

    // Release the pointer
    opencl_manager.ReleaseDeviceBuffer(dose_params_[j], dose_params_device, j);

    // Allocated buffers storing dose on OpenCL device
    dose_recording_.edep_[j] = opencl_manager.Allocate(nullptr, total_number_of_dosels_*sizeof(GGDosiType), j, CL_MEM_READ_WRITE, "GGEMSDosimetryCalculator");
    dose_recording_.dose_[j] = opencl_manager.Allocate(nullptr, total_number_of_dosels_*sizeof(GGfloat), j, CL_MEM_READ_WRITE, "GGEMSDosimetryCalculator");

    dose_recording_.uncertainty_dose_[j] = is_uncertainty_ ? opencl_manager.Allocate(nullptr, total_number_of_dosels_*sizeof(GGfloat), j, CL_MEM_READ_WRITE, "GGEMSDosimetryCalculator") : nullptr;
    dose_recording_.edep_squared_[j] = (is_edep_squared_||is_uncertainty_) ? opencl_manager.Allocate(nullptr, total_number_of_dosels_*sizeof(GGDosiType), j, CL_MEM_READ_WRITE, "GGEMSDosimetryCalculator") : nullptr;
    dose_recording_.hit_[j] = (is_hit_tracking_||is_uncertainty_) ? opencl_manager.Allocate(nullptr, total_number_of_dosels_*sizeof(GGint), j, CL_MEM_READ_WRITE, "GGEMSDosimetryCalculator") : nullptr;

    dose_recording_.photon_tracking_[j] = is_photon_tracking_ ? opencl_manager.Allocate(nullptr, total_number_of_dosels_*sizeof(GGint), j, CL_MEM_READ_WRITE, "GGEMSDosimetryCalculator") : nullptr;

    // Set buffer to zero
    opencl_manager.CleanBuffer(dose_recording_.edep_[j], total_number_of_dosels_*sizeof(GGDosiType), j);
    opencl_manager.CleanBuffer(dose_recording_.dose_[j], total_number_of_dosels_*sizeof(GGfloat), j);

    if (is_uncertainty_) opencl_manager.CleanBuffer(dose_recording_.uncertainty_dose_[j], total_number_of_dosels_*sizeof(GGfloat), j);
    if (is_edep_squared_||is_uncertainty_) opencl_manager.CleanBuffer(dose_recording_.edep_squared_[j], total_number_of_dosels_*sizeof(GGDosiType), j);
    if (is_hit_tracking_||is_uncertainty_) opencl_manager.CleanBuffer(dose_recording_.hit_[j], total_number_of_dosels_*sizeof(GGint), j);

    if (is_photon_tracking_) opencl_manager.CleanBuffer(dose_recording_.photon_tracking_[j], total_number_of_dosels_*sizeof(GGint), j);
  }

  if (nav_type == "VOXELIZED")
    InitializeVoxelizedKernel();
  else if (nav_type == "MESHED")
    InitializeMeshedKernel();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SaveResults(void) const
{
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

void GGEMSDosimetryCalculator::SavePhotonTracking(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device for dose parameters, take data from first device only
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_[0], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSDoseParams), 0);

  GGsize total_number_of_dosels = static_cast<GGsize>(dose_params_device->total_number_of_dosels_);
  GGint* photon_tracking = new GGint[total_number_of_dosels];
  std::memset(photon_tracking, 0, total_number_of_dosels*sizeof(GGint));

  GGsize3 dimensions;
  dimensions.x_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.s[0]);
  dimensions.y_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.s[1]);
  dimensions.z_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.s[2]);

  GGEMSMHDImage mhdImage;
  mhdImage.SetOutputFileName(dosimetry_output_filename_ + "_photon_tracking.mhd");
  mhdImage.SetDataType("MET_INT");
  mhdImage.SetDimensions(dimensions);
  mhdImage.SetElementSizes(dose_params_device->size_of_dosels_);

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_[0], dose_params_device, 0);

  // Loop over all activated device
  for (GGsize j = 0; j < number_activated_devices_; ++j) {
    GGint* photon_tracking_device = opencl_manager.GetDeviceBuffer<GGint>(dose_recording_.photon_tracking_[j], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, total_number_of_dosels*sizeof(GGint), j);

    for (GGsize i = 0; i < total_number_of_dosels; ++i) photon_tracking[i] += photon_tracking_device[i];
    opencl_manager.ReleaseDeviceBuffer(dose_recording_.photon_tracking_[j], photon_tracking_device, j);
  }

  // Writing data
  mhdImage.Write<GGint>(photon_tracking);
  delete[] photon_tracking;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SaveHit(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device for dose parameters, take data from first device only
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_[0], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSDoseParams), 0);

  GGsize total_number_of_dosels = static_cast<GGsize>(dose_params_device->total_number_of_dosels_);
  GGint* hit_tracking = new GGint[total_number_of_dosels];
  std::memset(hit_tracking, 0, total_number_of_dosels*sizeof(GGint));

  GGsize3 dimensions;
  dimensions.x_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.s[0]);
  dimensions.y_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.s[1]);
  dimensions.z_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.s[2]);

  GGEMSMHDImage mhdImage;
  mhdImage.SetOutputFileName(dosimetry_output_filename_ + "_hit.mhd");
  mhdImage.SetDataType("MET_INT");
  mhdImage.SetDimensions(dimensions);
  mhdImage.SetElementSizes(dose_params_device->size_of_dosels_);

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_[0], dose_params_device, 0);

  // Loop over all activated device
  for (GGsize j = 0; j < number_activated_devices_; ++j) {
    GGint* hit_device = opencl_manager.GetDeviceBuffer<GGint>(dose_recording_.hit_[j], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, total_number_of_dosels*sizeof(GGint), j);

    for (GGsize i = 0; i < total_number_of_dosels; ++i) hit_tracking[i] = hit_device[i];

    opencl_manager.ReleaseDeviceBuffer(dose_recording_.hit_[j], hit_device, j);
  }

  // Writing data
  mhdImage.Write<GGint>(hit_tracking);
  delete[] hit_tracking;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SaveEdep(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device for dose parameters, take data from first device only
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_[0], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSDoseParams), 0);

  GGsize total_number_of_dosels = static_cast<GGsize>(dose_params_device->total_number_of_dosels_);
  GGDosiType* edep_tracking = new GGDosiType[total_number_of_dosels];
  std::memset(edep_tracking, 0, total_number_of_dosels*sizeof(GGDosiType));

  GGsize3 dimensions;
  dimensions.x_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.s[0]);
  dimensions.y_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.s[1]);
  dimensions.z_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.s[2]);

  GGEMSMHDImage mhdImage;
  mhdImage.SetOutputFileName(dosimetry_output_filename_ + "_edep.mhd");
  if (sizeof(GGDosiType) == 4) mhdImage.SetDataType("MET_FLOAT");
  else if (sizeof(GGDosiType) == 8) mhdImage.SetDataType("MET_DOUBLE");
  mhdImage.SetDimensions(dimensions);
  mhdImage.SetElementSizes(dose_params_device->size_of_dosels_);

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_[0], dose_params_device, 0);

  // Loop over all activated device
  for (GGsize j = 0; j < number_activated_devices_; ++j) {
    GGDosiType* edep_device = opencl_manager.GetDeviceBuffer<GGDosiType>(dose_recording_.edep_[j], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, total_number_of_dosels*sizeof(GGDosiType), j);

    for (GGsize i = 0; i < total_number_of_dosels; ++i) edep_tracking[i] = edep_device[i];

    opencl_manager.ReleaseDeviceBuffer(dose_recording_.edep_[j], edep_device, j);
  }

  // Writing data
  mhdImage.Write<GGDosiType>(edep_tracking);
  delete[] edep_tracking;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SaveEdepSquared(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device for dose parameters, take data from first device only
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_[0], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSDoseParams), 0);

  GGsize total_number_of_dosels = static_cast<GGsize>(dose_params_device->total_number_of_dosels_);
  GGDosiType* edep_squared_tracking = new GGDosiType[total_number_of_dosels];
  std::memset(edep_squared_tracking, 0, total_number_of_dosels*sizeof(GGDosiType));

  GGsize3 dimensions;
  dimensions.x_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.s[0]);
  dimensions.y_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.s[1]);
  dimensions.z_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.s[2]);

  GGEMSMHDImage mhdImage;
  mhdImage.SetOutputFileName(dosimetry_output_filename_ + "_edep_squared.mhd");
  if (sizeof(GGDosiType) == 4) mhdImage.SetDataType("MET_FLOAT");
  else if (sizeof(GGDosiType) == 8) mhdImage.SetDataType("MET_DOUBLE");
  mhdImage.SetDimensions(dimensions);
  mhdImage.SetElementSizes(dose_params_device->size_of_dosels_);

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_[0], dose_params_device, 0);

  // Loop over all activated device
  for (GGsize j = 0; j < number_activated_devices_; ++j) {
    GGDosiType* edep_squared_device = opencl_manager.GetDeviceBuffer<GGDosiType>(dose_recording_.edep_squared_[j], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, total_number_of_dosels*sizeof(GGDosiType), j);

    for (GGsize i = 0; i < total_number_of_dosels; ++i) edep_squared_tracking[i] = edep_squared_device[i];

    opencl_manager.ReleaseDeviceBuffer(dose_recording_.edep_squared_[j], edep_squared_device, j);
  }

  // Writing data
  mhdImage.Write<GGDosiType>(edep_squared_tracking);
  delete[] edep_squared_tracking;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SaveDose(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device for dose parameters, take data from first device only
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_[0], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSDoseParams), 0);

  GGsize total_number_of_dosels = static_cast<GGsize>(dose_params_device->total_number_of_dosels_);
  GGfloat* dose = new GGfloat[total_number_of_dosels];
  std::memset(dose, 0, total_number_of_dosels*sizeof(GGfloat));

  GGsize3 dimensions;
  dimensions.x_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.s[0]);
  dimensions.y_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.s[1]);
  dimensions.z_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.s[2]);

  GGEMSMHDImage mhdImage;
  mhdImage.SetOutputFileName(dosimetry_output_filename_ + "_dose.mhd");
  mhdImage.SetDataType("MET_FLOAT");
  mhdImage.SetDimensions(dimensions);
  mhdImage.SetElementSizes(dose_params_device->size_of_dosels_);

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_[0], dose_params_device, 0);

  // Loop over all activated device
  for (GGsize j = 0; j < number_activated_devices_; ++j) {
    GGfloat* dose_device = opencl_manager.GetDeviceBuffer<GGfloat>(dose_recording_.dose_[j], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, total_number_of_dosels*sizeof(GGfloat), j);

    for (GGsize i = 0; i < total_number_of_dosels; ++i) dose[i] += dose_device[i];

    opencl_manager.ReleaseDeviceBuffer(dose_recording_.dose_[j], dose_device, j);
  }

  // Writing data
  mhdImage.Write<GGfloat>(dose);
  delete[] dose;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SaveUncertainty(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device for dose parameters, take data from first device only
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_[0], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSDoseParams), 0);

  GGsize total_number_of_dosels = static_cast<GGsize>(dose_params_device->total_number_of_dosels_);
  GGfloat* uncertainty = new GGfloat[total_number_of_dosels];
  std::memset(uncertainty, 0, total_number_of_dosels*sizeof(GGfloat));

  GGsize3 dimensions;
  dimensions.x_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.s[0]);
  dimensions.y_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.s[1]);
  dimensions.z_ = static_cast<GGsize>(dose_params_device->number_of_dosels_.s[2]);

  GGEMSMHDImage mhdImage;
  mhdImage.SetOutputFileName(dosimetry_output_filename_ + "_uncertainty.mhd");
  mhdImage.SetDataType("MET_FLOAT");
  mhdImage.SetDimensions(dimensions);
  mhdImage.SetElementSizes(dose_params_device->size_of_dosels_);

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_[0], dose_params_device, 0);

  // Loop over all activated device
  for (GGsize j = 0; j < number_activated_devices_; ++j) {
    GGfloat* uncertainty_device = opencl_manager.GetDeviceBuffer<GGfloat>(dose_recording_.uncertainty_dose_[j], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, total_number_of_dosels*sizeof(GGfloat), j);

    for (GGsize i = 0; i < total_number_of_dosels; ++i) uncertainty[i] = uncertainty_device[i];

    opencl_manager.ReleaseDeviceBuffer(dose_recording_.uncertainty_dose_[j], uncertainty_device, j);
  }

  // Writing data
  mhdImage.Write<GGfloat>(uncertainty);
  delete[] uncertainty;
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

void dose_tle_navigator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated)
{
 dose_calculator->SetTLE(is_activated);
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
