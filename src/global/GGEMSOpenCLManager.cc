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
  \file GGEMSOpenCLManager.cc

  \brief Singleton class storing all informations about OpenCL and managing GPU/CPU devices, contexts, kernels, command queues and events. In GGEMS the strategy is 1 context = 1 device.

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 23, 2021
*/

#include "GGEMS/tools/GGEMSRAMManager.hh"
#include "GGEMS/geometries/GGEMSVolumeCreatorManager.hh"
#include "GGEMS/tools/GGEMSProfilerManager.hh"
#include "GGEMS/navigators/GGEMSNavigatorManager.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/physics/GGEMSRangeCutsManager.hh"
#include "GGEMS/global/GGEMS.hh"
#include "GGEMS/physics/GGEMSProcessesManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenCLManager::GGEMSOpenCLManager(void)
{
  GGcout("GGEMSOpenCLManager", "GGEMSOpenCLManager", 3) << "GGEMSOpenCLManager creating..." << GGendl;

  GGcout("GGEMSOpenCLManager", "GGEMSOpenCLManager", 1) << "Retrieving OpenCL platform(s)..." << GGendl;
  CheckOpenCLError(cl::Platform::get(&platforms_), "GGEMSOpenCLManager", "GGEMSOpenCLManager");

  // Prevent cache kernel in OpenCL
  #ifndef OPENCL_CACHE_KERNEL_COMPILATION
  #ifdef _MSC_VER
  _putenv("CUDA_CACHE_DISABLE=1");
  #else
  std::string disable_cache("CUDA_CACHE_DISABLE=1");
  putenv(&disable_cache[0]);
  #endif
  #endif

  // Parameters reading infos from platform and device
  std::string info_string("");
  cl_device_type device_type;
  cl_device_fp_config device_fp_config;
  cl_device_exec_capabilities device_exec_capabilities;
  cl_device_mem_cache_type device_mem_cache_type;
  cl_device_local_mem_type device_local_mem_type;
  cl_device_affinity_domain device_affinity_domain;
  GGuint info_uint = 0;
  GGulong info_ulong = 0;
  GGsize info_size = 0;
  GGbool info_bool = false;
  char char_data[1024];
  GGsize size_data[3] = {0, 0, 0};

  // Getting infos about platform
  for (auto& p : platforms_) {
    CheckOpenCLError(p.getInfo(CL_PLATFORM_PROFILE, &info_string), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    platform_profile_.push_back(info_string);

    CheckOpenCLError(p.getInfo(CL_PLATFORM_VERSION, &info_string), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    platform_version_.push_back(info_string);

    CheckOpenCLError(p.getInfo(CL_PLATFORM_NAME, &info_string), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    platform_name_.push_back(info_string); 

    CheckOpenCLError(p.getInfo(CL_PLATFORM_VENDOR, &info_string), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    platform_vendor_.push_back(info_string);

    CheckOpenCLError(p.getInfo(CL_PLATFORM_EXTENSIONS, &info_string), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    platform_extensions_.push_back(info_string);
  }

  // Retrieve all the available devices
  GGcout("GGEMSOpenCLManager", "GGEMSOpenCLManager", 1) << "Retrieving OpenCL device(s)..." << GGendl;
  for (GGsize i = 0; i < platforms_.size(); ++i) {
    std::vector<cl::Device> all_devices;

    // Getting device from platform
    platforms_[i].getDevices(CL_DEVICE_TYPE_ALL, &all_devices);

    // Storing all devices from platform
    for (auto& d : all_devices) devices_.emplace_back(new cl::Device(d));
  }

  // Getting infos about device
  for (GGsize i = 0; i < devices_.size(); ++i) {
    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_TYPE, &device_type), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_type_.push_back(device_type);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_NAME, &info_string), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_name_.push_back(info_string);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_VENDOR, &info_string), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_vendor_.push_back(info_string);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_VENDOR_ID, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_vendor_id_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_PROFILE, &info_string), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_profile_.push_back(info_string);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_VERSION, &char_data), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_version_.push_back(std::string(char_data));

    CheckOpenCLError(devices_[i]->getInfo(CL_DRIVER_VERSION, &char_data), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_driver_version_.push_back(std::string(char_data));

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_OPENCL_C_VERSION, &char_data), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_opencl_c_version_.push_back(std::string(char_data));

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_native_vector_width_char_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_native_vector_width_short_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_native_vector_width_int_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_native_vector_width_long_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_native_vector_width_half_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_native_vector_width_float_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_native_vector_width_double_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_preferred_vector_width_char_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_preferred_vector_width_short_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_preferred_vector_width_int_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_preferred_vector_width_long_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_preferred_vector_width_half_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_preferred_vector_width_float_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_preferred_vector_width_double_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_ADDRESS_BITS, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_address_bits_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_AVAILABLE, &info_bool), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_available_.push_back(info_bool);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_COMPILER_AVAILABLE, &info_bool), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_compiler_available_.push_back(info_bool);

    if (device_native_vector_width_half_[i] != 0) {
      CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_HALF_FP_CONFIG, &device_fp_config), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
      device_half_fp_config_.push_back(device_fp_config);
    }

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_SINGLE_FP_CONFIG, &device_fp_config), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_single_fp_config_.push_back(device_fp_config);
    if (device_native_vector_width_double_[i] != 0) {
      CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_DOUBLE_FP_CONFIG, &device_fp_config), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
      device_double_fp_config_.push_back(device_fp_config);
    }

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_ENDIAN_LITTLE, &info_bool), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_endian_little_.push_back(info_bool);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_EXTENSIONS, &info_string), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_extensions_.push_back(info_string);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_ERROR_CORRECTION_SUPPORT, &info_bool), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_error_correction_support_.push_back(info_bool);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_EXECUTION_CAPABILITIES, &device_exec_capabilities), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_execution_capabilities_.push_back(device_exec_capabilities);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &info_ulong), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_global_mem_cache_size_.push_back(info_ulong);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, &device_mem_cache_type), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_global_mem_cache_type_.push_back(device_mem_cache_type);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_global_mem_cacheline_size_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &info_ulong), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_global_mem_size_.push_back(info_ulong);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &info_ulong), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_local_mem_size_.push_back(info_ulong);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_LOCAL_MEM_TYPE, &device_local_mem_type), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_local_mem_type_.push_back(device_local_mem_type);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_HOST_UNIFIED_MEMORY, &info_bool), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_host_unified_memory_.push_back(info_bool);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_IMAGE_SUPPORT, &info_bool), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_image_support_.push_back(info_bool);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, &info_size), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_image_max_array_size_.push_back(info_size);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, &info_size), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_image_max_buffer_size_.push_back(info_size);
    
    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_IMAGE2D_MAX_WIDTH, &info_size), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_image2D_max_width_.push_back(info_size);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT, &info_size), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_image2D_max_height_.push_back(info_size);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_IMAGE3D_MAX_WIDTH, &info_size), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_image3D_max_width_.push_back(info_size);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_IMAGE3D_MAX_HEIGHT, &info_size), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_image3D_max_height_.push_back(info_size);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_IMAGE3D_MAX_DEPTH, &info_size), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_image3D_max_depth_.push_back(info_size);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_MAX_READ_IMAGE_ARGS, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_max_read_image_args_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_MAX_WRITE_IMAGE_ARGS, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_max_write_image_args_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_max_clock_frequency_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_max_compute_units_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_MAX_CONSTANT_ARGS, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_max_constant_args_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, &info_ulong), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_max_constant_buffer_size_.push_back(info_ulong);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &info_ulong), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_max_mem_alloc_size_.push_back(info_ulong);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_MAX_PARAMETER_SIZE, &info_size), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_max_parameter_size_.push_back(info_size);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &info_size), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_max_work_group_size_.push_back(info_size);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_max_work_item_dimensions_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_MEM_BASE_ADDR_ALIGN, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_mem_base_addr_align_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &size_data), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    for (GGsize j = 0; j < 3; ++j) device_max_work_item_sizes_.push_back(size_data[j]);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_PRINTF_BUFFER_SIZE, &info_size), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_printf_buffer_size_.push_back(info_size);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_MAX_SAMPLERS, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_max_samplers_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_PARTITION_AFFINITY_DOMAIN, &device_affinity_domain), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_partition_affinity_domain_.push_back(device_affinity_domain);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_PARTITION_MAX_SUB_DEVICES, &info_uint), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_partition_max_sub_devices_.push_back(info_uint);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_PROFILING_TIMER_RESOLUTION, &info_size), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_profiling_timer_resolution_.push_back(info_size);
  }

  // Custom work group size, 64 seems a good trade-off
  work_group_size_ = 64;

  // Define the compilation options by default for OpenCL
  build_options_ = "-cl-std=CL1.2 -w -Werror -cl-fast-relaxed-math";

  // Give precision for dosimetry
  #ifdef DOSIMETRY_DOUBLE_PRECISION
  build_options_ += " -DDOSIMETRY_DOUBLE_PRECISION";
  #endif

  // Add auxiliary function path to OpenCL options
  #ifdef GGEMS_PATH
  build_options_ += " -I";
  build_options_ += GGEMS_PATH;
  build_options_ += "/include";
  #elif
  GGEMSMisc::ThrowException("GGEMSOpenCLManager","GGEMSOpenCLManager", "OPENCL_KERNEL_PATH not defined or not find!!!");
  #endif

  // Filling umap of vendors
  vendors_.insert(std::make_pair("nvidia", "NVIDIA Corporation"));
  vendors_.insert(std::make_pair("intel", "Intel(R) Corporation"));
  vendors_.insert(std::make_pair("amd", "Advanced Micro Devices, Inc."));

  // Create buffer of kernels for all devices
  kernels_.clear();
  kernel_compilation_options_.clear();

  GGcout("GGEMSOpenCLManager", "GGEMSOpenCLManager", 3) << "GGEMSOpenCLManager created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenCLManager::~GGEMSOpenCLManager(void)
{
  GGcout("GGEMSOpenCLManager", "~GGEMSOpenCLManager", 3) << "GGEMSOpenCLManager erasing..." << GGendl;

  GGcout("GGEMSOpenCLManager", "~GGEMSOpenCLManager", 3) << "GGEMSOpenCLManager erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::Clean(void)
{
  GGcout("GGEMSOpenCLManager", "Clean", 3) << "GGEMSOpenCLManager cleaning..." << GGendl;

  // Cleaning all singletons
  GGEMSRAMManager::GetInstance().Clean();
  GGEMSVolumeCreatorManager::GetInstance().Clean();
  GGEMSProfilerManager::GetInstance().Clean();
  GGEMSNavigatorManager::GetInstance().Clean();
  GGEMSSourceManager::GetInstance().Clean();
  GGEMSMaterialsDatabaseManager::GetInstance().Clean();
  GGEMSProcessesManager::GetInstance().Clean();
  GGEMSRangeCutsManager::GetInstance().Clean();

  // Freeing platforms, and platform infos
  platform_profile_.clear();
  platform_version_.clear();
  platform_name_.clear(); 
  platform_vendor_.clear();
  platform_extensions_.clear();
  platforms_.clear();

  // Freeing devices
  for (auto d : devices_) delete d;
  devices_.clear();
  device_indices_.clear();
  device_type_.clear();
  device_name_.clear();
  device_vendor_.clear();
  device_vendor_id_.clear();
  device_profile_.clear();
  device_version_.clear();
  device_driver_version_.clear();
  device_opencl_c_version_.clear();
  device_native_vector_width_char_.clear();
  device_native_vector_width_short_.clear();
  device_native_vector_width_int_.clear();
  device_native_vector_width_long_.clear();
  device_native_vector_width_half_.clear();
  device_native_vector_width_float_.clear();
  device_native_vector_width_double_.clear();
  device_preferred_vector_width_char_.clear();
  device_preferred_vector_width_short_.clear();
  device_preferred_vector_width_int_.clear();
  device_preferred_vector_width_long_.clear();
  device_preferred_vector_width_half_.clear();
  device_preferred_vector_width_float_.clear(); 
  device_preferred_vector_width_double_.clear();
  device_address_bits_.clear();
  device_available_.clear();
  device_compiler_available_.clear();
  device_half_fp_config_.clear();
  device_single_fp_config_.clear();
  device_double_fp_config_.clear();
  device_endian_little_.clear();
  device_extensions_.clear();
  device_error_correction_support_.clear();
  device_execution_capabilities_.clear();
  device_global_mem_cache_size_.clear();
  device_global_mem_cache_type_.clear();
  device_global_mem_cacheline_size_.clear();
  device_global_mem_size_.clear();
  device_local_mem_size_.clear();
  device_local_mem_type_.clear(); 
  device_host_unified_memory_.clear();
  device_image_max_array_size_.clear();
  device_image_max_buffer_size_.clear();
  device_image_support_.clear();
  device_image2D_max_width_.clear();
  device_image2D_max_height_.clear();
  device_image3D_max_width_.clear();
  device_image3D_max_height_.clear();
  device_image3D_max_depth_.clear();
  device_max_read_image_args_.clear();
  device_max_write_image_args_.clear();
  device_max_clock_frequency_.clear();
  device_max_compute_units_.clear();
  device_max_constant_args_.clear();
  device_max_constant_buffer_size_.clear();
  device_max_mem_alloc_size_.clear();
  device_max_parameter_size_.clear();
  device_max_samplers_.clear();
  device_max_work_group_size_.clear();
  device_max_work_item_sizes_.clear();
  device_mem_base_addr_align_.clear();
  device_max_work_item_dimensions_.clear();
  device_printf_buffer_size_.clear();
  device_partition_affinity_domain_.clear();
  device_partition_max_sub_devices_.clear();
  device_profiling_timer_resolution_.clear();

  // Freeing contexts, queues and events
  for (auto c : contexts_) delete c;
  contexts_.clear();
  for (auto q : queues_) delete q;
  queues_.clear();
  for (auto e : events_) delete e;
  events_.clear();

  // Deleting kernel
  for (auto k : kernels_) delete k;
  kernels_.clear();

  GGcout("GGEMSOpenCLManager", "Clean", 3) << "GGEMSOpenCLManager cleaned!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::PrintPlatformInfos(void) const
{
  for (GGsize i = 0; i < platforms_.size(); ++i) {
    GGcout("GGEMSOpenCLManager", "PrintPlatformInfos", 0) << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintPlatformInfos", 0) << "#### PLATFORM: " << i << " ####" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintPlatformInfos", 0) << "    + Platform: " << platform_profile_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintPlatformInfos", 0) << "    + Version: " << platform_version_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintPlatformInfos", 0) << "    + Name: " << platform_name_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintPlatformInfos", 0) << "    + Vendor: " << platform_vendor_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintPlatformInfos", 0) << "    + Extensions: " << platform_extensions_[i] << GGendl;
  }
  GGcout("GGEMSOpenCLManager", "PrintPlatformInfos", 0) << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::PrintDeviceInfos(void) const
{
  for (GGsize i = 0; i < devices_.size(); ++i) {
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "#### DEVICE: " << i << " ####" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Name: " << device_name_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Vendor: " << device_vendor_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Vendor ID: " << device_vendor_id_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Version: " << device_version_[i] << GGendl;
     GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Driver Version: " << device_driver_version_[i] << GGendl;
     GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + OpenCL C Version: " << device_opencl_c_version_[i] << GGendl;
    if (device_type_[i] == CL_DEVICE_TYPE_CPU) {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Device Type: " << "CL_DEVICE_TYPE_CPU" << GGendl;
    }
    else if (device_type_[i] == CL_DEVICE_TYPE_GPU) {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Device Type: " << "CL_DEVICE_TYPE_GPU" << GGendl;
    }
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Profile: " << device_profile_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Native Vector Width Char: " << device_native_vector_width_char_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Native Vector Width Short: " << device_native_vector_width_short_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Native Vector Width Int: " << device_native_vector_width_int_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Native Vector Width Long: " << device_native_vector_width_long_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Native Vector Width Half: " << device_native_vector_width_half_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Native Vector Width Float: " << device_native_vector_width_float_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Native Vector Width Double: " << device_native_vector_width_double_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Preferred Vector Width Char: " << device_preferred_vector_width_char_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Preferred Vector Width Short: " << device_preferred_vector_width_short_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Preferred Vector Width Int: " << device_preferred_vector_width_int_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Preferred Vector Width Long: " << device_preferred_vector_width_long_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Preferred Vector Width Half: " << device_preferred_vector_width_half_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Preferred Vector Width Float: " << device_preferred_vector_width_float_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Preferred Vector Width Double: " << device_preferred_vector_width_double_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Address Bits: " << device_address_bits_[i] << " bits" << GGendl;
    if (device_available_[i] == static_cast<GGbool>(true)) {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Device Available: ON" << GGendl;
    }
    else {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Device Available: OFF" << GGendl;
    }
    if (device_compiler_available_[i] == static_cast<GGbool>(true)) {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Compiler Available: ON" << GGendl;
    }
    else {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Compiler Available: OFF" << GGendl;
    }
    if (device_native_vector_width_half_[i] != 0) {
      std::string half_fp_capability("");
      half_fp_capability += device_half_fp_config_[i] & CL_FP_DENORM ? "DENORM " : "";
      half_fp_capability += device_half_fp_config_[i] & CL_FP_INF_NAN ? "INF_NAN " : "";
      half_fp_capability += device_half_fp_config_[i] & CL_FP_ROUND_TO_NEAREST ? "ROUND_TO_NEAREST " : "";
      half_fp_capability += device_half_fp_config_[i] & CL_FP_ROUND_TO_ZERO ? "ROUND_TO_ZERO " : "";
      half_fp_capability += device_half_fp_config_[i] & CL_FP_ROUND_TO_INF ? "ROUND_TO_INF " : "";
      half_fp_capability += device_half_fp_config_[i] & CL_FP_FMA ? "FMA " : "";
      half_fp_capability += device_half_fp_config_[i] & CL_FP_SOFT_FLOAT ? "SOFT_FLOAT " : "";
      half_fp_capability += device_half_fp_config_[i] & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT ? "CORRECTLY_ROUNDED_DIVIDE_SQRT" : "";
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Half Precision Capability: " << half_fp_capability << GGendl;
    }
    std::string single_fp_capability("");
    single_fp_capability += device_single_fp_config_[i] & CL_FP_DENORM ? "DENORM " : "";
    single_fp_capability += device_single_fp_config_[i] & CL_FP_INF_NAN ? "INF_NAN " : "";
    single_fp_capability += device_single_fp_config_[i] & CL_FP_ROUND_TO_NEAREST ? "ROUND_TO_NEAREST " : "";
    single_fp_capability += device_single_fp_config_[i] & CL_FP_ROUND_TO_ZERO ? "ROUND_TO_ZERO " : "";
    single_fp_capability += device_single_fp_config_[i] & CL_FP_ROUND_TO_INF ? "ROUND_TO_INF " : "";
    single_fp_capability += device_single_fp_config_[i] & CL_FP_FMA ? "FMA " : "";
    single_fp_capability += device_single_fp_config_[i] & CL_FP_SOFT_FLOAT ? "SOFT_FLOAT " : "";
    single_fp_capability += device_single_fp_config_[i] & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT ? "CORRECTLY_ROUNDED_DIVIDE_SQRT" : "";
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Single Precision Capability: " << single_fp_capability << GGendl;
    if (device_native_vector_width_double_[i] != 0) {
      std::string double_fp_capability("");
      double_fp_capability += device_double_fp_config_[i] & CL_FP_DENORM ? "DENORM " : "";
      double_fp_capability += device_double_fp_config_[i] & CL_FP_INF_NAN ? "INF_NAN " : "";
      double_fp_capability += device_double_fp_config_[i] & CL_FP_ROUND_TO_NEAREST ? "ROUND_TO_NEAREST " : "";
      double_fp_capability += device_double_fp_config_[i] & CL_FP_ROUND_TO_ZERO ? "ROUND_TO_ZERO " : "";
      double_fp_capability += device_double_fp_config_[i] & CL_FP_ROUND_TO_INF ? "ROUND_TO_INF " : "";
      double_fp_capability += device_double_fp_config_[i] & CL_FP_FMA ? "FMA " : "";
      double_fp_capability += device_double_fp_config_[i] & CL_FP_SOFT_FLOAT ? "SOFT_FLOAT " : "";
      double_fp_capability += device_double_fp_config_[i] & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT ? "CORRECTLY_ROUNDED_DIVIDE_SQRT" : "";
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Double Precision Capability: " << double_fp_capability << GGendl;
    }
    if (device_endian_little_[i] == static_cast<GGbool>(true)) {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Endian Little: ON" << GGendl;
    }
    else {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Endian Little: OFF" << GGendl;
    }
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Extensions: " << device_extensions_[i] << GGendl;
    if (device_error_correction_support_[i] == static_cast<GGbool>(true)) {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Error Correction Support: ON" << GGendl;
    }
    else {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Error Correction Support: OFF" << GGendl;
    }
    std::string execution_capabilities("");
    execution_capabilities += device_execution_capabilities_[i] & CL_EXEC_KERNEL ? "KERNEL " : "";
    execution_capabilities += device_execution_capabilities_[i] & CL_EXEC_NATIVE_KERNEL ? "NATIVE_KERNEL " : "";
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Execution Capabilities: " << execution_capabilities << GGendl;
    if (device_global_mem_cache_type_[i] == CL_NONE) {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Global Mem. Cache Type: " << "CL_NONE" << GGendl;
    }
    else if (device_global_mem_cache_type_[i] == CL_READ_ONLY_CACHE) {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Global Mem. Cache Type: " << "CL_READ_ONLY_CACHE" << GGendl;
    }
    else if (device_global_mem_cache_type_[i] == CL_READ_WRITE_CACHE) {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Global Mem. Cache Type: " << "CL_READ_WRITE_CACHE" << GGendl;
    }
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Global Mem. Cache Size: " << device_global_mem_cache_size_[i] << " bytes" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Global Mem. Line Cache Size: " << device_global_mem_cacheline_size_[i] << " bytes" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Global Mem. Size: " << device_global_mem_size_[i] << " bytes" << GGendl;
    if (device_local_mem_type_[i] == CL_LOCAL) {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Local Mem. Type: " << "CL_LOCAL" << GGendl;
    }
    else if (device_local_mem_type_[i] == CL_GLOBAL) {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Local Mem. Type: " << "CL_GLOBAL" << GGendl;
    }
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Local Mem. Size: " << device_local_mem_size_[i] << " bytes" << GGendl;
    if (device_host_unified_memory_[i] == static_cast<GGbool>(true)) {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Host Unified Memory: ON" << GGendl;
    }
    else {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Host Unified Memory: OFF" << GGendl;
    }
    if (device_image_support_[i] == static_cast<GGbool>(true)) {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Image Support: ON" << GGendl;
    }
    else {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Image Support: OFF" << GGendl;
    }
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Image Max Array Size: " << device_image_max_array_size_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Image Max Buffer Size: " << device_image_max_buffer_size_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Image 2D Max Width: " << device_image2D_max_width_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Image 2D Max Height: " << device_image2D_max_height_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Image 3D Max Width: " << device_image3D_max_width_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Image 3D Max Height: " << device_image3D_max_height_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Image 3D Max Depth: " << device_image3D_max_depth_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Max Simultaneous Read Image: " << device_max_read_image_args_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Max Simultaneous Write Image: " << device_max_write_image_args_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Max Clock Frequency: " << device_max_clock_frequency_[i] << " MHz" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Max Compute Units: " << device_max_compute_units_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Max Constant Argument In Kernel: " << device_max_constant_args_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Max Constant Buffer Size: " << device_max_constant_buffer_size_[i] << " bytes" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Mem. Alloc. Size: " << device_max_mem_alloc_size_[i] << " bytes" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Max Parameters Size In Kernel: " << device_max_parameter_size_[i] << " bytes" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Mem. Base Addr. Align.: " << device_mem_base_addr_align_[i] << " bytes" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Max Work Group Size: " << device_max_work_group_size_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Max Work Item Dimensions: " << device_max_work_item_dimensions_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Max Work Item Sizes: " << device_max_work_item_sizes_[0+i*3] << " " << device_max_work_item_sizes_[1+i*3] << " " << device_max_work_item_sizes_[2+i*3] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Printf Buffer Size: " << device_printf_buffer_size_[i] << " bytes" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Max Samplers: " << device_max_samplers_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Partition Max Sub-Devices: " << device_partition_max_sub_devices_[i] << GGendl;
    std::string partition_affinity("");
    partition_affinity += device_single_fp_config_[i] & CL_DEVICE_AFFINITY_DOMAIN_NUMA ? "NUMA " : "";
    partition_affinity += device_single_fp_config_[i] & CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE ? "L4_CACHE " : "";
    partition_affinity += device_single_fp_config_[i] & CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE ? "L3_CACHE " : "";
    partition_affinity += device_single_fp_config_[i] & CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE ? "L2_CACHE " : "";
    partition_affinity += device_single_fp_config_[i] & CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE ? "L1_CACHE " : "";
    partition_affinity += device_single_fp_config_[i] & CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE ? "NEXT_PARTITIONABLE " : "";
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Partition Affinity: " << partition_affinity << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + Timer Resolution: " << device_profiling_timer_resolution_[i] << " ns" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "    + GGEMS Custom Work Group Size: " << work_group_size_ << GGendl;
  }
  GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::PrintBuildOptions(void) const
{
  GGcout("GGEMSOpenCLManager", "PrintBuildOptions", 0) << "OpenCL building options: " << build_options_ << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::PrintActivatedDevices(void) const
{
  // Checking if activated context
  GGcout("GGEMSOpenCLManager", "PrintActivatedDevices", 3) << "Printing activated devices for GGEMS..." << GGendl;

  GGcout("GGEMSOpenCLManager", "PrintActivatedDevices", 0) << GGendl;
  GGcout("GGEMSOpenCLManager", "PrintActivatedDevices", 0) << "ACTIVATED DEVICES:" << GGendl;
  GGcout("GGEMSOpenCLManager", "PrintActivatedDevices", 0) << "------------------" << GGendl;

  // Loop over activated devices
  for (GGsize i = 0; i < device_indices_.size(); ++i) {
    GGcout("GGEMSOpenCLManager", "PrintActivatedDevices", 0) << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintActivatedDevices", 0) << "#### DEVICE: " << device_indices_[i] << " ####" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintActivatedDevices", 0) << "    -> Name: " << GetDeviceName(device_indices_[i]) << " ####" << GGendl;
    if (GetDeviceType(device_indices_[i]) == CL_DEVICE_TYPE_CPU)
      GGcout("GGEMSOpenCLManager", "PrintActivatedDevices", 0) << "    -> Type: CL_DEVICE_TYPE_CPU " << GGendl;
    else if (GetDeviceType(device_indices_[i]) == CL_DEVICE_TYPE_GPU)
      GGcout("GGEMSOpenCLManager", "PrintActivatedDevices", 0) << "    -> Type: CL_DEVICE_TYPE_GPU " << GGendl;
  }

  GGcout("GGEMSOpenCLManager", "PrintActivatedDevice", 0) << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::DeviceToActivate(std::string const& device_type, std::string const& device_vendor)
{
  // Transform all parameters in lower caracters
  std::string type = device_type;
  std::string vendor = device_vendor;
  std::transform(type.begin(), type.end(), type.begin(), ::tolower);
  std::transform(vendor.begin(), vendor.end(), vendor.begin(), ::tolower);

  // Checking if there is a number in type
  bool is_index_device = false;
  for (GGsize i = 0; i < type.size(); ++i) {
    if (isdigit(type[i]) != 0) {
      is_index_device = true;
      break;
    }
  }

  // Analyze all cases
  if (type == "all") { // Activating all available OpenCL devices
    for (GGsize i = 0; i < devices_.size(); ++i) DeviceToActivate(i);
  }
  else if (type == "cpu") { // Activating all CPU devices
    for (GGsize i = 0; i < devices_.size(); ++i) {
      if (device_type_[i] == CL_DEVICE_TYPE_CPU) DeviceToActivate(i);
    }
  }
  else if (type == "gpu") { // Activating all GPU devices or GPU by vendor name
    for (GGsize i = 0; i < devices_.size(); ++i) {
      if (device_type_[i] == CL_DEVICE_TYPE_GPU) {
        if (vendor.empty()) DeviceToActivate(i); // If vendor not specified, take all the GPUs
        else if (device_vendor_[i].find(vendors_[vendor]) != std::string::npos) DeviceToActivate(i); // Specify a vendor
      }
    }
  }
  else if (is_index_device) { // Activating device using index of device
    GGsize pos = 0;
    GGsize index = 0;
    std::string delimiter = ";";
    while ((pos = type.find(delimiter)) != std::string::npos) {
      index = static_cast<GGsize>(std::stoi(type.substr(0, pos)));
      DeviceToActivate(index);
      type.erase(0, pos + delimiter.length());
    }
    index = static_cast<GGsize>(std::stoi(type));
    DeviceToActivate(index);
  }
  else {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Unknown type of device '"<< type << "' !!!";
    GGEMSMisc::ThrowException("GGEMSOpenCLManager", "DeviceToActivate", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::DeviceToActivate(GGsize const& device_id)
{
  GGcout("GGEMSOpenCLManager", "DeviceToActivate", 3) << "Activating a device for GGEMS..." << GGendl;

  // Checking range of the index
  if (device_id >= devices_.size()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Your device index is out of range!!! " << devices_.size() << " device(s) detected. Index must be in the range [" << 0 << ";" << devices_.size() - 1 << "]!!!";
    GGEMSMisc::ThrowException("GGEMSOpenCLManager", "DeviceToActivate", oss.str());
  }

  // Checking if OpenCL device is CPU or GPU
  if (GetDeviceType(device_id) != CL_DEVICE_TYPE_CPU && GetDeviceType(device_id) != CL_DEVICE_TYPE_GPU) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Your device is not a GPU or CPU, please activate another device!!!";
    GGEMSMisc::ThrowException("GGEMSOpenCLManager", "DeviceToActivate", oss.str());
  }

  // Checking if device already activated
  if (std::find(device_indices_.begin(), device_indices_.end(), device_id) != device_indices_.end()) return;

  // Storing index of activated device
  device_indices_.push_back(device_id);

  // Checking double precision for dosimetry
  #ifdef DOSIMETRY_DOUBLE_PRECISION
  if (!IsDoublePrecision(device_id)) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Your OpenCL device '" << GetDeviceName(device_id) << "' does not support double precision!!! Please recompile GGEMS setting DOSIMETRY_DOUBLE_PRECISION to OFF.";
    GGEMSMisc::ThrowException("GGEMSOpenCLManager", "ContextToActivate", oss.str());
  }
  #endif

  // Printing name of activated device
  GGcout("GGEMSOpenCLManager", "DeviceToActivate", 2) << "Activated device: " << GetDeviceName(device_id) << GGendl;

  // Creating context, command queue and event
  contexts_.push_back(new cl::Context(*devices_.at(device_id)));
  queues_.push_back(new cl::CommandQueue(*contexts_.back(), *devices_.at(device_id), CL_QUEUE_PROFILING_ENABLE));
  events_.push_back(new cl::Event());
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::DeviceBalancing(std::string const& device_balancing)
{
  std::string tmp_device_load = device_balancing;
  GGsize pos = 0;
  GGfloat balancing = 0;
  std::string delimiter = ";";
  GGsize i = 0;
  GGfloat incr_balancing = 0.0f;
  while ((pos = tmp_device_load.find(delimiter)) != std::string::npos) {
    balancing = std::stof(tmp_device_load.substr(0, pos));
    device_balancing_.push_back(balancing);
    incr_balancing += balancing;
    tmp_device_load.erase(0, pos + delimiter.length());
    ++i;
  }
  balancing = std::stof(tmp_device_load.substr(0, pos));
  incr_balancing += balancing;
  device_balancing_.push_back(balancing);

  // Checking sum of balancing = 1;
  if (incr_balancing != 1.0f) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Device balancing has to be 1 !!! Please change your value. Current value is " << incr_balancing;
    GGEMSMisc::ThrowException("GGEMSOpenCLManager", "DeviceBalancing", oss.str());
  }

  // Checking number of device balancing value
  if (device_balancing_.size() != device_indices_.size()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Mismatch between number of device balancing values and number of activated devices!!!";
    GGEMSMisc::ThrowException("GGEMSOpenCLManager", "DeviceBalancing", oss.str());
  }

  // Printing device balancing
  for (GGsize i = 0; i < device_balancing_.size(); ++i) {
    GGcout("GGEMSOpenCLManager", "DeviceBalancing", 0) << "Balance on device " << GetDeviceName(device_indices_[i]) << ": " << device_balancing_[i]*100.0f << "%" << GGendl;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGsize GGEMSOpenCLManager::CheckKernel(std::string const& kernel_name, std::string const& compilation_options) const
{
  GGcout("GGEMSOpenCLManager","CheckKernel", 3) << "Checking if kernel has already been compiled..." << GGendl;

  // Parameters for kernel infos
  std::string registered_kernel_name("");

  // Loop over registered kernels
  for (GGsize i = 0; i < kernels_.size(); ++i) {
    CheckOpenCLError(kernels_.at(i)->getInfo(CL_KERNEL_FUNCTION_NAME, &registered_kernel_name), "GGEMSOpenCLManager", "CheckKernel");
    registered_kernel_name.erase(registered_kernel_name.end()-1); // Remove '\0' char from previous function
    if (kernel_name == registered_kernel_name && compilation_options == kernel_compilation_options_.at(i)) return i;
  }

  return KERNEL_NOT_COMPILED;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::CompileKernel(std::string const& kernel_filename, std::string const& kernel_name, cl::Kernel** kernel_list, char* const p_custom_options, char* const p_additional_options)
{
  GGcout("GGEMSOpenCLManager","CompileKernel", 3) << "Compiling a kernel on OpenCL activated context..." << GGendl;

  // Checking the compilation options
  if (p_custom_options && p_additional_options) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Custom and additional options can not by set in same time!!!";
    GGEMSMisc::ThrowException("GGEMSOpenCLManager", "CompileKernel", oss.str());
  }

  // Handling options to OpenCL compilation kernel
  char kernel_compilation_option[1024];
  if (p_custom_options) {
    #if defined _MSC_VER
    ::strcpy_s(kernel_compilation_option, p_custom_options);
    #else
    ::strcpy(kernel_compilation_option, p_custom_options);
    #endif
  }
  else if (p_additional_options) {
    #if defined _MSC_VER
    ::strcpy_s(kernel_compilation_option, build_options_.c_str());
    ::strcat_s(kernel_compilation_option, " ");
    ::strcat_s(kernel_compilation_option, p_additional_options);
    #else
    ::strcpy(kernel_compilation_option, build_options_.c_str());
    ::strcat(kernel_compilation_option, " ");
    ::strcat(kernel_compilation_option, p_additional_options);
    #endif
  }
  else {
    #if defined _MSC_VER
    ::strcpy_s(kernel_compilation_option, build_options_.c_str());
    #else
    ::strcpy(kernel_compilation_option, build_options_.c_str());
    #endif
  }

  // Checking if kernel already compiled
  GGsize kernel_index = CheckKernel(kernel_name, kernel_compilation_option);

  // if kernel already compiled return it
  if (kernel_index != KERNEL_NOT_COMPILED) {
    for (GGsize i = 0; i < device_indices_.size(); ++i) {
      kernel_list[i] = kernels_[kernel_index+i];
    }
  }
  else {
    // Check if the source kernel file exists
    std::ifstream source_file_stream(kernel_filename.c_str(), std::ios::in);
    GGEMSFileStream::CheckInputStream(source_file_stream, kernel_filename);

    // Store kernel in a std::string buffer
    std::string source_code(std::istreambuf_iterator<char>(source_file_stream), (std::istreambuf_iterator<char>()));

    // Creating an OpenCL program
    cl::Program::Sources program_source(1, std::make_pair(source_code.c_str(), source_code.length() + 1));

    // Loop over activated device
    for (GGsize i = 0; i < device_indices_.size(); ++i) {
      // Make program from source code in context
      cl::Program program = cl::Program(*contexts_[i], program_source);

      // Get device associated to context, in our case 1 context = 1 device
      std::vector<cl::Device> device;
      CheckOpenCLError(contexts_[i]->getInfo(CL_CONTEXT_DEVICES, &device), "GGEMSOpenCLManager", "CompileKernel");

      GGcout("GGEMSOpenCLManager", "CompileKernel", 2) << "Compile a new kernel '" << kernel_name << "' from file: " << kernel_filename << " on device: " << GetDeviceName(device_indices_[i]) << " with options: " << kernel_compilation_option << GGendl;

      // Compile source code on device
      GGint build_status = program.build(device, kernel_compilation_option);
      if (build_status != CL_SUCCESS) {
        std::ostringstream oss(std::ostringstream::out);
        std::string log;
        program.getBuildInfo(device[0], CL_PROGRAM_BUILD_LOG, &log);
        oss << ErrorType(build_status) << std::endl;
        oss << log;
        GGEMSMisc::ThrowException("GGEMSOpenCLManager", "CompileKernel", oss.str());
      }

      // Storing the kernel in the singleton
      kernels_.push_back(new cl::Kernel(program, kernel_name.c_str(), &build_status));
      kernel_list[i] = kernels_.back();
      CheckOpenCLError(build_status, "GGEMSOpenCLManager", "CompileKernel");

      // Storing the compilation options
      kernel_compilation_options_.push_back(kernel_compilation_option);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

cl::Buffer* GGEMSOpenCLManager::Allocate(void* host_ptr, GGsize const& size, GGsize const& thread_index, cl_mem_flags flags, std::string const& class_name)
{
  GGcout("GGEMSOpenCLManager","Allocate", 3) << "Allocating memory on OpenCL device memory..." << GGendl;

  // Get the RAM manager and check memory
  GGEMSRAMManager& ram_manager = GGEMSRAMManager::GetInstance();

  // Get index of the device
  GGsize device_index = GetIndexOfActivatedDevice(thread_index);

  // Check if buffer size depending on device parameters
  if (!ram_manager.IsBufferSizeCorrect(device_index, size)) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Size of buffer: " << size << " bytes, is too big!!! The maximum size is " << GetMaxBufferAllocationSize(device_index) << " bytes";
    GGEMSMisc::ThrowException("GGEMSOpenCLManager", "Allocate", oss.str());
  }

  // Check if enough space on device
  if (!ram_manager.IsEnoughAvailableRAMMemory(device_index, size)) {
    GGEMSMisc::ThrowException("GGEMSOpenCLManager", "Allocate", "Not enough RAM memory for buffer allocation!!!");
  }

  GGint error = 0;
  cl::Buffer* buffer = new cl::Buffer(*contexts_[thread_index], flags, size, host_ptr, &error);
  CheckOpenCLError(error, "GGEMSOpenCLManager", "Allocate");

  // Increment RAM memory
  ram_manager.IncrementRAMMemory(class_name, thread_index, size);

  return buffer;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::Deallocate(cl::Buffer* buffer, GGsize size, GGsize const& thread_index, std::string const& class_name)
{
  GGcout("GGEMSOpenCLManager","Deallocate", 3) << "Deallocating memory on OpenCL device memory..." << GGendl;

  // Get the RAM manager and check memory
  GGEMSRAMManager& ram_manager = GGEMSRAMManager::GetInstance();

  // Decrement RAM memory
  ram_manager.DecrementRAMMemory(class_name, thread_index, size);

  delete buffer;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::CleanBuffer(cl::Buffer* buffer, GGsize const& size, GGsize const& thread_index)
{
  GGcout("GGEMSOpenCLManager","CleanBuffer", 3) << "Cleaning OpenCL buffer..." << GGendl;

  GGint error = queues_[thread_index]->enqueueFillBuffer(*buffer, 0, 0, size, nullptr, nullptr);
  CheckOpenCLError(error, "GGEMSOpenCLManager", "CleanBuffer");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

bool GGEMSOpenCLManager::IsDoublePrecision(GGsize const& device_index) const
{
  if (device_extensions_[device_index].find("cl_khr_fp64") == std::string::npos) return false;
  else return true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

bool GGEMSOpenCLManager::IsDoublePrecisionAtomicAddition(GGsize const& device_index) const
{
  if (device_extensions_[device_index].find("cl_khr_int64_base_atomics") == std::string::npos) return false;
  else return true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGsize GGEMSOpenCLManager::GetBestWorkItem(GGsize const& number_of_elements) const
{
  if (number_of_elements%work_group_size_ == 0) {
    return number_of_elements;
  }
  else if (number_of_elements <= work_group_size_) {
    return work_group_size_;
  }
  else {
    return number_of_elements + (work_group_size_ - number_of_elements%work_group_size_);
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::CheckOpenCLError(GGint const& error, std::string const& class_name, std::string const& method_name) const
{
  if (error != CL_SUCCESS) {
    GGEMSMisc::ThrowException(class_name, method_name, ErrorType(error));
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

std::string GGEMSOpenCLManager::ErrorType(GGint const& error) const
{
  // Error description storing in a ostringstream
  std::ostringstream oss(std::ostringstream::out);
  oss << std::endl;

  // Case 0 -> -19: Run-time and JIT Compiler Errors (driver-dependent)
  // Case -30 -> -70: Compile-time Errors (driver-dependent)
  // Case -1000 -> -1009: Errors thrown by extensions
  // Case -9999: Errors thrown by Vendors
  switch (error) {
    case -1: {
      oss << "CL_DEVICE_NOT_FOUND:" << std::endl;
      oss << "    * if no OpenCL devices that matched device_type were found." << std::endl;
      return oss.str();
    }
    case -2: {
      oss << "CL_DEVICE_NOT_AVAILABLE:" << std::endl;
      oss << "    * if a device in devices is currently not available even though the device was returned by clGetDeviceIDs." << std::endl;
      return oss.str();
    }
    case -3: {
      oss << "CL_COMPILER_NOT_AVAILABLE:" << std::endl;
      oss << "    * if program is created with clCreateProgramWithSource and a compiler is not available i.e. CL_DEVICE_COMPILER_AVAILABLE specified in the table of OpenCL Device Queries for clGetDeviceInfo is set to CL_FALSE." << std::endl;
      return oss.str();
    }
    case -4: {
      oss << "CL_MEM_OBJECT_ALLOCATION_FAILURE:" << std::endl;
      oss << "    * if there is a failure to allocate memory for buffer object." << std::endl;
      return oss.str();
    }
    case -5: {
      oss << "CL_OUT_OF_RESOURCES:" << std::endl;
      oss << "    * if there is a failure to allocate resources required by the OpenCL implementation on the device." << std::endl;
      return oss.str();
    }
    case -6: {
      oss << "CL_OUT_OF_HOST_MEMORY:" << std::endl;
      oss << "    * if there is a failure to allocate resources required by the OpenCL implementation on the host." << std::endl;
      return oss.str();
    }
    case -7: {
      oss << "CL_PROFILING_INFO_NOT_AVAILABLE:" << std::endl;
      oss << "    * if the CL_QUEUE_PROFILING_ENABLE flag is not set for the command-queue, if the execution status of the command identified by event is not CL_COMPLETE or if event is a user event object." << std::endl;
      return oss.str();
    }
    case -8: {
      oss << "CL_MEM_COPY_OVERLAP:" << std::endl;
      oss << "    * if src_buffer and dst_buffer are the same buffer or subbuffer object and the source and destination regions overlap or if src_buffer and dst_buffer are different sub-buffers of the same associated buffer object and they overlap. The regions overlap if src_offset <= to dst_offset <= to src_offset + size – 1, or if dst_offset <= to src_offset <= to dst_offset + size – 1." << std::endl;
      return oss.str();
    }
    case -9: {
      oss << "CL_IMAGE_FORMAT_MISMATCH:" << std::endl;
      oss << "    * if src_image and dst_image do not use the same image format." << std::endl;
      return oss.str();
    }
    case -10: {
      oss << "CL_IMAGE_FORMAT_NOT_SUPPORTED:" << std::endl;
      oss << "    * if the image_format is not supported." << std::endl;
      return oss.str();
    }
    case -11: {
      oss << "CL_BUILD_PROGRAM_FAILURE:" << std::endl;
      oss << "    * if there is a failure to build the program executable. This error will be returned if clBuildProgram does not return until the build has completed." << std::endl;
      return oss.str();
    }
    case -12: {
      oss << "CL_MAP_FAILURE:" << std::endl;
      oss << "    * if there is a failure to map the requested region into the host address space. This error cannot occur for image objects created with CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR." << std::endl;
      return oss.str();
    }
    case -13: {
      oss << "CL_MISALIGNED_SUB_BUFFER_OFFSET:" << std::endl;
      oss << "    * if a sub-buffer object is specified as the value for an argument that is a buffer object and the offset specified when the sub-buffer object is created is not aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue." << std::endl;
      return oss.str();
    }
    case -14: {
      oss << "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:" << std::endl;
      oss << "    * if the execution status of any of the events in event_list is a negative integer value." << std::endl;
      return oss.str();
    }
    case -15: {
      oss << "CL_COMPILE_PROGRAM_FAILURE:" << std::endl;
      oss << "    * if there is a failure to compile the program source. This error will be returned if clCompileProgram does not return until the compile has completed." << std::endl;
      return oss.str();
    }
    case -16: {
      oss << "CL_LINKER_NOT_AVAILABLE:" << std::endl;
      oss << "    * if a linker is not available i.e. CL_DEVICE_LINKER_AVAILABLE specified in the table of allowed values for param_name for clGetDeviceInfo is set to CL_FALSE." << std::endl;
      return oss.str();
    }
    case -17: {
      oss << "CL_LINK_PROGRAM_FAILURE:" << std::endl;
      oss << "    * if there is a failure to link the compiled binaries and/or libraries." << std::endl;
      return oss.str();
    }
    case -18: {
      oss << "CL_DEVICE_PARTITION_FAILED:" << std::endl;
      oss << "    * if the partition name is supported by the implementation but in_device could not be further partitioned." << std::endl;
      return oss.str();
    }
    case -19: {
      oss << "CL_KERNEL_ARG_INFO_NOT_AVAILABLE:" << std::endl;
      oss << "    * if the argument information is not available for kernel." << std::endl;
      return oss.str();
    }
    case -30: {
      oss << "CL_INVALID_VALUE:" << std::endl;
      oss << "    * This depends on the function: two or more coupled parameters had errors." << std::endl;
      return oss.str();
    }
    case -31: {
      oss << "CL_INVALID_DEVICE_TYPE:" << std::endl;
      oss << "    * if an invalid device_type is given" << std::endl;
      return oss.str();
    }
    case -32: {
      oss << "CL_INVALID_PLATFORM:" << std::endl;
      oss << "    * if an invalid platform was given" << std::endl;
      return oss.str();
    }
    case -33: {
      oss << "CL_INVALID_DEVICE:" << std::endl;
      oss << "    * if devices contains an invalid device or are not associated with the specified platform." << std::endl;
      return oss.str();
    }
    case -34: {
      oss << "CL_INVALID_CONTEXT:" << std::endl;
      oss << "    * if context is not a valid context." << std::endl;
      return oss.str();
    }
    case -35: {
      oss << "CL_INVALID_QUEUE_PROPERTIES:" << std::endl;
      oss << "    * if specified command-queue-properties are valid but are not supported by the device." << std::endl;
      return oss.str();
    }
    case -36: {
      oss << "CL_INVALID_COMMAND_QUEUE:" << std::endl;
      oss << "    * if command_queue is not a valid command-queue." << std::endl;
      return oss.str();
    }
    case -37: {
      oss << "CL_INVALID_HOST_PTR:" << std::endl;
      oss << "    * This flag is valid only if host_ptr is not NULL. If specified, it indicates that the application wants the OpenCL implementation to allocate memory for the memory object and copy the data from memory referenced by host_ptr.CL_MEM_COPY_HOST_PTR and CL_MEM_USE_HOST_PTR are mutually exclusive.CL_MEM_COPY_HOST_PTR can be used with CL_MEM_ALLOC_HOST_PTR to initialize the contents of the cl_mem object allocated using host-accessible (e.g. PCIe) memory." << std::endl;
      return oss.str();
    }
    case -38: {
      oss << "CL_INVALID_MEM_OBJECT:" << std::endl;
      oss << "    * if memobj is not a valid OpenCL memory object." << std::endl;
      return oss.str();
    }
    case -39: {
      oss << "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:" << std::endl;
      oss << "    * if the OpenGL/DirectX texture internal format does not map to a supported OpenCL image format." << std::endl;
      return oss.str();
    }
    case -40: {
      oss << "CL_INVALID_IMAGE_SIZE:" << std::endl;
      oss << "    * if an image object is specified as an argument value and the image dimensions (image width, height, specified or compute row and/or slice pitch) are not supported by device associated with queue." << std::endl;
      return oss.str();
    }
    case -41: {
      oss << "CL_INVALID_SAMPLER:" << std::endl;
      oss << "    * if sampler is not a valid sampler object." << std::endl;
      return oss.str();
    }
    case -42: {
      oss << "CL_INVALID_BINARY:" << std::endl;
      oss << "    * The provided binary is unfit for the selected device.if program is created with clCreateProgramWithBinary and devices listed in device_list do not have a valid program binary loaded." << std::endl;
      return oss.str();
    }
    case -43: {
      oss << "CL_INVALID_BUILD_OPTIONS:" << std::endl;
      oss << "    * if the build options specified by options are invalid." << std::endl;
      return oss.str();
    }
    case -44: {
      oss << "CL_INVALID_PROGRAM:" << std::endl;
      oss << "    * if program is a not a valid program object." << std::endl;
      return oss.str();
    }
    case -45: {
      oss << "CL_INVALID_PROGRAM_EXECUTABLE:" << std::endl;
      oss << "    * if there is no successfully built program executable available for device associated with command_queue." << std::endl;
      return oss.str();
    }
    case -46: {
      oss << "CL_INVALID_KERNEL_NAME:" << std::endl;
      oss << "    * if kernel_name is not found in program."  << std::endl;
      return oss.str();
    }
    case -47: {
      oss << "CL_INVALID_KERNEL_DEFINITION:" << std::endl;
      oss << "    * if the function definition for __kernel function given by kernel_name such as the number of arguments, the argument types are not the same for all devices for which the program executable has been built." << std::endl;
      return oss.str();
    }
    case -48: {
      oss << "CL_INVALID_KERNEL:" << std::endl;
      oss << "    * if kernel is not a valid kernel object."  << std::endl;
      return oss.str();
    }
    case -49: {
      oss << "CL_INVALID_ARG_INDEX:" << std::endl;
      oss << "    * if arg_index is not a valid argument index." << std::endl;
      return oss.str();
    }
    case -50: {
      oss << "CL_INVALID_ARG_VALUE:" << std::endl;
      oss << "    * if arg_value specified is not a valid value." << std::endl;
      return oss.str();
    }
    case -51: {
      oss << "CL_INVALID_ARG_SIZE:" << std::endl;
      oss << "    * if arg_size does not match the size of the data type for an argument that is not a memory object or if the argument is a memory object and arg_size != sizeof(cl_mem) or if arg_size is zero and the argument is declared with the __local qualifier or if the argument is a sampler and arg_size != sizeof(cl_sampler)." << std::endl;
      return oss.str();
    }
    case -52: {
      oss << "CL_INVALID_KERNEL_ARGS:" << std::endl;
      oss << "    * if the kernel argument values have not been specified." << std::endl;
      return oss.str();
    }
    case -53: {
      oss << "CL_INVALID_WORK_DIMENSION:" << std::endl;
      oss << "    * if work_dim is not a valid value (i.e. a value between 1 and 3)." << std::endl;
      return oss.str();
    }
    case -54: {
      oss << "CL_INVALID_WORK_GROUP_SIZE:" << std::endl;
      oss << "    * if local_work_size is specified and number of work-items specified by global_work_size is not evenly divisable by size of work-group given by local_work_size or does not match the work-group size specified for kernel using the __attribute__((reqd_work_group_size(X, Y, Z))) qualifier in program source.if local_work_size is specified and the total number of work-items in the work-group computed as local_work_size[0] *... local_work_size[work_dim – 1] is greater than the value specified by CL_DEVICE_MAX_WORK_GROUP_SIZE in the table of OpenCL Device Queries for clGetDeviceInfo. if local_work_size is NULL and the __attribute__ ((reqd_work_group_size(X, Y, Z))) qualifier is used to declare the work-group size for kernel in the program source." << std::endl;
      return oss.str();
    }
    case -55: {
      oss << "CL_INVALID_WORK_ITEM_SIZE:" << std::endl;
      oss << "    * if the number of work-items specified in any of local_work_size[0], … local_work_size[work_dim – 1] is greater than the corresponding values specified by CL_DEVICE_MAX_WORK_ITEM_SIZES[0], ... CL_DEVICE_MAX_WORK_ITEM_SIZES[work_dim – 1]" << std::endl;
      return oss.str();
    }
    case -56: {
      oss << "CL_INVALID_GLOBAL_OFFSET:" << std::endl;
      oss << "    * if the value specified in global_work_size + the corresponding values in global_work_offset for any dimensions is greater than the sizeof(size_t) for the device on which the kernel execution will be enqueued." << std::endl;
      return oss.str();
    }
    case -57: {
      oss << "CL_INVALID_EVENT_WAIT_LIST:" << std::endl;
      oss << "    * if event_wait_list is NULL and num_events_in_wait_list > 0, or event_wait_list is not NULL and num_events_in_wait_list is 0, or if event objects in event_wait_list are not valid events." << std::endl;
      return oss.str();
    }
    case -58: {
      oss << "CL_INVALID_EVENT:" << std::endl;
      oss << "    * if event objects specified in event_list are not valid event objects." << std::endl;
      return oss.str();
    }
    case -59: {
      oss << "CL_INVALID_OPERATION:" << std::endl;
      oss << "    * if interoperability is specified by setting CL_CONTEXT_ADAPTER_D3D9_KHR, CL_CONTEXT_ADAPTER_D3D9EX_KHR or CL_CONTEXT_ADAPTER_DXVA_KHR to a non-NULL value, and interoperability with another graphics API is also specified. (only if the cl_khr_dx9_media_sharing extension is supported)." << std::endl;
      return oss.str();
    }
    case -60: {
      oss << "CL_INVALID_GL_OBJECT:" << std::endl;
      oss << "    * if texture is not a GL texture object whose type matches texture_target, if the specified miplevel of texture is not defined, or if the width or height of the specified miplevel is zero." << std::endl;
      return oss.str();
    }
    case -61: {
      oss << "CL_INVALID_BUFFER_SIZE:" << std::endl;
      oss << "    * if size is 0.Implementations may return CL_INVALID_BUFFER_SIZE if size is greater than the CL_DEVICE_MAX_MEM_ALLOC_SIZE value specified in the table of allowed values for param_name for clGetDeviceInfo for all devices in context." << std::endl;
      return oss.str();
    }
    case -62: {
      oss << "CL_INVALID_MIP_LEVEL:" << std::endl;
      oss << "    * if miplevel is greater than zero and the OpenGL implementation does not support creating from non-zero mipmap levels." << std::endl;
      return oss.str();
    }
    case -63: {
      oss << "CL_INVALID_GLOBAL_WORK_SIZE:" << std::endl;
      oss << "    * if global_work_size is NULL, or if any of the values specified in global_work_size[0], ... global_work_size [work_dim – 1] are 0 or exceed the range given by the sizeof(size_t) for the device on which the kernel execution will be enqueued." << std::endl;
      return oss.str();
    }
    case -64: {
      oss << "CL_INVALID_PROPERTY:" << std::endl;
      oss << "    * Vague error, depends on the function" << std::endl;
      return oss.str();
    }
    case -65: {
      oss << "CL_INVALID_IMAGE_DESCRIPTOR:" << std::endl;
      oss << "    * if values specified in image_desc are not valid or if image_desc is NULL." << std::endl;
      return oss.str();
    }
    case -66: {
      oss << "CL_INVALID_COMPILER_OPTIONS:" << std::endl;
      oss << "    * if the compiler options specified by options are invalid." << std::endl;
      return oss.str();
    }
    case -67: {
      oss << "CL_INVALID_LINKER_OPTIONS:" << std::endl;
      oss << "    * if the linker options specified by options are invalid." << std::endl;
      return oss.str();
    }
    case -68: {
      oss << "CL_INVALID_DEVICE_PARTITION_COUNT:" << std::endl;
      oss << "    * if the partition name specified in properties is CL_DEVICE_PARTITION_BY_COUNTS and the number of sub-devices requested exceeds CL_DEVICE_PARTITION_MAX_SUB_DEVICES or the total number of compute units requested exceeds CL_DEVICE_PARTITION_MAX_COMPUTE_UNITS for in_device, or the number of compute units requested for one or more sub-devices is less than zero or the number of sub-devices requested exceeds CL_DEVICE_PARTITION_MAX_COMPUTE_UNITS for in_device." << std::endl;
      return oss.str();
    }
    case -69: {
      oss << "CL_INVALID_PIPE_SIZE:" << std::endl;
      oss << "    * if pipe_packet_size is 0 or the pipe_packet_size exceeds CL_DEVICE_PIPE_MAX_PACKET_SIZE value for all devices in context or if pipe_max_packets is 0." << std::endl;
      return oss.str();
    }
    case -70: {
      oss << "CL_INVALID_DEVICE_QUEUE:" << std::endl;
      oss << "    * when an argument is of type queue_t when it’s not a valid device queue object." << std::endl;
      return oss.str();
    }
    case -1000: {
      oss << "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR:" << std::endl;
      oss << "    * CL and GL not on the same device (only when using a GPU)." << std::endl;
      return oss.str();
    }
    case -1001: {
      oss << "CL_PLATFORM_NOT_FOUND_KHR:" << std::endl;
      oss << "    * No valid ICDs found" << std::endl;
      return oss.str();
    }
    case -1002: {
      oss << "CL_INVALID_D3D10_DEVICE_KHR:" << std::endl;
      oss << "    * if the Direct3D 10 device specified for interoperability is not compatible with the devices against which the context is to be created." << std::endl;
      return oss.str();
    }
    case -1003: {
      oss << "CL_INVALID_D3D10_RESOURCE_KHR:" << std::endl;
      oss << "    * If the resource is not a Direct3D 10 buffer or texture object" << std::endl;
      return oss.str();
    }
    case -1004: {
      oss << "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR:" << std::endl;
      oss << "    * If a mem_object is already acquired by OpenCL" << std::endl;
      return oss.str();
    }
    case -1005: {
      oss << "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR:" << std::endl;
      oss << "    * If a mem_object is not acquired by OpenCL" << std::endl;
      return oss.str();
    }
    case -1006: {
      oss << "CL_INVALID_D3D11_DEVICE_KHR:" << std::endl;
      oss << "    * if the Direct3D 11 device specified for interoperability is not compatible with the devices against which the context is to be created." << std::endl;
      return oss.str();
    }
    case -1007: {
      oss << "CL_INVALID_D3D11_RESOURCE_KHR:" << std::endl;
      oss << "    * If the resource is not a Direct3D 11 buffer or texture object" << std::endl;
      return oss.str();
    }
    case -1008: {
      oss << "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR:" << std::endl;
      oss << "    * If a mem_object is already acquired by OpenCL" << std::endl;
      return oss.str();
    }
    case -1009: {
      oss << "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR:" << std::endl;
      oss << "    * If a mem_object is not acquired by OpenCL" << std::endl;
      return oss.str();
    }
    case -9999: {
      oss << "NVidia:" << std::endl;
      oss << "    * Illegal read or write to a buffer" << std::endl;
      return oss.str();
    }
    default: {
      oss << "Unknown OpenCL error" << std::endl;
      return oss.str();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenCLManager* get_instance_ggems_opencl_manager(void)
{
  return &GGEMSOpenCLManager::GetInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_infos_opencl_manager(GGEMSOpenCLManager* opencl_manager)
{
  opencl_manager->PrintPlatformInfos();
  opencl_manager->PrintDeviceInfos();
  opencl_manager->PrintBuildOptions();
  opencl_manager->PrintActivatedDevices();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_device_index_ggems_opencl_manager(GGEMSOpenCLManager* opencl_manager, GGsize const device_id)
{
  opencl_manager->DeviceToActivate(device_id);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_device_to_activate_opencl_manager(GGEMSOpenCLManager* opencl_manager, char const* device_type, char const* device_vendor)
{
  opencl_manager->DeviceToActivate(device_type, device_vendor);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void clean_opencl_manager(GGEMSOpenCLManager* opencl_manager)
{
  opencl_manager->Clean();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_device_balancing_opencl_manager(GGEMSOpenCLManager* opencl_manager, char const* device_balancing)
{
  opencl_manager->DeviceBalancing(device_balancing);
}
