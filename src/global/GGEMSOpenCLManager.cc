/*!
  \file GGEMSOpenCLManager.cc

  \brief Singleton class storing all informations about OpenCL and managing GPU/CPU contexts and kernels for GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday September 23, 2019
*/

#include <algorithm>
#include <sstream>

#include "GGEMS/global/GGEMSOpenCLManager.hh"

#include "GGEMS/global/GGEMSConfiguration.hh"
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/tools/GGEMSChrono.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenCLManager::GGEMSOpenCLManager(void)
: platforms_(0),
  platform_vendor_(0),
  devices_(0),
  device_device_type_(0),
  device_vendor_(0),
  device_version_(0),
  device_driver_version_(0),
  device_address_bits_(0),
  device_available_(0),
  device_compiler_available_(0),
  device_global_mem_cache_size_(0),
  device_global_mem_cacheline_size_(0),
  device_global_mem_size_(0),
  device_local_mem_size_(0),
  device_mem_base_addr_align_(0),
  device_name_(0),
  device_opencl_c_version_(0),
  device_max_clock_frequency_(0),
  device_max_compute_units_(0),
  device_constant_buffer_size_(0),
  device_mem_alloc_size_(0),
  device_native_vector_width_double_(0),
  build_options_(""),
  context_index_(0),
  contexts_(0),
  contexts_cpu_(0),
  contexts_gpu_(0),
  context_act_(nullptr),
  queues_(0),
  queue_act_(nullptr),
  events_(0),
  event_act_(nullptr),
  kernels_(0),
  used_ram_(0)
{
  GGcout("GGEMSOpenCLManager", "GGEMSOpenCLManager", 3) << "Allocation of GGEMS OpenCL manager..." << GGendl;

  GGcout("GGEMSOpenCLManager", "GGEMSOpenCLManager", 1) << "Retrieving OpenCL platform(s)..." << GGendl;
  CheckOpenCLError(cl::Platform::get(&platforms_), "GGEMSOpenCLManager", "GGEMSOpenCLManager");

  // Getting infos about platform(s)
  for (auto&& i : platforms_) {
    std::string platform_vendor("");
    CheckOpenCLError(i.getInfo(CL_PLATFORM_VENDOR, &platform_vendor), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    platform_vendor_.push_back(platform_vendor);
  }

  // Retrieve all the available devices
  GGcout("GGEMSOpenCLManager", "GGEMSOpenCLManager", 1) << "Retrieving OpenCL device(s)..." << GGendl;
  for (auto&& p : platforms_) {
    std::vector<cl::Device> v_current_device;
    CheckOpenCLError(p.getDevices(CL_DEVICE_TYPE_ALL, &v_current_device), "GGEMSOpenCLManager", "GGEMSOpenCLManager");

    // Storing device of each platform
    for (auto&& d : v_current_device) devices_.emplace_back(new cl::Device(d));
  }

  // Resizing vector storing infos for devices
  device_device_type_.resize(devices_.size());
  device_vendor_.resize(devices_.size());
  device_version_.resize(devices_.size());
  device_driver_version_.resize(devices_.size());
  device_address_bits_.resize(devices_.size());
  device_available_.resize(devices_.size());
  device_compiler_available_.resize(devices_.size());
  device_global_mem_cache_size_.resize(devices_.size());
  device_global_mem_cacheline_size_.resize(devices_.size());
  device_global_mem_size_.resize(devices_.size());
  device_local_mem_size_.resize(devices_.size());
  device_mem_base_addr_align_.resize(devices_.size());
  device_name_.resize(devices_.size());
  device_opencl_c_version_.resize(devices_.size());
  device_max_clock_frequency_.resize(devices_.size());
  device_max_compute_units_.resize(devices_.size());
  device_constant_buffer_size_.resize(devices_.size());
  device_mem_alloc_size_.resize(devices_.size());
  device_native_vector_width_double_.resize(devices_.size());

  // Getting infos for device
  GGcout("GGEMSOpenCLManager", "GGEMSOpenCLManager", 1) << "Retrieving OpenCL device informations..." << GGendl;

  // Make a char buffer reading char* data
  char char_data[1024];
  for (std::size_t i = 0; i < devices_.size(); ++i) {
    // Device Type
    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_TYPE, &device_device_type_[i]), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_VENDOR, &device_vendor_[i]), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_ADDRESS_BITS, &device_address_bits_[i]), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_AVAILABLE, &device_available_[i]), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_COMPILER_AVAILABLE, &device_compiler_available_[i]), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &device_global_mem_cache_size_[i]), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, &device_global_mem_cacheline_size_[i]), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    CheckOpenCLError(devices_[i]->getInfo( CL_DEVICE_GLOBAL_MEM_SIZE, &device_global_mem_size_[i]), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &device_local_mem_size_[i]), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_MEM_BASE_ADDR_ALIGN, &device_mem_base_addr_align_[i]), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_NAME, &device_name_[i]), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &device_max_clock_frequency_[i]), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &device_max_compute_units_[i]), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, &device_constant_buffer_size_[i]), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &device_mem_alloc_size_[i]), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, &device_native_vector_width_double_[i]), "GGEMSOpenCLManager", "GGEMSOpenCLManager");

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_VERSION, &char_data), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_version_[i] = std::string(char_data);

    CheckOpenCLError(devices_[i]->getInfo(CL_DRIVER_VERSION, &char_data), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_driver_version_[i] = std::string(char_data);

    CheckOpenCLError(devices_[i]->getInfo(CL_DEVICE_OPENCL_C_VERSION, &char_data), "GGEMSOpenCLManager", "GGEMSOpenCLManager");
    device_opencl_c_version_[i] = std::string(char_data);
  }

  // Define the compilation options by default for OpenCL
  build_options_ = "-cl-std=CL1.2 -w -Werror -DOPENCL_COMPILER";
  //build_options_ += " -cl-fast-relaxed-math";

  // Add auxiliary function path to OpenCL options
  #ifdef GGEMS_PATH
  build_options_ += " -I";
  build_options_ += GGEMS_PATH;
  build_options_ += "/include";
  #elif
  GGEMSMisc::ThrowException("GGEMSOpenCLManager","GGEMSOpenCLManager", "OPENCL_KERNEL_PATH not defined or not find!!!");
  #endif

  // Creating a context for each device
  CreateContext();

  // Creating the command queue for CPU and GPU
  CreateCommandQueue();

  // Creating the events for each context
  CreateEvent();

  // Initialization of the RAM manager
  InitializeRAMManager();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenCLManager::~GGEMSOpenCLManager(void)
{
  GGcout("GGEMSOpenCLManager", "~GGEMSOpenCLManager", 3) << "Deallocation of GGEMS OpenCL manager..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::Clean(void)
{
  GGcout("GGEMSOpenCLManager", "Clean", 3) << "Cleaning OpenCL platform, device, context, queue, event and kernel..." << GGendl;

  // Deleting platform(s) and infos
  GGcout("GGEMSOpenCLManager", "Clean", 1) << "Deleting OpenCL platform(s)..." << GGendl;

  // Freeing platforms
  platforms_.clear();
  platform_vendor_.clear();

  // Deleting devices(s) and infos
  GGcout("GGEMSOpenCLManager", "Clean", 1) << "Deleting OpenCL device(s)..." << GGendl;

  // Freeing devices
  devices_.clear();
  device_device_type_.clear();
  device_vendor_.clear();
  device_version_.clear();
  device_driver_version_.clear();
  device_address_bits_.clear();
  device_available_.clear();
  device_compiler_available_.clear();
  device_global_mem_cache_size_.clear();
  device_global_mem_cacheline_size_.clear();
  device_global_mem_size_.clear();
  device_local_mem_size_.clear();
  device_mem_base_addr_align_.clear();
  device_name_.clear();
  device_opencl_c_version_.clear();
  device_max_clock_frequency_.clear();
  device_max_compute_units_.clear();
  device_constant_buffer_size_.clear();
  device_mem_alloc_size_.clear();

  // Deleting context(s)
  GGcout("GGEMSOpenCLManager", "Clean", 1) << "Deleting OpenCL context(s)..." << GGendl;
  contexts_.clear();
  contexts_cpu_.clear();
  contexts_gpu_.clear();
  context_act_.reset(); 

  // Deleting command queue(s)
  GGcout("GGEMSOpenCLManager", "Clean", 1) << "Deleting OpenCL command queue(s)..." << GGendl;
  queues_.clear();
  queue_act_.reset();

  // Deleting event(s)
  GGcout("GGEMSOpenCLManager", "Clean", 1) << "Deleting OpenCL event(s)..." << GGendl;
  events_.clear();
  event_act_.reset();

  // Deleting kernel(s)
  GGcout("GGEMSOpenCLManager", "Clean", 1) << "Deleting OpenCL kernel(s)..." << GGendl;
  kernels_.clear();

  // Freeing memory of RAM manager
  used_ram_.clear();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::PrintPlatformInfos(void) const
{
  GGcout("GGEMSOpenCLManager", "PrintPlatformInfos", 3) << "Printing infos about OpenCL platform(s)..." << GGendl;

  // Loop over the platforms
  for (std::size_t i = 0; i < platforms_.size(); ++i) {
    GGcout("GGEMSOpenCLManager", "PrintPlatformInfos", 0) << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintPlatformInfos", 0) << "#### PLATFORM: " << i << " ####" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintPlatformInfos", 0) << "+ Vendor: " << platform_vendor_[i] << GGendl;
  }
  GGcout("GGEMSOpenCLManager", "PrintPlatformInfos", 0) << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::PrintDeviceInfos(void) const
{
  GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 3) << "Printing infos about OpenCL device(s)..." << GGendl;

  for (std::size_t i = 0; i < devices_.size(); ++i) {
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "#### DEVICE: " << i << " ####" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Name: " << device_name_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Vendor: " << device_vendor_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Version: " << device_version_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Driver Version: " << device_driver_version_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ OpenCL C Version: " << device_opencl_c_version_[i] << GGendl;
    if (device_device_type_[i] == CL_DEVICE_TYPE_CPU) {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Device Type: " << "+ Device Type: CL_DEVICE_TYPE_CPU" << GGendl;
    }
    else if (device_device_type_[i] == CL_DEVICE_TYPE_GPU) {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Device Type: " << "CL_DEVICE_TYPE_GPU" << GGendl;
    }
    else {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Device Type: " << "Unknown device type!!!" << GGendl;
    }
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Address Bits: " << device_address_bits_[i] << " bits" << GGendl;
    if (device_available_[i] == static_cast<GGbool>(true)) {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Device Available: ON" << GGendl;
    }
    else {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Device Available: OFF" << GGendl;
    }
    if (device_compiler_available_[i] == static_cast<GGbool>(true)) {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Compiler Available: ON" << GGendl;
    }
    else {
      GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Compiler Available: OFF" << GGendl;
    }
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Global Mem. Cache Size: " << device_global_mem_cache_size_[i] << " bytes" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Global Mem. Line Cache Size: " << device_global_mem_cacheline_size_[i] << " bytes" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Global Mem. Size: " << device_global_mem_size_[i] << " bytes" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Local Mem. Size: " << device_local_mem_size_[i] << " bytes" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Mem. Base Addr. Align.: " << device_mem_base_addr_align_[i] << " bytes" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Native Vector Width Double: " << device_native_vector_width_double_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Max Clock Frequency: " << device_max_clock_frequency_[i] << " MHz" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Max Compute Units: " << device_max_compute_units_[i] << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Constant Buffer Size: " << device_constant_buffer_size_[i] << " bytes" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << "+ Mem. Alloc. Size: " << device_mem_alloc_size_[i] << " bytes" << GGendl;
  }
  GGcout("GGEMSOpenCLManager", "PrintDeviceInfos", 0) << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::PrintBuildOptions(void) const
{
  GGcout("GGEMSOpenCLManager", "PrintBuildOptions", 3)
    << "Printing infos about OpenCL compilation options..." << GGendl;

  GGcout("GGEMSOpenCLManager", "PrintBuildOptions", 0)
    << "OpenCL building options: " << build_options_ << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::CreateContext(void)
{
  GGcout("GGEMSOpenCLManager", "CreateContext", 3)
    << "Creating context for CPU and/or GPU..." << GGendl;

  // Loop over the devices
  for (std::size_t i = 0; i < devices_.size(); ++i) {
    // Get GPU type
    if (device_device_type_[i] == CL_DEVICE_TYPE_GPU) {
      contexts_.emplace_back(new cl::Context(*devices_[i]));
      contexts_gpu_.emplace_back(contexts_.back());
    }

    if (device_device_type_[i] == CL_DEVICE_TYPE_CPU) {
      contexts_.emplace_back(new cl::Context(*devices_[i]));
      contexts_cpu_.emplace_back(contexts_.back());
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::ContextToActivate(GGuint const& context_id)
{
  GGcout("GGEMSOpenCLManager", "ContextToActivate", 3) << "Activating a context for GGEMS..." << GGendl;

  // Checking if a context has already been activated
  if (context_act_)
    GGEMSMisc::ThrowException("GGEMSOpenCLManager", "ContextToActivate", "A context has already been activated!!!");

  // Checking the index of the context
  if (context_id >= contexts_.size()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Your context index is out of range!!! " << contexts_.size() << " context(s) detected. Index must be in the range [" << 0 << ";" << contexts_.size() - 1 << "]!!!";
    GGEMSMisc::ThrowException("GGEMSOpenCLManager", "ContextToActivate", oss.str());
  }

  // Activate the context
  context_act_ = contexts_.at(context_id);

  // Storing the index of activated context
  context_index_ = context_id;

  // Checking if an Intel HD Graphics has been selected and send a warning
  std::size_t found_device = device_name_[context_id].find("HD Graphics");
  if (found_device != std::string::npos) {
    GGwarn("GGEMSOpenCLManager", "ContextToActivate", 0) << "##########################################" << GGendl;
    GGwarn("GGEMSOpenCLManager", "ContextToActivate", 0) << "#                WARNING!!!              #" << GGendl;
    GGwarn("GGEMSOpenCLManager", "ContextToActivate", 0) << "##########################################" << GGendl;
    GGwarn("GGEMSOpenCLManager", "ContextToActivate", 0) << "You are using a HD Graphics architecture, GGEMS could work on this device, but the computation is very slow and the result could be" << " hazardous. Please use a CPU architecture or a GPU." << GGendl;
  }

  // Checking if the double precision is activated on OpenCL device
  if (device_native_vector_width_double_[context_id] == 0) {
    GGEMSMisc::ThrowException("GGEMSOpenCLManager", "ContextToActivate", "Your OpenCL device does not support double precision!!!");
  }

  // Activate the command queue
  queue_act_ = queues_.at(context_id);

  // Activate the event
  event_act_ = events_.at(context_id);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::PrintActivatedContextInfos(void) const
{
  // Checking if activated context
  if (context_act_) {

    GGcout("GGEMSOpenCLManager", "PrintActivatedContextInfos", 3) << "Printing activated context for GGEMS..." << GGendl;

    GGuint context_num_devices = 0;
    std::vector<cl::Device> device;
    cl_device_type device_type = 0;
    std::string device_name;

    GGcout("GGEMSOpenCLManager", "PrintActivatedContextInfos", 0) << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintActivatedContextInfos", 0) << "ACTIVATED CONTEXT(S):" << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintActivatedContextInfos", 0) << "---------------------" << GGendl;

    // Loop over all the context
    GGcout("GGEMSOpenCLManager", "PrintActivatedContextInfos", 0) << GGendl;
    GGcout("GGEMSOpenCLManager", "PrintActivatedContextInfos", 0) << "#### CONTEXT: " << context_index_ << " ####" << GGendl;

    CheckOpenCLError(context_act_->getInfo(CL_CONTEXT_NUM_DEVICES, &context_num_devices), "GGEMSOpenCLManager", "PrintActivatedContextInfos");
    CheckOpenCLError(context_act_->getInfo(CL_CONTEXT_DEVICES, &device), "GGEMSOpenCLManager", "PrintActivatedContextInfos");

    GGcout("GGEMSOpenCLManager", "PrintActivatedContextInfos", 0) << "+ Type of device(s): " << GGendl;

    for (GGuint j = 0; j < context_num_devices; ++j) {
      CheckOpenCLError(device[ j ].getInfo(CL_DEVICE_NAME, &device_name), "GGEMSOpenCLManager", "PrintActivatedContextInfos");

      GGcout("GGEMSOpenCLManager", "PrintActivatedContextInfos", 0) << "    -> Name: " << device_name << GGendl;

      CheckOpenCLError(device[j].getInfo(CL_DEVICE_TYPE, &device_type), "GGEMSOpenCLManager", "PrintActivatedContextInfos");

      if (device_type == CL_DEVICE_TYPE_CPU) {
        GGcout("GGEMSOpenCLManager", "PrintActivatedContextInfos", 0) << "    -> Device [" << j << "]: CL_DEVICE_TYPE_CPU" << GGendl;
      }
      else if (device_type == CL_DEVICE_TYPE_GPU) {
        GGcout("GGEMSOpenCLManager", "PrintActivatedContextInfos", 0) << "    -> Device [" << j << "]: CL_DEVICE_TYPE_GPU" << GGendl;
      }
      else {
        GGcout("GGEMSOpenCLManager", "PrintActivatedContextInfos", 0) << "    -> Device [" << j << "]: Unknown device type!!!" << GGendl;
      }
    }
    GGcout("GGEMSOpenCLManager", "PrintActivatedContextInfos", 0) << GGendl;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::CreateCommandQueue(void)
{
  GGcout("GGEMSOpenCLManager","CreateCommandQueue", 3) << "Creating command queue for CPU and/or GPU..." << GGendl;

  // Vector of devices in the context
  std::vector<cl::Device> device;
  // Loop over the contexts
  for (std::size_t i = 0; i < contexts_.size(); ++i) {
    // Get the vector of devices include in the context
    CheckOpenCLError(contexts_[i]->getInfo(CL_CONTEXT_DEVICES, &device), "GGEMSOpenCLManager", "CreateCommandQueue");
    queues_.emplace_back(new cl::CommandQueue(*contexts_[i], device[0], CL_QUEUE_PROFILING_ENABLE));
  }
  device.clear();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::PrintCommandQueueInfos(void) const
{
  GGcout("GGEMSOpenCLManager","PrintCommandQueueInfos", 3) << "Printing infos about OpenCL command queue(s)..." << GGendl;

  cl::Device device;
  std::string device_name;

  // Loop over the queues
  for (std::size_t i = 0; i < queues_.size(); ++i) {
    GGcout("GGEMSOpenCLManager","PrintCommandQueueInfos", 0) << GGendl;
    GGcout("GGEMSOpenCLManager","PrintCommandQueueInfos", 0) << "#### COMMAND QUEUE: " << i << " ####" << GGendl;

    CheckOpenCLError(queues_[i]->getInfo(CL_QUEUE_DEVICE, &device), "GGEMSOpenCLManager", "PrintCommandQueueInfos");
    CheckOpenCLError(device.getInfo(CL_DEVICE_NAME, &device_name), "GGEMSOpenCLManager", "PrintCommandQueueInfos");

    GGcout("GGEMSOpenCLManager","PrintCommandQueueInfos", 0) << "+ Device Name: " << device_name << GGendl;
    GGcout("GGEMSOpenCLManager","PrintCommandQueueInfos", 0) << "+ Command Queue Index: " << i << GGendl;
    GGcout("GGEMSOpenCLManager","PrintCommandQueueInfos", 0) << GGendl;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::CreateEvent(void)
{
  GGcout("GGEMSOpenCLManager","CreateEvent", 3) << "Creating event for CPU and/or GPU..." << GGendl;

  for (std::size_t i = 0; i < contexts_.size(); ++i) {
    events_.emplace_back(new cl::Event());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<cl::Kernel> GGEMSOpenCLManager::CompileKernel(std::string const& kernel_filename, std::string const& kernel_name, char* const p_custom_options, char* const p_additional_options)
{
  GGcout("GGEMSOpenCLManager","CompileKernel", 3) << "Compiling a kernel on OpenCL activated context..." << GGendl;

  // Checking the compilation options
  if (p_custom_options && p_additional_options) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Custom and additional options can not by set in same time!!!";
    GGEMSMisc::ThrowException("GGEMSOpenCLManager", "CompileKernel", oss.str());
  }

  // Check if the source kernel file exists
  std::ifstream source_file_stream(kernel_filename.c_str(), std::ios::in);
  GGEMSFileStream::CheckInputStream(source_file_stream, kernel_filename);

  // Handling options to OpenCL compilation kernel
  char kernel_compilation_option[512];
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

  GGcout("GGEMSOpenCLManager", "CompileKernel", 1) << "Compile a new kernel '" << kernel_name << "' from file: " << kernel_filename << " on context: " << context_index_ << " with options: " << kernel_compilation_option << GGendl;

  // Store kernel in a std::string buffer
  std::string source_code(std::istreambuf_iterator<char>(source_file_stream), (std::istreambuf_iterator<char>()));

  // Creating an OpenCL program
  cl::Program::Sources program_source(1, std::make_pair(source_code.c_str(), source_code.length() + 1));

  // Make program from source code in specific context
  cl::Program program = cl::Program(*(context_act_.get()), program_source);

  // Vector storing all the devices from a context
  // In GGEMS a device is associated to a context
  std::vector<cl::Device> devices;

  // Get the vector of devices
  CheckOpenCLError(context_act_.get()->getInfo(CL_CONTEXT_DEVICES, &devices), "GGEMSOpenCLManager", "CompileKernel");

  // Compile source code on devices
  GGint build_status = program.build(devices, kernel_compilation_option);
  if (build_status != CL_SUCCESS) {
    std::ostringstream oss(std::ostringstream::out);
    std::string log;
    program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &log);
    oss << ErrorType(build_status) << std::endl;
    oss << log;
    GGEMSMisc::ThrowException("GGEMSOpenCLManager", "CompileKernel", oss.str());
  }

  GGcout("GGEMSOpenCLManager", "CompileKernel", 0) << "Compilation OK" << GGendl;

  // Storing the kernel in the singleton
  kernels_.emplace_back(new cl::Kernel(program, kernel_name.c_str(), &build_status));
  CheckOpenCLError(build_status, "GGEMSOpenCLManager", "CompileKernel");

  return kernels_.back();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::InitializeRAMManager(void)
{
  GGcout("GGEMSOpenCLManager", "InitializeRAMManager", 3) << "Initializing a RAM handler for each context..." << GGendl;

  // For each context we create a RAM handler
  used_ram_.resize(contexts_.size());
  for (auto&& i : used_ram_) i = 0;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::PrintRAMStatus(void) const
{
  GGcout("GGEMSOpenCLManager", "PrintRAMStatus", 3) << "Printing infos about RAM memory on OpenCL context(s)..." << GGendl;
  GGcout("GGEMSOpenCLManager", "PrintRAMStatus", 0) << "---------------------------" << GGendl;

  // Loop over the contexts
  for (std::size_t i = 0; i < contexts_.size(); ++i) {
    // Get the max. RAM memory by context
    GGulong const max_RAM = device_global_mem_size_[i];
    GGfloat const percent_RAM = static_cast<GGfloat>(used_ram_[i]) * 100.0f / static_cast<GGfloat>(max_RAM);
    GGcout("GGEMSOpenCLManager", "PrintRAMStatus", 0) << "Context " << i << ": " << used_ram_[i] << " / " << max_RAM << " bytes -> " << percent_RAM << " % used" << GGendl;
  }
  GGcout("GGEMSOpenCLManager", "PrintRAMStatus", 0) << "---------------------------" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::CheckRAMMemory(std::size_t const& size)
{
  GGcout("GGEMSOpenCLManager","CheckRAMMemory", 3) << "Checking RAM memory usage..." << GGendl;

  // Getting memory infos
  GGulong const max_RAM = device_global_mem_size_[context_index_];
  GGdouble const percent_RAM = static_cast<GGdouble>(used_ram_[context_index_] + size) * 100.0 / static_cast<GGdouble>(max_RAM);

  if (percent_RAM >= 80.0f && percent_RAM < 95.0f) {
    GGwarn("GGEMSOpenCLManager", "CheckRAMMemory", 0) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << GGendl;
    GGwarn("GGEMSOpenCLManager", "CheckRAMMemory", 0) << "!!!             MEMORY WARNING             !!!" << GGendl;
    GGwarn("GGEMSOpenCLManager", "CheckRAMMemory", 0) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << GGendl;
    GGwarn("GGEMSOpenCLManager", "CheckRAMMemory", 0) << "RAM allocation (" << percent_RAM << "%) is superior to 80%, the simulation will be automatically killed whether RAM allocation is superior to 95%" << GGendl;
  }
  else if (percent_RAM >= 95.0f) {
    GGcerr("GGEMSOpenCLManager", "CheckRAMMemory", 0) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << GGendl;
    GGcerr("GGEMSOpenCLManager", "CheckRAMMemory", 0) << "!!!             MEMORY ERROR             !!!" << GGendl;
    GGcerr("GGEMSOpenCLManager", "CheckRAMMemory", 0) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << GGendl;
    GGcerr("GGEMSOpenCLManager", "CheckRAMMemory", 0) << "RAM allocation (" << percent_RAM << "%) is superior to 95%, the simulation is killed!!!" << GGendl;
    GGEMSMisc::ThrowException("GGEMSOpenCLManager", "CheckRAMMemory", "Not enough RAM memory on OpenCL device!!!");
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::AddRAMMemory(GGulong const& size)
{
  GGcout("GGEMSOpenCLManager","AddRAMMemory", 3) << "Adding RAM memory on OpenCL activated context..." << GGendl;

  // Increment size
  used_ram_[context_index_] += size;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::SubRAMMemory(GGulong const& size)
{
  GGcout("GGEMSOpenCLManager","SubRAMMemory", 3) << "Substracting RAM memory on OpenCL activated context..." << GGendl;

  // Decrement size
  used_ram_[context_index_] -= size;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::DisplayElapsedTimeInKernel(std::string const& kernel_name) const
{
  GGcout("GGEMSOpenCLManager","DisplayElapsedTimeInKernel", 3) << "Displaying elapsed time in the last OpenCL kernel..." << GGendl;

  // Get the start and end of the activated event
  GGulong start = 0, end = 0;

  // Start
  CheckOpenCLError(event_act_.get()->getProfilingInfo(CL_PROFILING_COMMAND_START, &start), "GGEMSOpenCLManager", "DisplayElapsedTimeInKernel");

  // End
  CheckOpenCLError(event_act_.get()->getProfilingInfo(CL_PROFILING_COMMAND_END, &end), "GGEMSOpenCLManager", "DisplayElapsedTimeInKernel");

  DurationNano duration = static_cast<std::chrono::nanoseconds>((end-start));

  // Display time in kernel
  GGEMSChrono::DisplayTime(duration, kernel_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::PrintContextInfos(void) const
{
  GGcout("GGEMSOpenCLManager","PrintContextInfos", 3) << "Printing infos about OpenCL context(s)..." << GGendl;

  GGuint context_number_devices = 0;
  GGuint reference_count = 0;
  std::vector<cl::Device> device;
  cl_device_type device_type = 0;
  std::string device_name;

  GGcout("GGEMSOpenCLManager", "PrintContextInfos", 0) << GGendl;

  // Loop over all the context
  for (std::size_t i = 0; i < contexts_.size(); ++i) {
    GGcout("GGEMSOpenCLManager", "PrintContextInfos", 0) << "#### CONTEXT: " << i << " ####" << GGendl;

    CheckOpenCLError(contexts_[i]->getInfo(CL_CONTEXT_NUM_DEVICES, &context_number_devices), "GGEMSOpenCLManager", "PrintContextInfos");

    if (context_number_devices > 1) {
      GGEMSMisc::ThrowException("GGEMSOpenCLManager", "PrintContextInfos", "One device by context only!!!");
    }

    CheckOpenCLError(contexts_[i]->getInfo(CL_CONTEXT_REFERENCE_COUNT, &reference_count), "GGEMSOpenCLManager", "PrintContextInfos");
    CheckOpenCLError(contexts_[i]->getInfo(CL_CONTEXT_DEVICES, &device), "GGEMSOpenCLManager", "PrintContextInfos");

    GGcout("GGEMSOpenCLManager", "PrintContextInfos", 0) << "+ Type of device(s): " << GGendl;

    for (GGuint j = 0; j < context_number_devices; ++j) {
      CheckOpenCLError(device[j].getInfo(CL_DEVICE_NAME, &device_name), "GGEMSOpenCLManager", "PrintContextInfos");
      GGcout("GGEMSOpenCLManager", "PrintContextInfos", 0) << "    -> Name: " << device_name << GGendl;

      CheckOpenCLError(device[j].getInfo(CL_DEVICE_TYPE, &device_type), "GGEMSOpenCLManager", "PrintContextInfos");

      if (device_type == CL_DEVICE_TYPE_CPU) {
        GGcout("GGEMSOpenCLManager", "PrintContextInfos", 0) << "    -> Device [" << j << "]: CL_DEVICE_TYPE_CPU" << GGendl;
      }
      else if (device_type == CL_DEVICE_TYPE_GPU) {
        GGcout("GGEMSOpenCLManager", "PrintContextInfos", 0) << "    -> Device [" << j << "]: CL_DEVICE_TYPE_GPU" << GGendl;
      }
      else {
        GGcout("GGEMSOpenCLManager", "PrintContextInfos", 0) << "    -> Device [" << j << "]: Unknown device type!!!" << GGendl;
      }
      GGcout("GGEMSOpenCLManager", "PrintContextInfos", 0) << GGendl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<cl::Buffer> GGEMSOpenCLManager::Allocate(void* host_ptr, std::size_t size, cl_mem_flags flags)
{
  GGcout("GGEMSOpenCLManager","Allocate", 3) << "Allocating memory on OpenCL device memory..." << GGendl;

  CheckRAMMemory(size);

  GGint error = 0;
  std::unique_ptr<cl::Buffer> buffer(new cl::Buffer(*(context_act_.get()), flags, size, host_ptr, &error));
  CheckOpenCLError(error, "GGEMSOpenCLManager", "Allocate");
  return buffer;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenCLManager::CheckOpenCLError(GGint const& error, std::string const& class_name, std::string const& method_name) const
{
  if (error != CL_SUCCESS) GGEMSMisc::ThrowException(class_name, method_name, ErrorType(error));
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
  opencl_manager->PrintContextInfos();
  opencl_manager->PrintCommandQueueInfos();
  opencl_manager->PrintActivatedContextInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_RAM_ggems_opencl_manager(GGEMSOpenCLManager* opencl_manager)
{
  opencl_manager->PrintRAMStatus();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_context_index_ggems_opencl_manager(GGEMSOpenCLManager* opencl_manager, GGuint const context_index)
{
  opencl_manager->ContextToActivate(context_index);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void clean_opencl_manager(GGEMSOpenCLManager* opencl_manager)
{
  opencl_manager->Clean();
}
