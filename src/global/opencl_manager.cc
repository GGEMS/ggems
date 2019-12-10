/*!
  \file opencl_manager.cc

  \brief Singleton class storing all informations about OpenCL and managing
  GPU/CPU contexts and kernels for GGEMS
  IMPORTANT: Only 1 context has to be activated.

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday September 23, 2019
*/

#include "GGEMS/global/ggems_configuration.hh"
#include "GGEMS/global/opencl_manager.hh"
#include "GGEMS/tools/print.hh"
#include "GGEMS/tools/functions.hh"
#include "GGEMS/tools/memory.hh"
#include "GGEMS/tools/chrono.hh"
#include <algorithm>

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

OpenCLManager::OpenCLManager(void)
: v_platforms_cl_(0),
  p_platform_vendor_(nullptr),
  vp_devices_cl_(0),
  p_device_device_type_(nullptr),
  p_device_vendor_(nullptr),
  p_device_version_(nullptr),
  p_device_driver_version_(nullptr),
  p_device_address_bits_(nullptr),
  p_device_available_(nullptr),
  p_device_compiler_available_(nullptr),
  p_device_global_mem_cache_size_(nullptr),
  p_device_global_mem_cacheline_size_(nullptr),
  p_device_global_mem_size_(nullptr),
  p_device_local_mem_size_(nullptr),
  p_device_mem_base_addr_align_(nullptr),
  p_device_name_(nullptr),
  p_device_opencl_c_version_(nullptr),
  p_device_max_clock_frequency_(nullptr),
  p_device_max_compute_units_(nullptr),
  p_device_constant_buffer_size_(nullptr),
  p_device_mem_alloc_size_(nullptr),
  p_device_native_vector_width_double_(nullptr),
  p_device_preferred_vector_width_double_(nullptr),
  build_options_(""),
  context_index_(0),
  vp_contexts_cl_(0),
  vp_contexts_cpu_cl_(0),
  vp_contexts_gpu_cl_(0),
  vp_contexts_act_cl_(0),
  vp_queues_cl_(0),
  vp_queues_act_cl_(0),
  vp_event_cl_(0),
  vp_event_act_cl_(0),
  vp_kernel_cl_(0),
  p_used_ram_(nullptr)
{
  GGEMScout("OpenCLManager", "OpenCLManager", 3)
    << "Allocation of OpenCL Manager singleton..." << GGEMSendl;

  GGEMScout("OpenCLManager", "OpenCLManager", 1)
    << "Retrieving OpenCL platform(s)..." << GGEMSendl;
  CheckOpenCLError(cl::Platform::get(&v_platforms_cl_), "OpenCLManager",
    "OpenCLManager");

  // Getting infos about platform(s)
  p_platform_vendor_ = Memory::MemAlloc<std::string>(v_platforms_cl_.size());
  for (std::size_t i = 0; i < v_platforms_cl_.size(); ++i) {
    CheckOpenCLError(v_platforms_cl_[i].getInfo(CL_PLATFORM_VENDOR,
      &p_platform_vendor_[i]), "OpenCLManager", "OpenCLManager");
  }

  // Retrieve all the available devices
  GGEMScout("OpenCLManager", "OpenCLManager", 1)
    << "Retrieving OpenCL device(s)..." << GGEMSendl;
  for (auto& p : v_platforms_cl_) {
    std::vector<cl::Device> v_current_device;
    CheckOpenCLError(p.getDevices(CL_DEVICE_TYPE_ALL, &v_current_device),
      "OpenCLManager", "OpenCLManager");

    // Copy the current device to global device
    for (auto&& d : v_current_device) {
      vp_devices_cl_.push_back(new cl::Device(d));
    }
  }

  // Retrieve informations about the device
  p_device_device_type_ =
    Memory::MemAlloc<cl_device_type>(vp_devices_cl_.size());
  p_device_vendor_ = Memory::MemAlloc<std::string>(vp_devices_cl_.size());
  p_device_version_ = Memory::MemAlloc<std::string>(vp_devices_cl_.size());
  p_device_driver_version_ =
    Memory::MemAlloc<std::string>(vp_devices_cl_.size());
  p_device_address_bits_ = Memory::MemAlloc<cl_uint>(vp_devices_cl_.size());
  p_device_available_ = Memory::MemAlloc<cl_bool>(vp_devices_cl_.size());
  p_device_compiler_available_ =
    Memory::MemAlloc<cl_bool>(vp_devices_cl_.size());
  p_device_global_mem_cache_size_ =
    Memory::MemAlloc<cl_ulong>(vp_devices_cl_.size());
  p_device_global_mem_cacheline_size_ =
    Memory::MemAlloc<cl_uint>(vp_devices_cl_.size());
  p_device_global_mem_size_ = Memory::MemAlloc<cl_ulong>(vp_devices_cl_.size());
  p_device_local_mem_size_ = Memory::MemAlloc<cl_ulong>(vp_devices_cl_.size());
  p_device_mem_base_addr_align_ =
    Memory::MemAlloc<cl_uint>(vp_devices_cl_.size());
  p_device_name_ = Memory::MemAlloc<std::string>(vp_devices_cl_.size());
  p_device_opencl_c_version_ =
    Memory::MemAlloc<std::string>(vp_devices_cl_.size());
  p_device_max_clock_frequency_ =
    Memory::MemAlloc<cl_uint>(vp_devices_cl_.size());
  p_device_max_compute_units_ =
    Memory::MemAlloc<cl_uint>(vp_devices_cl_.size());
  p_device_constant_buffer_size_ =
    Memory::MemAlloc<cl_ulong>(vp_devices_cl_.size());
  p_device_mem_alloc_size_ = Memory::MemAlloc<cl_ulong>(vp_devices_cl_.size());
  p_device_native_vector_width_double_ =
    Memory::MemAlloc<cl_uint>(vp_devices_cl_.size());
  p_device_preferred_vector_width_double_ =
    Memory::MemAlloc<cl_uint>(vp_devices_cl_.size());

  GGEMScout("OpenCLManager", "OpenCLManager", 1)
    << "Retrieving OpenCL device informations..." << GGEMSendl;
  // Make a char buffer reading char* data
  char char_data[1024];
  for (std::size_t i = 0; i < vp_devices_cl_.size(); ++i) {
    // Device Type
    CheckOpenCLError(vp_devices_cl_[i]->getInfo(CL_DEVICE_TYPE,
      &p_device_device_type_[i]), "OpenCLManager", "OpenCLManager");

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(CL_DEVICE_VENDOR,
      &p_device_vendor_[i]), "OpenCLManager", "OpenCLManager");

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(CL_DEVICE_VERSION,
      &char_data), "OpenCLManager", "OpenCLManager");
    p_device_version_[i] = std::string(char_data);

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(CL_DRIVER_VERSION,
      &char_data), "OpenCLManager", "OpenCLManager");
    p_device_driver_version_[i] = std::string(char_data);

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(CL_DEVICE_ADDRESS_BITS,
      &p_device_address_bits_[i]), "OpenCLManager", "OpenCLManager");

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(CL_DEVICE_AVAILABLE,
      &p_device_available_[i]), "OpenCLManager", "OpenCLManager");

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(CL_DEVICE_COMPILER_AVAILABLE,
      &p_device_compiler_available_[i]), "OpenCLManager", "OpenCLManager");

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
      &p_device_global_mem_cache_size_[i]), "OpenCLManager", "OpenCLManager");

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(
      CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
      &p_device_global_mem_cacheline_size_[i]), "OpenCLManager",
      "OpenCLManager");

    CheckOpenCLError(vp_devices_cl_[i]->getInfo( CL_DEVICE_GLOBAL_MEM_SIZE,
      &p_device_global_mem_size_[i]), "OpenCLManager", "OpenCLManager");

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(CL_DEVICE_LOCAL_MEM_SIZE,
      &p_device_local_mem_size_[i]), "OpenCLManager", "OpenCLManager");

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(CL_DEVICE_MEM_BASE_ADDR_ALIGN,
      &p_device_mem_base_addr_align_[i]), "OpenCLManager", "OpenCLManager");

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(CL_DEVICE_NAME,
      &p_device_name_[i]), "OpenCLManager", "OpenCLManager");

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(CL_DEVICE_OPENCL_C_VERSION,
      &char_data), "OpenCLManager", "OpenCLManager");
    p_device_opencl_c_version_[i] = std::string(char_data);

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY,
      &p_device_max_clock_frequency_[i]), "OpenCLManager", "OpenCLManager");

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS,
      &p_device_max_compute_units_[i]), "OpenCLManager", "OpenCLManager");

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(
      CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
      &p_device_constant_buffer_size_[i]), "OpenCLManager", "OpenCLManager");

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE,
      &p_device_mem_alloc_size_[i]), "OpenCLManager", "OpenCLManager");

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(
      CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
      &p_device_native_vector_width_double_[i]), "OpenCLManager",
      "OpenCLManager");

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(
      CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
      &p_device_preferred_vector_width_double_[i]), "OpenCLManager",
      "OpenCLManager");

    CheckOpenCLError(vp_devices_cl_[i]->getInfo(
      CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
      &p_device_native_vector_width_double_[i]), "OpenCLManager",
      "OpenCLManager");
  }

  // Define the compilation options by default for OpenCL
  build_options_ = "-cl-std=CL1.2 -w -Werror";

  // Add auxiliary function path to OpenCL options
  #ifdef GGEMS_PATH
  build_options_ += " -I";
  build_options_ += GGEMS_PATH;
  build_options_ += "/include";
  #elif
  Misc::ThrowException("OpenCLManager","OpenCLManager",
    "OPENCL_KERNEL_PATH not defined or not find!!!");
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

OpenCLManager::~OpenCLManager(void)
{
  // RAM manager
  if (p_used_ram_) {
    Memory::MemFree(p_used_ram_);
    p_used_ram_ = nullptr;
  }

  // Deleting kernel(s)
 // for (std::size_t i = 0; i < vp_kernel_cl_.size(); ++i) {
 //   delete vp_kernel_cl_.at(i);
 // }

  GGEMScout("OpenCLManager", "~OpenCLManager", 3)
    << "Deallocation of OpenCL Manager singleton..." << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::Clean(void)
{
  GGEMScout("OpenCLManager", "Clean", 3)
    << "Cleaning OpenCL platform, device, context, queue, event and kernel..."
    << GGEMSendl;

  // Deleting platform(s) and infos
  GGEMScout("OpenCLManager", "Clean", 1)
    << "Deleting OpenCL platform(s)..." << GGEMSendl;
  if (p_platform_vendor_) {
    Memory::MemFree(p_platform_vendor_);
    p_platform_vendor_ = nullptr;
    v_platforms_cl_.clear();
  }

  // Deleting device(s)
  GGEMScout("OpenCLManager", "Clean", 1)
    << "Deleting OpenCL device(s)..." << GGEMSendl;
  for (auto d : vp_devices_cl_) {
    delete d;
    d = nullptr;
  }
  vp_devices_cl_.clear();

  // Deleting device infos
  if (p_device_device_type_) Memory::MemFree(p_device_device_type_);
  if (p_device_vendor_) Memory::MemFree(p_device_vendor_);
  if (p_device_version_) Memory::MemFree(p_device_version_);
  if (p_device_driver_version_) Memory::MemFree(p_device_driver_version_);
  if (p_device_address_bits_) Memory::MemFree(p_device_address_bits_);
  if (p_device_available_) Memory::MemFree(p_device_available_);
  if (p_device_compiler_available_)
    Memory::MemFree(p_device_compiler_available_);
  if (p_device_global_mem_cache_size_)
    Memory::MemFree(p_device_global_mem_cache_size_);
  if (p_device_global_mem_cacheline_size_)
    Memory::MemFree(p_device_global_mem_cacheline_size_);
  if (p_device_global_mem_size_) Memory::MemFree(p_device_global_mem_size_);
  if (p_device_local_mem_size_) Memory::MemFree(p_device_local_mem_size_);
  if (p_device_mem_base_addr_align_)
    Memory::MemFree(p_device_mem_base_addr_align_);
  if (p_device_name_) Memory::MemFree(p_device_name_);
  if (p_device_opencl_c_version_) Memory::MemFree(p_device_opencl_c_version_);
  if (p_device_max_clock_frequency_)
    Memory::MemFree(p_device_max_clock_frequency_);
  if (p_device_max_compute_units_) Memory::MemFree(p_device_max_compute_units_);
  if (p_device_mem_alloc_size_) Memory::MemFree(p_device_mem_alloc_size_);
  if (p_device_native_vector_width_double_)
    Memory::MemFree(p_device_native_vector_width_double_);
  if (p_device_preferred_vector_width_double_)
    Memory::MemFree(p_device_preferred_vector_width_double_);

  // Deleting context(s)
  GGEMScout("OpenCLManager", "Clean", 1)
    << "Deleting OpenCL context(s)..." << GGEMSendl;
  for (auto&& c : vp_contexts_cl_) {
    delete c;
    c = nullptr;
  }
  vp_contexts_cl_.clear();
  vp_contexts_cpu_cl_.clear();
  vp_contexts_gpu_cl_.clear();
  vp_contexts_act_cl_.clear();

  // Deleting command queue(s)
  GGEMScout("OpenCLManager", "Clean", 1)
    << "Deleting OpenCL command queue(s)..." << GGEMSendl;
  for (auto&& q : vp_queues_cl_) {
    delete q;
    q = nullptr;
  }
  vp_queues_cl_.clear();
  vp_queues_act_cl_.clear();

  // Deleting event(s)
  GGEMScout("OpenCLManager", "Clean", 1)
    << "Deleting OpenCL event(s)..." << GGEMSendl;
  for (auto&& e : vp_event_cl_) {
    delete e;
    e = nullptr;
  }
  vp_event_cl_.clear();
  vp_event_act_cl_.clear();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::PrintPlatformInfos(void) const
{
  GGEMScout("OpenCLManager", "PrintPlatformInfos", 3)
    << "Printing infos about OpenCL platform(s)..." << GGEMSendl;

  // Loop over the platforms
  for (std::size_t i = 0; i < v_platforms_cl_.size(); ++i) {
    GGEMScout("OpenCLManager", "PrintPlatformInfos", 0) << GGEMSendl;
    GGEMScout("OpenCLManager", "PrintPlatformInfos", 0) << "#### PLATFORM: "
      << i << " ####" << GGEMSendl;
    GGEMScout("OpenCLManager", "PrintPlatformInfos", 0) << "+ Vendor: "
      << p_platform_vendor_[i] << GGEMSendl;
  }
  GGEMScout("OpenCLManager", "PrintPlatformInfos", 0) << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::PrintDeviceInfos(void) const
{
  GGEMScout("OpenCLManager", "PrintDeviceInfos", 3)
    << "Printing infos about OpenCL device(s)..." << GGEMSendl;

  for (std::size_t i = 0; i < vp_devices_cl_.size(); ++i) {
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0) << GGEMSendl;
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0) << "#### DEVICE: " << i
      << " ####" << GGEMSendl;
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0) << "+ Name: "
      << p_device_name_[i] << GGEMSendl;
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0) << "+ Vendor: "
      << p_device_vendor_[i] << GGEMSendl;
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0) << "+ Version: "
      << p_device_version_[i] << GGEMSendl;
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0) << "+ Driver Version: "
      << p_device_driver_version_[i] << GGEMSendl;
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0) << "+ OpenCL C Version: "
      << p_device_opencl_c_version_[ i ] << GGEMSendl;
    if (p_device_device_type_[i] == CL_DEVICE_TYPE_CPU) {
      GGEMScout("OpenCLManager", "PrintDeviceInfos", 0) << "+ Device Type: "
        << "+ Device Type: CL_DEVICE_TYPE_CPU" << GGEMSendl;
    }
    else if (p_device_device_type_[i] == CL_DEVICE_TYPE_GPU) {
      GGEMScout("OpenCLManager", "PrintDeviceInfos", 0) << "+ Device Type: "
        << "CL_DEVICE_TYPE_GPU" << GGEMSendl;
    }
    else {
      GGEMScout("OpenCLManager", "PrintDeviceInfos", 0) << "+ Device Type: "
        << "Unknown device type!!!" << GGEMSendl;
    }
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0) << "+ Address Bits: "
      << p_device_address_bits_[i] << " bits" << GGEMSendl;
    if (p_device_available_[i] == static_cast<cl_bool>(true)) {
      GGEMScout("OpenCLManager", "PrintDeviceInfos", 0)
        << "+ Device Available: ON" << GGEMSendl;
    }
    else {
      GGEMScout("OpenCLManager", "PrintDeviceInfos", 0)
        << "+ Device Available: OFF" << GGEMSendl;
    }
    if (p_device_compiler_available_[i] == static_cast<cl_bool>(true)) {
      GGEMScout("OpenCLManager", "PrintDeviceInfos", 0)
        << "+ Compiler Available: ON" << GGEMSendl;
    }
    else {
      GGEMScout("OpenCLManager", "PrintDeviceInfos", 0)
        << "+ Compiler Available: OFF" << GGEMSendl;
    }
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0)
      << "+ Global Mem. Cache Size: " << p_device_global_mem_cache_size_[i]
      << " bytes" << GGEMSendl;
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0)
      << "+ Global Mem. Line Cache Size: "
      << p_device_global_mem_cacheline_size_[i] << " bytes" << GGEMSendl;
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0) << "+ Global Mem. Size: "
      << p_device_global_mem_size_[i] << " bytes" << GGEMSendl;
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0) << "+ Local Mem. Size: "
      << p_device_local_mem_size_[ i ] << " bytes" << GGEMSendl;
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0)
      << "+ Mem. Base Addr. Align.: " << p_device_mem_base_addr_align_[i]
      << " bytes" << GGEMSendl;
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0)
      << "+ Native Vector Width Double: "
      << p_device_native_vector_width_double_[i] << GGEMSendl;
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0)
      << "+ Preferred Vector Width Double: "
      << p_device_preferred_vector_width_double_[i] << GGEMSendl;
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0)
      << "+ Max Clock Frequency: " << p_device_max_clock_frequency_[i] << " MHz"
      << GGEMSendl;
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0) << "+ Max Compute Units: "
      << p_device_max_compute_units_[i] << GGEMSendl;
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0)
      << "+ Constant Buffer Size: " << p_device_constant_buffer_size_[i]
      << " bytes" << GGEMSendl;
    GGEMScout("OpenCLManager", "PrintDeviceInfos", 0) << "+ Mem. Alloc. Size: "
      << p_device_mem_alloc_size_[i] << " bytes" << GGEMSendl;
  }
  GGEMScout("OpenCLManager", "PrintDeviceInfos", 0) << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::PrintBuildOptions(void) const
{
  GGEMScout("OpenCLManager", "PrintBuildOptions", 3)
    << "Printing infos about OpenCL compilation options..." << GGEMSendl;

  GGEMScout("OpenCLManager", "PrintBuildOptions", 0)
    << "OpenCL building options: " << build_options_ << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::CreateContext(void)
{
  GGEMScout("OpenCLManager", "CreateContext", 3)
    << "Creating context for CPU and/or GPU..." << GGEMSendl;

  // Loop over the devices
  for (std::size_t i = 0; i < vp_devices_cl_.size(); ++i) {
    // Get GPU type
    if ((p_device_device_type_[i] == CL_DEVICE_TYPE_GPU)) {
      vp_contexts_cl_.push_back(new cl::Context(*vp_devices_cl_[i]));
      vp_contexts_gpu_cl_.push_back(vp_contexts_cl_.back());
    }

    if ((p_device_device_type_[i] == CL_DEVICE_TYPE_CPU)) {
      vp_contexts_cl_.push_back(new cl::Context(*vp_devices_cl_[i]));
      vp_contexts_cpu_cl_.push_back(vp_contexts_cl_.back());
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::ContextToActivate(uint32_t const& context_id)
{
  GGEMScout("OpenCLManager", "ContextToActivate", 3)
    << "Activating a context for GGEMS..." << GGEMSendl;

  // Checking if a context has already been activated
  if (!vp_contexts_act_cl_.empty())
  {
    Clean();
    Misc::ThrowException("OpenCLManager", "ContextToActivate",
      "A context has already been activated!!!");
  }

  // Checking the index of the context
  if (context_id >= vp_contexts_cl_.size()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Your context index is out of range!!! " << vp_contexts_cl_.size()
        << " context(s) detected. Index must be in the range [" << 0 << ";"
        << vp_contexts_cl_.size() - 1 << "]!!!";
    Clean();
    Misc::ThrowException("OpenCLManager", "ContextToActivate", oss.str());
  }

  // Activate the context
  vp_contexts_act_cl_.push_back(vp_contexts_cl_.at(context_id));

  // Storing the index of activated context
  context_index_ = context_id;

  // Checking if an Intel HD Graphics has been selected and send a warning
  std::size_t found_device = p_device_name_[context_id].find("HD Graphics");
  if (found_device != std::string::npos) {
    GGEMSwarn("OpenCLManager", "ContextToActivate", 0)
      << "##########################################" << GGEMSendl;
    GGEMSwarn("OpenCLManager", "ContextToActivate", 0)
      << "#                WARNING!!!              #" << GGEMSendl;
    GGEMSwarn("OpenCLManager", "ContextToActivate", 0)
      << "##########################################" << GGEMSendl;
    GGEMSwarn("OpenCLManager", "ContextToActivate", 0)
      << "You are using a HD Graphics architecture, GGEMS could work on this"
      << " device, but the computation is very slow and the result could be"
      << " hazardous. Please use a CPU architecture or a GPU." << GGEMSendl;
  }

  // Checking if the double precision is activated on OpenCL device
  if (p_device_native_vector_width_double_[context_id] == 0) {
    Clean();
    Misc::ThrowException("OpenCLManager", "ContextToActivate",
      "Your OpenCL device does not support double precision!!!");
  }

  // Activate the command queue
  vp_queues_act_cl_.push_back(vp_queues_cl_.at(context_id));

  // Activate the event
  vp_event_act_cl_.push_back(vp_event_cl_.at(context_id));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::PrintActivatedContextInfos(void) const
{
  GGEMScout("OpenCLManager", "PrintActivatedContextInfos", 3)
    << "Printing activated context for GGEMS..." << GGEMSendl;

  cl_uint context_num_devices = 0;
  std::vector<cl::Device> device;
  cl_device_type device_type = 0;
  std::string device_name;

  GGEMScout("OpenCLManager", "PrintActivatedContextInfos", 0) << GGEMSendl;
  GGEMScout("OpenCLManager", "PrintActivatedContextInfos", 0)
    << "ACTIVATED CONTEXT(S):" << GGEMSendl;
  GGEMScout("OpenCLManager", "PrintActivatedContextInfos", 0)
    << "---------------------" << GGEMSendl;

  // Loop over all the context
  GGEMScout("OpenCLManager", "PrintActivatedContextInfos", 0) << GGEMSendl;
  GGEMScout("OpenCLManager", "PrintActivatedContextInfos", 0)
    << "#### CONTEXT: " << context_index_ << " ####" << GGEMSendl;

  CheckOpenCLError(vp_contexts_act_cl_.at(0)->getInfo(CL_CONTEXT_NUM_DEVICES,
    &context_num_devices), "OpenCLManager", "PrintActivatedContextInfos");

  CheckOpenCLError(vp_contexts_act_cl_.at(0)->getInfo(CL_CONTEXT_DEVICES,
    &device), "OpenCLManager", "PrintActivatedContextInfos");

  GGEMScout("OpenCLManager", "PrintActivatedContextInfos", 0)
    << "+ Type of device(s): " << GGEMSendl;

  for (uint32_t j = 0; j < context_num_devices; ++j) {
    CheckOpenCLError(device[ j ].getInfo(CL_DEVICE_NAME, &device_name),
      "OpenCLManager", "PrintActivatedContextInfos");

    GGEMScout("OpenCLManager", "PrintActivatedContextInfos", 0)
      << "    -> Name: " << device_name << GGEMSendl;

    CheckOpenCLError(device[j].getInfo(CL_DEVICE_TYPE, &device_type),
      "OpenCLManager", "PrintActivatedContextInfos");

    if( device_type == CL_DEVICE_TYPE_CPU )
    {
      GGEMScout("OpenCLManager", "PrintActivatedContextInfos", 0)
        << "    -> Device [" << j << "]: "
        << "CL_DEVICE_TYPE_CPU" << GGEMSendl;
    }
    else if( device_type == CL_DEVICE_TYPE_GPU )
    {
      GGEMScout("OpenCLManager", "PrintActivatedContextInfos", 0)
        << "    -> Device [" << j << "]: "
        << "CL_DEVICE_TYPE_GPU" << GGEMSendl;
    }
    else
    {
      GGEMScout("OpenCLManager", "PrintActivatedContextInfos", 0)
        << "    -> Device [" << j << "]: "
        << "Unknown device type!!!" << GGEMSendl;
    }
  }
  GGEMScout("OpenCLManager", "PrintActivatedContextInfos", 0) << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::CreateCommandQueue(void)
{
  GGEMScout("OpenCLManager","CreateCommandQueue", 3)
    << "Creating command queue for CPU and/or GPU..." << GGEMSendl;

  // Vector of devices in the context
  std::vector<cl::Device> device;
  // Loop over the contexts
  for (std::size_t i = 0; i < vp_contexts_cl_.size(); ++i) {
    // Get the vector of devices include in the context
    CheckOpenCLError(vp_contexts_cl_[i]->getInfo(CL_CONTEXT_DEVICES, &device),
      "OpenCLManager", "CreateCommandQueue");

    vp_queues_cl_.push_back(new cl::CommandQueue(*vp_contexts_cl_[i], device[0],
      CL_QUEUE_PROFILING_ENABLE));
  }
  device.clear();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::PrintCommandQueueInfos(void) const
{
  GGEMScout("OpenCLManager","PrintCommandQueueInfos", 3)
    << "Printing infos about OpenCL command queue(s)..." << GGEMSendl;

  cl::Device device;
  std::string device_name;

  // Loop over the queues
  for (std::size_t i = 0; i < vp_queues_cl_.size(); ++i) {
    GGEMScout("OpenCLManager","PrintCommandQueueInfos", 0) << GGEMSendl;

    GGEMScout("OpenCLManager","PrintCommandQueueInfos", 0)
      << "#### COMMAND QUEUE: " << i << " ####" << GGEMSendl;

    CheckOpenCLError(vp_queues_cl_[i]->getInfo(CL_QUEUE_DEVICE, &device),
      "OpenCLManager", "PrintCommandQueueInfos");
    CheckOpenCLError(device.getInfo(CL_DEVICE_NAME, &device_name),
      "OpenCLManager", "PrintCommandQueueInfos");

    GGEMScout("OpenCLManager","PrintCommandQueueInfos", 0)
      << "+ Device Name: " << device_name << GGEMSendl;
    GGEMScout("OpenCLManager","PrintCommandQueueInfos", 0)
      << "+ Command Queue Index: " << i << GGEMSendl;
    GGEMScout("OpenCLManager","PrintCommandQueueInfos", 0) << GGEMSendl;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::CreateEvent(void)
{
  GGEMScout("OpenCLManager","CreateEvent", 3)
    << "Creating event for CPU and/or GPU..." << GGEMSendl;

  for (std::size_t i = 0; i < vp_contexts_cl_.size(); ++i) {
    vp_event_cl_.push_back(new cl::Event());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

cl::Kernel* OpenCLManager::CompileKernel(std::string const& kernel_filename,
  std::string const& kernel_name, char* const p_custom_options,
  char* const p_additional_options)
{
  GGEMScout("OpenCLManager","CompileKernel", 3)
    << "Compiling a kernel on OpenCL activated context..." << GGEMSendl;

  // Checking the compilation options
  if (p_custom_options && p_additional_options) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Custom and additional options can not by set in same time!!!";
    Misc::ThrowException("OpenCLManager", "CompileKernel", oss.str());
  }

  // Check if the source kernel file exists
  std::ifstream source_file_stream(kernel_filename.c_str(), std::ios::in);
  Stream::CheckInputStream(source_file_stream, kernel_filename);

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

  GGEMScout("OpenCLManager", "CompileKernel", 0) << "Compile a new kernel '"
    << kernel_name << "' from file: " << kernel_filename << " on context: "
    << context_index_ << " with options: " << kernel_compilation_option
    << GGEMSendl;

  // Store kernel in a std::string buffer
  std::string source_code(std::istreambuf_iterator<char>(source_file_stream),
    (std::istreambuf_iterator<char>()));

  // Creating an OpenCL program
  cl::Program::Sources program_source(1,
    std::make_pair(source_code.c_str(), source_code.length() + 1));

  // Make program from source code in specific context
  cl::Program program = cl::Program(*vp_contexts_act_cl_.at(0), program_source);

  // Vector storing all the devices from a context
  // In GGEMS a device is associated to a context
  std::vector<cl::Device> devices;

  // Get the vector of devices
  CheckOpenCLError(vp_contexts_act_cl_.at(0)->getInfo(CL_CONTEXT_DEVICES,
    &devices), "OpenCLManager", "CompileKernel");

  // Compile source code on devices
  cl_int build_status = program.build(devices, kernel_compilation_option);
  if (build_status != CL_SUCCESS) {
    std::ostringstream oss(std::ostringstream::out);
    std::string log;
    program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &log);
    oss << ErrorType(build_status) << std::endl;
    oss << log;
    Clean();
    Misc::ThrowException("OpenCLManager", "CompileKernel", oss.str());
  }

  GGEMScout("OpenCLManager", "CompileKernel", 0) << "Compilation OK"
    << GGEMSendl;

  // Storing the kernel in the singleton
  vp_kernel_cl_.push_back(new cl::Kernel(program, kernel_name.c_str(),
    &build_status));
  CheckOpenCLError(build_status, "OpenCLManager", "CompileKernel");

  return vp_kernel_cl_.back();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::InitializeRAMManager(void)
{
  GGEMScout("OpenCLManager", "InitializeRAMManager", 3)
    << "Initializing a RAM handler for each context..." << GGEMSendl;

  // For each context we create a RAM handler
  p_used_ram_ = Memory::MemAlloc<cl_ulong>(vp_contexts_cl_.size());
  for (std::size_t i = 0; i < vp_contexts_cl_.size(); ++i) p_used_ram_[i] = 0;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::PrintRAMStatus(void) const
{
  GGEMScout("OpenCLManager", "PrintRAMStatus", 3)
    << "Printing infos about RAM memory on OpenCL context(s)..." << GGEMSendl;

  GGEMScout("OpenCLManager", "PrintRAMStatus", 0)
    << "---------------------------" << GGEMSendl;

  // Loop over the contexts
  for (std::size_t i = 0; i < vp_contexts_cl_.size(); ++i) {
    // Get the max. RAM memory by context
    cl_ulong const max_RAM = p_device_global_mem_size_[i];
    float const percent_RAM = static_cast<float>(p_used_ram_[i])
      * 100.0f / static_cast<float>(max_RAM);
    GGEMScout("OpenCLManager", "PrintRAMStatus", 0) << "Context " << i << ": "
      << p_used_ram_[i] << " / " << max_RAM
      << " bytes -> " << percent_RAM << " % used" << GGEMSendl;
  }
  GGEMScout("OpenCLManager", "PrintRAMStatus", 0)
    << "---------------------------" << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::CheckRAMMemory(std::size_t const& size)
{
  GGEMScout("OpenCLManager","CheckRAMMemory", 3)
    << "Checking RAM memory usage..." << GGEMSendl;

  // Getting memory infos
  cl_ulong const max_RAM = p_device_global_mem_size_[context_index_];
  float const percent_RAM =
    static_cast<float>(p_used_ram_[context_index_] + size)
    * 100.0f / static_cast<float>(max_RAM);

  if (percent_RAM >= 80.0f && percent_RAM < 95.0f) {
    GGEMSwarn("OpenCLManager", "CheckRAMMemory", 0)
      << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << GGEMSendl;
    GGEMSwarn("OpenCLManager", "CheckRAMMemory", 0)
      << "!!!             MEMORY WARNING             !!!" << GGEMSendl;
    GGEMSwarn("OpenCLManager", "CheckRAMMemory", 0)
      << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << GGEMSendl;
    GGEMSwarn("OpenCLManager", "CheckRAMMemory", 0)
      << "RAM allocation (" << percent_RAM <<
      "%) is superior to 80%, the simulation will be "
      << "automatically killed whether RAM allocation is superior to 95%"
      << GGEMSendl;
  }
  else if (percent_RAM >= 95.0f) {
    GGEMScerr("OpenCLManager", "CheckRAMMemory", 0)
      << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << GGEMSendl;
    GGEMScerr("OpenCLManager", "CheckRAMMemory", 0)
      << "!!!             MEMORY ERROR             !!!" << GGEMSendl;
    GGEMScerr("OpenCLManager", "CheckRAMMemory", 0)
      << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << GGEMSendl;
    GGEMScerr("OpenCLManager", "CheckRAMMemory", 0)
      << "RAM allocation (" << percent_RAM <<
      "%) is superior to 95%, the simulation is killed!!!" << GGEMSendl;
    Clean();
    Misc::ThrowException("OpenCLManager", "CheckRAMMemory",
      "Not enough RAM memory on OpenCL device!!!");
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::AddRAMMemory(cl_ulong const& size)
{
  GGEMScout("OpenCLManager","AddRAMMemory", 3)
    << "Adding RAM memory on OpenCL activated context..." << GGEMSendl;

  // Increment size
  p_used_ram_[context_index_] += size;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::SubRAMMemory(cl_ulong const& size)
{
  GGEMScout("OpenCLManager","SubRAMMemory", 3)
    << "Substracting RAM memory on OpenCL activated context..." << GGEMSendl;

  // Decrement size
  p_used_ram_[context_index_] -= size;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::DisplayElapsedTimeInKernel(
  std::string const& kernel_name) const
{
  GGEMScout("OpenCLManager","DisplayElapsedTimeInKernel", 3)
    << "Displaying elapsed time in the last OpenCL kernel..." << GGEMSendl;

  // Get the activated event
  cl::Event* p_event = vp_event_act_cl_.at(0);

  // Get the start and end of the activated event
  cl_ulong start = 0, end = 0;

  // Start
  CheckOpenCLError(p_event->getProfilingInfo(
    CL_PROFILING_COMMAND_START, &start), "OpenCLManager",
    "DisplayElapsedTimeInKernel");

  // End
  CheckOpenCLError(p_event->getProfilingInfo(
    CL_PROFILING_COMMAND_END, &end), "OpenCLManager",
    "DisplayElapsedTimeInKernel");

  DurationNano duration = static_cast<std::chrono::nanoseconds>((end-start));

  // Display time in kernel
  Chrono::DisplayTime(duration, kernel_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::PrintContextInfos(void) const
{
  GGEMScout("OpenCLManager","PrintContextInfos", 3)
    << "Printing infos about OpenCL context(s)..." << GGEMSendl;

  cl_uint context_number_devices = 0;
  cl_uint reference_count = 0;
  std::vector<cl::Device> device;
  cl_device_type device_type = 0;
  std::string device_name;

  GGEMScout("OpenCLManager", "PrintContextInfos", 0) << GGEMSendl;

  // Loop over all the context
  for (std::size_t i = 0; i < vp_contexts_cl_.size(); ++i) {
    GGEMScout("OpenCLManager", "PrintContextInfos", 0) << "#### CONTEXT: " << i
      << " ####" << GGEMSendl;

    CheckOpenCLError(vp_contexts_cl_[i]->getInfo(CL_CONTEXT_NUM_DEVICES,
      &context_number_devices), "OpenCLManager", "PrintContextInfos");

    if (context_number_devices > 1) {
      Misc::ThrowException("OpenCLManager", "PrintContextInfos",
        "One device by context only!!!");
    }

    CheckOpenCLError(vp_contexts_cl_[i]->getInfo(CL_CONTEXT_REFERENCE_COUNT,
      &reference_count), "OpenCLManager", "PrintContextInfos");

    CheckOpenCLError(vp_contexts_cl_[i]->getInfo(CL_CONTEXT_DEVICES, &device),
      "OpenCLManager", "PrintContextInfos");

    GGEMScout("OpenCLManager", "PrintContextInfos", 0)
      << "+ Type of device(s): " << GGEMSendl;
    for (uint32_t j = 0; j < context_number_devices; ++j) {
      CheckOpenCLError(device[j].getInfo(CL_DEVICE_NAME, &device_name),
        "OpenCLManager", "PrintContextInfos");
      GGEMScout("OpenCLManager", "PrintContextInfos", 0) << "    -> Name: "
        << device_name << GGEMSendl;

      CheckOpenCLError(device[j].getInfo(CL_DEVICE_TYPE, &device_type),
        "OpenCLManager", "PrintContextInfos");

      if (device_type == CL_DEVICE_TYPE_CPU) {
        GGEMScout("OpenCLManager", "PrintContextInfos", 0) << "    -> Device ["
          << j << "]: CL_DEVICE_TYPE_CPU" << GGEMSendl;
      }
      else if (device_type == CL_DEVICE_TYPE_GPU) {
        GGEMScout("OpenCLManager", "PrintContextInfos", 0) << "    -> Device ["
          << j << "]: CL_DEVICE_TYPE_GPU" << GGEMSendl;
      }
      else {
        GGEMScout("OpenCLManager", "PrintContextInfos", 0) << "    -> Device ["
          << j << "]: Unknown device type!!!" << GGEMSendl;
      }
      GGEMScout("OpenCLManager", "PrintContextInfos", 0) << GGEMSendl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

cl::Buffer* OpenCLManager::Allocate(void* p_host_ptr, std::size_t size,
  cl_mem_flags flags)
{
  GGEMScout("OpenCLManager","Allocate", 3)
    << "Allocating memory on OpenCL device memory..." << GGEMSendl;

  ///////////////
  // Checking if enough memory on device!!!
  ///////////////////
  // Checking memory usage
  CheckRAMMemory(size);

  cl_int error = 0;
  cl::Buffer* p_buffer = new cl::Buffer(*vp_contexts_act_cl_.at(0), flags,
    size, p_host_ptr, &error);
  CheckOpenCLError(error, "OpenCLManager", "Allocate");
  AddRAMMemory(size);
  return p_buffer;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::Deallocate(cl::Buffer* p_buffer, std::size_t size)
{
  GGEMScout("OpenCLManager","Deallocate", 3)
    << "Deallocating memory on OpenCL device memory..." << GGEMSendl;

  SubRAMMemory(size);
  delete p_buffer;
  p_buffer = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void OpenCLManager::CheckOpenCLError(cl_int const& error,
  std::string const& class_name, std::string const& method_name) const
{
  if (error != CL_SUCCESS) {
    Misc::ThrowException(class_name, method_name, ErrorType(error));
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

std::string OpenCLManager::ErrorType(cl_int const& error) const
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
      oss << "    * if no OpenCL devices that matched device_type were found."
          << std::endl;
      return oss.str();
    }
    case -2: {
      oss << "CL_DEVICE_NOT_AVAILABLE:" << std::endl;
      oss << "    * if a device in devices is currently not available even\n"
          << "      though the device was returned by clGetDeviceIDs."
          << std::endl;
      return oss.str();
    }
    case -3: {
      oss << "CL_COMPILER_NOT_AVAILABLE:" << std::endl;
      oss << "    * if program is created with clCreateProgramWithSource and\n"
          << "      a compiler is not available i.e.\n"
          << "      CL_DEVICE_COMPILER_AVAILABLE specified in the table of\n"
          << "      OpenCL Device Queries for clGetDeviceInfo is set to\n"
          << "      CL_FALSE."
          << std::endl;
      return oss.str();
    }
    case -4: {
      oss << "CL_MEM_OBJECT_ALLOCATION_FAILURE:" << std::endl;
      oss << "    * if there is a failure to allocate memory for buffer\n"
          << "      object."
          << std::endl;
      return oss.str();
    }
    case -5: {
      oss << "CL_OUT_OF_RESOURCES:" << std::endl;
      oss << "    * if there is a failure to allocate resources required by\n"
          << "      the OpenCL implementation on the device."
          << std::endl;
      return oss.str();
    }
    case -6: {
      oss << "CL_OUT_OF_HOST_MEMORY:" << std::endl;
      oss << "    * if there is a failure to allocate resources required by\n"
          << "      the OpenCL implementation on the host."
          << std::endl;
      return oss.str();
    }
    case -7: {
      oss << "CL_PROFILING_INFO_NOT_AVAILABLE:" << std::endl;
      oss << "    * if the CL_QUEUE_PROFILING_ENABLE flag is not set for the\n"
          << "      command-queue, if the execution status of the command\n"
          << "      identified by event is not CL_COMPLETE or if event is a\n"
          << "      user event object."
          << std::endl;
      return oss.str();
    }
    case -8: {
      oss << "CL_MEM_COPY_OVERLAP:" << std::endl;
      oss << "    * if src_buffer and dst_buffer are the same buffer or\n"
          << "      subbuffer object and the source and destination regions\n"
          << "      overlap or if src_buffer and dst_buffer are different\n"
          << "      sub-buffers of the same associated buffer object and they\n"
          << "      overlap. The regions overlap if src_offset <= to\n"
          << "      dst_offset <= to src_offset + size – 1, or if dst_offset\n"
          << "      <= to src_offset <= to dst_offset + size – 1."
          << std::endl;
      return oss.str();
    }
    case -9: {
      oss << "CL_IMAGE_FORMAT_MISMATCH:" << std::endl;
      oss << "    * if src_image and dst_image do not use the same image\n"
          << "      format."
          << std::endl;
      return oss.str();
    }
    case -10: {
      oss << "CL_IMAGE_FORMAT_NOT_SUPPORTED:" << std::endl;
      oss << "    * if the image_format is not supported." << std::endl;
      return oss.str();
    }
    case -11: {
      oss << "CL_BUILD_PROGRAM_FAILURE:" << std::endl;
      oss << "    * if there is a failure to build the program executable.\n"
          << "      This error will be returned if clBuildProgram does not\n"
          << "      return until the build has completed."
          << std::endl;
      return oss.str();
    }
    case -12: {
      oss << "CL_MAP_FAILURE:" << std::endl;
      oss << "    * if there is a failure to map the requested region into\n"
          << "      the host address space. This error cannot occur for\n"
          << "      image objects created with CL_MEM_USE_HOST_PTR or\n"
          << "      CL_MEM_ALLOC_HOST_PTR."
          << std::endl;
      return oss.str();
    }
    case -13: {
      oss << "CL_MISALIGNED_SUB_BUFFER_OFFSET:" << std::endl;
      oss << "    * if a sub-buffer object is specified as the value for an\n"
          << "      argument that is a buffer object and the offset\n"
          << "      specified when the sub-buffer object is created is not\n"
          << "      aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for\n"
          << "      device associated with queue."
          << std::endl;
      return oss.str();
    }
    case -14: {
      oss << "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:" << std::endl;
      oss << "    * if the execution status of any of the events in\n"
          << "      event_list is a negative integer value."
          << std::endl;
      return oss.str();
    }
    case -15: {
      oss << "CL_COMPILE_PROGRAM_FAILURE:" << std::endl;
      oss << "    * if there is a failure to compile the program source.\n"
          << "      This error will be returned if clCompileProgram does\n"
          << "      not return until the compile has completed."
          << std::endl;
      return oss.str();
    }
    case -16: {
      oss << "CL_LINKER_NOT_AVAILABLE:" << std::endl;
      oss << "    * if a linker is not available i.e.\n"
          << "      CL_DEVICE_LINKER_AVAILABLE specified in the table of\n"
          << "      allowed values for param_name for clGetDeviceInfo is set\n"
          << "      to CL_FALSE."
          << std::endl;
      return oss.str();
    }
    case -17: {
      oss << "CL_LINK_PROGRAM_FAILURE:" << std::endl;
      oss << "    * if there is a failure to link the compiled binaries\n"
          << "      and/or libraries."
          << std::endl;
      return oss.str();
    }
    case -18: {
      oss << "CL_DEVICE_PARTITION_FAILED:" << std::endl;
      oss << "    * if the partition name is supported by the implementation\n"
          << "      but in_device could not be further partitioned."
          << std::endl;
      return oss.str();
    }
    case -19: {
      oss << "CL_KERNEL_ARG_INFO_NOT_AVAILABLE:" << std::endl;
      oss << "    * if the argument information is not available for kernel."
          << std::endl;
      return oss.str();
    }
    case -30: {
      oss << "CL_INVALID_VALUE:" << std::endl;
      oss << "    * This depends on the function: two or more coupled\n"
          << "      parameters had errors."
          << std::endl;
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
      oss << "    * if devices contains an invalid device or are not\n"
          << "      associated with the specified platform."
          << std::endl;
      return oss.str();
    }
    case -34: {
      oss << "CL_INVALID_CONTEXT:" << std::endl;
      oss << "    * if context is not a valid context." << std::endl;
      return oss.str();
    }
    case -35: {
      oss << "CL_INVALID_QUEUE_PROPERTIES:" << std::endl;
      oss << "    * if specified command-queue-properties are valid but are\n"
          << "      not supported by the device."
          << std::endl;
      return oss.str();
    }
    case -36: {
      oss << "CL_INVALID_COMMAND_QUEUE:" << std::endl;
      oss << "    * if command_queue is not a valid command-queue."
          << std::endl;
      return oss.str();
    }
    case -37: {
      oss << "CL_INVALID_HOST_PTR:" << std::endl;
      oss << "    * This flag is valid only if host_ptr is not NULL. If\n"
          << "      specified, it indicates that the application wants the\n"
          << "      OpenCL implementation to allocate memory for the memory\n"
          << "      object and copy the data from memory referenced by\n"
          << "      host_ptr.CL_MEM_COPY_HOST_PTR and CL_MEM_USE_HOST_PTR\n"
          << "      are mutually exclusive.CL_MEM_COPY_HOST_PTR can be used\n"
          << "      with CL_MEM_ALLOC_HOST_PTR to initialize the contents of\n"
          << "      the cl_mem object allocated using host-accessible\n"
          << "      (e.g. PCIe) memory."
          << std::endl;
      return oss.str();
    }
    case -38: {
      oss << "CL_INVALID_MEM_OBJECT:" << std::endl;
      oss << "    * if memobj is not a valid OpenCL memory object."
          << std::endl;
      return oss.str();
    }
    case -39: {
      oss << "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:" << std::endl;
      oss << "    * if the OpenGL/DirectX texture internal format does not\n"
          << "      map to a supported OpenCL image format."
          << std::endl;
      return oss.str();
    }
    case -40: {
      oss << "CL_INVALID_IMAGE_SIZE:" << std::endl;
      oss << "    * if an image object is specified as an argument value and\n"
          << "      the image dimensions (image width, height, specified or\n"
          << "      compute row and/or slice pitch) are not supported by\n"
          << "      device associated with queue."
          << std::endl;
      return oss.str();
    }
    case -41: {
      oss << "CL_INVALID_SAMPLER:" << std::endl;
      oss << "    * if sampler is not a valid sampler object." << std::endl;
      return oss.str();
    }
    case -42: {
      oss << "CL_INVALID_BINARY:" << std::endl;
      oss << "    * The provided binary is unfit for the selected device.if\n"
          << "      program is created with clCreateProgramWithBinary and\n"
          << "      devices listed in device_list do not have a valid\n"
          << "      program binary loaded."
          << std::endl;
      return oss.str();
    }
    case -43: {
      oss << "CL_INVALID_BUILD_OPTIONS:" << std::endl;
      oss << "    * if the build options specified by options are invalid."
          << std::endl;
      return oss.str();
    }
    case -44: {
      oss << "CL_INVALID_PROGRAM:" << std::endl;
      oss << "    * if program is a not a valid program object." << std::endl;
      return oss.str();
    }
    case -45: {
      oss << "CL_INVALID_PROGRAM_EXECUTABLE:" << std::endl;
      oss << "    * if there is no successfully built program executable\n"
          << "      available for device associated with command_queue."
          << std::endl;
      return oss.str();
    }
    case -46: {
      oss << "CL_INVALID_KERNEL_NAME:" << std::endl;
      oss << "    * if kernel_name is not found in program."  << std::endl;
      return oss.str();
    }
    case -47: {
      oss << "CL_INVALID_KERNEL_DEFINITION:" << std::endl;
      oss << "    * if the function definition for __kernel function given\n"
          << "      by kernel_name such as the number of arguments, the\n"
          << "      argument types are not the same for all devices for\n"
          << "      which the program executable has been built."
          << std::endl;
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
      oss << "    * if arg_size does not match the size of the data type for\n"
          << "      an argument that is not a memory object or if the\n"
          << "      argument is a memory object and arg_size !=\n"
          << "      sizeof(cl_mem) or if arg_size is zero and the argument\n"
          << "      is declared with the __local qualifier or if the\n"
          << "      argument is a sampler and arg_size != sizeof(cl_sampler)."
          << std::endl;
      return oss.str();
    }
    case -52: {
      oss << "CL_INVALID_KERNEL_ARGS:" << std::endl;
      oss << "    * if the kernel argument values have not been specified."
          << std::endl;
      return oss.str();
    }
    case -53: {
      oss << "CL_INVALID_WORK_DIMENSION:" << std::endl;
      oss << "    * if work_dim is not a valid value (i.e. a value between\n"
          << "      1 and 3)."
          << std::endl;
      return oss.str();
    }
    case -54: {
      oss << "CL_INVALID_WORK_GROUP_SIZE:" << std::endl;
      oss << "    * if local_work_size is specified and number of\n"
          << "      work-items specified by global_work_size is not evenly\n"
          << "      divisable by size of work-group given by local_work_size\n"
          << "      or does not match the work-group size specified for\n"
          << "      kernel using the\n"
          << "      __attribute__((reqd_work_group_size(X, Y, Z)))\n"
          << "      qualifier in program source.if local_work_size is\n"
          << "      specified and the total number of work-items in the\n"
          << "      work-group computed as local_work_size[0] *...\n"
          << "      local_work_size[work_dim – 1] is greater than the value\n"
          << "      specified by CL_DEVICE_MAX_WORK_GROUP_SIZE in the table\n"
          << "      of OpenCL Device Queries for clGetDeviceInfo. if\n"
          << "      local_work_size is NULL and the\n"
          << "      __attribute__ ((reqd_work_group_size(X, Y, Z)))\n"
          << "      qualifier is used to declare the work-group size for\n"
          << "      kernel in the program source."
          << std::endl;
      return oss.str();
    }
    case -55: {
      oss << "CL_INVALID_WORK_ITEM_SIZE:" << std::endl;
      oss << "    * if the number of work-items specified in any of\n"
          << "      local_work_size[0], … local_work_size[work_dim – 1] is\n"
          << "      greater than the corresponding values specified by\n"
          << "      CL_DEVICE_MAX_WORK_ITEM_SIZES[0], ....\n"
          << "      CL_DEVICE_MAX_WORK_ITEM_SIZES[work_dim – 1]"
          << std::endl;
      return oss.str();
    }
    case -56: {
      oss << "CL_INVALID_GLOBAL_OFFSET:" << std::endl;
      oss << "    * if the value specified in global_work_size + the\n"
          << "      corresponding values in global_work_offset for any\n"
          << "      dimensions is greater than the sizeof(size_t) for the\n"
          << "      device on which the kernel execution will be enqueued."
          << std::endl;
      return oss.str();
    }
    case -57: {
      oss << "CL_INVALID_EVENT_WAIT_LIST:" << std::endl;
      oss << "    * if event_wait_list is NULL and num_events_in_wait_list\n"
          << "      > 0, or event_wait_list is not NULL and\n"
          << "      num_events_in_wait_list is 0, or if event objects in\n"
          << "      event_wait_list are not valid events."
          << std::endl;
      return oss.str();
    }
    case -58: {
      oss << "CL_INVALID_EVENT:" << std::endl;
      oss << "    * if event objects specified in event_list are not valid\n"
          << "      event objects."
          << std::endl;
      return oss.str();
    }
    case -59: {
      oss << "CL_INVALID_OPERATION:" << std::endl;
      oss << "    * if interoperability is specified by setting\n"
          << "      CL_CONTEXT_ADAPTER_D3D9_KHR,\n"
          << "      CL_CONTEXT_ADAPTER_D3D9EX_KHR or\n"
          << "      CL_CONTEXT_ADAPTER_DXVA_KHR to a non-NULL value, and\n"
          << "      interoperability with another graphics API is also\n"
          << "      specified. (only if the cl_khr_dx9_media_sharing\n"
          << "      extension is supported)."
          << std::endl;
      return oss.str();
    }
    case -60: {
      oss << "CL_INVALID_GL_OBJECT:" << std::endl;
      oss << "    * if texture is not a GL texture object whose type\n"
          << "      matches texture_target, if the specified miplevel of\n"
          << "      texture is not defined, or if the width or height of the\n"
          << "      specified miplevel is zero."
          << std::endl;
      return oss.str();
    }
    case -61: {
      oss << "CL_INVALID_BUFFER_SIZE:" << std::endl;
      oss << "    * if size is 0.Implementations may return\n"
          << "      CL_INVALID_BUFFER_SIZE if size is greater than the\n"
          << "      CL_DEVICE_MAX_MEM_ALLOC_SIZE value specified in the\n"
          << "      table of allowed values for param_name for\n"
          << "      clGetDeviceInfo for all devices in context."
          << std::endl;
      return oss.str();
    }
    case -62: {
      oss << "CL_INVALID_MIP_LEVEL:" << std::endl;
      oss << "    * if miplevel is greater than zero and the OpenGL\n"
          << "      implementation does not support creating from non-zero\n"
          << "      mipmap levels."
          << std::endl;
      return oss.str();
    }
    case -63: {
      oss << "CL_INVALID_GLOBAL_WORK_SIZE:" << std::endl;
      oss << "    * if global_work_size is NULL, or if any of the values\n"
          << "      specified in global_work_size[0], ...\n"
          << "      global_work_size [work_dim – 1] are 0 or exceed the\n"
          << "      range given by the sizeof(size_t) for the device on\n"
          << "      which the kernel execution will be enqueued."
          << std::endl;
      return oss.str();
    }
    case -64: {
      oss << "CL_INVALID_PROPERTY:" << std::endl;
      oss << "    * Vague error, depends on the function" << std::endl;
      return oss.str();
    }
    case -65: {
      oss << "CL_INVALID_IMAGE_DESCRIPTOR:" << std::endl;
      oss << "    * if values specified in image_desc are not valid or if\n"
          << "      image_desc is NULL."
          << std::endl;
      return oss.str();
    }
    case -66: {
      oss << "CL_INVALID_COMPILER_OPTIONS:" << std::endl;
      oss << "    * if the compiler options specified by options are invalid."
          << std::endl;
      return oss.str();
    }
    case -67: {
      oss << "CL_INVALID_LINKER_OPTIONS:" << std::endl;
      oss << "    * if the linker options specified by options are invalid."
          << std::endl;
      return oss.str();
    }
    case -68: {
      oss << "CL_INVALID_DEVICE_PARTITION_COUNT:" << std::endl;
      oss << "    * if the partition name specified in properties is\n"
          << "      CL_DEVICE_PARTITION_BY_COUNTS and the number of\n"
          << "      sub-devices requested exceeds\n"
          << "      CL_DEVICE_PARTITION_MAX_SUB_DEVICES or the total number\n"
          << "      of compute units requested exceeds\n"
          << "      CL_DEVICE_PARTITION_MAX_COMPUTE_UNITS for in_device, or\n"
          << "      the number of compute units requested for one or more\n"
          << "      sub-devices is less than zero or the number of\n"
          << "      sub-devices requested exceeds\n"
          << "      CL_DEVICE_PARTITION_MAX_COMPUTE_UNITS for in_device."
          << std::endl;
      return oss.str();
    }
    case -69: {
      oss << "CL_INVALID_PIPE_SIZE:" << std::endl;
      oss << "    * if pipe_packet_size is 0 or the pipe_packet_size exceeds\n"
          << "      CL_DEVICE_PIPE_MAX_PACKET_SIZE value for all devices\n"
          << "      in context or if pipe_max_packets is 0."
          << std::endl;
      return oss.str();
    }
    case -70: {
      oss << "CL_INVALID_DEVICE_QUEUE:" << std::endl;
      oss << "    * when an argument is of type queue_t when it’s not a valid\n"
          << "      device queue object."
          << std::endl;
      return oss.str();
    }
    case -1000: {
      oss << "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR:" << std::endl;
      oss << "    * CL and GL not on the same device (only when using a GPU)."
          << std::endl;
      return oss.str();
    }
    case -1001: {
      oss << "CL_PLATFORM_NOT_FOUND_KHR:" << std::endl;
      oss << "    * No valid ICDs found" << std::endl;
      return oss.str();
    }
    case -1002: {
      oss << "CL_INVALID_D3D10_DEVICE_KHR:" << std::endl;
      oss << "    * if the Direct3D 10 device specified for interoperability\n"
          << "      is not compatible with the devices against which the\n"
          << "      context is to be created."
          << std::endl;
      return oss.str();
    }
    case -1003: {
      oss << "CL_INVALID_D3D10_RESOURCE_KHR:" << std::endl;
      oss << "    * If the resource is not a Direct3D 10 buffer or texture\n"
          << "      object"
          << std::endl;
      return oss.str();
    }
    case -1004: {
      oss << "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR:" << std::endl;
      oss << "    * If a mem_object is already acquired by OpenCL"
          << std::endl;
      return oss.str();
    }
    case -1005: {
      oss << "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR:" << std::endl;
      oss << "    * If a mem_object is not acquired by OpenCL"
          << std::endl;
      return oss.str();
    }
    case -1006: {
      oss << "CL_INVALID_D3D11_DEVICE_KHR:" << std::endl;
      oss << "    * if the Direct3D 11 device specified for interoperability\n"
          << "      is not compatible with the devices against which the\n"
          << "      context is to be created."
          << std::endl;
      return oss.str();
    }
    case -1007: {
      oss << "CL_INVALID_D3D11_RESOURCE_KHR:" << std::endl;
      oss << "    * If the resource is not a Direct3D 11 buffer or texture\n"
          << "      object"
          << std::endl;
      return oss.str();
    }
    case -1008: {
      oss << "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR:" << std::endl;
      oss << "    * If a mem_object is already acquired by OpenCL"
          << std::endl;
      return oss.str();
    }
    case -1009: {
      oss << "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR:" << std::endl;
      oss << "    * If a mem_object is not acquired by OpenCL"
          << std::endl;
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

OpenCLManager* get_instance_opencl_manager(void)
{
  return &OpenCLManager::GetInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_platform(OpenCLManager* p_opencl_manager)
{
  p_opencl_manager->PrintPlatformInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_device(OpenCLManager* p_opencl_manager)
{
  p_opencl_manager->PrintDeviceInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_build_options(OpenCLManager* p_opencl_manager)
{
  p_opencl_manager->PrintBuildOptions();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_context(OpenCLManager* p_opencl_manager)
{
  p_opencl_manager->PrintContextInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_RAM(OpenCLManager* p_opencl_manager)
{
  p_opencl_manager->PrintRAMStatus();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_command_queue(OpenCLManager* p_opencl_manager)
{
  p_opencl_manager->PrintCommandQueueInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_context_index(OpenCLManager* p_opencl_manager,
  uint32_t const context_index)
{
  p_opencl_manager->ContextToActivate(context_index);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_activated_context(OpenCLManager* p_opencl_manager)
{
  p_opencl_manager->PrintActivatedContextInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void clean_opencl_manager(OpenCLManager* p_opencl_manager)
{
  p_opencl_manager->Clean();
}
