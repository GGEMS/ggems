#ifndef GUARD_GGEMS_GLOBAL_GGOPENCLMANAGER_HH
#define GUARD_GGEMS_GLOBAL_GGOPENCLMANAGER_HH

/*!
  \file GGOpenCLManager.hh

  \brief Singleton class storing all informations about OpenCL and managing
  GPU/CPU contexts and kernels for GGEMS
  IMPORTANT: Only 1 context has to be activated.

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday September 23, 2019
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include <iostream>
#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "GGEMS/global/GGExport.hh"
#include "GGEMS/tools/GGPrint.hh"

/*!
  \class GGOpenCLManager
  \brief Singleton class storing all information about OpenCL in GGEMS
*/
class GGEMS_EXPORT GGOpenCLManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGOpenCLManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGOpenCLManager(void);

  public:
    /*!
      \fn static GGOpenCLManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGOpenCLManager
    */
    static GGOpenCLManager& GetInstance(void)
    {
      static GGOpenCLManager instance;
      return instance;
    }

  public:
    /*!
      \fn GGOpenCLManager(GGOpenCLManager const& opencl_manager) = delete
      \param opencl_manager - reference on the singleton
      \brief Avoid copy of the singleton by reference
    */
    GGOpenCLManager(GGOpenCLManager const& opencl_manager) = delete;

    /*!
      \fn GGOpenCLManager& operator=(GGOpenCLManager const& opencl_manager) = delete
      \param opencl_manager - reference on the singleton
      \brief Avoid assignement of the singleton by reference
    */
    GGOpenCLManager& operator=(GGOpenCLManager const& opencl_manager) = delete;

    /*!
      \fn OpenCLManager(OpenCLManager const&& opencl_manager) = delete
      \param opencl_manager - rvalue reference on the singleton
      \brief Avoid copy of the singleton by rvalue reference
    */
    GGOpenCLManager(GGOpenCLManager const&& opencl_manager) = delete;

    /*!
      \fn GGOpenCLManager& operator=(GGOpenCLManager const&& opencl_manager) = delete
      \param opencl_manager - rvalue reference on the singleton
      \brief Avoid copy of the singleton by rvalue reference
    */
    GGOpenCLManager& operator=(GGOpenCLManager const&& opencl_manager) = delete;

  public: // Clean memory of correctly
    /*!
      \fn void Clean(void)
      \brief Clean correctly OpenCL platform, device, context, command queue, event and kernel
    */
    void Clean(void);

  public: // Error stream management
    /*!
      \fn void CheckOpenCLError(GGint const& error, std::string const& class_name, std::string const& method_name) const
      \param error - error index
      \param class_name - name of the class
      \param method_name - name of the method
      \brief check the OpenCL error
    */
    void CheckOpenCLError(GGint const& error, std::string const& class_name,
      std::string const& method_name) const;

  private:
    /*!
      \fn std::string ErrorType(GGint const& error) const
      \param error - error index from OpenCL library
      \brief get the error description
    */
    std::string ErrorType(GGint const& error) const;

  public: // OpenCL infos
    /*!
      \fn void PrintPlatformInfos() const
      \brief print all the informations about the platform
    */
    void PrintPlatformInfos(void) const;

    /*!
      \fn void PrintDeviceInfos(void) const
      \brief print all informations about devices
    */
   void PrintDeviceInfos(void) const;

    /*!
      \fn void PrintBuildOptions(void) const
      \brief print global build options used during kernel compilation
    */
   void PrintBuildOptions(void) const;

  private: // Context management
    /*!
      \fn void CreateContext(void)
      \brief Create a context for GPU or CPU
    */
    void CreateContext(void);

  public:
    /*!
      \fn void ContextToActivate(GGuint const& context_id)
      \param context_id - context index
      \brief set the index of the context to activate
    */
    void ContextToActivate(GGuint const& context_id);

  public:
    /*!
      \fn void PrintContextInfos(void) const
      \brief print infos about each context
    */
    void PrintContextInfos(void) const;

    /*!
      \fn void PrintActivatedContextInfos(void) const
      \brief print infos about each activated context
    */
    void PrintActivatedContextInfos(void) const;

    /*!
      \fn std::size_t GetContext() const
      \brief return the activated context
    */
    inline cl::Context* GetContext(void) const
    {
      return vp_contexts_act_cl_.at(0);
    };

  private: // OpenCL command queue
    /*!
      \fn void CreateCommandQueue(void)
      \brief create a command queue for each context
    */
    void CreateCommandQueue(void);

  public:
    /*!
      \fn void PrintCommandQueueInfos(void) const
      \brief print the informations about the command queue
    */
    void PrintCommandQueueInfos(void) const;

    /*!
      \fn cl::CommandQueue* GetCommandQueue(void) const
      \brief Return the command queue to activated context
    */
    inline cl::CommandQueue* GetCommandQueue(void) const
    {
      return vp_queues_act_cl_.at(0);
    }

  private: // OpenCL event
    /*!
      \fn void CreateEvent()
      \brief creating an event for each context
    */
    void CreateEvent();

  public:
    /*!
      \fn cl::Event* GetEvent(void) const
      \brief return an event to activated context
    */
    inline cl::Event* GetEvent(void) const
    {
      return vp_event_act_cl_.at(0);
    }

  public:
    /*!
      \fn cl::Kernel* CompileKernel(std::string const& kernel_filename, std::string const& kernel_name, GGchar* const p_custom_options = nullptr, GGchar* const p_additional_options = nullptr)
      \param kernel_filename - filename where is declared the kernel
      \param kernel_name - name of the kernel
      \param p_custom_options - new compilation option for the kernel
      \param p_additionnal_options - additionnal compilation option
      \brief Compile the OpenCL kernel on the activated context
      \return the pointer on the OpenCL kernel
    */
    cl::Kernel* CompileKernel(std::string const& kernel_filename,
      std::string const& kernel_name, char* const p_custom_options = nullptr,
      char* const p_additional_options = nullptr);

  public: // OpenCL buffer handling
    /*!
      \fn cl::Buffer* Allocate(void* p_host_ptr, std::size_t size, cl_mem_flags flags)
      \param p_host_ptr - pointer to buffer in host memory
      \param size - size of the buffer in bytes
      \param flags - mode to open the buffer
      \brief Allocation of OpenCL memory
      \return a pointer to an OpenCL buffer
    */
    cl::Buffer* Allocate(void* p_host_ptr, std::size_t size,
      cl_mem_flags flags);

    /*
      \fn void Deallocate(cl::Buffer* p_buffer, std::size_t size)
      \param p_buffer - pointer to a buffer
      \param size - size of the buffer in bytes
      \brief Deallocation of OpenCL memory
    */
    void Deallocate(cl::Buffer* p_buffer, std::size_t size);

    /*
      \fn T* GetDeviceBuffer(cl::Buffer* p_device_ptr, std::size_t const size) const
      \tparam T - type of the returned pointer on host memory
      \param p_device_ptr - pointer on device memory
      \param size - size of region to map
      \brief Get the device pointer on host to write on it. ReleaseDeviceBuffer must be used after this method!!!
      \return the pointer on host memory on write/read mode
    */
    template <typename T>
    T* GetDeviceBuffer(cl::Buffer* const p_device_ptr,
      std::size_t const size) const;

    /*
      \fn void ReleaseDeviceBuffer(cl::Buffer* const p_device_ptr, T* p_host_ptr) const
      \tparam T - type of host memory pointer to release
      \param p_device_ptr - pointer on device memory
      \param p_host_ptr - pointer on host memory mapped on device memory
      \brief Get the device pointer on host to write on it. Mandatory after a
      GetDeviceBufferWrite ou GetDeviceBufferRead!!!
      \return the pointer on host memory on write mode
    */
    template <typename T>
    void ReleaseDeviceBuffer(cl::Buffer* const p_device_ptr,
      T* p_host_ptr) const;

  private: // RAM manager
    /*!
      \fn void InitializeRAMManager()
      \brief initialization of the RAM manager by context
    */
    void InitializeRAMManager(void);

  public:
    /*!
      \fn void PrintRAMStatus() const
      \brief print the RAM memory status for each context
    */
    void PrintRAMStatus(void) const;

    /*!
      \fn void AddRAMMemory(GGulong const& size)
      \param size - size of the allocated buffer in byte
      \brief store the size of the allocated buffer
    */
    void AddRAMMemory(GGulong const& size);

    /*!
      \fn SubRAMMemory(GGulong const& size)
      \param size - size of the allocated buffer in byte
      \brief substract the size of the allocated buffer
    */
    void SubRAMMemory(GGulong const& size);

    /*!
      \fn GGulong GetMaxRAMMemoryOnActivatedDevice(void) const
      \brief Get the maximum RAM memory on activated OpenCL device
    */
    inline GGulong GetMaxRAMMemoryOnActivatedDevice(void) const
    {
      return p_device_global_mem_size_[context_index_];
    }

  public:
    /*!
      \fn void DisplayElapsedTimeInKernel(std::string const& kernel_name) const
      \brief Compute and display elapsed time in kernel for an activated context
    */
    void DisplayElapsedTimeInKernel(std::string const& kernel_name) const;

  private:
    /*!
      \fn void CheckRAMMemory(std::size_t const& size)
      \param size - size in bytes to allocate
      \brief Checking RAM memory allocation
    */
    void CheckRAMMemory(std::size_t const& size);

  private: // Platforms
    std::vector<cl::Platform> v_platforms_cl_; /*!< Vector of platforms */
    std::string *p_platform_vendor_; /*!< Vendor of the platform */

  private: // Devices
    std::vector<cl::Device*> vp_devices_cl_; /*!< Vector of pointers of devices */
    cl_device_type *p_device_device_type_; /*!< Type of device */
    std::string *p_device_vendor_; /*!< Vendor of the device */
    std::string *p_device_version_; /*!< Version of the device */
    std::string *p_device_driver_version_; /*!< Driver version of the device */
    GGuint *p_device_address_bits_; /*!< Address Bits */
    GGbool *p_device_available_; /*!< Flag on device availability */
    GGbool *p_device_compiler_available_; /*!< Flag on compiler availability */
    GGulong *p_device_global_mem_cache_size_; /*!< Global memory cache size */
    GGuint *p_device_global_mem_cacheline_size_; /*!< Global memory cache line size */
    GGulong *p_device_global_mem_size_; /*!< Global memory size */
    GGulong *p_device_local_mem_size_; /*!< Local memory size */
    GGuint *p_device_mem_base_addr_align_; /*!< Alignment memory */
    std::string *p_device_name_; /*!< Name of the device */
    std::string *p_device_opencl_c_version_; /*!< OpenCL C version */
    GGuint *p_device_max_clock_frequency_; /*!< Max clock frequency */
    GGuint *p_device_max_compute_units_; /*!< Max compute units */
    GGulong *p_device_constant_buffer_size_; /*!< Constant buffer size */
    GGulong *p_device_mem_alloc_size_; /*!< Memory allocation size */
    GGuint *p_device_native_vector_width_double_; /*!< Native size of the double */
    GGuint *p_device_preferred_vector_width_double_; /*!< Preferred size of the double */

  private: // OpenCL compilation options
    std::string build_options_; /*!< list of option to OpenCL compiler */

  private: // Context and informations about them
    GGuint context_index_; /*!< Index of the activated context */
    std::vector<cl::Context*> vp_contexts_cl_; /*!< Vector of context */
    std::vector<cl::Context*> vp_contexts_cpu_cl_; /*!< Vector of CPU context */
    std::vector<cl::Context*> vp_contexts_gpu_cl_; /*!< Vector of GPU context */
    std::vector<cl::Context*> vp_contexts_act_cl_; /*!< Activated context */

  private: // Command queue informations
    std::vector<cl::CommandQueue*> vp_queues_cl_; /*!< Command queue for all the context */
    std::vector<cl::CommandQueue*> vp_queues_act_cl_; /*!< Activated command queue */

  private: // OpenCL event
    std::vector<cl::Event*> vp_event_cl_; /*!< List of pointer to OpenCL event, for profiling */
    std::vector<cl::Event*> vp_event_act_cl_; /*!< Activated event */

  private: // Kernels
    std::vector<cl::Kernel*> vp_kernel_cl_; /*!< List of pointer to OpenCL kernel */

  private: // RAM handler
    GGulong *p_used_ram_; /*!< Memory RAM used by context */
};

template <typename T>
T* GGOpenCLManager::GetDeviceBuffer(cl::Buffer* const p_device_ptr,
  std::size_t const size) const
{
  GGEMScout("OpenCLManager", "GetDeviceBuffer", 3)
    << "Getting mapped memory buffer on OpenCL device..." << GGEMSendl;

  GGint err = 0;
  T* ptr = static_cast<T*>(vp_queues_act_cl_.at(0)->enqueueMapBuffer(
    *p_device_ptr, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, size, nullptr,
    nullptr, &err));
  CheckOpenCLError(err, "OpenCLManager", "GetDeviceBuffer");
  return ptr;
}

template <typename T>
void GGOpenCLManager::ReleaseDeviceBuffer(cl::Buffer* const p_device_ptr,
  T* p_host_ptr) const
{
  GGEMScout("OpenCLManager", "ReleaseDeviceBuffer", 3)
    << "Releasing mapped memory buffer on OpenCL device..." << GGEMSendl;

  // Unmap the memory
  CheckOpenCLError(vp_queues_act_cl_.at(0)->enqueueUnmapMemObject(*p_device_ptr,
    p_host_ptr), "OpenCLManager", "ReleaseDeviceBuffer");
}

/*!
  \fn GGOpenCLManager* get_instance_opencl_manager(void)
  \brief Get the OpenCLManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGOpenCLManager* get_instance_opencl_manager(void);

/*!
  \fn void print_platform(OpenCLManager* p_opencl_manager)
  \param p_opencl_manager - pointer on the singleton
  \brief Print information about OpenCL platform
*/
extern "C" GGEMS_EXPORT void print_platform(GGOpenCLManager* p_opencl_manager);

/*!
  \fn void print_device(OpenCLManager* p_opencl_manager)
  \param p_opencl_manager - pointer on the singleton
  \brief Print information about OpenCL device
*/
extern "C" GGEMS_EXPORT void print_device(GGOpenCLManager* p_opencl_manager);

/*!
  \fn void print_build_options(OpenCLManager* p_opencl_manager)
  \param p_opencl_manager - pointer on the singleton
  \brief Print information about OpenCL compilation option
*/
extern "C" GGEMS_EXPORT void print_build_options(
  GGOpenCLManager* p_opencl_manager);

/*!
  \fn void print_context(OpenCLManager* p_opencl_manager)
  \param p_opencl_manager - pointer on the singleton
  \brief Print information about OpenCL context
*/
extern "C" GGEMS_EXPORT void print_context(GGOpenCLManager* p_opencl_manager);

/*!
  \fn void print_RAM(OpenCLManager* p_opencl_manager)
  \param p_opencl_manager - pointer on the singleton
  \brief Print information about RAM in OpenCL context
*/
extern "C" GGEMS_EXPORT void print_RAM(GGOpenCLManager* p_opencl_manager);

/*!
  \fn void print_command_queue(OpenCLManager* p_opencl_manager)
  \param p_opencl_manager - pointer on the singleton
  \brief Print information about command in OpenCL for each context
*/
extern "C" GGEMS_EXPORT void print_command_queue(
  GGOpenCLManager* p_opencl_manager);

/*!
  \fn void set_context_index(OpenCLManager* p_opencl_manager, uint32_t const context_id)
  \param p_opencl_manager - pointer on the singleton
  \param context_id - index of the context
  \brief Set the context index to activate
*/
extern "C" GGEMS_EXPORT void set_context_index(
  GGOpenCLManager* p_opencl_manager, GGuint const context_id);

/*!
  \fn void print_activated_context(OpenCLManager* p_opencl_manager)
  \param p_opencl_manager - pointer on the singleton
  \brief Print information about activated context
*/
extern "C" GGEMS_EXPORT void print_activated_context(
  GGOpenCLManager* p_opencl_manager);

/*!
  \fn void clean_opencl_manager(OpenCLManager* p_opencl_manager)
  \param p_opencl_manager - pointer on the singleton
  \brief Clean correctly OpenCL platform, device, context, command queue, event and kernel
*/
extern "C" GGEMS_EXPORT void clean_opencl_manager(
  GGOpenCLManager* p_opencl_manager);

#endif // GUARD_GGEMS_GLOBAL_OPENCL_MANAGER_HH
