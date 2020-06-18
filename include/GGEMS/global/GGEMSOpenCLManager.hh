#ifndef GUARD_GGEMS_GLOBAL_GGEMSOPENCLMANAGER_HH
#define GUARD_GGEMS_GLOBAL_GGEMSOPENCLMANAGER_HH

/*!
  \file GGEMSOpenCLManager.hh

  \brief Singleton class storing all informations about OpenCL and managing GPU/CPU contexts and kernels for GGEMS
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

#include <memory>

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "GGEMS/tools/GGEMSPrint.hh"

/*!
  \class GGEMSOpenCLManager
  \brief Singleton class storing all information about OpenCL in GGEMS
*/
class GGEMS_EXPORT GGEMSOpenCLManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGEMSOpenCLManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGEMSOpenCLManager(void);

  public:
    /*!
      \fn static GGEMSOpenCLManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSOpenCLManager
    */
    static GGEMSOpenCLManager& GetInstance(void)
    {
      static GGEMSOpenCLManager instance;
      return instance;
    }

    /*!
      \fn GGEMSOpenCLManager(GGEMSOpenCLManager const& opencl_manager) = delete
      \param opencl_manager - reference on the singleton
      \brief Avoid copy of the singleton by reference
    */
    GGEMSOpenCLManager(GGEMSOpenCLManager const& opencl_manager) = delete;

    /*!
      \fn GGEMSOpenCLManager& operator=(GGEMSOpenCLManager const& opencl_manager) = delete
      \param opencl_manager - reference on the singleton
      \brief Avoid assignement of the singleton by reference
    */
    GGEMSOpenCLManager& operator=(GGEMSOpenCLManager const& opencl_manager) = delete;

    /*!
      \fn GGEMSOpenCLManager(GGEMSOpenCLManager const&& opencl_manager) = delete
      \param opencl_manager - rvalue reference on the singleton
      \brief Avoid copy of the singleton by rvalue reference
    */
    GGEMSOpenCLManager(GGEMSOpenCLManager const&& opencl_manager) = delete;

    /*!
      \fn GGEMSOpenCLManager& operator=(GGEMSOpenCLManager const&& opencl_manager) = delete
      \param opencl_manager - rvalue reference on the singleton
      \brief Avoid copy of the singleton by rvalue reference
    */
    GGEMSOpenCLManager& operator=(GGEMSOpenCLManager const&& opencl_manager) = delete;

    /*!
      \fn void Clean(void)
      \brief Clean correctly OpenCL platform, device, context, command queue, event and kernel
    */
    void Clean(void);

    /*!
      \fn GGbool IsReady(void) const
      \return true if OpenCL manager is ready, otherwize false
      \brief Checking if the OpenCL manager is ready, it means if a context is set
    */
    inline GGbool IsReady(void) const
    {
      if (context_act_cl_) return true;
      else return false;
    }

    /*!
      \fn void PrintPlatformInfos(void) const
      \brief print all the informations about the platform
    */
    void PrintPlatformInfos(void) const;

    /*!
      \fn void PrintDeviceInfos(void) const
      \brief print all informations about devices
    */
    void PrintDeviceInfos(void) const;

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
      \fn void PrintCommandQueueInfos(void) const
      \brief print the informations about the command queue
    */
    void PrintCommandQueueInfos(void) const;

    /*!
      \fn void PrintBuildOptions(void) const
      \brief print global build options used during kernel compilation
    */
    void PrintBuildOptions(void) const;

    /*!
      \fn void ContextToActivate(GGuint const& context_id)
      \param context_id - context index
      \brief set the index of the context to activate
    */
    void ContextToActivate(GGuint const& context_id);

    /*!
      \fn GGulong GetMaxRAMMemoryOnActivatedContext(void) const
      \return Max RAM memory on a context
      \brief Get the maximum RAM memory on activated OpenCL context
    */
    inline GGulong GetMaxRAMMemoryOnActivatedContext(void) const {return device_global_mem_size_[context_index_];}

    /*!
      \fn std::string GetNameOfActivatedContext(void) const
      \return name of activated context
      \brief Get the name of the activated context
    */
    inline std::string GetNameOfActivatedContext(void) const {return device_name_[context_index_];}

    /*!
      \fn cl::Context* GetContext(void) const
      \return the pointer on activated context
      \brief return the activated context
    */
    inline cl::Context* GetContext(void) const {return context_act_cl_.get();}

    /*!
      \fn cl::CommandQueue* GetCommandQueue(void) const
      \return the pointer on activated command queue
      \brief Return the command queue to activated context
    */
    inline cl::CommandQueue* GetCommandQueue(void) const {return queue_act_cl_.get();}

    /*!
      \fn cl::Event* GetEvent(void) const
      \return the pointer on activated event
      \brief return an event to activated context
    */
    inline cl::Event* GetEvent(void) const {return event_act_cl_.get();}

    /*!
      \fn std::weak_ptr<cl::Kernel> CompileKernel(std::string const& kernel_filename, std::string const& kernel_name, char* const custom_options = nullptr, char* const additional_options = nullptr)
      \param kernel_filename - filename where is declared the kernel
      \param kernel_name - name of the kernel
      \param custom_options - new compilation option for the kernel
      \param additional_options - additionnal compilation option
      \brief Compile the OpenCL kernel on the activated context
      \return the pointer on the OpenCL kernel
    */
    std::weak_ptr<cl::Kernel> CompileKernel(std::string const& kernel_filename, std::string const& kernel_name, char* const custom_options = nullptr, char* const additional_options = nullptr);

    /*!
      \fn std::unique_ptr<cl::Buffer> Allocate(void* host_ptr, std::size_t size, cl_mem_flags flags)
      \param host_ptr - pointer to buffer in host memory
      \param size - size of the buffer in bytes
      \param flags - mode to open the buffer
      \brief Allocation of OpenCL memory
      \return an unique pointer to an OpenCL buffer
    */
    std::unique_ptr<cl::Buffer> Allocate(void* host_ptr, std::size_t size, cl_mem_flags flags);

    /*!
      \return the pointer on host memory on write/read mode
      \brief Get the device pointer on host to write on it. ReleaseDeviceBuffer must be used after this method!!!
      \param device_ptr - pointer on device memory
      \param size - size of region to map
      \tparam T - type of the returned pointer on host memory
    */
    template <typename T>
    T* GetDeviceBuffer(cl::Buffer* device_ptr, std::size_t const size) const;

    /*!
      \brief Get the device pointer on host to write on it. Mandatory after a GetDeviceBufferWrite ou GetDeviceBufferRead!!!
      \param device_ptr - pointer on device memory
      \param host_ptr - pointer on host memory mapped on device memory
      \tparam T - type of host memory pointer to release
    */
    template <typename T>
    void ReleaseDeviceBuffer(cl::Buffer* const device_ptr, T* host_ptr) const;

    /*!
      \fn void DisplayElapsedTimeInKernel(std::string const& kernel_name) const
      \param kernel_name - name of the kernel for time displaying
      \brief Compute and display elapsed time in kernel for an activated context
    */
    void DisplayElapsedTimeInKernel(std::string const& kernel_name) const;

    /*!
      \fn void CheckOpenCLError(GGint const& error, std::string const& class_name, std::string const& method_name) const
      \param error - error index
      \param class_name - name of the class
      \param method_name - name of the method
      \brief check the OpenCL error
    */
    void CheckOpenCLError(GGint const& error, std::string const& class_name, std::string const& method_name) const;

  private:
    /*!
      \fn void CreateContext(void)
      \brief Create a context for GPU or CPU
    */
    void CreateContext(void);

    /*!
      \fn void CreateCommandQueue(void)
      \brief create a command queue for each context
    */
    void CreateCommandQueue(void);

    /*!
      \fn void CreateEvent(void)
      \brief creating an event for each context
    */
    void CreateEvent(void);

    /*!
      \fn std::string ErrorType(GGint const& error) const
      \param error - error index from OpenCL library
      \return the message error
      \brief get the error description
    */
    std::string ErrorType(GGint const& error) const;

  private:
    // Platforms
    std::vector<cl::Platform> platforms_cl_; /*!< Vector of platforms */
    std::vector<std::string> platform_vendor_; /*!< Vendor of the platform */

    // Devices
    std::vector<std::unique_ptr<cl::Device>> devices_cl_; /*!< Vector of devices */
    std::vector<cl_device_type> device_device_type_; /*!< Type of device */
    std::vector<std::string> device_vendor_; /*!< Vendor of the device */
    std::vector<std::string> device_version_; /*!< Version of the device */
    std::vector<std::string> device_driver_version_; /*!< Driver version of the device */
    std::vector<GGuint> device_address_bits_; /*!< Address Bits */
    std::vector<GGbool> device_available_; /*!< Flag on device availability */
    std::vector<GGbool> device_compiler_available_; /*!< Flag on compiler availability */
    std::vector<GGulong> device_global_mem_cache_size_; /*!< Global memory cache size */
    std::vector<GGuint> device_global_mem_cacheline_size_; /*!< Global memory cache line size */
    std::vector<GGulong> device_global_mem_size_; /*!< Global memory size */
    std::vector<GGulong> device_local_mem_size_; /*!< Local memory size */
    std::vector<GGuint> device_mem_base_addr_align_; /*!< Alignment memory */
    std::vector<std::string> device_name_; /*!< Name of the device */
    std::vector<std::string> device_opencl_c_version_; /*!< OpenCL C version */
    std::vector<GGuint> device_max_clock_frequency_; /*!< Max frequency of device */
    std::vector<GGuint> device_max_compute_units_; /*!< Max compute units */
    std::vector<GGulong> device_constant_buffer_size_; /*!< Constant buffer size */
    std::vector<GGulong> device_mem_alloc_size_; /*!< Memory allocation size */
    std::vector<GGuint> device_native_vector_width_double_; /*!< Native size of double */

    // OpenCL compilation options
    std::string build_options_; /*!< list of option to OpenCL compiler */

    // Context and informations about them
    GGuint context_index_; /*!< Index of the activated context */
    std::vector<std::shared_ptr<cl::Context>> contexts_cl_; /*!< Vector of context */
    std::vector<std::shared_ptr<cl::Context>> contexts_cpu_cl_; /*!< Vector of CPU context */
    std::vector<std::shared_ptr<cl::Context>> contexts_gpu_cl_; /*!< Vector of GPU context */
    std::shared_ptr<cl::Context> context_act_cl_; /*!< Activated context */

    // Command queue informations
    std::vector<std::shared_ptr<cl::CommandQueue>> queues_cl_; /*!< Command queue for all the context */
    std::shared_ptr<cl::CommandQueue> queue_act_cl_; /*!< Activated command queue */

    // OpenCL event
    std::vector<std::shared_ptr<cl::Event>> events_cl_; /*!< List of pointer to OpenCL event, for profiling */
    std::shared_ptr<cl::Event> event_act_cl_; /*!< Activated event */

    // Kernels
    std::vector<std::shared_ptr<cl::Kernel>> kernels_cl_; /*!< List of pointer to OpenCL kernel */
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
T* GGEMSOpenCLManager::GetDeviceBuffer(cl::Buffer* const device_ptr, std::size_t const size) const
{
  GGcout("GGEMSOpenCLManager", "GetDeviceBuffer", 3) << "Getting mapped memory buffer on OpenCL device..." << GGendl;

  GGint err = 0;
  T* ptr = static_cast<T*>(queue_act_cl_.get()->enqueueMapBuffer(*device_ptr, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, size, nullptr, nullptr, &err));
  CheckOpenCLError(err, "GGEMSOpenCLManager", "GetDeviceBuffer");
  return ptr;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
void GGEMSOpenCLManager::ReleaseDeviceBuffer(cl::Buffer* const device_ptr, T* host_ptr) const
{
  GGcout("GGEMSOpenCLManager", "ReleaseDeviceBuffer", 3) << "Releasing mapped memory buffer on OpenCL device..." << GGendl;

  // Unmap the memory
  CheckOpenCLError(queue_act_cl_.get()->enqueueUnmapMemObject(*device_ptr, host_ptr), "GGEMSOpenCLManager", "ReleaseDeviceBuffer");
}

/*!
  \fn GGEMSOpenCLManager* get_instance_ggems_opencl_manager(void)
  \return the pointer on the singleton
  \brief Get the GGEMSOpenCLManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSOpenCLManager* get_instance_ggems_opencl_manager(void);

/*!
  \fn void print_infos_opencl_manager(GGEMSOpenCLManager* opencl_manager)
  \param opencl_manager - pointer on the singleton
  \brief Print information about OpenCL
*/
extern "C" GGEMS_EXPORT void print_infos_opencl_manager(GGEMSOpenCLManager* opencl_manager);

/*!
  \fn void set_context_index_ggems_opencl_manager(GGEMSOpenCLManager* opencl_manager, GGuint const context_id)
  \param opencl_manager - pointer on the singleton
  \param context_id - index of the context
  \brief Set the context index to activate
*/
extern "C" GGEMS_EXPORT void set_context_index_ggems_opencl_manager(GGEMSOpenCLManager* opencl_manager, GGuint const context_id);

/*!
  \fn void clean_opencl_manager(GGEMSOpenCLManager* opencl_manager)
  \param opencl_manager - pointer on the singleton
  \brief Clean OpenCL manager safely with python
*/
extern "C" GGEMS_EXPORT void clean_opencl_manager(GGEMSOpenCLManager* opencl_manager);

#endif // GUARD_GGEMS_GLOBAL_GGEMSOPENCLMANAGER_HH
