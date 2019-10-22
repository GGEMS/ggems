#ifndef GUARD_GGEMS_GLOBAL_OPENCL_MANAGER_HH
#define GUARD_GGEMS_GLOBAL_OPENCL_MANAGER_HH

/*!
  \file opencl_manager.hh

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

#include "GGEMS/global/ggems_export.hh"

/*!
  \class OpenCLManager
  \brief Singleton class storing all information about OpenCL in GGEMS
*/
class GGEMS_EXPORT OpenCLManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    OpenCLManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~OpenCLManager(void);

  public:
    /*!
      \fn static OpenCLManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type OpenCLManager
    */
    static OpenCLManager& GetInstance(void)
    {
      static OpenCLManager instance;
      return instance;
    }

  public:
    /*!
      \fn OpenCLManager(OpenCLManager const& opencl_manager) = delete
      \param opencl_manager - reference on the singleton
      \brief Avoid copy of the singleton by reference
    */
    OpenCLManager(OpenCLManager const& opencl_manager) = delete;

    /*!
      \fn OpenCLManager& operator=(OpenCLManager const& opencl_manager) = delete
      \param opencl_manager - reference on the singleton
      \brief Avoid assignement of the singleton by reference
    */
    OpenCLManager& operator=(OpenCLManager const& opencl_manager) = delete;

    /*!
      \fn OpenCLManager(OpenCLManager const&& opencl_manager) = delete
      \param opencl_manager - rvalue reference on the singleton
      \brief Avoid copy of the singleton by rvalue reference
    */
    OpenCLManager(OpenCLManager const&& opencl_manager) = delete;

    /*!
      \fn OpenCLManager& operator=(OpenCLManager const&& opencl_manager) = delete
      \param opencl_manager - rvalue reference on the singleton
      \brief Avoid copy of the singleton by rvalue reference
    */
    OpenCLManager& operator=(OpenCLManager const&& opencl_manager) = delete;

  public: // Error stream management
    /*!
      \fn void CheckOpenCLError( cl_int const& error ) const
      \param error - error index
      \brief check the OpenCL error
    */
    void CheckOpenCLError(cl_int const& error) const;

  private:
    /*!
      \fn std::string ErrorType(cl_int const& error) const
      \param error - error index from OpenCL library
      \brief get the error description
    */
    std::string ErrorType(cl_int const& error) const;

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

  private: // Informations about device
    /*!
      \fn cl_device_type GetDeviceType(std::size_t const& device_id) const
      \param device_id - index of device
      \brief return the device type
    */
    inline cl_device_type GetDeviceType(std::size_t const& device_id) const
    {
      return p_device_device_type_[device_id];
    }

  private: // Context management
    /*!
      \fn void CreateContextCPUGPU(void)
      \brief Create a context for GPU or CPU
    */
    void CreateContextCPUGPU(void);

    /*!
      \fn cl_ulong GetGlobalMemoryContext(std::size_t const& context_id) const
      \param context_id - index of context
      \brief return the global memory in a specific context
    */
    inline cl_ulong GetGlobalMemoryContext(std::size_t const& context_id) const
    {
      return p_device_global_mem_size_[context_id];
    }

  public:
    /*!
      \fn std::size_t GetGlobalContextID(cl::Context* const p_context) const
      \param p_context - pointer on the context
      \return the global id of the context
    */
    inline std::size_t GetGlobalContextID(cl::Context* const p_context) const
    {
      std::size_t context_id = 0;
      // Loop over the contexts to find the good one
      for (std::size_t i = 0; i < vp_contexts_cl_.size(); ++i) {
        if (p_context == vp_contexts_cl_.at(i)) {
          context_id = i;
          break;
        }
      }
      return context_id;
    }

    /*!
      \fn void ContextToActivate(uint32_t const& context_id)
      \param context_id - context index
      \brief set the index of the context to activate
    */
    void ContextToActivate(uint32_t const& context_id);

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
      \fn std::size_t GetNumberOfActivatedContext() const
      \brief return the number of the activated contexts
    */
    inline std::size_t GetNumberOfActivatedContext(void) const
    {
      return vp_contexts_act_cl_.size();
    }

    /*!
      \fn std::size_t GetContext() const
      \brief return the activated context
    */
    inline cl::Context* GetContext(void) const
    {
      return vp_contexts_act_cl_.front();
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
      return vp_queues_cl_.at(GetGlobalContextID(vp_contexts_act_cl_.at(0)));
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
      return vp_event_cl_.at(GetGlobalContextID(vp_contexts_act_cl_.at(0)));
    }

  public:
    /*!
      \fn cl::Kernel* CompileKernel(std::string const& kernel_filename, std::string const& kernel_name, char* const p_custom_options = nullptr, char* const p_additional_options = nullptr)
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
      \fn T* GetDeviceBufferWrite(cl::Buffer* p_device_ptr) const
      \tparam T - type of the returned pointer on host memory
      \param p_device_ptr - pointer on device memory
      \brief Get the device pointer on host to write on it. ReleaseDeviceBuffer
      must be used after this method!!!
      \return the pointer on host memory on write mode
    */
    template <typename T>
    T* GetDeviceBufferWrite(cl::Buffer* const p_device_ptr) const;

    /*
      \fn T* GetDeviceBufferRead(cl::Buffer* p_device_ptr) const
      \tparam T - type of the returned pointer on host memory
      \param p_device_ptr - pointer on device memory
      \brief Get the device pointer on host to read it. ReleaseDeviceBuffer
      must be used after this method!!!
      \return the pointer on host memory on read mode
    */
    template <typename T>
    T* GetDeviceBufferRead(cl::Buffer* const p_device_ptr) const;

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
      \fn void AddRAMMemory(cl_ulong const size)
      \param size - size of the allocated buffer in byte
      \brief store the size of the allocated buffer
    */
    void AddRAMMemory(cl_ulong const size);

    /*!
      \fn SubRAMMemory(cl_ulong const size)
      \param size - size of the allocated buffer in byte
      \brief substract the size of the allocated buffer
    */
    void SubRAMMemory(cl_ulong const size);

  public:
    /*!
      \fn void DisplayElapsedTimeInKernel(std::string const& kernel_name) const
      \brief Compute and display elapsed time in kernel for an activated context
    */
    void DisplayElapsedTimeInKernel(std::string const& kernel_name) const;

  private: // Platforms
    std::vector<cl::Platform> v_platforms_cl_; /*!< Vector of platforms */
    std::string *p_platform_vendor_; /*!< Vendor of the platform */

  private: // Devices
    std::vector<cl::Device*> vp_devices_cl_; /*!< Vector of pointers of devices */
    cl_device_type *p_device_device_type_; /*!< Type of device */
    std::string *p_device_vendor_; /*!< Vendor of the device */
    std::string *p_device_version_; /*!< Version of the device */
    std::string *p_device_driver_version_; /*!< Driver version of the device */
    cl_uint *p_device_address_bits_; /*!< Address Bits */
    cl_bool *p_device_available_; /*!< Flag on device availability */
    cl_bool *p_device_compiler_available_; /*!< Flag on compiler availability */
    cl_ulong *p_device_global_mem_cache_size_; /*!< Global memory cache size */
    cl_uint *p_device_global_mem_cacheline_size_; /*!< Global memory cache line size */
    cl_ulong *p_device_global_mem_size_; /*!< Global memory size */
    cl_ulong *p_device_local_mem_size_; /*!< Local memory size */
    cl_uint *p_device_mem_base_addr_align_; /*!< Alignment memory */
    std::string *p_device_name_; /*!< Name of the device */
    std::string *p_device_opencl_c_version_; /*!< OpenCL C version */
    size_t *p_device_printf_buffer_size_; /*!< Printf buffer size */
    size_t *p_device_image_max_array_size_; /*!< Image max array size */
    size_t *p_device_image_max_buffer_size_; /*!< Image max buffer size */
    cl_bool *p_device_image_support_; /*!< Flag on image support */
    size_t *p_device_image_2d_max_w_; /*!< Image2D max width */
    size_t *p_device_image_2d_max_h_; /*!< Image2D max height */
    size_t *p_device_image_3d_max_w_; /*!< Image3D max width */
    size_t *p_device_image_3d_max_h_; /*!< Image3D max height */
    size_t *p_device_image_3d_max_d_; /*!< Image3D max depth */
    cl_uint *p_device_max_clock_frequency_; /*!< Max clock frequency */
    cl_uint *p_device_max_compute_units_; /*!< Max compute units */
    cl_ulong *p_device_constant_buffer_size_; /*!< Constant buffer size */
    cl_ulong *p_device_mem_alloc_size_; /*!< Memory allocation size */
    cl_uint *p_device_max_read_image_args_; /*!< Max read image arguments */
    cl_uint *p_device_max_write_image_args_; /*!< Max write image arguments */
    size_t *p_device_parameter_size_; /*!< Parameter size */
    cl_uint *p_device_samplers_; /*!< Number of maximum samplers */
    cl_uint *p_device_workitem_dimensions_; /*!< Work item dimensions */
    size_t *p_device_workgroup_size_; /*!< Work group size */
    std::vector<size_t> *p_device_workitem_size_; /*!< Work item size */
    cl_platform_id *p_device_platform_id_; /*!< Platform ID */
    cl_uint *p_device_native_vector_width_char_; /*!< Native size of the char */
    cl_uint *p_device_preferred_vector_width_char_; /*!< Preferred size of the char */
    cl_uint *p_device_native_vector_width_int_; /*!< Native size of the int */
    cl_uint *p_device_preferred_vector_width_int_; /*!< Preferred size of the int */
    cl_uint *p_device_native_vector_width_long_; /*!< Native size of the long */
    cl_uint *p_device_preferred_vector_width_long_; /*!< Preferred size of the long */
    cl_uint *p_device_native_vector_width_short_; /*!< Native size of the short */
    cl_uint *p_device_preferred_vector_width_short_; /*!< Preferred size of the short */
    cl_uint *p_device_native_vector_width_half_; /*!< Native size of the half */
    cl_uint *p_device_preferred_vector_width_half_; /*!< Preferred size of the half */
    cl_uint *p_device_native_vector_width_float_; /*!< Native size of the float */
    cl_uint *p_device_preferred_vector_width_float_; /*!< Preferred size of the float */
    cl_uint *p_device_native_vector_width_double_; /*!< Native size of the double */
    cl_uint *p_device_preferred_vector_width_double_; /*!< Preferred size of the double */

  private: // OpenCL compilation options
    std::string build_options_; /*!< list of option to OpenCL compiler */

  private: // Context and informations about them
    std::vector<cl::Context*> vp_contexts_cl_; /*!< Vector of context */
    std::vector<cl::Context*> vp_contexts_cpu_cl_; /*!< Vector of CPU context */
    std::vector<cl::Context*> vp_contexts_gpu_cl_; /*!< Vector of GPU context */
    std::vector<cl::Context*> vp_contexts_act_cl_; /*!< List of activated context */

  private: // Command queue informations
    std::vector<cl::CommandQueue*> vp_queues_cl_; /*!< Command queue for all the context */

  private: // OpenCL event
    std::vector<cl::Event*> vp_event_cl_; /*!< List of pointer to OpenCL event, for profiling */

  private: // Kernels
    std::vector<cl::Kernel*> vp_kernel_cl_; /*!< List of pointer to OpenCL kernel */

  private: // RAM handler
    cl_ulong *p_used_ram_; /*!< Memory RAM used by context */
};

template <typename T>
T* OpenCLManager::GetDeviceBufferWrite(cl::Buffer* const p_device_ptr) const
{
  return static_cast<T*>(GetCommandQueue()->enqueueMapBuffer(*p_device_ptr,
    CL_TRUE, CL_MAP_WRITE, 0, sizeof(T), nullptr, nullptr, nullptr));
}

template <typename T>
T* OpenCLManager::GetDeviceBufferRead(cl::Buffer* const p_device_ptr) const
{
  return static_cast<T*>(GetCommandQueue()->enqueueMapBuffer(*p_device_ptr,
    CL_TRUE, CL_MAP_READ, 0, sizeof(T), nullptr, nullptr, nullptr));
}

template <typename T>
void OpenCLManager::ReleaseDeviceBuffer(cl::Buffer* const p_device_ptr,
  T* p_host_ptr) const
{
  // Unmap the memory
  GetCommandQueue()->enqueueUnmapMemObject(*p_device_ptr, p_host_ptr);
  GetCommandQueue()->finish();
}

/*!
  \fn OpenCLManager* get_instance_opencl_manager(void)
  \brief Get the OpenCLManager pointer for python user.
*/
extern "C" GGEMS_EXPORT OpenCLManager* get_instance_opencl_manager(void);

/*!
  \fn void print_platform(OpenCLManager* opencl_manager)
  \param opencl_manager - pointer on the singleton
  \brief Print information about OpenCL platform
*/
extern "C" GGEMS_EXPORT void print_platform(OpenCLManager* opencl_manager);

/*!
  \fn void print_device(OpenCLManager* opencl_manager)
  \param opencl_manager - pointer on the singleton
  \brief Print information about OpenCL device
*/
extern "C" GGEMS_EXPORT void print_device(OpenCLManager* opencl_manager);

/*!
  \fn void print_build_options(OpenCLManager* opencl_manager)
  \param opencl_manager - pointer on the singleton
  \brief Print information about OpenCL compilation option
*/
extern "C" GGEMS_EXPORT void print_build_options(OpenCLManager* opencl_manager);

/*!
  \fn void print_context(OpenCLManager* opencl_manager)
  \param opencl_manager - pointer on the singleton
  \brief Print information about OpenCL context
*/
extern "C" GGEMS_EXPORT void print_context(OpenCLManager* opencl_manager);

/*!
  \fn void print_RAM(OpenCLManager* opencl_manager)
  \param opencl_manager - pointer on the singleton
  \brief Print information about RAM in OpenCL context
*/
extern "C" GGEMS_EXPORT void print_RAM(OpenCLManager* opencl_manager);

/*!
  \fn void print_command_queue(OpenCLManager* opencl_manager)
  \param opencl_manager - pointer on the singleton
  \brief Print information about command in OpenCL for each context
*/
extern "C" GGEMS_EXPORT void print_command_queue(OpenCLManager* opencl_manager);

/*!
  \fn void set_context_index(OpenCLManager* opencl_manager, uint32_t const context_id)
  \param opencl_manager - pointer on the singleton
  \param context_id - index of the context
  \brief Set the context index to activate
*/
extern "C" GGEMS_EXPORT void set_context_index(OpenCLManager* opencl_manager,
  uint32_t const context_id);

/*!
  \fn void print_activated_context(OpenCLManager* opencl_manager)
  \param opencl_manager - pointer on the singleton
  \brief Print information about activated context
*/
extern "C" GGEMS_EXPORT void print_activated_context(
  OpenCLManager* opencl_manager);

#endif // GUARD_GGEMS_GLOBAL_OPENCL_MANAGER_HH
