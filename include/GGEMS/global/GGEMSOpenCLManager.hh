#ifndef GUARD_GGEMS_GLOBAL_GGEMSOPENCLMANAGER_HH
#define GUARD_GGEMS_GLOBAL_GGEMSOPENCLMANAGER_HH

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
  \file GGEMSOpenCLManager.hh

  \brief Singleton class storing all informations about OpenCL and managing GPU/CPU contexts and kernels, and 1 context is associated to 1 device

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

#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/tools/GGEMSChrono.hh"

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
      \fn GGbool IsReady(void) const
      \return true if OpenCL manager is ready, otherwize false
      \brief Checking if the OpenCL manager is ready, it means if a context is set
    */
    inline GGbool IsReady(void) const
    {
      if (is_context_activated_) return true;
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
      \fn void ContextToActivate(GGint const& context_id)
      \param context_id - context index
      \brief set the index of the context to activate
    */
    void ContextToActivate(GGint const& context_id);

    /*!
      \fn GGulong GetMaxRAMMemoryOnActivatedContext(void) const
      \return Max RAM memory on a context
      \brief Get the maximum RAM memory on activated OpenCL context
    */
    inline GGulong GetMaxRAMMemoryOnActivatedContext(void) const {return device_global_mem_size_[context_index_];}

    /*!
      \fn inline GGulong GetMaxBufferAllocationSize(void) const
      \return Max buffer allocation size
      \brief Get the max buffer size in bytes on activated OpenCL context
    */
    inline GGulong GetMaxBufferAllocationSize(void) const {return device_max_mem_alloc_size_[context_index_];}

    /*!
      \fn inline std::size_t GetMaxWorkGroupSize(void) const
      \return Max work group size
      \brief Get the max work group size on activated OpenCL context
    */
    inline std::size_t GetMaxWorkGroupSize(void) const { return device_max_work_group_size_[context_index_];}

    /*!
      \fn inline std::size_t GetWorkGroupSize(void) const
      \return Work group size
      \brief Get the work group size defined in GGEMS on activated OpenCL context
    */
    inline std::size_t GetWorkGroupSize(void) const { return work_group_size_;}

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
    inline cl::Context* GetContext(void) const {return contexts_.at(context_index_).get();}

    /*!
      \fn cl::CommandQueue* GetCommandQueue(void) const
      \return the pointer on activated command queue
      \brief Return the command queue to activated context
    */
    inline cl::CommandQueue* GetCommandQueue(void) const {return queue_act_.get();}

    /*!
      \fn cl::Event* GetEvent(void) const
      \return the pointer on activated event
      \brief return an event to activated context
    */
    inline cl::Event* GetEvent(void) const {return event_act_.get();}

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
      \fn void Deallocate(std::shared_ptr<cl::Buffer> buffer, std::size_t size)
      \param buffer - pointer to buffer in host memory
      \param size - size of the buffer in bytes
      \brief Deallocation of OpenCL memory
    */
    void Deallocate(std::shared_ptr<cl::Buffer> buffer, std::size_t size);

    /*!
      \fn void Clean(std::shared_ptr<cl::Buffer> buffer, std::size_t size)
      \param buffer - pointer to buffer in host memory
      \param size - size of the buffer in bytes
      \brief Cleaning buffer on OpenCL device
    */
    void Clean(std::shared_ptr<cl::Buffer> buffer, std::size_t size);

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
      \fn DurationNano GetElapsedTimeInKernel(void) const
      \return elapsed time in kernel
      \brief Get the elapsed time in the last used kernel
    */
    DurationNano GetElapsedTimeInKernel(void) const;

    /*!
      \fn void CheckOpenCLError(GGint const& error, std::string const& class_name, std::string const& method_name) const
      \param error - error index
      \param class_name - name of the class
      \param method_name - name of the method
      \brief check the OpenCL error
    */
    void CheckOpenCLError(GGint const& error, std::string const& class_name, std::string const& method_name) const;

    /*!
      \fn std::size_t GetBestWorkItem(GGulong const& number_of_elements) const
      \param number_of_elements - number of elements for the kernel computation
      \return best number of work item
      \brief get the best number of work item
    */
    std::size_t GetBestWorkItem(GGulong const& number_of_elements) const;

  private:
    /*!
      \fn std::string ErrorType(GGint const& error) const
      \param error - error index from OpenCL library
      \return the message error
      \brief get the error description
    */
    std::string ErrorType(GGint const& error) const;

    /*!
      \fn std::size_t CheckKernel(std::string const& kernel_name, std::string const& compilation_options) const
      \param kernel_name - name of the kernel
      \param compilation_options - arguments of compilation
      \brief check if a kernel has been already compiled
      \return index of kernel if already compiled
    */
    std::size_t CheckKernel(std::string const& kernel_name, std::string const& compilation_options) const;

    /*!
      \fn bool IsDoublePrecisionAtomicAddition(void) const
      \return true if double precision atomic addition is supported by OpenCL device, otherwize false
      \brief checking double precision atomic addition on OpenCL device
    */
    bool IsDoublePrecisionAtomicAddition(void) const;

    /*!
      \fn bool CheckDoublePrecision(void) const
      \return true if double precision is supported by OpenCL device, otherwize false
      \brief checking double precision on OpenCL device
    */
    bool IsDoublePrecision(void) const;

  private:
    // Platforms
    std::vector<cl::Platform> platforms_; /*!< List of detected platforms */
    std::vector<std::string> platform_profile_; /*!< OpenCL profile */
    std::vector<std::string> platform_version_; /*!< OpenCL version supported by the implementation */
    std::vector<std::string> platform_name_; /*!< Platform name */
    std::vector<std::string> platform_vendor_; /*!< Vendor of the platform */
    std::vector<std::string> platform_extensions_; /*!< List of the extension names */

    // Devices
    std::vector<std::unique_ptr<cl::Device>> devices_; /*!< Vector of devices */
    std::vector<cl_device_type> device_type_; /*!< Type of device */
    std::vector<std::string> device_name_; /*!< Name of the device */
    std::vector<std::string> device_vendor_; /*!< Vendor of the device */
    std::vector<GGuint> device_vendor_id_; /*!< Vendor ID of the device */
    std::vector<std::string> device_profile_; /*!< Profile of the device */
    std::vector<std::string> device_version_; /*!< Version of the device */
    std::vector<std::string> device_driver_version_; /*!< Driver version of the device */
    std::vector<std::string> device_opencl_c_version_; /*!< OpenCL C version */
    std::vector<GGuint> device_native_vector_width_char_; /*!< Native vector for char integer */
    std::vector<GGuint> device_native_vector_width_short_; /*!< Native vector for short integer */
    std::vector<GGuint> device_native_vector_width_int_; /*!< Native vector for int integer */
    std::vector<GGuint> device_native_vector_width_long_; /*!< Native vector for long integer */
    std::vector<GGuint> device_native_vector_width_half_; /*!< Native vector for half precision */
    std::vector<GGuint> device_native_vector_width_float_; /*!< Native vector for single precision */
    std::vector<GGuint> device_native_vector_width_double_; /*!< Native vector for double precision */
    std::vector<GGuint> device_preferred_vector_width_char_; /*!< Preferred vector for char integer */
    std::vector<GGuint> device_preferred_vector_width_short_; /*!< Preferred vector for short integer */
    std::vector<GGuint> device_preferred_vector_width_int_; /*!< Preferred vector for int integer */
    std::vector<GGuint> device_preferred_vector_width_long_; /*!< Preferred vector for long integer */
    std::vector<GGuint> device_preferred_vector_width_half_; /*!< Preferred vector for half precision */
    std::vector<GGuint> device_preferred_vector_width_float_; /*!< Preferred vector for single precision */
    std::vector<GGuint> device_preferred_vector_width_double_; /*!< Preferred vector for double precision */
    std::vector<GGuint> device_address_bits_; /*!< Address Bits */
    std::vector<GGbool> device_available_; /*!< Flag on device availability */
    std::vector<GGbool> device_compiler_available_; /*!< Flag on compiler availability */
    std::vector<cl_device_fp_config> device_half_fp_config_; /*!< Half precision capability */
    std::vector<cl_device_fp_config> device_single_fp_config_; /*!< Single precision capability */
    std::vector<cl_device_fp_config> device_double_fp_config_; /*!< Double precision capability */
    std::vector<GGbool> device_endian_little_; /*!< Endian little */
    std::vector<std::string> device_extensions_; /*!< Extensions */
    std::vector<GGbool> device_error_correction_support_; /*!< Error correction support */
    std::vector<cl_device_exec_capabilities> device_execution_capabilities_; /*!< Execution capabilities */
    std::vector<GGulong> device_global_mem_cache_size_; /*!< Global memory cache size */
    std::vector<cl_device_mem_cache_type> device_global_mem_cache_type_; /*!< Global memory cache type */
    std::vector<GGuint> device_global_mem_cacheline_size_; /*!< Global memory cache line size */
    std::vector<GGulong> device_global_mem_size_; /*!< Global memory size */
    std::vector<GGulong> device_local_mem_size_; /*!< Local memory size */
    std::vector<cl_device_local_mem_type> device_local_mem_type_; /*!< Local memory type */
    std::vector<GGbool> device_host_unified_memory_; /*!< Host unified memory */
    std::vector<std::size_t> device_image_max_array_size_; /*!< Max size of image array */
    std::vector<std::size_t> device_image_max_buffer_size_; /*!< Max size of image buffer */
    std::vector<GGbool> device_image_support_; /*!< Image support */
    std::vector<std::size_t> device_image2D_max_width_; /*!< Max width of image 2D */
    std::vector<std::size_t> device_image2D_max_height_; /*!< Max height of image 2D */
    std::vector<std::size_t> device_image3D_max_width_; /*!< Max width of image 3D */
    std::vector<std::size_t> device_image3D_max_height_; /*!< Max height of image 3D */
    std::vector<std::size_t> device_image3D_max_depth_; /*!< Max depth of image 3D */
    std::vector<GGuint> device_max_read_image_args_; /*!< Max read image read by kernel in same time */
    std::vector<GGuint> device_max_write_image_args_; /*!< Max write image read by kernel in same time */
    std::vector<GGuint> device_max_clock_frequency_; /*!< Max frequency of device */
    std::vector<GGuint> device_max_compute_units_; /*!< Max compute units */
    std::vector<GGuint> device_max_constant_args_; /*!< Max constant arguments in kernel */
    std::vector<GGulong> device_max_constant_buffer_size_; /*!< Max constant buffer size */
    std::vector<GGulong> device_max_mem_alloc_size_; /*!< Max memory allocation size */
    std::vector<std::size_t> device_max_parameter_size_; /*!< Max Parameter size in kernel */
    std::vector<GGuint> device_max_samplers_; /*!< Max number of samplers in kernel */
    std::vector<std::size_t> device_max_work_group_size_; /*!< Max Work group size */
    std::vector<GGuint> device_max_work_item_dimensions_; /*!< Maximum work item dimensions */
    std::vector<std::size_t> device_max_work_item_sizes_; /*!< Maximum work item sizes */
    std::vector<GGuint> device_mem_base_addr_align_; /*!< Alignment memory */
    std::vector<std::size_t> device_printf_buffer_size_; /*!< Size of buffer for printf in kernel */
    std::vector<cl_device_affinity_domain> device_partition_affinity_domain_; /*!< Partition affinity domain */
    std::vector<GGuint> device_partition_max_sub_devices_; /*!< Partition affinity domain */
    std::vector<std::size_t> device_profiling_timer_resolution_; /*!< Timer resolution */

    // Custom work group size
    std::size_t work_group_size_; /*!< Work group size by GGEMS, here 64 */

    // OpenCL compilation options
    std::string build_options_; /*!< list of option to OpenCL compiler */

    // Context and informations about them
    bool is_context_activated_; /*!< Check if context already activated */
    GGint context_index_; /*!< Index of the activated context */
    std::vector<std::shared_ptr<cl::Context>> contexts_; /*!< Vector of context */

    // Command queue informations
    std::vector<std::shared_ptr<cl::CommandQueue>> queues_; /*!< Command queue for all the context */
    std::shared_ptr<cl::CommandQueue> queue_act_; /*!< Activated command queue */

    // OpenCL event
    std::vector<std::shared_ptr<cl::Event>> events_; /*!< List of pointer to OpenCL event, for profiling */
    std::shared_ptr<cl::Event> event_act_; /*!< Activated event */

    // Kernels
    std::vector<std::shared_ptr<cl::Kernel>> kernels_; /*!< List of pointer to OpenCL kernel */
    std::vector<std::string> kernel_compilation_options_; /*!< List of compilation options for kernel */
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
T* GGEMSOpenCLManager::GetDeviceBuffer(cl::Buffer* const device_ptr, std::size_t const size) const
{
  GGcout("GGEMSOpenCLManager", "GetDeviceBuffer", 3) << "Getting mapped memory buffer on OpenCL device..." << GGendl;

  GGint err = 0;
  T* ptr = static_cast<T*>(queue_act_.get()->enqueueMapBuffer(*device_ptr, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, size, nullptr, nullptr, &err));
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
  CheckOpenCLError(queue_act_.get()->enqueueUnmapMemObject(*device_ptr, host_ptr), "GGEMSOpenCLManager", "ReleaseDeviceBuffer");
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
  \fn void set_context_index_ggems_opencl_manager(GGEMSOpenCLManager* opencl_manager, GGint const context_id)
  \param opencl_manager - pointer on the singleton
  \param context_id - index of the context
  \brief Set the context index to activate
*/
extern "C" GGEMS_EXPORT void set_context_index_ggems_opencl_manager(GGEMSOpenCLManager* opencl_manager, GGint const context_id);

#endif // GUARD_GGEMS_GLOBAL_GGEMSOPENCLMANAGER_HH
