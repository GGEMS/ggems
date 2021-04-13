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

  \brief Singleton class storing all informations about OpenCL and managing GPU/CPU devices, contexts, kernels, command queues and events. In GGEMS the strategy is 1 context = 1 device.

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 23, 2021
*/

#include <unordered_map>
#include "GGEMS/tools/GGEMSPrint.hh"

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#if __GNUC__ >= 6 
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

typedef std::unordered_map<std::string, std::string> VendorUMap; /*!< Alias to OpenCL vendors */

#define KERNEL_NOT_COMPILED 0x100000000 /*!< value if OpenCL kernel is not compiled */

/*!
  \class GGEMSOpenCLManager
  \brief Singleton class storing all informations about OpenCL and managing GPU/CPU devices, contexts, kernels, command queues and events. In GGEMS the strategy is 1 context = 1 device.
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
      \fn void CheckOpenCLError(GGint const& error, std::string const& class_name, std::string const& method_name) const
      \param error - error index
      \param class_name - name of the class
      \param method_name - name of the method
      \brief check the OpenCL error
    */
    void CheckOpenCLError(GGint const& error, std::string const& class_name, std::string const& method_name) const;

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
      \fn void PrintBuildOptions(void) const
      \brief print global build options used during kernel compilation
    */
    void PrintBuildOptions(void) const;

    /*!
      \fn void PrintActivatedDevices(void) const
      \brief print infos about activated devices
    */
    void PrintActivatedDevices(void) const;

    /*!
      \fn std::string GetNameOfDevice(GGsize const& index) const
      \param index - index of device
      \return name of activated device
      \brief Get the name of the activated device
    */
    inline std::string GetDeviceName(GGsize const& index) const {return device_name_[index];}

    /*!
      \fn cl_device_type GetDeviceType(GGsize const& index) const
      \param index - index of device
      \return name of activated device
      \brief Get the type of the activated device
    */
    inline cl_device_type GetDeviceType(GGsize const& index) const {return device_type_[index];}

    /*!
      \fn inline GGsize GetNumberOfDetectedDevice(void) const
      \return number of detected device
      \brief get the number of detected devices
    */
    inline GGsize GetNumberOfDetectedDevice(void) const {return devices_.size();}

    /*!
      \fn inline GGsize GetNumberOfActivatedDevice(void) const
      \return number of activated device
      \brief get the number of activated devices
    */
    inline GGsize GetNumberOfActivatedDevice(void) const {return device_indices_.size();}

    /*!
      \fn inline GGsize GetIndexOfActivatedDevice(GGsize const& device_index) const
      \param device_index - index of device
      \return index of activated device
      \brief get the index of activated device
    */
    inline GGsize GetIndexOfActivatedDevice(GGsize const& device_index) const {return device_indices_[device_index];}

    /*!
      \fn inline GGsize GetMaxBufferAllocationSize(GGsize const& device_index) const
      \param device_index - index of activated devices
      \return Max buffer allocation size
      \brief Get the max buffer size in bytes on activated OpenCL device
    */
    inline GGsize GetMaxBufferAllocationSize(GGsize const& device_index) const {return static_cast<GGsize>(device_max_mem_alloc_size_[device_index]);}

    /*!
      \fn inline GGsize GetRAMMemory(GGsize const& device_index) const
      \param device_index - index of activated devices
      \return RAM memory on a specific device
      \brief Get the RAM in bytes on OpenCL device
    */
    inline GGsize GetRAMMemory(GGsize const& device_index) const {return static_cast<GGsize>(device_global_mem_size_[device_index]);}

    /*!
      \fn inline GGsize GetWorkGroupSize(void) const
      \return Work group size
      \brief Get the work group size defined in GGEMS on activated OpenCL context
    */
    inline GGsize GetWorkGroupSize(void) const { return work_group_size_;}

    /*!
      \fn GGsize GetBestWorkItem(GGsize const& number_of_elements) const
      \param number_of_elements - number of elements for the kernel computation
      \return best number of work item
      \brief get the best number of work item
    */
    GGsize GetBestWorkItem(GGsize const& number_of_elements) const;

    /*!
      \fn cl::Context* GetContext(GGsize const& thread_index) const
      \param thread_index - index of the thread (= activated device index)
      \return the pointer on activated context
      \brief return the activated context
    */
    inline cl::Context* GetContext(GGsize const& thread_index) const {return contexts_[thread_index];}

    /*!
      \fn cl::CommandQueue* GetCommandQueue(GGsize const& thread_index) const
      \param thread_index - index of the thread (= activated device index)
      \return the pointer on activated command queue
      \brief Return the command queue to activated context
    */
    inline cl::CommandQueue* GetCommandQueue(GGsize const& thread_index) const {return queues_[thread_index];}

    /*!
      \fn cl::Event* GetEvent(GGsize const& thread_index) const
      \param thread_index - index of the thread (= activated device index)
      \return the pointer on activated event
      \brief return an event to activated context
    */
    inline cl::Event* GetEvent(GGsize const& thread_index) const {return events_[thread_index];}

    /*!
      \fn void DeviceToActivate(GGsize const& device_id)
      \param device_id - device index
      \brief set the index of the device to activate
    */
    void DeviceToActivate(GGsize const& device_id);

    /*!
      \fn void DeviceToActivate(std::string const& device_type, std::string const& device_vendor = "")
      \param device_type - type of device : all, gpu or cpu
      \param device_vendor - vendor : nvidia, intel, or amd
      \brief activate specific device
    */
    void DeviceToActivate(std::string const& device_type, std::string const& device_vendor = "");

    /*!
      \fn inline bool IsReady(void) const
      \return true is OpenCL manager is ready to use, it means a device is activated
      \brief check if an OpenCL device is activated
    */
    inline bool IsReady(void) const {return !device_indices_.empty();}

    /*!
      \fn void Clean(void)
      \brief clean OpenCL data
    */
    void Clean(void);

    /*!
      \fn void CompileKernel(std::string const& kernel_filename, std::string const& kernel_name, cl::Kernel** kernel_list, char* const custom_options = nullptr, char* const additional_options = nullptr)
      \param kernel_filename - filename where is declared the kernel
      \param kernel_name - name of the kernel
      \param kernel_list - list of kernel by device
      \param custom_options - new compilation option for the kernel
      \param additional_options - additionnal compilation option
      \brief Compile the OpenCL kernel on the activated device
      \return the pointer on the OpenCL kernel
    */
    void CompileKernel(std::string const& kernel_filename, std::string const& kernel_name, cl::Kernel** kernel_list, char* const custom_options = nullptr, char* const additional_options = nullptr);

    /*!
      \return the pointer on host memory on write/read mode
      \brief Get the device pointer on host to write on it. ReleaseDeviceBuffer must be used after this method!!!
      \param device_ptr - pointer on device memory
      \param size - size of region to map
      \param thread_index - index of the thread (= activated device index)
      \tparam T - type of the returned pointer on host memory
    */
    template <typename T>
    T* GetDeviceBuffer(cl::Buffer* device_ptr, GGsize const& size, GGsize const& thread_index);

    /*!
      \brief Get the device pointer on host to write on it. Mandatory after a GetDeviceBufferWrite ou GetDeviceBufferRead!!!
      \param device_ptr - pointer on device memory
      \param host_ptr - pointer on host memory mapped on device memory
      \param thread_index - index of the thread (= activated device index)
      \tparam T - type of host memory pointer to release
    */
    template <typename T>
    void ReleaseDeviceBuffer(cl::Buffer* const device_ptr, T* host_ptr, GGsize const& thread_index);

    /*!
      \fn cl::Buffer* Allocate(void* host_ptr, GGsize const& size, GGsize const& thread_index, cl_mem_flags flags, std::string const& class_name = "Undefined")
      \param host_ptr - pointer to buffer in host memory
      \param size - size of the buffer in bytes
      \param thread_index - index of the thread (= activated device index)
      \param flags - mode to open the buffer
      \param class_name - name of class allocating memory
      \brief Allocation of OpenCL memory
      \return an unique pointer to an OpenCL buffer
    */
    cl::Buffer* Allocate(void* host_ptr, GGsize const& size, GGsize const& thread_index, cl_mem_flags flags, std::string const& class_name = "Undefined");

    /*!
      \fn void Deallocate(cl::Buffer* buffer, GGsize size, GGsize const& thread_index)
      \param buffer - pointer to buffer in host memory
      \param size - size of the buffer in bytes
      \param thread_index - index of the thread (= activated device index)
      \param class_name - name of class deallocating memory
      \brief Deallocation of OpenCL memory
    */
    void Deallocate(cl::Buffer* buffer, GGsize size, GGsize const& thread_index, std::string const& class_name = "Undefined");

    /*!
      \fn void CleanBuffer(cl::Buffer* buffer, GGsize const& size, GGsize const& thread_index)
      \param buffer - pointer to buffer in host memory
      \param size - size of the buffer in bytes
      \param thread_index - index of the thread (= activated device index)
      \brief Cleaning buffer on OpenCL device
    */
    void CleanBuffer(cl::Buffer* buffer, GGsize const& size, GGsize const& thread_index);

  private:
    /*!
      \fn std::string ErrorType(GGint const& error) const
      \param error - error index from OpenCL library
      \return the message error
      \brief get the error description
    */
    std::string ErrorType(GGint const& error) const;

    /*!
      \fn std::GGsize CheckKernel(std::string const& kernel_name, std::string const& compilation_options) const
      \param kernel_name - name of the kernel
      \param compilation_options - arguments of compilation
      \brief check if a kernel has been already compiled
      \return index of kernel if already compiled
    */
    GGsize CheckKernel(std::string const& kernel_name, std::string const& compilation_options) const;

    /*!
      \fn bool IsDoublePrecision(GGsize const& index) const
      \param index - index of device
      \return true if double precision is supported by OpenCL device, otherwize false
      \brief checking double precision on OpenCL device
    */
    bool IsDoublePrecision(GGsize const& index) const;

  private:
    // OpenCL platform
    std::vector<cl::Platform> platforms_; /*!< List of detected platform */

    std::vector<std::string> platform_profile_; /*!< OpenCL profile */
    std::vector<std::string> platform_version_; /*!< OpenCL version supported by the implementation */
    std::vector<std::string> platform_name_; /*!< Platform name */
    std::vector<std::string> platform_vendor_; /*!< Vendor of the platform */
    std::vector<std::string> platform_extensions_; /*!< List of the extension names */

    // OpenCL device
    std::vector<cl::Device*> devices_; /*!< List of detected device */
    std::vector<GGsize> device_indices_; /*!< Index of the activated device */
    GGsize work_group_size_; /*!< Work group size by GGEMS, here 64 */
    VendorUMap vendors_; /*!< UMap storing vendor name and an alias */

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
    std::vector<GGsize> device_image_max_array_size_; /*!< Max size of image array */
    std::vector<GGsize> device_image_max_buffer_size_; /*!< Max size of image buffer */
    std::vector<GGbool> device_image_support_; /*!< Image support */
    std::vector<GGsize> device_image2D_max_width_; /*!< Max width of image 2D */
    std::vector<GGsize> device_image2D_max_height_; /*!< Max height of image 2D */
    std::vector<GGsize> device_image3D_max_width_; /*!< Max width of image 3D */
    std::vector<GGsize> device_image3D_max_height_; /*!< Max height of image 3D */
    std::vector<GGsize> device_image3D_max_depth_; /*!< Max depth of image 3D */
    std::vector<GGuint> device_max_read_image_args_; /*!< Max read image read by kernel in same time */
    std::vector<GGuint> device_max_write_image_args_; /*!< Max write image read by kernel in same time */
    std::vector<GGuint> device_max_clock_frequency_; /*!< Max frequency of device */
    std::vector<GGuint> device_max_compute_units_; /*!< Max compute units */
    std::vector<GGuint> device_max_constant_args_; /*!< Max constant arguments in kernel */
    std::vector<GGulong> device_max_constant_buffer_size_; /*!< Max constant buffer size */
    std::vector<GGulong> device_max_mem_alloc_size_; /*!< Max memory allocation size */
    std::vector<GGsize> device_max_parameter_size_; /*!< Max Parameter size in kernel */
    std::vector<GGuint> device_max_samplers_; /*!< Max number of samplers in kernel */
    std::vector<GGsize> device_max_work_group_size_; /*!< Max Work group size */
    std::vector<GGuint> device_max_work_item_dimensions_; /*!< Maximum work item dimensions */
    std::vector<GGsize> device_max_work_item_sizes_; /*!< Maximum work item sizes */
    std::vector<GGuint> device_mem_base_addr_align_; /*!< Alignment memory */
    std::vector<GGsize> device_printf_buffer_size_; /*!< Size of buffer for printf in kernel */
    std::vector<cl_device_affinity_domain> device_partition_affinity_domain_; /*!< Partition affinity domain */
    std::vector<GGuint> device_partition_max_sub_devices_; /*!< Partition affinity domain */
    std::vector<GGsize> device_profiling_timer_resolution_; /*!< Timer resolution */

    // OpenCL compilation options
    std::string build_options_; /*!< list of default option to OpenCL compiler */

    // OpenCL context + command queue + event
    std::vector<cl::Context*> contexts_; /*!< OpenCL contexts */
    std::vector<cl::CommandQueue*> queues_; /*!< OpenCL command queues */
    std::vector<cl::Event*> events_; /*!< OpenCL events */

    // OpenCL kernels
    std::vector<cl::Kernel*> kernels_; /*!< List of kernels for each device */
    std::vector<std::string> kernel_compilation_options_; /*!< List of compilation options for kernel */
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
T* GGEMSOpenCLManager::GetDeviceBuffer(cl::Buffer* const device_ptr, GGsize const& size, GGsize const& thread_index)
{
  GGcout("GGEMSOpenCLManager", "GetDeviceBuffer", 3) << "Getting mapped memory buffer on OpenCL device..." << GGendl;

  GGint err = 0;
  T* ptr = static_cast<T*>(queues_[thread_index]->enqueueMapBuffer(*device_ptr, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, size, nullptr, nullptr, &err));
  CheckOpenCLError(err, "GGEMSOpenCLManager", "GetDeviceBuffer");
  return ptr;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
void GGEMSOpenCLManager::ReleaseDeviceBuffer(cl::Buffer* const device_ptr, T* host_ptr, GGsize const& thread_index)
{
  GGcout("GGEMSOpenCLManager", "ReleaseDeviceBuffer", 3) << "Releasing mapped memory buffer on OpenCL device..." << GGendl;

  // Unmap the memory
  CheckOpenCLError(queues_[thread_index]->enqueueUnmapMemObject(*device_ptr, host_ptr), "GGEMSOpenCLManager", "ReleaseDeviceBuffer");
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
  \fn void set_device_index_ggems_opencl_manager(GGEMSOpenCLManager* opencl_manager, GGsize const device_id)
  \param opencl_manager - pointer on the singleton
  \param device_id - index of the device
  \brief Set the device index to activate
*/
extern "C" GGEMS_EXPORT void set_device_index_ggems_opencl_manager(GGEMSOpenCLManager* opencl_manager, GGsize const device_id);

/*!
  \fn void set_device_index_ggems_opencl_manager(GGEMSOpenCLManager* opencl_manager, GGsize const device_id)
  \param opencl_manager - pointer on the singleton
  \param device_id - index of the device
  \brief Set the device index to activate
*/
extern "C" GGEMS_EXPORT void set_device_to_activate_opencl_manager(GGEMSOpenCLManager* opencl_manager, char const* device_type, char const* device_vendor = "");

/*!
  \fn void clean_opencl_manager(GGEMSOpenCLManager* opencl_manager)
  \param opencl_manager - pointer on the singleton
  \brief Clean OpenCL manager for python user
*/
extern "C" GGEMS_EXPORT void clean_opencl_manager(GGEMSOpenCLManager* opencl_manager);

#endif // GUARD_GGEMS_GLOBAL_GGEMSOpenCLManager_HH
