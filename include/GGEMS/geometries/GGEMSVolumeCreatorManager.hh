#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSVOLUMECREATORMANAGER_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSVOLUMECREATORMANAGER_HH

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
  \file GGEMSVolumeCreatorManager.hh

  \brief Singleton class generating voxelized volume from analytical volume

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thursday January 9, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include <map>

#include "GGEMS/global/GGEMSOpenCLManager.hh"

typedef std::map<GGfloat, std::string> LabelToMaterialMap; /*!< Map of label value to material */

/*!
  \class GGEMSVolumeCreatorManager
  \brief Singleton class handling convertion from analytical volume to voxelized volume
*/
class GGEMS_EXPORT GGEMSVolumeCreatorManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGEMSVolumeCreatorManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGEMSVolumeCreatorManager(void);

  public:
    /*!
      \fn static GGEMSVolumeCreatorManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSVolumeCreatorManager
    */
    static GGEMSVolumeCreatorManager& GetInstance(void)
    {
      static GGEMSVolumeCreatorManager instance;
      return instance;
    }

    /*!
      \fn GGEMSVolumeCreatorManager(GGEMSVolumeCreatorManager const& volume_creator_manager) = delete
      \param volume_creator_manager - reference on the singleton
      \brief Avoid copy of the singleton by reference
    */
    GGEMSVolumeCreatorManager(GGEMSVolumeCreatorManager const& volume_creator_manager) = delete;

    /*!
      \fn GGEMSVolumeCreatorManager& operator=(GGEMSVolumeCreatorManager const& volume_creator_manager) = delete
      \param volume_creator_manager - reference on the singleton
      \brief Avoid assignement of the singleton by reference
    */
    GGEMSVolumeCreatorManager& operator=(GGEMSVolumeCreatorManager const& volume_creator_manager) = delete;

    /*!
      \fn GGEMSVolumeCreatorManager(GGEMSVolumeCreatorManager const&& volume_creator_manager) = delete
      \param volume_creator_manager - rvalue reference on the singleton
      \brief Avoid copy of the singleton by rvalue reference
    */
    GGEMSVolumeCreatorManager(GGEMSVolumeCreatorManager const&& volume_creator_manager) = delete;

    /*!
      \fn GGEMSVolumeCreatorManager& operator=(GGEMSVolumeCreatorManager const&& volume_creator_manager) = delete
      \param volume_creator_manager - rvalue reference on the singleton
      \brief Avoid copy of the singleton by rvalue reference
    */
    GGEMSVolumeCreatorManager& operator=(GGEMSVolumeCreatorManager const&& volume_creator_manager) = delete;

    /*!
      \fn void SetElementSizes(GGfloat const& voxel_width, GGfloat const& voxel_height, GGfloat const& voxel_depth, char const* unit = "mm")
      \param voxel_width - voxel width
      \param voxel_height - voxel height
      \param voxel_depth - voxel depth
      \param unit - unit of the distance
      \brief Set the size of the elements for the voxelized volume
    */
    void SetElementSizes(GGfloat const& voxel_width, GGfloat const& voxel_height, GGfloat const& voxel_depth, char const* unit = "mm");

    /*!
      \fn GGfloat3 GetElementsSizes(void) const
      \return a 3d float with the size of voxel in voxelized volume
      \brief size of voxels in the voxelized volume
    */
    inline GGfloat3 GetElementsSizes(void) const {return element_sizes_;}

    /*!
      \fn void SetVolumeDimensions(GGuint const& volume_width, GGuint const& volume_height, GGuint const& volume_depth)
      \param volume_width - volume width
      \param volume_height - volume height
      \param volume_depth - volume depth
      \brief Set the dimension of the volume for the voxelized volume
    */
    void SetVolumeDimensions(GGuint const& volume_width, GGuint const& volume_height, GGuint const& volume_depth);

    /*!
      \fn GGuint3 GetVolumeDimensions(void) const
      \return a 3d int with the dimenstion of the voxelized volume
      \brief dimensions of volume
    */
    inline GGuint3 GetVolumeDimensions(void) const {return volume_dimensions_;};

    /*!
      \fn void SetMaterial(char const* material)
      \param material - name of the material
      \brief set the material, Air by default
    */
    void SetMaterial(char const* material = "Air");

    /*!
      \fn GGulong GetNumberElements(void) const
      \return number of voxel in the voxelized volume
      \brief Return the total number of voxels
    */
    inline GGulong GetNumberElements(void) const {return number_elements_;}

    /*!
      \fn std::string GetDataType(void) const
      \brief get the type of data
      \return the type of the data
    */
    inline std::string GetDataType(void) const {return data_type_;};

    /*!
      \fn cl::Buffer* GetVoxelizedVolume(void) const
      \return pointer on OpenCL device storing voxelized volume
      \brief Return the voxelized volume on OpenCL device
    */
    inline cl::Buffer* GetVoxelizedVolume(void) const {return voxelized_volume_cl_.get();}

    /*!
      \fn void SetOutputImageFilename(char const* output_image_filename)
      \param output_image_filename - output image filename
      \brief Set the filename of MHD output
    */
    void SetOutputImageFilename(char const* output_image_filename);

    /*!
      \fn void SetRangeToMaterialDataFilename(char const* output_range_to_material_filename)
      \param output_range_to_material_filename - output range to material filename
      \brief Set the filename of range to material data
    */
    void SetRangeToMaterialDataFilename(char const* output_range_to_material_filename);

    /*!
      \fn void AddLabelAndMaterial(GGfloat const& label, std::string const& material)
      \param label - label of material
      \param material - material of the volume
      \brief add the label and the material
    */
    void AddLabelAndMaterial(GGfloat const& label, std::string const& material);

    /*!
      \fn void SetDataType(std::string const& data_type = "MET_FLOAT")
      \param data_type - type of data
      \brief set the type of data
    */
    void SetDataType(std::string const& data_type = "MET_FLOAT");

    /*!
      \fn void Initialize(void)
      \brief Initialize the volume Creator manager
    */
    void Initialize(void);

    /*!
      \fn void Write(void)
      \brief Save the voxelized volume to raw data in mhd file
    */
    void Write(void);

  private:
    /*!
      \fn void CheckParameters(void) const
      \brief Check the mandatory parameters
    */
    void CheckParameters(void) const;

    /*!
      \fn void WriteMHDImage(void) const
      \brief Write output MHD file
    */
    void WriteMHDImage(void) const;

    /*!
      \fn void WriteRangeToMaterialFile(void)
      \brief Write the file with range to material data
    */
    void WriteRangeToMaterialFile(void);

    /*!
      \fn template <typename T> void AllocateImage(void)
      \tparam T - type of data
      \brief allocating buffer storing volume
    */
    template <typename T>
    void AllocateImage(void);

  private:
    GGfloat3 element_sizes_; /*!< Size of voxels of voxelized volume */
    GGuint3 volume_dimensions_; /*!< Dimension of volume X, Y, Z */
    GGulong number_elements_; /*!< Total number of elements */
    std::string data_type_; /*!< Type of data */
    std::string material_; /*!< Material for background, air by default */
    std::string output_image_filename_; /*!< Output MHD where is stored the voxelized volume */
    std::string output_range_to_material_filename_; /*!< Output text file with range to material data */
    std::shared_ptr<cl::Buffer> voxelized_volume_cl_; /*!< Voxelized volume on OpenCL device */
    LabelToMaterialMap label_to_material_; /*!< Map of label to material */
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
void GGEMSVolumeCreatorManager::AllocateImage(void)
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Allocation of memory on OpenCL device depending of type
  voxelized_volume_cl_ = opencl_manager.Allocate(nullptr, number_elements_ * sizeof(T), CL_MEM_READ_WRITE);

  // Initialize the buffer to zero
  T* voxelized_volume_device = opencl_manager.GetDeviceBuffer<T>(voxelized_volume_cl_.get(), number_elements_ * sizeof(T));

  for (GGulong i = 0; i < number_elements_; ++i) voxelized_volume_device[i] = static_cast<T>(0);

  // Release the pointers
  opencl_manager.ReleaseDeviceBuffer(voxelized_volume_cl_.get(), voxelized_volume_device);
}

/*!
  \fn GGEMSVolumeCreatorManager* get_instance_volume_creator_manager(void)
  \return the pointer on the singleton
  \brief Get the GGEMSVolumeCreatorManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSVolumeCreatorManager* get_instance_volume_creator_manager(void);

/*!
  \fn void set_volume_dimension_volume_creator_manager(GGEMSVolumeCreatorManager* volume_creator_manager, GGuint const volume_width, GGuint const volume_height, GGuint const volume_depth)
  \param volume_creator_manager - pointer on the singleton
  \param volume_width - volume width
  \param volume_height - volume height
  \param volume_depth - volume depth
  \brief Set the dimension of the volume for the voxelized volume
*/
extern "C" GGEMS_EXPORT void set_volume_dimension_volume_creator_manager(GGEMSVolumeCreatorManager* volume_creator_manager, GGuint const volume_width, GGuint const volume_height, GGuint const volume_depth);

/*!
  \fn void set_element_sizes_volume_creator_manager(GGEMSVolumeCreatorManager* volume_creator_manager, GGfloat const voxel_width, GGfloat const voxel_height, GGfloat const voxel_depth, char const* unit)
  \param volume_creator_manager - pointer on the singleton
  \param voxel_width - voxel width
  \param voxel_height - voxel height
  \param voxel_depth - voxel depth
  \param unit - unit of the distance
  \brief Set the size of the elements for the voxelized volume
*/
extern "C" GGEMS_EXPORT void set_element_sizes_volume_creator_manager(GGEMSVolumeCreatorManager* volume_creator_manager, GGfloat const voxel_width, GGfloat const voxel_height, GGfloat const voxel_depth, char const* unit);

/*!
  \fn void set_output_image_filename_volume_creator_manager(GGEMSVolumeCreatorManager* volume_creator_manager, char const* output_image_filename)
  \param volume_creator_manager - pointer on the singleton
  \param output_image_filename - output MHD filename
  \brief Set the filename of MHD output
*/
extern "C" GGEMS_EXPORT void set_output_image_filename_volume_creator_manager(GGEMSVolumeCreatorManager* volume_creator_manager,char const* output_image_filename);

/*!
  \fn void set_output_range_to_material_filename_volume_creator_manager(GGEMSVolumeCreatorManager* volume_creator_manager, char const* output_range_to_material_filename)
  \param volume_creator_manager - pointer on the singleton
  \param output_range_to_material_filename - output range to material filename
  \brief Set the filename of range to material data
*/
extern "C" GGEMS_EXPORT void set_output_range_to_material_filename_volume_creator_manager(GGEMSVolumeCreatorManager* volume_creator_manager,char const* output_range_to_material_filename);

/*!
  \fn void initialize_volume_creator_manager(GGEMSVolumeCreatorManager* volume_creator_manager)
  \param volume_creator_manager - pointer on the singleton
  \brief Initialize the volume creator manager
*/
extern "C" GGEMS_EXPORT void initialize_volume_creator_manager(GGEMSVolumeCreatorManager* volume_creator_manager);

/*!
  \fn void write_volume_creator_manager(GGEMSVolumeCreatorManager* volume_creator_manager)
  \param volume_creator_manager - pointer on the singleton
  \brief Save the voxelized volume to raw data in mhd file
*/
extern "C" GGEMS_EXPORT void write_volume_creator_manager(GGEMSVolumeCreatorManager* volume_creator_manager);

/*!
  \fn void set_material_volume_creator_manager(GGEMSVolumeCreatorManager* volume_creator_manager, char const* material)
  \param volume_creator_manager - pointer on the singleton
  \param material - name of the material
  \brief set the material of the global (background volume)
*/
extern "C" GGEMS_EXPORT void set_material_volume_creator_manager(GGEMSVolumeCreatorManager* volume_creator_manager, char const* material);

/*!
  \fn void set_data_type_volume_creator_manager(GGEMSVolumeCreatorManager* volume_creator_manager, char const* data_type)
  \param volume_creator_manager - pointer on the singleton
  \param data_type - type of data
  \brief set the type of data
*/
extern "C" GGEMS_EXPORT void set_data_type_volume_creator_manager(GGEMSVolumeCreatorManager* volume_creator_manager, char const* data_type);

#endif // GUARD_GGEMS_GEOMETRIES_GGEMSVOLUMECREATORMANAGER_HH
