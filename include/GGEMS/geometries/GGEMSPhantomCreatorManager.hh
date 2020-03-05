#ifndef GUARD_GGEMS_GEOMETRY_GGEMSPHANTOMCREATORMANAGER_HH
#define GUARD_GGEMS_GEOMETRY_GGEMSPHANTOMCREATORMANAGER_HH

/*!
  \file GGEMSPhantomCreatorManager.hh

  \brief Singleton class generating voxelized phantom from analytical volume

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

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/global/GGEMSOpenCLManager.hh"

typedef std::map<float, std::string> LabelToMaterialMap;

/*!
  \class GGEMSPhantomCreatorManager
  \brief Singleton class handling convertion from analytical phantom to voxelized phantom
*/
class GGEMS_EXPORT GGEMSPhantomCreatorManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGEMSPhantomCreatorManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGEMSPhantomCreatorManager(void);

  public:
    /*!
      \fn static GGEMSPhantomCreatorManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSPhantomCreatorManager
    */
    static GGEMSPhantomCreatorManager& GetInstance(void)
    {
      static GGEMSPhantomCreatorManager instance;
      return instance;
    }

    /*!
      \fn GGEMSPhantomCreatorManager(GGEMSPhantomCreatorManager const& opencl_manager) = delete
      \param opencl_manager - reference on the singleton
      \brief Avoid copy of the singleton by reference
    */
    GGEMSPhantomCreatorManager(GGEMSPhantomCreatorManager const& phantom_creator_manager) = delete;

    /*!
      \fn GGEMSPhantomCreatorManager& operator=(GGEMSPhantomCreatorManager const& opencl_manager) = delete
      \param opencl_manager - reference on the singleton
      \brief Avoid assignement of the singleton by reference
    */
    GGEMSPhantomCreatorManager& operator=(GGEMSPhantomCreatorManager const& phantom_creator_manager) = delete;

    /*!
      \fn GGEMSOpenCLManager(GGEMSPhantomCreatorManager const&& opencl_manager) = delete
      \param opencl_manager - rvalue reference on the singleton
      \brief Avoid copy of the singleton by rvalue reference
    */
    GGEMSPhantomCreatorManager(GGEMSPhantomCreatorManager const&& phantom_creator_manager) = delete;

    /*!
      \fn GGEMSOpenCLManager& operator=(GGEMSOpenCLManager const&& opencl_manager) = delete
      \param opencl_manager - rvalue reference on the singleton
      \brief Avoid copy of the singleton by rvalue reference
    */
    GGEMSPhantomCreatorManager& operator=(GGEMSPhantomCreatorManager const&& phantom_creator_manager) = delete;

    /*!
      \fn void SetElementSizes(double const& voxel_width, double const& voxel_height, double const& voxel_depth, char const* unit = "mm")
      \param voxel_width - voxel width
      \param voxel_height - voxel height
      \param voxel_depth - voxel depth
      \param unit - unit of the distance
      \brief Set the size of the elements for the voxelized phantom
    */
    void SetElementSizes(GGdouble const& voxel_width, GGdouble const& voxel_height, GGdouble const& voxel_depth, char const* unit = "mm");

    /*!
      \fn GGdouble3 GetElementsSizes(void) const
      \return a 3d double with the size of voxel in voxelized phantom
      \brief size of voxels in the voxelized phantom
    */
    inline GGdouble3 GetElementsSizes(void) const {return element_sizes_;}

    /*!
      \fn void SetPhantomDimensions(GGuint const& phantom_width, GGuint const& phantom_height, GGuint const& phantom_depth)
      \param phantom_width - phantom width
      \param phantom_height - phatom height
      \param phantom_depth - phantom depth
      \brief Set the dimension of the phantom for the voxelized phantom
    */
    void SetPhantomDimensions(GGuint const& phantom_width, GGuint const& phantom_height, GGuint const& phantom_depth);

    /*!
      \fn GGuint3 GetPhantomDimensions(void) const
      \return a 3d int with the dimenstion of the voxelized phantom
      \brief dimensions of phantom
    */
    inline GGuint3 GetPhantomDimensions(void) const {return phantom_dimensions_;};

    /*!
      \fn void SetMaterial(char const* material)
      \param material - name of the material
      \brief set the material, Air by default
    */
    void SetMaterial(char const* material = "Air");

    /*!
      \fn GGulong GetNumberElements(void) const
      \return number of voxel in the voxelized phantom
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
      \fn cl::Buffer* GetVoxelizedPhantom(void) const
      \return pointer on OpenCL device storing voxelized phantom
      \brief Return the voxelized phantom on OpenCL device
    */
    inline cl::Buffer* GetVoxelizedPhantom(void) const {return voxelized_phantom_.get();}

    /*!
      \fn void SetOutputImageFilename(char const* output_image_filename)
      \param output_image_filename - output image filename
      \brief Set the filename of MHD output
    */
    void SetOutputImageFilename(char const* output_image_filename);

    /*!
      \fn void SetOutputImageFilename(char const* output_range_to_material_filename)
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
      \brief Initialize the Phantom Creator manager
    */
    void Initialize(void);

    /*!
      \fn void Write(void)
      \brief Save the voxelized phantom to raw data in mhd file
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
    GGdouble3 element_sizes_; /*!< Size of voxels of voxelized phantom */
    GGuint3 phantom_dimensions_; /*!< Dimension of phantom X, Y, Z */
    GGulong number_elements_; /*!< Total number of elements */
    std::string data_type_; /*!< Type of data */
    std::string material_; /*!< Material for background, air by default */
    std::string output_image_filename_; /*!< Output MHD where is stored the voxelized phantom */
    std::string output_range_to_material_filename_; /*!< Output text file with range to material data */
    std::shared_ptr<cl::Buffer> voxelized_phantom_; /*!< Voxelized phantom on OpenCL device */
    LabelToMaterialMap label_to_material_; /*!< Map of label to material */
    GGEMSOpenCLManager& opencl_manager_; /*!< Reference to opencl manager singleton */
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
void GGEMSPhantomCreatorManager::AllocateImage(void)
{
  // Allocation of memory on OpenCL device depending of type
  voxelized_phantom_ = opencl_manager_.Allocate(nullptr, number_elements_ * sizeof(T), CL_MEM_READ_WRITE);

  // Initialize the buffer to zero
  T* voxelized_phantom = opencl_manager_.GetDeviceBuffer<T>(voxelized_phantom_, number_elements_ * sizeof(T));

  for (GGulong i = 0; i < number_elements_; ++i) voxelized_phantom[i] = static_cast<T>(0);

  // Release the pointers
  opencl_manager_.ReleaseDeviceBuffer(voxelized_phantom_, voxelized_phantom);
}

/*!
  \fn GGEMSPhantomCreatorManager* get_instance_ggems_phantom_creator_manager(void)
  \brief Get the GGEMSOpenCLManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSPhantomCreatorManager* get_instance_phantom_creator_manager(void);

/*!
  \fn void set_phantom_dimension_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager, GGuint const phantom_width, GGuint const phantom_height, GGuint const phantom_depth)
  \param phantom_creator_manager - pointer on the singleton
  \param phantom_width - phantom width
  \param phantom_height - phatom height
  \param phantom_depth - phantom depth
  \brief Set the dimension of the phantom for the voxelized phantom
*/
extern "C" GGEMS_EXPORT void set_phantom_dimension_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager, GGuint const phantom_width, GGuint const phantom_height, GGuint const phantom_depth);

/*!
  \fn void set_element_sizes_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager, GGdouble const voxel_width, GGdouble const voxel_height, GGdouble const voxel_depth)
  \param phantom_creator_manager - pointer on the singleton
  \param voxel_width - voxel width
  \param voxel_height - voxel height
  \param voxel_depth - voxel depth
  \param unit - unit of the distance
  \brief Set the size of the elements for the voxelized phantom
*/
extern "C" GGEMS_EXPORT void set_element_sizes_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager, GGdouble const voxel_width, GGdouble const voxel_height, GGdouble const voxel_depth, char const* unit);

/*!
  \fn void set_output_basename_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager, char const* output_image_filename)
  \param phantom_creator_manager - pointer on the singleton
  \param output_image_filename - output MHD filename
  \brief Set the filename of MHD output
*/
extern "C" GGEMS_EXPORT void set_output_image_filename_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager,char const* output_image_filename);

/*!
  \fn void set_output_range_to_material_filename_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager, char const* output_range_to_material_filename)
  \param phantom_creator_manager - pointer on the singleton
  \param output_range_to_material_filename - output range to material filename
  \brief Set the filename of range to material data
*/
extern "C" GGEMS_EXPORT void set_output_range_to_material_filename_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager,char const* output_range_to_material_filename);

/*!
  \fn void initialize_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager)
  \param phantom_creator_manager - pointer on the singleton
  \brief Initialize the phantom creator manager
*/
extern "C" GGEMS_EXPORT void initialize_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager);

/*!
  \fn void write_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager)
  \param phantom_creator_manager - pointer on the singleton
  \brief Save the voxelized phantom to raw data in mhd file
*/
extern "C" GGEMS_EXPORT void write_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager);

/*!
  \fn void set_material_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager, char const* material)
  \param phantom_creator_manager - pointer on the singleton
  \param material - name of the material
  \brief set the material of the global (background phantom)
*/
extern "C" GGEMS_EXPORT void set_material_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager, char const* material);

/*!
  \fn void set_data_type_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager, char const* data_type)
  \param phantom_creator_manager - pointer on the singleton
  \param data_type - type of data
  \brief set the type of data
*/
extern "C" GGEMS_EXPORT void set_data_type_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager, char const* data_type);

#endif // GUARD_GGEMS_GEOMETRY_GGEMSPHANTOMCREATORMANAGER_HH
