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

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/global/GGEMSOpenCLManager.hh"

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

  public:
    /*!
      \fn GGEMSPhantomCreatorManager(GGEMSPhantomCreatorManager const& opencl_manager) = delete
      \param opencl_manager - reference on the singleton
      \brief Avoid copy of the singleton by reference
    */
    GGEMSPhantomCreatorManager(
      GGEMSPhantomCreatorManager const& phantom_creator_manager) = delete;

    /*!
      \fn GGEMSPhantomCreatorManager& operator=(GGEMSPhantomCreatorManager const& opencl_manager) = delete
      \param opencl_manager - reference on the singleton
      \brief Avoid assignement of the singleton by reference
    */
    GGEMSPhantomCreatorManager& operator=(
      GGEMSPhantomCreatorManager const& phantom_creator_manager) = delete;

    /*!
      \fn GGEMSOpenCLManager(GGEMSPhantomCreatorManager const&& opencl_manager) = delete
      \param opencl_manager - rvalue reference on the singleton
      \brief Avoid copy of the singleton by rvalue reference
    */
    GGEMSPhantomCreatorManager(
      GGEMSPhantomCreatorManager const&& phantom_creator_manager) = delete;

    /*!
      \fn GGEMSOpenCLManager& operator=(GGEMSOpenCLManager const&& opencl_manager) = delete
      \param opencl_manager - rvalue reference on the singleton
      \brief Avoid copy of the singleton by rvalue reference
    */
    GGEMSPhantomCreatorManager& operator=(
      GGEMSPhantomCreatorManager const&& phantom_creator_manager) = delete;

  public:
    /*!
      \fn void SetElementSizes(double const& voxel_width, double const& voxel_height, double const& voxel_depth)
      \param voxel_width - voxel width
      \param voxel_height - voxel height
      \param voxel_depth - voxel depth
      \brief Set the size of the elements for the voxelized phantom
    */
    void SetElementSizes(GGdouble const& voxel_width,
      GGdouble const& voxel_height, GGdouble const& voxel_depth);

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
    void SetPhantomDimensions(GGuint const& phantom_width,
      GGuint const& phantom_height, GGuint const& phantom_depth);

    /*!
      \fn GGuint3 GetPhantomDimensions(void) const
      \return a 3d int with the dimenstion of the voxelized phantom
      \brief dimensions of phantom
    */
    inline GGuint3 GetPhantomDimensions(void) const
    {
      return phantom_dimensions_;
    };

    /*!
      \fn void SetIsocenterPositions(GGdouble const& iso_pos_x, GGdouble const& iso_pos_y, GGdouble const& iso_pos_z)
      \param iso_pos_x - Isocenter position in X
      \param iso_pos_x - Isocenter position in X
      \param iso_pos_x - Isocenter position in X
      \brief Set isocenter position of the phantom
    */
    void SetIsocenterPositions(GGdouble const& iso_pos_x,
      GGdouble const& iso_pos_y, GGdouble const& iso_pos_z);

    /*!
      \fn GGdouble3 GetIsocenterPositions(void) const
      \return double buffer with offsets
      \brief return offset of phantom taking account isocenter position
    */
    inline GGdouble3 GetOffsets(void) const {return offsets_;}

    /*!
      \fn GGulong GetNumberElements(void)
      \return number of voxel in the voxelized phantom
      \brief Return the total number of voxels
    */
    inline GGulong GetNumberElements(void) const {return number_elements_;}

    /*!
      \fn cl::Buffer* GetVoxelizedPhantom(void) const
      \return pointer on OpenCL device storing voxelized phantom
      \brief Return the voxelized phantom on OpenCL device
    */
    inline cl::Buffer* GetVoxelizedPhantom(void) const
    {
      return p_voxelized_phantom_;
    }

    /*!
      \fn void SetOutputBasename(char const* output_basename, char const* format)
      \param output_basename - output basename
      \param format - format of the output file
      \brief Set the basename of MHD output
    */
    void SetOutputBasename(char const* output_basename, char const* format);

    /*!
      \fn void Initialize(void)
      \brief Initialize the Phantom Creator manager
    */
    void Initialize(void);

    /*!
      \fn void Write(void) const
      \brief Save the voxelized phantom to raw data in mhd file
    */
    void Write(void) const;

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

  private:
    GGdouble3 element_sizes_; /*!< Size of voxels of voxelized phantom */
    GGuint3 phantom_dimensions_; /*!< Dimension of phantom X, Y, Z */
    GGulong number_elements_; /*!< Total number of elements */
    GGdouble3 offsets_; /*!< Offset of the phantom taking account of isocenter position */
    GGdouble3 isocenter_position_; /*!< Isocenter position of the phantom */
    std::string output_basename_; /*!< Output MHD where is stored the voxelized phantom */
    std::string format_; /*!< Format of the output file */

  private: // OpenCL buffer
    cl::Buffer* p_voxelized_phantom_; /*!< Voxelized phantom on OpenCL device */

  private:
    GGEMSOpenCLManager& opencl_manager_; /*!< Reference to opencl manager singleton */
};

/*!
  \fn GGEMSPhantomCreatorManager* get_instance_ggems_phantom_creator_manager(void)
  \brief Get the GGEMSOpenCLManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSPhantomCreatorManager*
  get_instance_phantom_creator_manager(void);

/*!
  \fn void set_phantom_dimension_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager, GGuint const phantom_width, GGuint const phantom_height, GGuint const phantom_depth)
  \param phantom_creator_manager - pointer on the singleton
  \param phantom_width - phantom width
  \param phantom_height - phatom height
  \param phantom_depth - phantom depth
  \brief Set the dimension of the phantom for the voxelized phantom
*/
extern "C" GGEMS_EXPORT void set_phantom_dimension_phantom_creator_manager(
  GGEMSPhantomCreatorManager* phantom_creator_manager,
  GGuint const phantom_width, GGuint const phantom_height,
  GGuint const phantom_depth);

/*!
  \fn void set_element_sizes_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager, GGdouble const voxel_width, GGdouble const voxel_height, GGdouble const voxel_depth)
  \param phantom_creator_manager - pointer on the singleton
  \param voxel_width - voxel width
  \param voxel_height - voxel height
  \param voxel_depth - voxel depth
  \brief Set the size of the elements for the voxelized phantom
*/
extern "C" GGEMS_EXPORT void set_element_sizes_phantom_creator_manager(
  GGEMSPhantomCreatorManager* phantom_creator_manager,
  GGdouble const voxel_width, GGdouble const voxel_height,
  GGdouble const voxel_depth);

/*!
  \fn void set_isocenter_positions(GGEMSPhantomCreatorManager* phantom_creator_manager, GGdouble const& iso_pos_x, GGdouble const& iso_pos_y, GGdouble const& iso_pos_z)
  \param iso_pos_x - Isocenter position in X
  \param iso_pos_x - Isocenter position in X
  \param iso_pos_x - Isocenter position in X
  \brief Set isocenter position of the phantom
*/
extern "C" GGEMS_EXPORT void set_isocenter_positions(
  GGEMSPhantomCreatorManager* phantom_creator_manager,
  GGdouble const iso_pos_x, GGdouble const iso_pos_y,
  GGdouble const iso_pos_z);

/*!
  \fn void set_output_basename_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager, char const* output_basename)
  \param phantom_creator_manager - pointer on the singleton
  \param output_MHD_basename - output MHD basename
  \brief Set the basename of MHD output
*/
extern "C" GGEMS_EXPORT void set_output_basename_phantom_creator_manager(
  GGEMSPhantomCreatorManager* phantom_creator_manager,
  char const* output_basename, char const* format);

/*!
  \fn void initialize_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager)
  \param phantom_creator_manager - pointer on the singleton
  \brief Initialize the phantom creator manager
*/
extern "C" GGEMS_EXPORT void initialize_phantom_creator_manager(
  GGEMSPhantomCreatorManager* phantom_creator_manager);

/*!
  \fn void write_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager)
  \param phantom_creator_manager - pointer on the singleton
  \brief Save the voxelized phantom to raw data in mhd file
*/
extern "C" GGEMS_EXPORT void write_phantom_creator_manager(
  GGEMSPhantomCreatorManager* phantom_creator_manager);

#endif // GUARD_GGEMS_GEOMETRY_GGEMSPHANTOMCREATORMANAGER_HH
