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
      \fn void SetPhantomDimensions(GGuint const& phantom_width, GGuint const& phantom_height, GGuint const& phantom_depth)
      \param phantom_width - phantom width
      \param phantom_height - phatom height
      \param phantom_depth - phantom depth
      \brief Set the dimension of the phantom for the voxelized phantom
    */
    void SetPhantomDimensions(GGuint const& phantom_width,
      GGuint const& phantom_height, GGuint const& phantom_depth);

    /*!
      \fn GGdouble3 GetElementsSizes(void) const
      \return a 3d double with the size of voxel in voxelized phantom
      \brief size of voxels in the voxelized phantom
    */
    inline GGdouble3 GetElementsSizes(void) const {return element_sizes_;};

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
      \fn void SetOutputBasename(char const* output_MHD_basename)
      \param output_MHD_basename - output MHD basename
      \brief Set the basename of MHD output
    */
    void SetOutputBasename(char const* output_MHD_basename);

  private:
    GGdouble3 element_sizes_; /*!< Size of voxels of voxelized phantom */
    GGuint3 phantom_dimensions_; /*!< Dimension of phantom X, Y, Z */
    std::string output_MHD_basename_; /*!< Output MHD where is stored the voxelized phantom */
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
  \fn void set_output_basename_phantom_creator_manager(GGEMSPhantomCreatorManager* phantom_creator_manager, char const* output_basename)
  \param phantom_creator_manager - pointer on the singleton
  \param output_MHD_basename - output MHD basename
  \brief Set the basename of MHD output
*/
extern "C" GGEMS_EXPORT void set_output_basename_phantom_creator_manager(
  GGEMSPhantomCreatorManager* phantom_creator_manager,
  char const* output_MHD_basename);

#endif // GUARD_GGEMS_GEOMETRY_GGEMSPHANTOMCREATORMANAGER_HH
