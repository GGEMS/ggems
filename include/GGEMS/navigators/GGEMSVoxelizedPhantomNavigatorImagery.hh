#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSVOXELIZEDPHANTOMNAVIGATORIMAGERY_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSVOXELIZEDPHANTOMNAVIGATORIMAGERY_HH

/*!
  \file GGEMSVoxelizedPhantomNavigatorImagery.hh

  \brief GGEMS class managing voxelized phantom navigator for imagery application

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday February 11, 2020
*/

#include "GGEMS/navigators/GGEMSPhantomNavigator.hh"

/*!
  \class GGEMSVoxelizedPhantomNavigatorImagery
  \brief GGEMS class managing voxelized phantom navigator for imagery application
*/
class GGEMS_EXPORT GGEMSVoxelizedPhantomNavigatorImagery : public GGEMSPhantomNavigator
{
  public:
    /*!
      \brief GGEMSVoxelizedPhantomNavigatorImagery constructor
    */
    GGEMSVoxelizedPhantomNavigatorImagery(void);

    /*!
      \brief GGEMSVoxelizedPhantomNavigatorImagery destructor
    */
    ~GGEMSVoxelizedPhantomNavigatorImagery(void);

    /*!
      \fn GGEMSVoxelizedPhantomNavigatorImagery(GGEMSVoxelizedPhantomNavigatorImagery const& voxelized_phantom_navigator_imagery) = delete
      \param voxelized_phantom_navigator_imagery - reference on the GGEMS voxelized phantom navigator imagery
      \brief Avoid copy by reference
    */
    GGEMSVoxelizedPhantomNavigatorImagery(GGEMSVoxelizedPhantomNavigatorImagery const& voxelized_phantom_navigator_imagery) = delete;

    /*!
      \fn GGEMSVoxelizedPhantomNavigatorImagery& operator=(GGEMSVoxelizedPhantomNavigatorImagery const& voxelized_phantom_navigator_imagery) = delete
      \param voxelized_phantom_navigator_imagery - reference on the GGEMS voxelized phantom navigator imagery
      \brief Avoid assignement by reference
    */
    GGEMSVoxelizedPhantomNavigatorImagery& operator=(GGEMSVoxelizedPhantomNavigatorImagery const& voxelized_phantom_navigator_imagery) = delete;

    /*!
      \fn GGEMSVoxelizedPhantomNavigatorImagery(GGEMSVoxelizedPhantomNavigatorImagery const&& voxelized_phantom_navigator_imagery) = delete
      \param voxelized_phantom_navigator_imagery - rvalue reference on the GGEMS voxelized phantom navigator imagery
      \brief Avoid copy by rvalue reference
    */
    GGEMSVoxelizedPhantomNavigatorImagery(GGEMSVoxelizedPhantomNavigatorImagery const&& voxelized_phantom_navigator_imagery) = delete;

    /*!
      \fn GGEMSVoxelizedPhantomNavigatorImagery& operator=(GGEMSVoxelizedPhantomNavigatorImagery const&& voxelized_phantom_navigator_imagery) = delete
      \param voxelized_phantom_navigator_imagery - rvalue reference on the GGEMS voxelized phantom navigator imagery
      \brief Avoid copy by rvalue reference
    */
    GGEMSVoxelizedPhantomNavigatorImagery& operator=(GGEMSVoxelizedPhantomNavigatorImagery const&& voxelized_phantom_navigator_imagery) = delete;

    /*!
      \fn void Initialize(void)
      \brief Initialize a GGEMS phantom
    */
    void Initialize(void) override;

    /*!
      \fn void PrintInfos(void) const
      \brief Printing infos about the phantom navigator
    */
    void PrintInfos(void) const override;
};

/*!
  \fn GGEMSVoxelizedPhantomNavigatorImagery* create_ggems_voxelized_phantom_navigator_imagery(void)
  \return the pointer on the singleton
  \brief Get the GGEMSVoxelizedPhantomNavigatorImagery pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSVoxelizedPhantomNavigatorImagery* create_ggems_voxelized_phantom_navigator_imagery(void);

/*!
  \fn void set_phantom_name_ggems_voxelized_phantom_navigator_imagery(GGEMSVoxelizedPhantomNavigatorImagery* voxelized_phantom_navigator_imagery, char const* phantom_navigator_name)
  \param voxelized_phantom_navigator_imagery - pointer on the navigator
  \param phantom_navigator_name - name of the phantom
  \brief set the name of navigator
*/
extern "C" GGEMS_EXPORT void set_phantom_name_ggems_voxelized_phantom_navigator_imagery(GGEMSVoxelizedPhantomNavigatorImagery* voxelized_phantom_navigator_imagery, char const* phantom_navigator_name);

/*!
  \fn void set_phantom_file_ggems_voxelized_phantom_navigator_imagery(GGEMSVoxelizedPhantomNavigatorImagery* voxelized_phantom_navigator_imagery, char const* phantom_filename)
  \param voxelized_phantom_navigator_imagery - pointer on the navigator
  \param phantom_filename - filename of the phantom
  \brief set the filename of phantom
*/
extern "C" GGEMS_EXPORT void set_phantom_file_ggems_voxelized_phantom_navigator_imagery(GGEMSVoxelizedPhantomNavigatorImagery* voxelized_phantom_navigator_imagery, char const* phantom_filename);

/*!
  \fn void set_range_to_material_filename_ggems_voxelized_phantom_navigator_imagery(GGEMSVoxelizedPhantomNavigatorImagery* voxelized_phantom_navigator_imagery, char const* range_data_filename)
  \param voxelized_phantom_navigator_imagery - pointer on the navigator
  \param range_data_filename - range to material filename
  \brief set the filename of range to material data
*/
extern "C" GGEMS_EXPORT void set_range_to_material_filename_ggems_voxelized_phantom_navigator_imagery(GGEMSVoxelizedPhantomNavigatorImagery* voxelized_phantom_navigator_imagery, char const* range_data_filename);

/*!
  \fn void set_geometry_tolerance_ggems_voxelized_phantom_navigator_imagery(GGEMSVoxelizedPhantomNavigatorImagery* voxelized_phantom_navigator_imagery, GGfloat const distance, char const* unit)
  \param voxelized_phantom_navigator_imagery - pointer on the navigator
  \param distance - distance for the geometry tolerance
  \param unit - unit of the distance
  \brief set the filename of range to material data
*/
extern "C" GGEMS_EXPORT void set_geometry_tolerance_ggems_voxelized_phantom_navigator_imagery(GGEMSVoxelizedPhantomNavigatorImagery* voxelized_phantom_navigator_imagery, GGfloat const distance, char const* unit);

/*!
  \fn void set_offset_ggems_voxelized_phantom_navigator_imagery(GGEMSVoxelizedPhantomNavigatorImagery* voxelized_phantom_navigator_imagery, GGfloat const offset_x, GGfloat const offset_y, GGfloat const offset_z, char const* unit)
  \param voxelized_phantom_navigator_imagery - pointer on the navigator
  \param offset_x - offset in X
  \param offset_y - offset in Y
  \param offset_z - offset in Z
  \param unit - unit of the distance
  \brief set the offset of the phantom in X, Y and Z
*/
extern "C" GGEMS_EXPORT void set_offset_ggems_voxelized_phantom_navigator_imagery(GGEMSVoxelizedPhantomNavigatorImagery* voxelized_phantom_navigator_imagery, GGfloat const offset_x, GGfloat const offset_y, GGfloat const offset_z, char const* unit);

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSVOXELIZEDPHANTOMNAVIGATORIMAGERY_HH
