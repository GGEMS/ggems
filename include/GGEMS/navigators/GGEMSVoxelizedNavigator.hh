#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSVOXELIZEDNAVIGATOR_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSVOXELIZEDNAVIGATOR_HH

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
  \file GGEMSVoxelizedNavigator.hh

  \brief GGEMS class managing voxelized navigator

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday February 11, 2020
*/

// #include "GGEMS/navigators/GGEMSNavigator.hh"

// /*!
//   \class GGEMSVoxelizedNavigator
//   \brief GGEMS class managing voxelized navigator
// */
// class GGEMS_EXPORT GGEMSVoxelizedNavigator : public GGEMSNavigator
// {
//   public:
//     /*!
//       \brief GGEMSVoxelizedNavigator constructor
//     */
//     GGEMSVoxelizedNavigator(void);

//     /*!
//       \brief GGEMSVoxelizedNavigator destructor
//     */
//     ~GGEMSVoxelizedNavigator(void);

//     /*!
//       \fn GGEMSVoxelizedNavigator(GGEMSVoxelizedNavigator const& voxelized_navigator) = delete
//       \param voxelized_navigator - reference on the GGEMS voxelized navigator
//       \brief Avoid copy by reference
//     */
//     GGEMSVoxelizedNavigator(GGEMSVoxelizedNavigator const& voxelized_navigator) = delete;

//     /*!
//       \fn GGEMSVoxelizedNavigator& operator=(GGEMSVoxelizedNavigator const& voxelized_navigator) = delete
//       \param voxelized_navigator - reference on the GGEMS voxelized navigator
//       \brief Avoid assignement by reference
//     */
//     GGEMSVoxelizedNavigator& operator=(GGEMSVoxelizedNavigator const& voxelized_navigator) = delete;

//     /*!
//       \fn GGEMSVoxelizedNavigator(GGEMSVoxelizedNavigator const&& voxelized_navigator) = delete
//       \param voxelized_navigator - rvalue reference on the GGEMS voxelized navigator
//       \brief Avoid copy by rvalue reference
//     */
//     GGEMSVoxelizedNavigator(GGEMSVoxelizedNavigator const&& voxelized_navigator) = delete;

//     /*!
//       \fn GGEMSVoxelizedNavigator& operator=(GGEMSVoxelizedNavigator const&& voxelized_navigator) = delete
//       \param voxelized_navigator - rvalue reference on the GGEMS voxelized navigator
//       \brief Avoid copy by rvalue reference
//     */
//     GGEMSVoxelizedNavigator& operator=(GGEMSVoxelizedNavigator const&& voxelized_navigator) = delete;

//     /*!
//       \fn void SetPhantomFile(std::string const& filename)
//       \param filename - filename of MHD file for phantom
//       \brief set the mhd filename for phantom
//     */
//     void SetPhantomFile(std::string const& filename);

//     /*!
//       \fn void SetRangeToMaterialFile(std::string const& range_data_filename)
//       \param range_data_filename - filename with range to material data
//       \brief set the range to material filename
//     */
//     void SetRangeToMaterialFile(std::string const& range_data_filename);

//     /*!
//       \fn void Initialize(void)
//       \brief Initialize a GGEMS phantom
//     */
//     void Initialize(void) override;

//     /*!
//       \fn void PrintInfos(void) const
//       \brief Printing infos about the phantom navigator
//     */
//     void PrintInfos(void) const override;

//     /*!
//       \fn void CheckParameters(void) const
//       \brief Checking parameters for voxelized navigator
//     */
//     void CheckParameters(void) const override;

//   private:
//     std::string phantom_mhd_header_filename_; /*!< Filename of MHD file for phantom */
//     std::string range_data_filename_; /*!< Filename of file for range data */
// };

// /*!
//   \fn GGEMSVoxelizedNavigator* create_ggems_voxelized_navigator(void)
//   \return the pointer on the singleton
//   \brief Get the GGEMSVoxelizedNavigator pointer for python user.
// */
// extern "C" GGEMS_EXPORT GGEMSVoxelizedNavigator* create_ggems_voxelized_navigator(void);

// /*!
//   \fn void set_phantom_name_ggems_voxelized_navigator(GGEMSVoxelizedNavigator* voxelized_navigator, char const* phantom_navigator_name)
//   \param voxelized_navigator - pointer on the navigator
//   \param phantom_navigator_name - name of the phantom
//   \brief set the name of navigator
// */
// extern "C" GGEMS_EXPORT void set_phantom_name_ggems_voxelized_navigator(GGEMSVoxelizedNavigator* voxelized_navigator, char const* phantom_navigator_name);

// /*!
//   \fn void set_phantom_file_ggems_voxelized_navigator(GGEMSVoxelizedNavigator* voxelized_navigator, char const* phantom_filename)
//   \param voxelized_navigator - pointer on the navigator
//   \param phantom_filename - filename of the phantom
//   \brief set the filename of phantom
// */
// extern "C" GGEMS_EXPORT void set_phantom_file_ggems_voxelized_navigator(GGEMSVoxelizedNavigator* voxelized_navigator, char const* phantom_filename);

// /*!
//   \fn void set_range_to_material_filename_ggems_voxelized_navigator(GGEMSVoxelizedNavigator* voxelized_navigator, char const* range_data_filename)
//   \param voxelized_navigator - pointer on the navigator
//   \param range_data_filename - range to material filename
//   \brief set the filename of range to material data
// */
// extern "C" GGEMS_EXPORT void set_range_to_material_filename_ggems_voxelized_navigator(GGEMSVoxelizedNavigator* voxelized_navigator, char const* range_data_filename);

// /*!
//   \fn void set_geometry_tolerance_ggems_voxelized_navigator(GGEMSVoxelizedNavigator* voxelized_navigator, GGfloat const distance, char const* unit)
//   \param voxelized_navigator - pointer on the navigator
//   \param distance - distance for the geometry tolerance
//   \param unit - unit of the distance
//   \brief set the filename of range to material data
// */
// extern "C" GGEMS_EXPORT void set_geometry_tolerance_ggems_voxelized_navigator(GGEMSVoxelizedNavigator* voxelized_navigator, GGfloat const distance, char const* unit);

// /*!
//   \fn void set_position_ggems_voxelized_navigator(GGEMSVoxelizedNavigator* voxelized_navigator, GGfloat const position_x, GGfloat const position_y, GGfloat const position_z, char const* unit)
//   \param voxelized_navigator - pointer on the navigator
//   \param position_x - offset in X
//   \param position_y - offset in Y
//   \param position_z - offset in Z
//   \param unit - unit of the distance
//   \brief set the position of the phantom in X, Y and Z
// */
// extern "C" GGEMS_EXPORT void set_position_ggems_voxelized_navigator(GGEMSVoxelizedNavigator* voxelized_navigator, GGfloat const position_x, GGfloat const position_y, GGfloat const position_z, char const* unit);

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSVOXELIZEDNAVIGATOR_HH
