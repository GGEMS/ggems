#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSVOXELIZEDPHANTOM_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSVOXELIZEDPHANTOM_HH

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
  \file GGEMSVoxelizedPhantom.hh

  \brief Child GGEMS class handling voxelized phantom

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Tuesday October 20, 2020
*/

#include "GGEMS/navigators/GGEMSNavigator.hh"

/*!
  \class GGEMSVoxelizedPhantom
  \brief Child GGEMS class handling voxelized phantom
*/
class GGEMS_EXPORT GGEMSVoxelizedPhantom : public GGEMSNavigator
{
  public:
    /*!
      \param voxelized_phantom_name - name of the voxelized phantom
      \brief GGEMSVoxelizedPhantom constructor
    */
    explicit GGEMSVoxelizedPhantom(std::string const& voxelized_phantom_name);

    /*!
      \brief GGEMSVoxelizedPhantom destructor
    */
    ~GGEMSVoxelizedPhantom(void);

    /*!
      \fn GGEMSVoxelizedPhantom(GGEMSVoxelizedPhantom const& voxelized_phantom) = delete
      \param voxelized_phantom - reference on the GGEMS voxelized phantom
      \brief Avoid copy by reference
    */
    GGEMSVoxelizedPhantom(GGEMSVoxelizedPhantom const& voxelized_phantom) = delete;

    /*!
      \fn GGEMSVoxelizedPhantom& operator=(GGEMSVoxelizedPhantom const& voxelized_phantom) = delete
      \param voxelized_phantom - reference on the GGEMS voxelized phantom
      \brief Avoid assignement by reference
    */
    GGEMSVoxelizedPhantom& operator=(GGEMSVoxelizedPhantom const& voxelized_phantom) = delete;

    /*!
      \fn GGEMSVoxelizedPhantom(GGEMSVoxelizedPhantom const&& voxelized_phantom) = delete
      \param voxelized_phantom - rvalue reference on the GGEMS voxelized phantom
      \brief Avoid copy by rvalue reference
    */
    GGEMSVoxelizedPhantom(GGEMSVoxelizedPhantom const&& voxelized_phantom) = delete;

    /*!
      \fn GGEMSVoxelizedPhantom& operator=(GGEMSVoxelizedPhantom const&& voxelized_phantom) = delete
      \param voxelized_phantom - rvalue reference on the GGEMS voxelized phantom
      \brief Avoid copy by rvalue reference
    */
    GGEMSVoxelizedPhantom& operator=(GGEMSVoxelizedPhantom const&& voxelized_phantom) = delete;

    /*!
      \fn void SetPhantomFile(std::string const& voxelized_phantom_filename, std::string const& range_data_filename)
      \param voxelized_phantom_filename - MHD filename for voxelized phantom
      \param range_data_filename - text file with range to material data
      \brief set the mhd filename for voxelized phantom and the range data file
    */
    void SetPhantomFile(std::string const& voxelized_phantom_filename, std::string const& range_data_filename);

    /*!
      \fn void Initialize(void) override
      \brief Initialize the voxelized phantom
    */
    void Initialize(void) override;

    /*!
      \fn void SaveResults
      \brief save all results from solid
    */
    void SaveResults(void) override;

  private:
    /*!
      \fn void CheckParameters(void) const
      \return no returned value
    */
    void CheckParameters(void) const override;

  private:
    std::string voxelized_phantom_filename_; /*!< MHD file storing the voxelized phantom */
    std::string range_data_filename_; /*!< File for label to material matching */
};

/*!
  \fn GGEMSVoxelizedPhantom* create_ggems_voxelized_phantom(char const* voxelized_phantom_name)
  \param voxelized_phantom_name - name of voxelized phantom
  \return the pointer on the voxelized phantom
  \brief Get the GGEMSVoxelizedPhantom pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSVoxelizedPhantom* create_ggems_voxelized_phantom(char const* voxelized_phantom_name);

/*!
  \fn void set_phantom_file_ggems_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, char const* phantom_filename, char const* range_data_filename)
  \param voxelized_phantom - pointer on voxelized_phantom
  \param phantom_filename - filename of the voxelized phantom
  \param range_data_filename - range to material filename
  \brief set the filename of voxelized phantom and the range data file
*/
extern "C" GGEMS_EXPORT void set_phantom_file_ggems_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, char const* phantom_filename, char const* range_data_filename);

/*!
  \fn void set_position_ggems_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, GGfloat const position_x, GGfloat const position_y, GGfloat const position_z, char const* unit)
  \param voxelized_phantom - pointer on voxelized phantom
  \param position_x - offset in X
  \param position_y - offset in Y
  \param position_z - offset in Z
  \param unit - unit of the distance
  \brief set the position of the voxelized phantom in X, Y and Z
*/
extern "C" GGEMS_EXPORT void set_position_ggems_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, GGfloat const position_x, GGfloat const position_y, GGfloat const position_z, char const* unit);

/*!
  \fn void set_rotation_ggems_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, GGfloat const rx, GGfloat const ry, GGfloat const rz, char const* unit)
  \param voxelized_phantom - pointer on voxelized phantom
  \param rx - Rotation around X along local axis
  \param ry - Rotation around Y along local axis
  \param rz - Rotation around Z along local axis
  \param unit - unit of the angle
  \brief Set the rotation of the voxelized phantom around local axis
*/
extern "C" GGEMS_EXPORT void set_rotation_ggems_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, GGfloat const rx, GGfloat const ry, GGfloat const rz, char const* unit);

/*!
  \fn void set_visible_ggems_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, bool const flag)
  \param voxelized_phantom - pointer on voxelized phantom
  \param flag - flag drawing voxelized phantom
  \brief Set flag drawing voxelized phantom
*/
extern "C" GGEMS_EXPORT void set_visible_ggems_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, bool const flag);

/*!
  \fn void set_material_visible_ggems_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, char const* material_name, bool const flag)
  \param voxelized_phantom - pointer on voxelized phantom
  \param material_name - name of material to draw (or not)
  \param flag - flag drawing voxelized phantom
  \brief Set flag for each material to draw
*/
extern "C" GGEMS_EXPORT void set_material_visible_ggems_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, char const* material_name, bool const flag);

/*!
  \fn void set_material_color_ggems_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, char const* material_name, unsigned char const red, unsigned char const green, unsigned char const blue)
  \param voxelized_phantom - pointer on voxelized phantom
  \param material_name - name of material to draw (or not)
  \param red - red value
  \param green - green value
  \param blue - blue value
  \brief Set a new rgb color for a material
*/
extern "C" GGEMS_EXPORT void set_material_color_ggems_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, char const* material_name, unsigned char const red, unsigned char const green, unsigned char const blue);

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSVOXELIZEDPHANTOM_HH
