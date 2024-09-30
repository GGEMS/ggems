#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSMESHEDPHANTOM_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSMESHEDPHANTOM_HH

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
  \file GGEMSMeshedPhantom.hh

  \brief Child GGEMS class handling meshed phantom

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Tuesday June 14, 2022
*/

#include "GGEMS/navigators/GGEMSNavigator.hh"

/*!
  \class GGEMSMeshedPhantom
  \brief Child GGEMS class handling meshed phantom
*/
class GGEMS_EXPORT GGEMSMeshedPhantom : public GGEMSNavigator
{
  public:
    /*!
      \param meshed_phantom_name - name of the meshed phantom
      \brief GGEMSMeshedPhantom constructor
    */
    explicit GGEMSMeshedPhantom(std::string const& meshed_phantom_name);

    /*!
      \brief GGEMSMeshedPhantom destructor
    */
    ~GGEMSMeshedPhantom(void) override;

    /*!
      \fn GGEMSMeshedPhantom(GGEMSMeshedPhantom const& meshed_phantom_name) = delete
      \param meshed_phantom_name - reference on the GGEMS meshed phantom
      \brief Avoid copy by reference
    */
    GGEMSMeshedPhantom(GGEMSMeshedPhantom const& meshed_phantom_name) = delete;

    /*!
      \fn GGEMSMeshedPhantom& operator=(GGEMSMeshedPhantom const& meshed_phantom_name) = delete
      \param meshed_phantom_name - reference on the GGEMS meshed phantom
      \brief Avoid assignement by reference
    */
    GGEMSMeshedPhantom& operator=(GGEMSMeshedPhantom const& meshed_phantom_name) = delete;

    /*!
      \fn GGEMSMeshedPhantom(GGEMSMeshedPhantom const&& meshed_phantom_name) = delete
      \param meshed_phantom_name - rvalue reference on the GGEMS meshed phantom
      \brief Avoid copy by rvalue reference
    */
    GGEMSMeshedPhantom(GGEMSMeshedPhantom const&& meshed_phantom_name) = delete;

    /*!
      \fn GGEMSMeshedPhantom& operator=(GGEMSMeshedPhantom const&& meshed_phantom_name) = delete
      \param meshed_phantom_name - rvalue reference on the GGEMS meshed phantom
      \brief Avoid copy by rvalue reference
    */
    GGEMSMeshedPhantom& operator=(GGEMSMeshedPhantom const&& meshed_phantom_name) = delete;

    /*!
      \fn void SetPhantomFile(std::string const& voxelized_phantom_filename, std::string const& range_data_filename)
      \param voxelized_phantom_filename - MHD filename for voxelized phantom
      \param range_data_filename - text file with range to material data
      \brief set the mhd filename for voxelized phantom and the range data file
    */
    void SetPhantomFile(std::string const& meshed_phantom_filename);

    /*!
      \fn void SetMeshOctreeDepth(GGint const& depth)
      \param depth - depth of octree
      \brief set the mesh octree depth
    */
    void SetMeshOctreeDepth(GGint const& depth);

    /*!
      \fn void Initialize(void) override
      \brief Initialize the meshed phantom
    */
    void Initialize(void) override;

    /*!
      \fn void SaveResults
      \brief save all results from solid
    */
    void SaveResults(void) override;

  private:
    /*!
      \fn void CheckParameters(void) const override
      \brief checking parameters
    */
    void CheckParameters(void) const override;

  private:
    std::string meshed_phantom_filename_; /*!< Mesh file storing the meshed phantom */
    GGint mesh_octree_depth_; /*!< Depth of octree for mesh */
};

/*!
  \fn GGEMSMeshedPhantom* create_ggems_meshed_phantom(char const* meshed_phantom_name)
  \param meshed_phantom_name - name of meshed phantom
  \return the pointer on the meshed phantom
  \brief Get the GGEMSMeshedPhantom pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSMeshedPhantom* create_ggems_meshed_phantom(char const* meshed_phantom_name);

/*!
  \fn void set_phantom_file_ggems_meshed_phantom(GGEMSMeshedPhantom* meshed_phantom, char const* phantom_filename)
  \param meshed_phantom - pointer on meshed_phantom
  \param phantom_filename - filename of the meshed phantom
  \brief set the filename of meshed phantom
*/
extern "C" GGEMS_EXPORT void set_phantom_file_ggems_meshed_phantom(GGEMSMeshedPhantom* meshed_phantom, char const* phantom_filename);

/*!
  \fn void set_position_ggems_meshed_phantom(GGEMSMeshedPhantom* meshed_phantom, GGfloat const position_x, GGfloat const position_y, GGfloat const position_z, char const* unit)
  \param meshed_phantom - pointer on meshed phantom
  \param position_x - offset in X
  \param position_y - offset in Y
  \param position_z - offset in Z
  \param unit - unit of the distance
  \brief set the position of the meshed phantom in X, Y and Z
*/
extern "C" GGEMS_EXPORT void set_position_ggems_meshed_phantom(GGEMSMeshedPhantom* meshed_phantom, GGfloat const position_x, GGfloat const position_y, GGfloat const position_z, char const* unit);

/*!
  \fn void set_rotation_ggems_meshed_phantom(GGEMSMeshedPhantom* meshed_phantom, GGfloat const rx, GGfloat const ry, GGfloat const rz, char const* unit)
  \param meshed_phantom - pointer on meshed phantom
  \param rx - Rotation around X along local axis
  \param ry - Rotation around Y along local axis
  \param rz - Rotation around Z along local axis
  \param unit - unit of the angle
  \brief Set the rotation of the meshed phantom around local axis
*/
extern "C" GGEMS_EXPORT void set_rotation_ggems_meshed_phantom(GGEMSMeshedPhantom* meshed_phantom, GGfloat const rx, GGfloat const ry, GGfloat const rz, char const* unit);

/*!
  \fn void set_visible_ggems_meshed_phantom(GGEMSMeshedPhantom* meshed_phantom, bool const flag)
  \param meshed_phantom - pointer on meshed phantom
  \param flag - flag drawing meshed phantom
  \brief Set flag drawing meshed phantom
*/
extern "C" GGEMS_EXPORT void set_visible_ggems_meshed_phantom(GGEMSMeshedPhantom* meshed_phantom, bool const flag);

/*!
  \fn void set_material_name_ggems_meshed_phantom(GGEMSMeshedPhantom* meshed_phantom, char const* material_name)
  \param meshed_phantom - pointer on meshed phantom
  \param material_name - name of the material
  \brief set the material name for mesh phantom
*/
extern "C" GGEMS_EXPORT void set_material_name_ggems_meshed_phantom(GGEMSMeshedPhantom* meshed_phantom, char const* material_name);

/*!
  \fn void set_material_color_ggems_meshed_phantom(GGEMSMeshedPhantom* meshed_phantom, char const* material_name, GGuchar const red, GGuchar const green, GGuchar const blue)
  \param meshed_phantom - pointer on meshed phantom
  \param material_name - name of material to draw (or not)
  \param red - red value
  \param green - green value
  \param blue - blue value
  \brief Set a new rgb color for a material
*/
extern "C" GGEMS_EXPORT void set_material_color_ggems_meshed_phantom(GGEMSMeshedPhantom* meshed_phantom, char const* material_name, GGuchar const red, GGuchar const green, GGuchar const blue);

/*!
  \fn void set_material_color_name_ggems_meshed_phantom(GGEMSMeshedPhantom* ct_system, char const* material_name, char const* color_name)
  \param meshed_phantom - pointer on meshed phantom
  \param material_name - name of material to draw (or not)
  \param color_name - color name
  \brief Set a color for material
*/
extern "C" GGEMS_EXPORT void set_material_color_name_ggems_meshed_phantom(GGEMSMeshedPhantom* meshed_phantom, char const* material_name, char const* color_name);

/*!
  \fn void set_mesh_octree_depth_ggems_meshed_phantom(GGEMSMeshedPhantom* meshed_phantom, GGint const depth)
  \param meshed_phantom - pointer on meshed phantom
  \param depth - name of material to draw (or not)
  \brief Set depth
*/
extern "C" GGEMS_EXPORT void set_mesh_octree_depth_ggems_meshed_phantom(GGEMSMeshedPhantom* meshed_phantom, GGint const depth);
#endif
