#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSTUBE_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSTUBE_HH

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
  \file GGEMSTube.hh

  \brief Class GGEMSTube inheriting from GGEMSVolume handling Tube solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday January 13, 2020
*/

#include "GGEMS/geometries/GGEMSVolume.hh"

/*!
  \class GGEMSTube
  \brief Class GGEMSTube inheriting from GGEMSVolume handling Tube solid
*/
class GGEMS_EXPORT GGEMSTube : public GGEMSVolume
{
  public:
    /*!
      \param radius_x - Radius of the tube in X axis
      \param radius_y - Radius of the tube in Y axis
      \param height - Height of the tube
      \param unit - Unit of distance
      \brief GGEMSTube constructor
    */
    GGEMSTube(GGfloat const& radius_x, GGfloat const& radius_y, GGfloat const& height, std::string const& unit = "mm");

    /*!
      \brief GGEMSTube destructor
    */
    ~GGEMSTube(void);

    /*!
      \fn GGEMSTube(GGEMSTube const& tube) = delete
      \param tube - reference on the tube solid volume
      \brief Avoid copy of the class by reference
    */
    GGEMSTube(GGEMSTube const& tube) = delete;

    /*!
      \fn GGEMSTube& operator=(GGEMSTube const& tube) = delete
      \param tube - reference on the tube solid volume
      \brief Avoid assignement of the class by reference
    */
    GGEMSTube& operator=(GGEMSTube const& tube) = delete;

    /*!
      \fn GGEMSTube(GGEMSTube const&& tube) = delete
      \param tube - rvalue reference on the tube solid volume
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSTube(GGEMSTube const&& tube) = delete;

    /*!
      \fn GGEMSTube& operator=(GGEMSTube const&& tube) = delete
      \param tube - rvalue reference on the tube solid volume
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSTube& operator=(GGEMSTube const&& tube) = delete;

    /*!
      \fn void Initialize(void) override
      \brief Initialize the solid and store it in Phantom creator manager
    */
    void Initialize(void) override;

    /*!
      \fn void Draw(void)
      \brief Draw analytical volume in voxelized phantom
    */
    void Draw(void) override;

  private:
    GGfloat height_; /*!< Height of the cylinder */
    GGfloat radius_x_; /*!< Radius of the cylinder in X axis */
    GGfloat radius_y_; /*!< Radius of the cylinder in X axis */
};

/*!
  \fn GGEMSTube* create_tube(GGfloat const radius_x, GGfloat const radius_y, GGfloat const height, char const* unit)
  \param radius_x - Radius of the tube in X axis
  \param radius_y - Radius of the tube in Y axis
  \param height - Height of the tube
  \param unit - unit of the distance
  \return the pointer on the singleton
  \brief Create instance of GGEMSTube
*/
extern "C" GGEMS_EXPORT GGEMSTube* create_tube(GGfloat const radius_x, GGfloat const radius_y, GGfloat const height, char const* unit = "mm");

/*!
  \fn GGEMSTube* delete_tube(GGEMSTube* tube)
  \param tube - pointer on the solid tube
  \brief Delete instance of GGEMSTube
*/
extern "C" GGEMS_EXPORT void delete_tube(GGEMSTube* tube);

/*!
  \fn void set_position_tube(GGEMSTube* tube, GGfloat const pos_x, GGfloat const pos_y, GGfloat const pos_z, char const* unit)
  \param tube - pointer on the solid tube
  \param pos_x - radius of the tube
  \param pos_y - radius of the tube
  \param pos_z - radius of the tube
  \param unit - unit of the distance
  \brief Set the position of the tube
*/
extern "C" GGEMS_EXPORT void set_position_tube(GGEMSTube* tube, GGfloat const pos_x, GGfloat const pos_y, GGfloat const pos_z, char const* unit = "mm");

/*!
  \fn void set_material_tube(GGEMSTube* tube, char const* material)
  \param tube - pointer on the solid tube
  \param material - material of the tube
  \brief Set the material of the tube
*/
extern "C" GGEMS_EXPORT void set_material_tube(GGEMSTube* tube, char const* material);

/*!
  \fn void set_label_value_tube(GGEMSTube* tube, GGfloat const label_value)
  \param tube - pointer on the solid tube
  \param label_value - label value in tube
  \brief Set the label value in tube
*/
extern "C" GGEMS_EXPORT void set_label_value_tube(GGEMSTube* tube, GGfloat const label_value);

/*!
  \fn void initialize_tube(GGEMSTube* tube)
  \param tube - pointer on the solid tube
  \brief Initialize the solid and store it in Phantom creator manager
*/
extern "C" GGEMS_EXPORT void initialize_tube(GGEMSTube* tube);

/*!
  \fn void draw_tube(GGEMSTube* tube)
  \param tube - pointer on the solid tube
  \brief Draw analytical volume in voxelized phantom
*/
extern "C" GGEMS_EXPORT void draw_tube(GGEMSTube* tube);

#endif // End of GUARD_GGEMS_GEOMETRY_GGEMSTUBE_HH
