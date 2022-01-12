#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSSPHERE_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSSPHERE_HH

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
  \file GGEMSSphere.hh

  \brief Class GGEMSSphere inheriting from GGEMSVolume handling Sphere solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday November 4, 2020
*/

#include "GGEMS/geometries/GGEMSVolume.hh"

/*!
  \class GGEMSSphere
  \brief Class GGEMSSphere inheriting from GGEMSVolume handling Sphere solid
*/
class GGEMS_EXPORT GGEMSSphere : public GGEMSVolume
{
  public:
    /*!
      \param radius - Radius of the sphere
      \param unit - Unit of distance
      \brief GGEMSSphere constructor
    */
    GGEMSSphere(GGfloat const& radius, std::string const& unit = "mm");

    /*!
      \brief GGEMSSphere destructor
    */
    ~GGEMSSphere(void) override;

    /*!
      \fn GGEMSSphere(GGEMSSphere const& sphere) = delete
      \param sphere - reference on the sphere solid volume
      \brief Avoid copy of the class by reference
    */
    GGEMSSphere(GGEMSSphere const& sphere) = delete;

    /*!
      \fn GGEMSSphere& operator=(GGEMSSphere const& sphere) = delete
      \param sphere - reference on the sphere solid volume
      \brief Avoid assignement of the class by reference
    */
    GGEMSSphere& operator=(GGEMSSphere const& sphere) = delete;

    /*!
      \fn GGEMSSphere(GGEMSSphere const&& sphere) = delete
      \param sphere - rvalue reference on the sphere solid volume
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSSphere(GGEMSSphere const&& sphere) = delete;

    /*!
      \fn GGEMSSphere& operator=(GGEMSSphere const&& sphere) = delete
      \param sphere - rvalue reference on the sphere solid volume
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSSphere& operator=(GGEMSSphere const&& sphere) = delete;

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
    GGfloat radius_; /*!< Radius of the sphere */
};

/*!
  \fn GGEMSSphere* create_sphere(GGfloat const radius, char const* unit = "mm")
  \param radius - Radius of the sphere
  \param unit - unit of the distance
  \return the pointer on the singleton
  \brief Create instance of GGEMSSphere
*/
extern "C" GGEMS_EXPORT GGEMSSphere* create_sphere(GGfloat const radius, char const* unit = "mm");

/*!
  \fn GGEMSSphere* delete_sphere(GGEMSSphere* sphere)
  \param sphere - pointer on the solid sphere
  \brief Delete instance of GGEMSSphere
*/
extern "C" GGEMS_EXPORT void delete_sphere(GGEMSSphere* sphere);

/*!
  \fn void set_position_sphere(GGEMSSphere* sphere, GGfloat const pos_x, GGfloat const pos_y, GGfloat const pos_z, char const* unit = "mm")
  \param sphere - pointer on the solid sphere
  \param pos_x - radius of the sphere
  \param pos_y - radius of the sphere
  \param pos_z - radius of the sphere
  \param unit - unit of the distance
  \brief Set the position of the sphere
*/
extern "C" GGEMS_EXPORT void set_position_sphere(GGEMSSphere* sphere, GGfloat const pos_x, GGfloat const pos_y, GGfloat const pos_z, char const* unit = "mm");

/*!
  \fn void set_material_sphere(GGEMSSphere* sphere, char const* material)
  \param sphere - pointer on the solid sphere
  \param material - material of the sphere
  \brief Set the material of the sphere
*/
extern "C" GGEMS_EXPORT void set_material_sphere(GGEMSSphere* sphere, char const* material);

/*!
  \fn void set_label_value_sphere(GGEMSSphere* sphere, GGfloat const label_value)
  \param sphere - pointer on the solid sphere
  \param label_value - label value in sphere
  \brief Set the label value in sphere
*/
extern "C" GGEMS_EXPORT void set_label_value_sphere(GGEMSSphere* sphere, GGfloat const label_value);

/*!
  \fn void initialize_sphere(GGEMSSphere* sphere)
  \param sphere - pointer on the solid sphere
  \brief Initialize the solid and store it in Phantom creator manager
*/
extern "C" GGEMS_EXPORT void initialize_sphere(GGEMSSphere* sphere);

/*!
  \fn void draw_sphere(GGEMSSphere* sphere)
  \param sphere - pointer on the solid sphere
  \brief Draw analytical volume in voxelized phantom
*/
extern "C" GGEMS_EXPORT void draw_sphere(GGEMSSphere* sphere);

#endif // End of GUARD_GGEMS_GEOMETRY_GGEMSSPHERE_HH
