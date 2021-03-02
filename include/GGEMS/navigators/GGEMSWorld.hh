#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSWORLD_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSWORLD_HH

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
  \file GGEMSWorld.hh

  \brief GGEMS class handling global world (space between navigators) in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 11, 2021
*/

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \class GGEMSWorld
  \brief GGEMS class handling global world (space between navigators) in GGEMS
*/
class GGEMS_EXPORT GGEMSWorld
{
  public:
    /*!
      \brief GGEMSWorld constructor
    */
    GGEMSWorld(void);

    /*!
      \brief GGEMSWorld destructor
    */
    ~GGEMSWorld(void);

    /*!
      \fn GGEMSWorld(GGEMSWorld const& world) = delete
      \param world - reference on the GGEMS world
      \brief Avoid copy by reference
    */
    GGEMSWorld(GGEMSWorld const& world) = delete;

    /*!
      \fn GGEMSWorld& operator=(GGEMSWorld const& world) = delete
      \param world - reference on the GGEMS world
      \brief Avoid assignement by reference
    */
    GGEMSWorld& operator=(GGEMSWorld const& world) = delete;

    /*!
      \fn GGEMSWorld(GGEMSWorld const&& world) = delete
      \param world - rvalue reference on the GGEMS world
      \brief Avoid copy by rvalue reference
    */
    GGEMSWorld(GGEMSWorld const&& world) = delete;

    /*!
      \fn GGEMSWorld& operator=(GGEMSWorld const&& world) = delete
      \param world - rvalue reference on the GGEMS world
      \brief Avoid copy by rvalue reference
    */
    GGEMSWorld& operator=(GGEMSWorld const&& world) = delete;

    /*!
      \fn void SetDimension(GGint const& dimension_x, GGfloat const& dimension_y, GGfloat const& dimension_z)
      \param dimension_x - dimension in X
      \param dimension_y - dimension in Y
      \param dimension_z - dimension in Z
      \brief set the dimension of the world in X, Y and Z
    */
    void SetDimension(GGint const& dimension_x, GGint const& dimension_y, GGint const& dimension_z);

    /*!
      \fn void SetElementSize(GGfloat const& size_x, GGfloat const& size_y, GGfloat const& size_z, std::string const& unit = "mm")
      \param size_x - size in X
      \param size_y - size in Y
      \param size_z - size in Z
      \param unit - unit of the distance
      \brief set the size of elements of the world in X, Y and Z
    */
    void SetElementSize(GGfloat const& size_x, GGfloat const& size_y, GGfloat const& size_z, std::string const& unit = "mm");

    /*!
      \fn void Initialize(void)
      \brief initialize and check parameters for world
    */
    void Initialize(void);

  private:
    /*!
      \fn void CheckParameters(void) const
      \brief check parameters for world volume
    */
    void CheckParameters(void) const;

  private:
    GGint3 dimensions_; /*!< Dimensions of world */
    GGfloat3 sizes_; /*!< Sizes of elements in world */
};

/*!
  \fn GGEMSWorld* create_ggems_world(void)
  \return the pointer on the world
  \brief Get the GGEMSWorld pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSWorld* create_ggems_world(void);

/*!
  \fn void set_dimension_ggems_world(GGEMSWorld* world, GGint const dimension_x, GGint const dimension_y, GGint const dimension_z)
  \param world - pointer on world volume
  \param dimension_x - dimension in X
  \param dimension_y - dimension in Y
  \param dimension_z - dimension in Z
  \brief set the dimenstions of the world in X, Y and Z
*/
extern "C" GGEMS_EXPORT void set_dimension_ggems_world(GGEMSWorld* world, GGint const dimension_x, GGint const dimension_y, GGint const dimension_z);

/*!
  \fn void set_size_ggems_world(GGEMSWorld* world, GGfloat const size_x, GGfloat const size_y, GGfloat const size_z, char const* unit)
  \param world - pointer on world volume
  \param size_x - size of X elements of world
  \param size_y - size of Y elements of world
  \param size_z - size of Z elements of world
  \param unit - unit of the distance
  \brief set the element sizes of the world
*/
extern "C" GGEMS_EXPORT void set_size_ggems_world(GGEMSWorld* world, GGfloat const size_x, GGfloat const size_y, GGfloat const size_z, char const* unit);

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSWORLD_HH
