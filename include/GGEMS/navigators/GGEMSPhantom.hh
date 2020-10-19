#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSPHANTOM_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSPHANTOM_HH

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
  \file GGEMSPhantom.hh

  \brief GGEMS class initializing a phantom and setting type of navigator

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Thrusday October 15, 2020
*/

#include <string>
#include <memory>

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

class GGEMSNavigator;

/*!
  \class GGEMSPhantom
  \brief GGEMS class initializing a phantom and setting type of navigator
*/
class GGEMSPhantom
{
  public:
    /*!
      \brief GGEMSPhantom constructor
    */
    GGEMSPhantom(std::string const& phantom_name, std::string const& phantom_type);

    /*!
      \brief GGEMSPhantom destructor
    */
    ~GGEMSPhantom(void);

    /*!
      \fn GGEMSPhantom(GGEMSPhantom const& phantom) = delete
      \param phantom - reference on the GGEMS phantom
      \brief Avoid copy by reference
    */
    GGEMSPhantom(GGEMSPhantom const& phantom) = delete;

    /*!
      \fn GGEMSPhantom& operator=(GGEMSPhantom const& phantom) = delete
      \param phantom - reference on the GGEMS phantom
      \brief Avoid assignement by reference
    */
    GGEMSPhantom& operator=(GGEMSPhantom const& phantom) = delete;

    /*!
      \fn GGEMSPhantom(GGEMSPhantom const&& phantom) = delete
      \param phantom - rvalue reference on the GGEMS phantom
      \brief Avoid copy by rvalue reference
    */
    GGEMSPhantom(GGEMSPhantom const&& phantom) = delete;

    /*!
      \fn GGEMSPhantom& operator=(GGEMSPhantom const&& phantom) = delete
      \param phantom - rvalue reference on the GGEMS phantom
      \brief Avoid copy by rvalue reference
    */
    GGEMSPhantom& operator=(GGEMSPhantom const&& phantom) = delete;

    /*!
      \fn void SetPosition(GGfloat const& position_x, GGfloat const& position_y, GGfloat const& position_z, std::string const& unit = "mm")
      \param position_x - position in X
      \param position_y - position in Y
      \param position_z - position in Z
      \param unit - unit of the distance
      \brief set the position of the phantom in X, Y and Z
    */
    void SetPosition(GGfloat const& position_x, GGfloat const& position_y, GGfloat const& position_z, std::string const& unit = "mm");

    /*!
      \fn void SetVoxelizedPhantomFile(std::string const& filename, std::string const& range_data_filename)
      \param filename - MHD filename for voxelized phantom
      \param range_data_filename - text file with range to material data
      \brief set the mhd filename for phantom and the range data file
    */
    void SetVoxelizedPhantomFile(std::string const& filename, std::string const& range_data_filename);

  private:
    std::weak_ptr<GGEMSNavigator> navigator_; /*!< Pointer on navigator, this pointer is stored and deleted by GGEMSNavigatorManager */
};

/*!
  \fn GGEMSPhantom* create_ggems_phantom(char const* phantom_name, char const* phantom_type)
  \param phantom_name - name of phantom
  \param phantom_type - type of phantom
  \return the pointer on the phantom
  \brief Get the GGEMSPhantom pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSPhantom* create_ggems_phantom(char const* phantom_name, char const* phantom_type);

/*!
  \fn void set_phantom_file_ggems_phantom(GGEMSPhantom* phantom, char const* phantom_filename, char const* range_data_filename)
  \param phantom - pointer on phantom
  \param phantom_filename - filename of the phantom
  \param range_data_filename - range to material filename
  \brief set the filename of phantom and the range data file
*/
extern "C" GGEMS_EXPORT void set_phantom_file_ggems_phantom(GGEMSPhantom* phantom, char const* phantom_filename, char const* range_data_filename);

/*!
  \fn void set_position_ggems_phantom(GGEMSPhantom* phantom, GGfloat const position_x, GGfloat const position_y, GGfloat const position_z, char const* unit)
  \param phantom - pointer on phantom
  \param position_x - offset in X
  \param position_y - offset in Y
  \param position_z - offset in Z
  \param unit - unit of the distance
  \brief set the position of the phantom in X, Y and Z
*/
extern "C" GGEMS_EXPORT void set_position_ggems_phantom(GGEMSPhantom* phantom, GGfloat const position_x, GGfloat const position_y, GGfloat const position_z, char const* unit);

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSPHANTOM_HH
