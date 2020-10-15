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

#include "GGEMS/navigators/GGEMSNavigator.hh"
#include "GGEMS/global/GGEMSExport.hh"

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
    GGEMSPhantom(void);

    /*!
      \brief GGEMSPhantom destructor
    */
    ~GGEMSPhantom(void);

    /*!
      \fn GGEMSPhantom(GGEMSPhantom const& phantom) = delete
      \param phantom - reference on the GGEMS voxelized navigator
      \brief Avoid copy by reference
    */
    GGEMSPhantom(GGEMSPhantom const& phantom) = delete;

    /*!
      \fn GGEMSPhantom& operator=(GGEMSPhantom const& phantom) = delete
      \param phantom - reference on the GGEMS voxelized navigator
      \brief Avoid assignement by reference
    */
    GGEMSPhantom& operator=(GGEMSPhantom const& phantom) = delete;

    /*!
      \fn GGEMSPhantom(GGEMSPhantom const&& phantom) = delete
      \param phantom - rvalue reference on the GGEMS voxelized navigator
      \brief Avoid copy by rvalue reference
    */
    GGEMSPhantom(GGEMSPhantom const&& phantom) = delete;

    /*!
      \fn GGEMSPhantom& operator=(GGEMSPhantom const&& phantom) = delete
      \param phantom - rvalue reference on the GGEMS voxelized navigator
      \brief Avoid copy by rvalue reference
    */
    GGEMSPhantom& operator=(GGEMSPhantom const&& phantom) = delete;

    /*!
      \fn void SetNavigatorType(std::string const& navigator_type)
      \param navigator_type - type of navigator
      \brief set type of navigator: voxelized (only for moment)
    */
    void SetNavigatorType(std::string const& navigator_type);

    /*!
      \fn void SetPhantomName(std::string const& phantom_name)
      \param phantom_name - name of the phantom
      \brief set name of phantom
    */
    void SetPhantomName(std::string const& phantom_name);

  private:
    GGEMSNavigator* navigator_; /*!< Pointer on navigator, this pointer is stored and deleted by GGEMSNavigatorManager */
};

/*!
  \fn GGEMSPhantom* create_ggems_phantom(void)
  \return the pointer on the phantom
  \brief Get the GGEMSVoxelizedNavigator pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSPhantom* create_ggems_phantom(void);

/*!
  \fn void set_phantom_name_ggems_phantom(GGEMSPhantom* phantom, char const* phantom_name)
  \param phantom - pointer on the phantom
  \param phantom_name - name of the phantom
  \brief set the name of phantom
*/
extern "C" GGEMS_EXPORT void set_phantom_name_ggems_phantom(GGEMSPhantom* phantom, char const* phantom_name);

/*!
  \fn void set_phantom_type_ggems_phantom(GGEMSPhantom* phantom, char const* phantom_type)
  \param phantom - pointer on the phantom
  \param phantom_type - type of the phantom
  \brief set the type of phantom
*/
extern "C" GGEMS_EXPORT void set_phantom_type_ggems_phantom(GGEMSPhantom* phantom, char const* phantom_type);

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSPHANTOM_HH
