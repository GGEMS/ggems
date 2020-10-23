#ifndef GUARD_GGEMS_SYSTEMS_GGEMSSYSTEM_HH
#define GUARD_GGEMS_SYSTEMS_GGEMSSYSTEM_HH

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
  \file GGEMSSystem.hh

  \brief Child GGEMS class managing detector system in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Monday October 19, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include <string>
#include <memory>

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

#include "GGEMS/navigators/GGEMSNavigator.hh"

/*!
  \class GGEMSSystem
  \brief Child GGEMS class managing detector system in GGEMS
*/
class GGEMS_EXPORT GGEMSSystem : public GGEMSNavigator
{
  public:
    /*!
      \param system_name - name of the system
      \brief GGEMSSystem constructor
    */
    explicit GGEMSSystem(std::string const& system_name);

    /*!
      \brief GGEMSSystem destructor
    */
    virtual ~GGEMSSystem(void);

    /*!
      \fn GGEMSSystem(GGEMSSystem const& system) = delete
      \param system - reference on the GGEMS system
      \brief Avoid copy by reference
    */
    GGEMSSystem(GGEMSSystem const& system) = delete;

    /*!
      \fn GGEMSSystem& operator=(GGEMSSystem const& system) = delete
      \param system - reference on the GGEMS system
      \brief Avoid assignement by reference
    */
    GGEMSSystem& operator=(GGEMSSystem const& system) = delete;

    /*!
      \fn GGEMSSystem(GGEMSSystem const&& system) = delete
      \param system - rvalue reference on the GGEMS system
      \brief Avoid copy by rvalue reference
    */
    GGEMSSystem(GGEMSSystem const&& system) = delete;

    /*!
      \fn GGEMSSystem& operator=(GGEMSSystem const&& system) = delete
      \param system - rvalue reference on the GGEMS system
      \brief Avoid copy by rvalue reference
    */
    GGEMSSystem& operator=(GGEMSSystem const&& system) = delete;

    /*!
      \fn void SetNumberOfModules(GGuint const& module_x, GGuint const& module_y)
      \param module_x - Number of module in X (local axis of detector)
      \param module_y - Number of module in Y (local axis of detector)
      \brief set the number of module in X, Y of local axis of detector
    */
    void SetNumberOfModules(GGuint const& module_x, GGuint const& module_y);

  protected:
    GGuint2 number_of_modules_; /*!< Number of the detection modules */
    GGuint2 number_of_pixels_in_modules_; /*!< Number of pixels (X,Y) in a modules */
    GGfloat3 size_of_pixels_; /*!< Size of pixel in each direction */
};

#endif // End of GUARD_GGEMS_SYSTEMS_GGEMSSYSTEM_HH
