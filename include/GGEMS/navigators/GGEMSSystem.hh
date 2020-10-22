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
      \fn void SetPosition(GGfloat const& position_x, GGfloat const& position_y, GGfloat const& position_z, std::string const& unit = "mm")
      \param position_x - position in X
      \param position_y - position in Y
      \param position_z - position in Z
      \param unit - unit of the distance
      \brief set the position of the system in X, Y and Z
    */
    //void SetPosition(GGfloat const& position_x, GGfloat const& position_y, GGfloat const& position_z, std::string const& unit = "mm");

    /*!
      \fn void PrintInfos(void) const
      \return no returned value
    */
    void PrintInfos(void) const {;};

 // protected:
   // std::weak_ptr<GGEMSNavigator> navigator_; /*!< Pointer on navigator, this pointer is stored and deleted by GGEMSNavigatorManager */
};

#endif // End of GUARD_GGEMS_SYSTEMS_GGEMSSYSTEM_HH
