#ifndef GUARD_GGEMS_PHYSICS_GGEMSRANGECUTSMANAGER_HH
#define GUARD_GGEMS_PHYSICS_GGEMSRANGECUTSMANAGER_HH

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
  \file GGEMSRangeCutsManager.hh

  \brief GGEMS class managing the range cuts in GGEMS simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Friday March 6, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/global/GGEMSExport.hh"


/*!
  \class GGEMSRangeCutsManager
  \brief GGEMS class managing the range cuts in GGEMS simulation
*/
class GGEMS_EXPORT GGEMSRangeCutsManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGEMSRangeCutsManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGEMSRangeCutsManager(void);

  public:
    /*!
      \fn static GGEMSRangeCutsManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSRangeCutsManager
    */
    static GGEMSRangeCutsManager& GetInstance(void)
    {
      static GGEMSRangeCutsManager instance;
      return instance;
    }

    /*!
      \fn GGEMSRangeCutsManager(GGEMSRangeCutsManager const& range_cuts_manager) = delete
      \param range_cuts_manager - reference on the range cuts manager
      \brief Avoid copy of the class by reference
    */
    GGEMSRangeCutsManager(GGEMSRangeCutsManager const& range_cuts_manager) = delete;

    /*!
      \fn GGEMSRangeCutsManager& operator=(GGEMSRangeCutsManager const& range_cuts_manager) = delete
      \param range_cuts_manager - reference on the range cuts manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSRangeCutsManager& operator=(GGEMSRangeCutsManager const& range_cuts_manager) = delete;

    /*!
      \fn GGEMSRangeCutsManager(GGEMSRangeCutsManager const&& range_cuts_manager) = delete
      \param range_cuts_manager - rvalue reference on the range cuts manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSRangeCutsManager(GGEMSRangeCutsManager const&& range_cuts_manager) = delete;

    /*!
      \fn GGEMSRangeCutsManager& operator=(GGEMSRangeCutsManager const&& range_cuts_manager) = delete
      \param range_cuts_manager - rvalue reference on the range cuts manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSRangeCutsManager& operator=(GGEMSRangeCutsManager const&& range_cuts_manager) = delete;

    /*!
      \fn void SetLengthCut(std::string const& phantom_name, std::string const& particle_name, GGfloat const& value, std::string const& unit = "mm")
      \param phantom_name - name of the phantom
      \param particle_name - name of the particle
      \param value - value of the cut
      \param unit - unit of the cut in length
      \brief set the range cut length for a phantom and a particle
    */
    void SetLengthCut(std::string const& phantom_name, std::string const& particle_name, GGfloat const& value, std::string const& unit = "mm");

    /*!
      \fn void PrintInfos(void) const
      \brief print infos about range cut manager
    */
    void PrintInfos(void) const;

    /*!
      \fn void Clean(void)
      \brief clean OpenCL data if necessary
    */
    void Clean(void);
};

/*!
  \fn GGEMSRangeCutsManager* get_instance_range_cuts_manager(void)
  \return the pointer on the singleton
  \brief Get the GGEMSRangeCutsManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSRangeCutsManager* get_instance_range_cuts_manager(void);

/*!
  \fn void set_cut_range_cuts_manager(GGEMSRangeCutsManager* range_cut_manager, char const* phantom_name, char const* particle_name, GGfloat const value, char const* unit)
  \param range_cut_manager - pointer on the range cut manager
  \param phantom_name - name of the phantom
  \param particle_name - name of the particle
  \param value - value of the cut
  \param unit - unit of distance
  \brief set the range cut for a phantom and a particle
*/
extern "C" GGEMS_EXPORT void set_cut_range_cuts_manager(GGEMSRangeCutsManager* range_cut_manager, char const* phantom_name, char const* particle_name, GGfloat const value, char const* unit);

#endif // GUARD_GGEMS_PHYSICS_GGEMSRANGECUTSMANAGER_HH
