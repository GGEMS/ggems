#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSNAVIGATORMANAGER_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSNAVIGATORMANAGER_HH

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
  \file GGEMSNavigatorManager.hh

  \brief GGEMS class handling the navigators (detector + phantom) in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday February 11, 2020
*/

#include "GGEMS/navigators/GGEMSNavigator.hh"
#include "GGEMS/navigators/GGEMSWorld.hh"

/*!
  \class GGEMSNavigatorManager
  \brief GGEMS class handling the navigators (detector + phantom) in GGEMS
*/
class GGEMS_EXPORT GGEMSNavigatorManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGEMSNavigatorManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGEMSNavigatorManager(void);

  public:
    /*!
      \fn static GGEMSNavigatorManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSNavigatorManager
    */
    static GGEMSNavigatorManager& GetInstance(void)
    {
      static GGEMSNavigatorManager instance;
      return instance;
    }

    /*!
      \fn GGEMSNavigatorManager(GGEMSNavigatorManager const& navigator_manager) = delete
      \param navigator_manager - reference on the navigator manager
      \brief Avoid copy of the class by reference
    */
    GGEMSNavigatorManager(GGEMSNavigatorManager const& navigator_manager) = delete;

    /*!
      \fn GGEMSNavigatorManager& operator=(GGEMSNavigatorManager const& navigator_manager) = delete
      \param navigator_manager - reference on the navigator manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSNavigatorManager& operator=(GGEMSNavigatorManager const& navigator_manager) = delete;

    /*!
      \fn GGEMSNavigatorManager(GGEMSNavigatorManager const&& navigator_manager) = delete
      \param navigator_manager - rvalue reference on the navigator manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSNavigatorManager(GGEMSNavigatorManager const&& navigator_manager) = delete;

    /*!
      \fn GGEMSNavigatorManager& operator=(GGEMSNavigatorManager const&& navigator_manager) = delete
      \param navigator_manager - rvalue reference on the navigator manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSNavigatorManager& operator=(GGEMSNavigatorManager const&& navigator_manager) = delete;

    /*!
      \fn void Store(GGEMSNavigator* navigator)
      \param navigator - pointer to GGEMS navigator
      \brief storing the navigator pointer to navigator manager
    */
    void Store(GGEMSNavigator* navigator);

    /*!
      \fn void StoreWorld(GGEMSWorld* world)
      \param world - pointer to GGEMS world
      \brief storing the world pointer
    */
    void StoreWorld(GGEMSWorld* world);

    /*!
      \fn void Initialize(bool const& is_tracking = false) const
      \param is_tracking - flag activating tracking
      \brief Initialize a GGEMS navigators
    */
    void Initialize(bool const& is_tracking = false) const;

    /*!
      \fn void PrintInfos(void)
      \brief Printing infos about the navigators
    */
    void PrintInfos(void) const;

    /*!
      \fn GGsize GetNumberOfNavigators(void) const
      \brief Get the number of navigators
      \return the number of navigators
    */
    inline GGsize GetNumberOfNavigators(void) const {return number_of_navigators_;}

    /*!
      \fn inline GGEMSNavigator** GetNavigators(void) const
      \return the list of navigators
      \brief get the list of navigators
    */
    inline GGEMSNavigator** GetNavigators(void) const {return navigators_;}

    /*!
      \fn inline GGEMSNavigator* GetNavigator(std::string const& navigator_name) const
      \param navigator_name - name of the navigator
      \return the navigator by the name
      \brief get the navigator by the name
    */
    inline GGEMSNavigator* GetNavigator(std::string const& navigator_name) const
    {
      // Loop over the navigator
      for (GGsize i = 0; i < number_of_navigators_; ++i) {
        if (navigator_name == navigators_[i]->GetNavigatorName()) {
          return navigators_[i];
        }
      }
      GGEMSMisc::ThrowException("GGEMSNavigatorManager", "GetNavigator", "Name of the navigator unknown!!!");
      return nullptr;
    }

    /*!
      \fn inline GGsize GetNumberOfRegisteredSolids(void) const
      \brief get the number of current registered solid
      \return number of current registered solid
    */
    inline GGsize GetNumberOfRegisteredSolids(void) const
    {
      GGsize number_of_registered_solid = 0;
      // Loop over number of navigator
      for (GGsize i = 0; i < number_of_navigators_; ++i) {
        number_of_registered_solid += navigators_[i]->GetNumberOfSolids();
      }

      return number_of_registered_solid;
    }

    /*!
      \fn void FindSolid(GGsize const& thread_index) const
      \param thread_index - index of activated device (thread index)
      \brief Find closest solid before project particle to it
    */
    void FindSolid(GGsize const& thread_index) const;

    /*!
      \fn void ProjectToSolid(GGsize const& thread_index) const
      \param thread_index - index of activated device (thread index)
      \brief Project particle to selected solid
    */
    void ProjectToSolid(GGsize const& thread_index) const;

    /*!
      \fn void TrackThroughSolid(GGsize const& thread_index) const
      \param thread_index - index of activated device (thread index)
      \brief Track particles through selected solid
    */
    void TrackThroughSolid(GGsize const& thread_index) const;

    /*!
      \fn void SaveResults(void) const
      \brief save all results from navigator in files
    */
    void SaveResults(void) const;

    /*!
      \fn void WorldTracking(GGsize const& thread_index) const
      \param thread_index - index of activated device (thread index)
      \brief Tracking particles through world
    */
    void WorldTracking(GGsize const& thread_index) const;

    /*!
      \fn void ComputeDose(GGsize const& thread_index)
      \param thread_index - index of activated device (thread index)
      \brief Compute dose in volume
    */
    void ComputeDose(GGsize const& thread_index);

    /*!
      \fn void Clean(void)
      \brief clean OpenCL data if necessary
    */
    void Clean(void);

  private:
    GGEMSNavigator** navigators_; /*!< Pointer on the navigators */
    GGsize number_of_navigators_; /*!< Number of navigators */
    GGEMSWorld* world_; /*!< Pointer on world volume */
};

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSNAVIGATORMANAGER_HH
