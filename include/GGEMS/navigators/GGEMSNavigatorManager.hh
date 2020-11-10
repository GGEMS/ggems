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

#include "GGEMS/tools/GGEMSTools.hh"

#include "GGEMS/navigators/GGEMSNavigator.hh"

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
      \fn void Initialize(void) const
      \brief Initialize a GGEMS navigators
    */
    void Initialize(void) const;

    /*!
      \fn void PrintInfos(void)
      \brief Printing infos about the navigators
    */
    void PrintInfos(void) const;

    /*!
      \fn std::size_t GetNumberOfNavigators(void) const
      \brief Get the number of navigators
      \return the number of navigators
    */
    inline std::size_t GetNumberOfNavigators(void) const {return navigators_.size();}

    /*!
      \fn inline std::vector< std::shared_ptr<GGEMSNavigator> > GetNavigators(void) const
      \return the list of navigators
      \brief get the list of navigators
    */
    inline std::vector<std::shared_ptr<GGEMSNavigator>> GetNavigators(void) const {return navigators_;}

    /*!
      \fn inline std::shared_ptr<GGEMSNavigator> GetNavigator(std::string const& navigator_name) const
      \param navigator_name - name of the navigator
      \return the navigator by the name
      \brief get the navigator by the name
    */
    inline std::shared_ptr<GGEMSNavigator> GetNavigator(std::string const& navigator_name) const
    {
      // Loop over the navigator
      for (std::size_t i = 0; i < navigators_.size(); ++i) {
        if (navigator_name == (navigators_.at(i))->GetNavigatorName()) {
          return navigators_.at(i);
        }
      }
      GGEMSMisc::ThrowException("GGEMSNavigatorManager", "GetNavigator", "Name of the navigator unknown!!!");
      return nullptr;
    }

    /*!
      \fn inline std::size_t GetNumberOfRegisteredSolids(void) const
      \brief get the number of current registered solid
      \return number of current registered solid
    */
    inline std::size_t GetNumberOfRegisteredSolids(void) const
    {
      std::size_t number_of_registered_solid = 0;
      // Loop over number of navigator
      for (std::size_t i = 0; i < navigators_.size(); ++i) {
        number_of_registered_solid += (navigators_.at(i))->GetNumberOfSolids();
      }

      return number_of_registered_solid;
    }

    /*!
      \fn void FindClosestSolid(void) const
      \brief Find closest solid before project particle to it
    */
    void FindClosestSolid(void) const;

    /*!
      \fn void TrackToIn(void) const
      \brief Project particle to a navigator
    */
    void TrackToIn(void) const;

    /*!
      \fn void TrackToOut(void) const
      \brief Track particles inside navigator
    */
    void TrackToOut(void) const;

  private:
    /*!
      \fn bool CheckOverlap(std::weak_ptr<GGEMSNavigator> navigator_a, std::weak_ptr<GGEMSNavigator> navigator_b) const
      \param navigator_a - point on a navigator A
      \param navigator_b - point on a navigator B
      \brief check the overlap between navigator A and B
      \return true if there is an overlap and stop simulation
      \todo method only for voxelized navigator, generalize it!!!
    */
    //bool CheckOverlap(std::weak_ptr<GGEMSNavigator> navigator_a, std::weak_ptr<GGEMSNavigator> navigator_b) const;

  private:
    std::vector<std::shared_ptr<GGEMSNavigator>> navigators_; /*!< Pointer on the navigators */
};

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSNAVIGATORMANAGER_HH
