#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSNAVIGATOR_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSNAVIGATOR_HH

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
  \file GGEMSNavigator.hh

  \brief GGEMS mother class for navigation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Tuesday February 11, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include "GGEMS/physics/GGEMSRangeCuts.hh"
#include "GGEMS/geometries/GGEMSGeometryConstants.hh"

class GGEMSSolid;
class GGEMSMaterials;
class GGEMSCrossSections;

/*!
  \class GGEMSNavigator
  \brief GGEMS mother class for navigator
*/
class GGEMS_EXPORT GGEMSNavigator
{
  public:
    /*!
      \brief GGEMSNavigator constructor
    */
    GGEMSNavigator(void);

    /*!
      \brief GGEMSNavigator destructor
    */
    virtual ~GGEMSNavigator(void);

    /*!
      \fn GGEMSNavigator(GGEMSNavigator const& navigator) = delete
      \param navigator - reference on the GGEMS navigator
      \brief Avoid copy by reference
    */
    GGEMSNavigator(GGEMSNavigator const& navigator) = delete;

    /*!
      \fn GGEMSNavigator& operator=(GGEMSNavigator const& navigator) = delete
      \param navigator - reference on the GGEMS navigator
      \brief Avoid assignement by reference
    */
    GGEMSNavigator& operator=(GGEMSNavigator const& navigator) = delete;

    /*!
      \fn GGEMSNavigator(GGEMSNavigator const&& navigator) = delete
      \param navigator - rvalue reference on the GGEMS navigator
      \brief Avoid copy by rvalue reference
    */
    GGEMSNavigator(GGEMSNavigator const&& navigator) = delete;

    /*!
      \fn GGEMSNavigator& operator=(GGEMSNavigator const&& navigator) = delete
      \param navigator - rvalue reference on the GGEMS navigator
      \brief Avoid copy by rvalue reference
    */
    GGEMSNavigator& operator=(GGEMSNavigator const&& navigator) = delete;

    /*!
      \fn void SetNavigatorName(std::string const& navigator_name)
      \param navigator_name - name of the navigator
      \brief save the name of the navigator
    */
    void SetNavigatorName(std::string const& navigator_name);

    /*!
      \fn void SetGeometryTolerance(GGfloat const& distance, std::string const& unit)
      \param distance - geometry distance
      \param unit - unit of the distance
      \brief Set the geometry tolerance in distance
    */
    void SetGeometryTolerance(GGfloat const& distance, std::string const& unit = "mm");

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
      \fn inline std::string GetNavigatorName(void) const
      \brief Get the name of the navigator
      \return the name of the navigator
    */
    inline std::string GetNavigatorName(void) const {return navigator_name_;}

    /*!
      \fn inline std::weak_ptr<GGEMSSolid> GetSolid(void) const
      \brief get the pointer on solid
      \return the pointer on solid
    */
    inline std::weak_ptr<GGEMSSolid> GetSolid(void) const {return solid_;}

    /*!
      \fn inline std::weak_ptr<GGEMSMaterials> GetMaterials(void) const
      \brief get the pointer on materials
      \return the pointer on materials
    */
    inline std::weak_ptr<GGEMSMaterials> GetMaterials(void) const {return materials_;}

    /*!
      \fn inline std::weak_ptr<GGEMSCrossSections> GetCrossSections(void) const
      \brief get the pointer on cross sections
      \return the pointer on cross sections
    */
    inline std::weak_ptr<GGEMSCrossSections> GetCrossSections(void) const {return cross_sections_;}

    /*!
      \fn void SetNavigatorID(std::size_t const& navigator_id)
      \param navigator_id - index of the navigator
      \brief set the navigator index
    */
    void SetNavigatorID(std::size_t const& navigator_id);

    /*!
      \fn void EnableTracking(bool const& is_tracking)
      \param is_tracking - boolean enabling tracking infos for navigators
      \brief Enabling tracking infos during simulation
    */
    void EnableTracking(bool const& is_tracking);

    /*!
      \fn void ParticleNavigatorDistance(void) const
      \brief Compute distance between particle and navigator
    */
    void ParticleNavigatorDistance(void) const;

    /*!
      \fn void ParticleToNavigator(void) const
      \brief Project particle to entry of navigator
    */
    void ParticleToNavigator(void) const;

    /*!
      \fn void ParticleThroughNavigator(void) const
      \brief Move particle through navigator
    */
    void ParticleThroughNavigator(void) const;

    /*!
      \fn void PrintInfos(void) const
      \return no returned value
    */
    virtual void PrintInfos(void) const;

    /*!
      \fn void CheckParameters(void) const
      \return no returned value
    */
    virtual void CheckParameters(void) const;

    /*!
      \fn void Initialize(void)
      \return no returned value
    */
    virtual void Initialize(void);

  protected:
    std::string navigator_name_; /*!< Name of the navigator */
    GGfloat3 position_xyz_; /*!< Position of the navigator in X, Y and Z */
    std::size_t navigator_id_; /*!< Index of the navigator */
    bool is_tracking_; /*!< Boolean enabling tracking for debugging */
    bool is_update_pos_; /*!< Updating navigator position */

    std::shared_ptr<GGEMSSolid> solid_; /*!< Solid with geometric infos and label */
    std::shared_ptr<GGEMSMaterials> materials_; /*!< Materials of phantom */
    std::shared_ptr<GGEMSCrossSections> cross_sections_; /*!< Cross section table for process */
};

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSNAVIGATOR_HH
