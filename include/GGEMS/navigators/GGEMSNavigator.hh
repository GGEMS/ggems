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

  \brief Parent GGEMS class for navigation

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

#include "GGEMS/maths/GGEMSMatrixTypes.hh"

class GGEMSSolid;
class GGEMSMaterials;
class GGEMSCrossSections;

/*!
  \class GGEMSNavigator
  \brief Parent GGEMS class for navigator
*/
class GGEMS_EXPORT GGEMSNavigator
{
  public:
    /*!
      \param navigator_name - name of the navigator
      \brief GGEMSNavigator constructor
    */
    explicit GGEMSNavigator(std::string const& navigator_name);

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
      \fn void SetPosition(GGfloat const& position_x, GGfloat const& position_y, GGfloat const& position_z, std::string const& unit = "mm")
      \param position_x - position in X
      \param position_y - position in Y
      \param position_z - position in Z
      \param unit - unit of the distance
      \brief set the position of the global navigator in X, Y and Z
    */
    void SetPosition(GGfloat const& position_x, GGfloat const& position_y, GGfloat const& position_z, std::string const& unit = "mm");

    /*!
      \fn void SetRotation(GGfloat const& rx, GGfloat const& ry, GGfloat const& rz, std::string const& unit)
      \param rx - Rotation around X along global axis
      \param ry - Rotation around Y along global axis
      \param rz - Rotation around Z along global axis
      \param unit - unit of the angle
      \brief Set the rotation of the global navigator around global axis
    */
    void SetRotation(GGfloat const& rx, GGfloat const& ry, GGfloat const& rz, std::string const& unit = "deg");

    /*!
      \fn void SetThreshold(GGfloat const& threshold, std::string const& unit = "keV")
      \param threshold - threshold applyied to the navigator
      \param unit - unit of the energy
      \brief Set the energy threshold to navigator
    */
    void SetThreshold(GGfloat const& threshold, std::string const& unit = "keV");

    /*!
      \fn void SetNavigatorID(std::size_t const& navigator_id)
      \param navigator_id - index of the navigator
      \brief set the navigator index
    */
    void SetNavigatorID(std::size_t const& navigator_id);

    /*!
      \fn inline std::string GetNavigatorName(void) const
      \brief Get the name of the navigator
      \return the name of the navigator
    */
    inline std::string GetNavigatorName(void) const {return navigator_name_;}

    /*!
      \fn inline std::size_t GetNumberOfSolids(void) const
      \brief get the number of solids inside the navigator
      \return the number of solids
    */
    inline std::size_t GetNumberOfSolids(void) const {return solids_.size();}

    /*!
      \fn inline std::vector<std::shared_ptr<GGEMSSolid>> GetSolids(void) const
      \brief get the list of solids
      \return the list of solids
    */
    inline std::vector<std::shared_ptr<GGEMSSolid>> GetSolids(void) const {return solids_;}

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
      \fn void ParticleSolidDistance(void)
      \brief Compute distance between particle and solid
    */
    void ParticleSolidDistance(void);

    /*!
      \fn void ProjectToSolid(void)
      \brief Project particle to entry of closest solid
    */
    void ProjectToSolid(void);

    /*!
      \fn void TrackThroughSolid(void)
      \brief Move particle through solid
    */
    void TrackThroughSolid(void);

    /*!
      \fn void PrintInfos(void) const
      \brief Print infos about navigator
    */
    void PrintInfos(void) const;

    /*!
      \fn void Initialize(void)
      \return no returned value
    */
    virtual void Initialize(void);

    /*!
      \fn inline DurationNano GetKernelParticleSolidDistanceTimer(void) const
      \return elapsed time in particle solid distance kernel in all solids
      \brief get the elapsed time in particle solid distance kernel in all solids
    */
    inline DurationNano GetKernelParticleSolidDistanceTimer(void) const {return kernel_particle_solid_distance_timer_;};

    /*!
      \fn inline DurationNano GetKernelProjectToSolidTimer(void) const
      \return elapsed time in kernel computing projection to closest solid
      \brief get the elapsed time in kernel computing projection to closest solid
    */
    inline DurationNano GetKernelProjectToSolidTimer(void) const {return kernel_project_to_solid_timer_;};

    /*!
      \fn inline DurationNano GetKernelTrackThroughSolidTimer(void) const
      \return elapsed time in kernel tracking particle inside solid
      \brief get the elapsed time in kernel tracking particle inside solid
    */
    inline DurationNano GetKernelTrackThroughSolidTimer(void) const {return kernel_track_through_solid_timer_;};

    /*!
      \fn void SaveResults(void)
      \brief save all results from solid
    */
    virtual void SaveResults(void) = 0;

    /*!
      \fn void StoreOutput(std::string basename)
      \param basename - basename of the output file
      \brief Storing the basename and format of the output file
    */
    void StoreOutput(std::string basename);

  protected:
    /*!
      \fn void CheckParameters(void) const
      \return no returned value
    */
    virtual void CheckParameters(void) const;

  protected:
    std::string navigator_name_; /*!< Name of the navigator */

    // Global navigation members
    GGfloat3 position_xyz_; /*!< Position of the navigator in X, Y and Z */
    GGfloat3 rotation_xyz_; /*!< Rotation of the navigator in X, Y and Z */
    std::size_t navigator_id_; /*!< Index of the navigator */
    bool is_update_pos_; /*!< Updating navigator position */
    bool is_update_rot_; /*!< Updating navigator rotation */
    GGfloat threshold_; /*!< Threshold in energy applyied to navigator */

    // Output
    std::string output_basename_; /*!< Basename of output file */

    std::vector<std::shared_ptr<GGEMSSolid>> solids_; /*!< Solid with geometric infos and label */
    std::shared_ptr<GGEMSMaterials> materials_; /*!< Materials of phantom */
    std::shared_ptr<GGEMSCrossSections> cross_sections_; /*!< Cross section table for process */

    // Timers for kernel computation
    DurationNano kernel_particle_solid_distance_timer_; /*!< Timer for kernel computing particle solid distance */
    DurationNano kernel_project_to_solid_timer_; /*!< Timer for kernel computing projection to closest solid */
    DurationNano kernel_track_through_solid_timer_; /*!< Timer for kernel computing tracking through closest solid */
};

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSNAVIGATOR_HH
