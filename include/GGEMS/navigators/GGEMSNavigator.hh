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

#define NAVIGATOR_NOT_INITIALIZED 0x100000000 /*!< value if OpenCL kernel is not compiled */

#include "GGEMS/physics/GGEMSRangeCuts.hh"

#include "GGEMS/geometries/GGEMSGeometryConstants.hh"

#include "GGEMS/maths/GGEMSMatrixTypes.hh"

#include "GGEMS/maths/GGEMSMathAlgorithms.hh"
#include "GGEMS/physics/GGEMSMuData.hh"
#include "GGEMS/physics/GGEMSMuDataConstants.hh"

class GGEMSSolid;
class GGEMSMaterials;
class GGEMSCrossSections;
class GGEMSDosimetryCalculator;

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
      \fn void SetShiftedPosition(GGfloat const& position_x, GGfloat const& shifted_position_y, GGfloat const& shifted_position_z, std::string const& unit = "mm")
      \param shifted_position_x - shifted position in X
      \param shifted_position_y - shifted position in Y
      \param shifted_position_z - shifted position in Z
      \param unit - unit of the distance
      \brief set the shifted position of the global navigator in X, Y and Z
    */
    void SetShiftedPosition(GGfloat const& shifted_position_x, GGfloat const& shifted_position_y, GGfloat const& shifted_position_z, std::string const& unit = "mm");

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
      \fn void SetNavigatorID(GGsize const& navigator_id)
      \param navigator_id - index of the navigator
      \brief set the navigator index
    */
    void SetNavigatorID(GGsize const& navigator_id);

    /*!
      \fn inline std::string GetNavigatorName(void) const
      \brief Get the name of the navigator
      \return the name of the navigator
    */
    inline std::string GetNavigatorName(void) const {return navigator_name_;}

    /*!
      \fn inline GGsize GetNumberOfSolids(void) const
      \brief get the number of solids inside the navigator
      \return the number of solids
    */
    inline GGsize GetNumberOfSolids(void) const {return number_of_solids_;}

    /*!
      \fn inline GGEMSSolid* GetSolids(GGsize const& solid_index) const
      \param solid_index - index of solid
      \brief get the list of solids
      \return the list of solids
    */
    inline GGEMSSolid* GetSolids(GGsize const& solid_index) const {return solids_[solid_index];}

    /*!
      \fn inline GGEMSMaterials* GetMaterials(void) const
      \brief get the pointer on materials
      \return the pointer on materials
    */
    inline GGEMSMaterials* GetMaterials(void) const {return materials_;}

    /*!
      \fn inline GGEMSCrossSections* GetCrossSections(void) const
      \brief get the pointer on cross sections
      \return the pointer on cross sections
    */
    inline GGEMSCrossSections* GetCrossSections(void) const {return cross_sections_;}

    /*!
      \fn void ParticleSolidDistance(GGsize const& thread_index)
      \param thread_index - index of activated device (thread index)
      \brief Compute distance between particle and solid
    */
    void ParticleSolidDistance(GGsize const& thread_index);

    /*!
      \fn void ProjectToSolid(GGsize const& thread_index)
      \param thread_index - index of activated device (thread index)
      \brief Project particle to entry of closest solid
    */
    void ProjectToSolid(GGsize const& thread_index);

    /*!
      \fn void TrackThroughSolid(GGsize const& thread_index)
      \param thread_index - index of activated device (thread index)
      \brief Move particle through solid
    */
    void TrackThroughSolid(GGsize const& thread_index);

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
      \fn void SaveResults(void)
      \brief save all results from solid
    */
    virtual void SaveResults(void) = 0;

    /*!
      \fn void ComputeDose(GGsize const& thread_index)
      \param thread_index - index of activated device (thread index)
      \brief Compute dose in volume
    */
    void ComputeDose(GGsize const& thread_index);

    /*!
      \fn void StoreOutput(std::string basename)
      \param basename - basename of the output file
      \brief Storing the basename and format of the output file
    */
    void StoreOutput(std::string basename);

    /*!
      \fn void SetDosimetryCalculator(GGEMSDosimetryCalculator* dosimetry_calculator)
      \param dosimetry_calculator - pointer on dosimetry calculator
      \brief give adress of dosimetry calculator to navigator
    */
    void SetDosimetryCalculator(GGEMSDosimetryCalculator* dosimetry_calculator);

    /*!
      \fn void EnableTracking(void)
      \brief Enable tracking during simulation
    */
    void EnableTracking(void);

    /*!
      \fn void EnableTLE(void)
      \brief Enable track length estimator (TLE) during particle navigation
    */
    void EnableTLE(bool const& is_activated);

    /*!
      \fn void Init_Mu_Table(void)
      \brief Enable track length estimator (TLE) during particle navigation
    */
    void Init_Mu_Table(void);

    /*!
      \fn void SetVisible(bool const& is_visible)
      \param is_visible - true if navigator is drawn using OpenGL
      \brief set to true to draw navigator
    */
    void SetVisible(bool const& is_visible);

    /*!
      \fn void SetMaterialColor(std::string const& material_name, GGuchar const& red, GGuchar const& green, GGuchar const& blue)
      \param material_name - material name
      \param red - red part of RGB color
      \param green - green part of RGB color
      \param blue - blue part of RGB color
      \brief set custom color for opengl volume
    */
    void SetMaterialColor(std::string const& material_name, GGuchar const& red, GGuchar const& green, GGuchar const& blue);

    /*!
      \fn void SetMaterialColor(std::string const& material_name, std::string const& color_name)
      \param material_name - material name
      \param color_name - name of color from GGEMS color list
      \brief set custom color for opengl volume
    */
    void SetMaterialColor(std::string const& material_name, std::string const& color_name);

    /*!
      \fn void SetMaterialVisible(std::string const& material_name, bool const& is_material_visible)
      \param material_name - name of material
      \param is_material_visible - flag drawing material
      \brief set visibility of material for OpenGL
    */
    void SetMaterialVisible(std::string const& material_name, bool const& is_material_visible);

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
    GGfloat3 shifted_position_xyz_; /*!< Shifted position of the navigator in X, Y and Z */

    GGsize navigator_id_; /*!< Index of the navigator */
    bool is_update_pos_; /*!< Updating navigator position */
    bool is_update_rot_; /*!< Updating navigator rotation */
    GGfloat threshold_; /*!< Threshold in energy applyied to navigator */
    bool is_tracking_; /*!< Boolean activating tracking */

    // Output
    std::string output_basename_; /*!< Basename of output file */

    GGEMSSolid** solids_; /*!< Solid with geometric infos and label */
    GGsize number_of_solids_; /*!< Number of solids in navigator */
    GGEMSMaterials* materials_; /*!< Materials of phantom */
    GGEMSCrossSections* cross_sections_; /*!< Cross section table for process */

    // Dosimetry
    GGEMSDosimetryCalculator* dose_calculator_; /*!< Dose calculator pointer */
    bool is_dosimetry_mode_; /*!< Boolean checking if dosimetry mode is activated */
    GGint is_tle_;  /*!< Boolean checking if tle mode is activated */
    GGsize number_activated_devices_; /*!< Number of activated device */

    cl::Buffer** mu_tables_; /*!< Buffer for the Mu Table */

    // OpenGL
    bool is_visible_; /*!< flag for opengl */
    MaterialRGBColorUMap custom_material_rgb_; /*!< Custom color for material */
    MaterialVisibleUMap material_visible_; /*!< list of flag to draw volume or not */
};

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSNAVIGATOR_HH
