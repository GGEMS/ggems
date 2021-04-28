#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSCTSYSTEM_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSCTSYSTEM_HH

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
  \file GGEMSCTSystem.hh

  \brief Child GGEMS class managing CT/CBCT detector in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Monday October 19, 2020
*/

#include "GGEMS/navigators/GGEMSSystem.hh"

/*!
  \class GGEMSCTSystem
  \brief Child GGEMS class managing CT/CBCT detector in GGEMS
*/
class GGEMS_EXPORT GGEMSCTSystem : public GGEMSSystem
{
  public:
    /*!
      \param ct_system_name - name of the CT system
      \brief GGEMSCTSystem constructor
    */
    explicit GGEMSCTSystem(std::string const& ct_system_name);

    /*!
      \brief GGEMSCTSystem destructor
    */
    ~GGEMSCTSystem(void);

    /*!
      \fn GGEMSCTSystem(GGEMSCTSystem const& ct_system_name) = delete
      \param ct_system_name - reference on the GGEMS ct system name
      \brief Avoid copy by reference
    */
    GGEMSCTSystem(GGEMSCTSystem const& ct_system_name) = delete;

    /*!
      \fn GGEMSCTSystem& operator=(GGEMSCTSystem const& ct_system_name) = delete
      \param ct_system_name - reference on the GGEMS ct system name
      \brief Avoid assignement by reference
    */
    GGEMSCTSystem& operator=(GGEMSCTSystem const& ct_system_name) = delete;

    /*!
      \fn GGEMSCTSystem(GGEMSCTSystem const&& ct_system_name) = delete
      \param ct_system_name - rvalue reference on the GGEMS ct system name
      \brief Avoid copy by rvalue reference
    */
    GGEMSCTSystem(GGEMSCTSystem const&& ct_system_name) = delete;

    /*!
      \fn GGEMSCTSystem& operator=(GGEMSCTSystem const&& ct_system_name) = delete
      \param ct_system_name - rvalue reference on the GGEMS ct system name
      \brief Avoid copy by rvalue reference
    */
    GGEMSCTSystem& operator=(GGEMSCTSystem const&& ct_system_name) = delete;

    /*!
      \fn void Initialize(void) override
      \param is_tracking - flag activating tracking
      \brief Initialize CT system
    */
    void Initialize(void) override;

    /*!
      \fn void SetCTSystemType(std::string const& ct_system_type)
      \param ct_system_type - type of CT system
      \brief type of CT system: flat or curved
    */
    void SetCTSystemType(std::string const& ct_system_type);

    /*!
      \fn void SetSourceIsocenterDistance(GGfloat const& source_isocenter_distance, std::string const& unit)
      \param source_isocenter_distance - source isocenter distance
      \param unit - distance unit
      \brief set the source isocenter distance
    */
    void SetSourceIsocenterDistance(GGfloat const& source_isocenter_distance, std::string const& unit = "mm");

    /*!
      \fn void SetSourceDetectorDistance(GGfloat const& source_detector_distance, std::string const& unit)
      \param source_detector_distance - source detector distance
      \param unit - distance unit
      \brief set the source detector distance
    */
    void SetSourceDetectorDistance(GGfloat const& source_detector_distance, std::string const& unit = "mm");

  private:
    /*!
      \fn void CheckParameters(void) const override
      \return no returned value
    */
    void CheckParameters(void) const override;

    /*!
      \fn void InitializeCurvedGeometry(void)
      \brief Initialize the curved CT geometry
    */
    void InitializeCurvedGeometry(void);

    /*!
      \fn void InitializeFlatGeometry(void)
      \brief Initialize the flat CT geometry
    */
    void InitializeFlatGeometry(void);

  private:
    std::string ct_system_type_; /*!< Type of CT scanner, here: flat or curved */
    GGfloat source_isocenter_distance_; /*!< Distance from source to isocenter (SID) */
    GGfloat source_detector_distance_; /*!< Distance from source to detector (SDD) */
};

/*!
  \fn GGEMSPhantom* create_ggems_ct_system(char const* ct_system_name)
  \param ct_system_name - name of ct system
  \return the pointer on the ct system
  \brief Get the GGEMSCTSystem pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSCTSystem* create_ggems_ct_system(char const* ct_system_name);

/*!
  \fn void set_number_of_modules_ggems_ct_system(GGEMSCTSystem* ct_system, GGsize const module_x, GGsize const module_y)
  \param ct_system - pointer on ct system
  \param module_x - Number of module in X (local axis of detector)
  \param module_y - Number of module in Y (local axis of detector)
  \brief set the number of module in X, Y of local axis of detector
*/
extern "C" GGEMS_EXPORT void set_number_of_modules_ggems_ct_system(GGEMSCTSystem* ct_system, GGsize const module_x, GGsize const module_y);

/*!
  \fn void set_ct_system_type_ggems_ct_system(GGEMSCTSystem* ct_system, char const* ct_system_type)
  \param ct_system - pointer on ct system
  \param ct_system_type - type of CT system
  \brief set the type of CT system
*/
extern "C" GGEMS_EXPORT void set_ct_system_type_ggems_ct_system(GGEMSCTSystem* ct_system, char const* ct_system_type);

/*!
  \fn void set_number_of_detection_elements_ggems_ct_system(GGEMSCTSystem* ct_system, GGsize const n_detection_element_x, GGsize const n_detection_element_y, GGsize const n_detection_element_z)
  \param ct_system - pointer on ct system
  \param n_detection_element_x - Number of detection element inside a module in X (local axis of detector)
  \param n_detection_element_y - Number of detection element inside a module in Y (local axis of detector)
  \param n_detection_element_z - Number of detection element inside a module in Z (local axis of detector)
  \brief set the number of detection element inside a module
*/
extern "C" GGEMS_EXPORT void set_number_of_detection_elements_ggems_ct_system(GGEMSCTSystem* ct_system, GGsize const n_detection_element_x, GGsize const n_detection_element_y, GGsize const n_detection_element_z);

/*!
  \fn void set_size_of_detection_elements_ggems_ct_system(GGEMSCTSystem* ct_system, GGfloat const size_of_detection_element_x, GGfloat const size_of_detection_element_y, GGfloat const size_of_detection_element_z, char const* unit)
  \param ct_system - pointer on ct system
  \param size_of_detection_element_x - Size of detection element in X
  \param size_of_detection_element_y - Size of detection element in Y
  \param size_of_detection_element_z - Size of detection element in Z
  \param unit - unit of the distance
  \brief set the size of detection element in X, Y, Z
*/
extern "C" GGEMS_EXPORT void set_size_of_detection_elements_ggems_ct_system(GGEMSCTSystem* ct_system, GGfloat const size_of_detection_element_x, GGfloat const size_of_detection_element_y, GGfloat const size_of_detection_element_z, char const* unit);

/*!
  \fn void set_material_name_ggems_ct_system(GGEMSCTSystem* ct_system, char const* material_name)
  \param ct_system - pointer on ct system
  \param material_name - name of the material
  \brief set the material name for detection element
*/
extern "C" GGEMS_EXPORT void set_material_name_ggems_ct_system(GGEMSCTSystem* ct_system, char const* material_name);

/*!
  \fn void set_source_isocenter_distance_ggems_ct_system(GGEMSCTSystem* ct_system, GGfloat const source_isocenter_distance, char const* unit)
  \param ct_system - pointer on ct system
  \param source_isocenter_distance - Source isocenter distance
  \param unit - unit of the distance
  \brief set source isocenter distance (SID)
*/
extern "C" GGEMS_EXPORT void set_source_isocenter_distance_ggems_ct_system(GGEMSCTSystem* ct_system, GGfloat const source_isocenter_distance, char const* unit);

/*!
  \fn void set_source_detector_distance_ggems_ct_system(GGEMSCTSystem* ct_system, GGfloat const source_detector_distance, char const* unit)
  \param ct_system - pointer on ct system
  \param source_detector_distance - Source detector distance
  \param unit - unit of the distance
  \brief set source detector distance (SDD)
*/
extern "C" GGEMS_EXPORT void set_source_detector_distance_ggems_ct_system(GGEMSCTSystem* ct_system, GGfloat const source_detector_distance, char const* unit);

/*!
  \fn void set_rotation_ggems_ct_system(GGEMSCTSystem* ct_system, GGfloat const rx, GGfloat const ry, GGfloat const rz, char const* unit)
  \param ct_system - pointer on ct system
  \param rx - Rotation around X along local axis
  \param ry - Rotation around Y along local axis
  \param rz - Rotation around Z along local axis
  \param unit - unit of the angle
  \brief Set the rotation of the voxelized phantom around local axis
*/
extern "C" GGEMS_EXPORT void set_rotation_ggems_ct_system(GGEMSCTSystem* ct_system, GGfloat const rx, GGfloat const ry, GGfloat const rz, char const* unit);

/*!
  \fn void set_threshold_ggems_ct_system(GGEMSCTSystem* ct_system, GGfloat const threshold, char const* unit)
  \param ct_system - pointer on ct system
  \param threshold - threshold applyied to navigator
  \param unit - unit of the energy
  \brief Set the threshold applyied to navigator
*/
extern "C" GGEMS_EXPORT void set_threshold_ggems_ct_system(GGEMSCTSystem* ct_system, GGfloat const threshold, char const* unit);

/*!
  \fn void set_save_ggems_ct_system(GGEMSCTSystem* ct_system, char const* basename)
  \param ct_system - pointer on ct system
  \param basename - basename (without suffix) or output file
  \brief Set the output file and format
*/
extern "C" GGEMS_EXPORT void set_save_ggems_ct_system(GGEMSCTSystem* ct_system, char const* basename);

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSSYSTEM_HH
