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
      \fn void SetNumberOfModules(GGsize const& n_module_x, GGsize const& n_module_y)
      \param n_module_x - Number of module in X (local axis of detector)
      \param n_module_y - Number of module in Y (local axis of detector)
      \brief set the number of module in X, Y of local axis of detector
    */
    void SetNumberOfModules(GGsize const& n_module_x, GGsize const& n_module_y);

    /*!
      \fn void SetNumberOfDetectionElementsInsideModule(GGsize const& n_detection_element_x, GGsize const& n_detection_element_y, GGsize const& n_detection_element_z)
      \param n_detection_element_x - Detection element in X
      \param n_detection_element_y - Detection element in Y
      \param n_detection_element_z - Detection element in Z
      \brief set the number of detection elements in X and Y and Z
    */
    void SetNumberOfDetectionElementsInsideModule(GGsize const& n_detection_element_x, GGsize const& n_detection_element_y, GGsize const& n_detection_element_z);

    /*!
      \fn void SetSizeOfDetectionElements(GGfloat const& detection_element_x, GGfloat const& detection_element_y, GGfloat const& detection_element_z, std::string const& unit)
      \param detection_element_x - Detection element in X axis
      \param detection_element_y - Detection element in Y axis
      \param detection_element_z - Detection element in Z axis
      \param unit - unit of detection element
      \brief set the detection elements in each direction
    */
    void SetSizeOfDetectionElements(GGfloat const& size_of_detection_element_x, GGfloat const& size_of_detection_element_y, GGfloat const& size_of_detection_element_z, std::string const& unit = "mm");

    /*!
      \fn void SetGlobalPosition(GGfloat const& global_position_x, GGfloat const& global_position_y, GGfloat const& global_position_z, std::string const& unit = "mm")
      \param global_position_x - global position of the system in X (global axis)
      \param global_position_y - global position of the system in Y (global axis)
      \param global_position_z - global position of the system in Z (global axis)
      \param unit - distance unit
      \brief set the global position of the system
    */
    void SetGlobalPosition(GGfloat const& global_position_x, GGfloat const& global_position_y, GGfloat const& global_position_z, std::string const& unit = "mm");

    /*!
      \fn void SetMaterialName(std::string const& material_name)
      \param material_name - name of the material for detection element
      \brief set the name of the material
    */
    void SetMaterialName(std::string const& material_name);

    /*!
      \fn void SetScatter(bool const& is_scatter)
      \param is_scatter - true to store scatter image
      \brief set to true to activate scatter registration
    */
    void SetScatter(bool const& is_scatter);

    /*!
      \fn void SaveResults(void)
      \brief save all results from solid
    */
    void SaveResults(void);

  protected:
    /*!
      \fn void CheckParameters(void) const
      \return no returned value
    */
    virtual void CheckParameters(void) const;

  protected:
    GGsize2 number_of_modules_xy_; /*!< Number of the detection modules */
    GGsize3 number_of_detection_elements_inside_module_xyz_; /*!< Number of virtual elements (X,Y,Z) in a module */
    GGfloat3 size_of_detection_elements_xyz_; /*!< Size of pixel in each direction */
    bool is_scatter_; /*!< Boolean storing scatter infos */
};

#endif // End of GUARD_GGEMS_SYSTEMS_GGEMSSYSTEM_HH
