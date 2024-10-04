#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSSOLIDBOX_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSSOLIDBOX_HH

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
  \file GGEMSSolidBox.hh

  \brief GGEMS class for solid box

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 27, 2020
*/

#include "GGEMS/geometries/GGEMSSolid.hh"

/*!
  \class GGEMSSolidBox
  \brief GGEMS class for solid box
*/
class GGEMS_EXPORT GGEMSSolidBox : public GGEMSSolid
{
  public:
    /*!
      \param virtual_element_number_x - virtual element number in X (local axis)
      \param virtual_element_number_y - virtual element number in Y (local axis)
      \param virtual_element_number_z - virtual element number in Z (local axis)
      \param element_size_x - element size along X
      \param element_size_y - element size along Y
      \param element_size_z - element size along Z
      \param data_reg_type - type of registration "HIT", "SINGLE", "DOSE"
      \brief GGEMSSolidBox constructor
    */
    GGEMSSolidBox(GGsize const& virtual_element_number_x, GGsize const& virtual_element_number_y, GGsize const& virtual_element_number_z, GGfloat const& element_size_x, GGfloat const& element_size_y, GGfloat const& element_size_z, std::string const& data_reg_type);

    /*!
      \brief GGEMSSolidBox destructor
    */
    ~GGEMSSolidBox(void) override;

    /*!
      \fn GGEMSSolidBox(GGEMSSolidBox const& solid_box) = delete
      \param solid_box - reference on the GGEMS solid box
      \brief Avoid copy by reference
    */
    GGEMSSolidBox(GGEMSSolidBox const& solid_box) = delete;

    /*!
      \fn GGEMSSolidBox& operator=(GGEMSSolidBox const& solid_box) = delete
      \param solid_box - reference on the GGEMS solid box
      \brief Avoid assignement by reference
    */
    GGEMSSolidBox& operator=(GGEMSSolidBox const& solid_box) = delete;

    /*!
      \fn GGEMSSolidBox(GGEMSSolidBox const&& solid_box) = delete
      \param solid_box - rvalue reference on the GGEMS solid box
      \brief Avoid copy by rvalue reference
    */
    GGEMSSolidBox(GGEMSSolidBox const&& solid_box) = delete;

    /*!
      \fn GGEMSSolidBox& operator=(GGEMSSolidBox const&& solid_box) = delete
      \param solid_box - rvalue reference on the GGEMS solid box
      \brief Avoid copy by rvalue reference
    */
    GGEMSSolidBox& operator=(GGEMSSolidBox const&& solid_box) = delete;

    /*!
      \fn void Initialize(GGEMSMaterials* materials)
      \param materials - pointer on materials
      \brief Initialize solid for geometric navigation
    */
    void Initialize(GGEMSMaterials* materials) override;

    /*!
      \fn void EnableScatter(void)
      \brief Activate scatter registration
    */
    void EnableScatter(void) override;

    /*!
      \fn void PrintInfos(void) const
      \brief printing infos about voxelized solid
    */
    void PrintInfos(void) const override;

    /*!
      \fn void UpdateTransformationMatrix(GGsize const& thread_index)
      \param thread_index - index of the thread (= activated device index)
      \brief Update transformation matrix for solid box object
    */
    void UpdateTransformationMatrix(GGsize const& thread_index) override;

    GGfloat3 GetVoxelSizes(GGsize const& thread_index) const override {return {{0.0f, 0.0f, 0.0f}};}
    GGEMSOBB GetOBBGeometry(GGsize const& thread_index) const override {return GGEMSOBB{};}

  private:
    /*!
      \fn void InitializeKernel(void)
      \brief Initialize kernel for particle solid distance
    */
    void InitializeKernel(void) override;

  private:
};

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSSOLIDBOX_HH
