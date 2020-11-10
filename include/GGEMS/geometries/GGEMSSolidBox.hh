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

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/geometries/GGEMSSolid.hh"

/*!
  \class GGEMSSolidBox
  \brief GGEMS class for solid box
*/
class GGEMS_EXPORT GGEMSSolidBox : public GGEMSSolid
{
  public:
    /*!
      \param length_x - length along X for the solid box
      \param length_y - length along Y for the solid box
      \param length_z - length along Z for the solid box
      \brief GGEMSSolidBox constructor
    */
    GGEMSSolidBox(GGfloat const& length_x, GGfloat const& length_y, GGfloat const& length_z);

    /*!
      \brief GGEMSSolidBox destructor
    */
    ~GGEMSSolidBox(void);

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
      \fn void Initialize(std::weak_ptr<GGEMSMaterials> materials)
      \param materials - pointer on materials
      \brief Initialize solid for geometric navigation
    */
    void Initialize(std::weak_ptr<GGEMSMaterials> materials) override;

    /*!
      \fn void PrintInfos(void) const
      \brief printing infos about voxelized solid
    */
    void PrintInfos(void) const override;

    /*!
      \fn void GetTransformationMatrix(void)
      \brief Get the transformation matrix for solid box object
    */
    void GetTransformationMatrix(void) override;

  private:
    /*!
      \fn void InitializeKernel(void)
      \brief Initialize kernel for particle solid distance
    */
    void InitializeKernel(void) override;

  private:
};

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSSOLIDBOX_HH
