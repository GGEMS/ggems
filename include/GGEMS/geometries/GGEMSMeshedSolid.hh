#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSMESHEDSOLID_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSMESHEDSOLID_HH

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
  \file GGEMSMeshedSolid.hh

  \brief GGEMS class for meshed solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday June 22, 2022
*/

// #include "GGEMS/geometries/GGEMSVoxelizedSolidData.hh"
#include "GGEMS/geometries/GGEMSSolid.hh"

/*!
  \class GGEMSMeshedSolid
  \brief GGEMS class for meshed solid
*/
class GGEMS_EXPORT GGEMSMeshedSolid : public GGEMSSolid
{
  public:
    /*!
      \param meshed_phantom_name - header file for volume
      \brief GGEMSMeshedSolid constructor
    */
    GGEMSMeshedSolid(std::string const& meshed_phantom_name, std::string const& data_reg_type = "");

    /*!
      \brief GGEMSMeshedSolid destructor
    */
    ~GGEMSMeshedSolid(void) override;

    /*!
      \fn GGEMSMeshedSolid(GGEMSMeshedSolid const& meshed_solid) = delete
      \param meshed_solid - reference on the GGEMS meshed solid
      \brief Avoid copy by reference
    */
    GGEMSMeshedSolid(GGEMSMeshedSolid const& meshed_solid) = delete;

    /*!
      \fn GGEMSMeshedSolid& operator=(GGEMSMeshedSolid const& meshed_solid) = delete
      \param meshed_solid - reference on the GGEMS meshed solid
      \brief Avoid assignement by reference
    */
    GGEMSMeshedSolid& operator=(GGEMSMeshedSolid const& meshed_solid) = delete;

    /*!
      \fn GGEMSMeshedSolid(GGEMSMeshedSolid const&& meshed_solid) = delete
      \param meshed_solid - rvalue reference on the GGEMS meshed solid
      \brief Avoid copy by rvalue reference
    */
    GGEMSMeshedSolid(GGEMSMeshedSolid const&& meshed_solid) = delete;

    /*!
      \fn GGEMSMeshedSolid& operator=(GGEMSMeshedSolid const&& meshed_solid) = delete
      \param meshed_solid - rvalue reference on the GGEMS meshed solid
      \brief Avoid copy by rvalue reference
    */
    GGEMSMeshedSolid& operator=(GGEMSMeshedSolid const&& meshed_solid) = delete;

    /*!
      \fn void Initialize(GGEMSMaterials* materials)
      \param materials - pointer on materials
      \brief Initialize solid for geometric navigation
    */
  //   void Initialize(GGEMSMaterials* materials) override;

  //   /*!
  //     \fn void EnableScatter(void)
  //     \brief Activate scatter registration
  //   */
  //   void EnableScatter(void) override {}

  //   /*!
  //     \fn void PrintInfos(void) const
  //     \brief printing infos about voxelized solid
  //   */
  //   void PrintInfos(void) const override;

  //   /*!
  //     \fn void LoadVolumeImage(GGEMSMaterials* materials)
  //     \param materials - pointer on material for a phantom
  //     \brief load volume image to GGEMS and create a volume of label in GGEMS for voxelized solid
  //   */
  //   void LoadVolumeImage(GGEMSMaterials* materials);

  //   /*!
  //     \fn void UpdateTransformationMatrix(GGsize const& thread_index)
  //     \param thread_index - index of the thread (= activated device index)
  //     \brief Update transformation matrix for solid box object
  //   */
  //   void UpdateTransformationMatrix(GGsize const& thread_index) override;

  //   /*!
  //     \fn GGfloat3 GetVoxelSizes(GGsize const& thread_index) const
  //     \param thread_index - index of the thread (= activated device index)
  //     \return size of voxels in voxelized solid
  //     \brief get the size of voxels in voxelized solid
  //   */
  //   GGfloat3 GetVoxelSizes(GGsize const& thread_index) const;

  //   /*!
  //     \fn GGEMSOBB GetOBBGeometry(GGsize const& thread_index) const
  //     \param thread_index - index of the thread (= activated device index)
  //     \return OBB params for the object
  //     \brief return the parameters about OBB geometry
  //   */
  //   GGEMSOBB GetOBBGeometry(GGsize const& thread_index) const;

  private:
  //   /*!
  //     \fn void InitializeKernel(void)
  //     \brief Initialize kernel for particle solid distance
  //   */
  //   void InitializeKernel(void) override;

  private:
    std::string meshed_phantom_name; /*!< Filename of STL file for mesh */
};

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSVOXELIZEDSOLID_HH
