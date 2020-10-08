#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSSOLID_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSSOLID_HH

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
  \file GGEMSSolid.hh

  \brief GGEMS class for solid. This class store geometry about phantom or detector

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 2, 2020
*/

#include <limits>
#include <algorithm>

#include "GGEMS/io/GGEMSTextReader.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolidStack.hh"
#include "GGEMS/tools/GGEMSRAMManager.hh"
#include "GGEMS/navigators/GGEMSNavigatorManager.hh"

/*!
  \class GGEMSSolid
  \brief GGEMS class for solid (voxelized or analytical) informations
*/
class GGEMS_EXPORT GGEMSSolid
{
  public:
    /*!
      \brief GGEMSSolid constructor
    */
    GGEMSSolid(void);

    /*!
      \brief GGEMSSolid destructor
    */
    ~GGEMSSolid(void);

    /*!
      \fn GGEMSSolid(GGEMSSolid const& solid) = delete
      \param solid - reference on the GGEMS solid
      \brief Avoid copy by reference
    */
    GGEMSSolid(GGEMSSolid const& solid) = delete;

    /*!
      \fn GGEMSSolid& operator=(GGEMSSolid const& solid) = delete
      \param solid - reference on the GGEMS solid
      \brief Avoid assignement by reference
    */
    GGEMSSolid& operator=(GGEMSSolid const& solid) = delete;

    /*!
      \fn GGEMSSolid(GGEMSSolid const&& solid) = delete
      \param solid - rvalue reference on the GGEMS solid
      \brief Avoid copy by rvalue reference
    */
    GGEMSSolid(GGEMSSolid const&& solid) = delete;

    /*!
      \fn GGEMSSolid& operator=(GGEMSSolid const&& solid) = delete
      \param solid - rvalue reference on the GGEMS solid
      \brief Avoid copy by rvalue reference
    */
    GGEMSSolid& operator=(GGEMSSolid const&& solid) = delete;

    /*!
      \fn void SetGeometryTolerance(GGfloat const& tolerance)
      \param tolerance - geometry tolerance for computation
      \brief set the geometry tolerance
    */
    void SetGeometryTolerance(GGfloat const& tolerance);

    /*!
      \fn void SetNavigatorID(std::size_t const& navigator_id)
      \param navigator_id - index of the navigator
      \brief set the navigator index in solid data
    */
    void SetNavigatorID(std::size_t const& navigator_id);

    /*!
      \fn void EnableTracking(void)
      \brief Enabling tracking infos during simulation
    */
    void EnableTracking(void);

    /*!
      \fn inline cl::Buffer* GetSolidData(void) const
      \brief get the informations about the solid geometry
      \return header data OpenCL pointer about solid
    */
    inline cl::Buffer* GetSolidData(void) const {return solid_data_cl_.get();};

    /*!
      \fn void Distance(void)
      \brief compute distance from particle position to solid and store this distance in OpenCL particle buffer
    */
    void Distance(void);

    /*!
      \fn void ProjectTo(void)
      \brief Move particles at an entry of solid
    */
    void ProjectTo(void);

    /*!
      \fn void TrackThrough(std::weak_ptr<GGEMSCrossSections> cross_sections, std::weak_ptr<GGEMSMaterials> materials)
      \param cross_sections - pointer storing cross sections values
      \param materials - pointer storing materials values
      \brief Track particles through solid
    */
    void TrackThrough(std::weak_ptr<GGEMSCrossSections> cross_sections, std::weak_ptr<GGEMSMaterials> materials);

    /*!
      \fn void Initialize(std::shared_ptr<GGEMSMaterials> materials)
      \param materials - pointer on GGEMS materials
      \brief Initialize solid for geometric navigation
    */
    virtual void Initialize(std::shared_ptr<GGEMSMaterials> materials) = 0;

    /*!
      \fn void SetPosition(GGfloat3 const& position_xyz)
      \param position_xyz - position in X, Y and Z
      \brief set a position for solid
    */
    virtual void SetPosition(GGfloat3 const& position_xyz) = 0;

    /*!
      \fn void PrintInfos(void) const
      \brief printing infos about solid
    */
    virtual void PrintInfos(void) const = 0;

  protected:
    /*!
      \fn void InitializeKernel(void)
      \brief Initialize kernel for particle solid distance
    */
    virtual void InitializeKernel(void) = 0;

  protected:
    std::shared_ptr<cl::Buffer> solid_data_cl_; /*!< Data about solid */
    std::shared_ptr<cl::Buffer> label_data_cl_; /*!< Pointer storing the buffer about label data */
    std::weak_ptr<cl::Kernel> kernel_distance_cl_; /*!< OpenCL kernel computing distance between particles and solid */
    std::weak_ptr<cl::Kernel> kernel_project_to_cl_; /*!< OpenCL kernel moving particles to solid */
    std::weak_ptr<cl::Kernel> kernel_track_through_cl_; /*!< OpenCL kernel tracking particles through a solid */
    std::string tracking_kernel_option_; /*!< Preprocessor option for tracking */
    bool is_tracking_; /*!< Boolean enabling tracking */
};

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSSOLID_HH
