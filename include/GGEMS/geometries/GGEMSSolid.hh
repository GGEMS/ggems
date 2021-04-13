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

#include "GGEMS/io/GGEMSTextReader.hh"
#include "GGEMS/io/GGEMSHistogramMode.hh"
#include "GGEMS/tools/GGEMSRAMManager.hh"
#include "GGEMS/navigators/GGEMSNavigatorManager.hh"

class GGEMSGeometryTransformation;

/*!
  \class GGEMSSolid
  \brief GGEMS class for solid informations
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
    virtual ~GGEMSSolid(void);

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
      \fn void EnableTracking(void)
      \brief Enabling tracking infos during simulation
    */
    void EnableTracking(void);

    /*!
      \fn inline cl::Buffer* GetSolidData(GGsize const& thread_index) const
      \param thread_index - index of the thread (= activated device index)
      \brief get the informations about the solid geometry
      \return header data OpenCL pointer about solid
    */
    inline cl::Buffer* GetSolidData(GGsize const& thread_index) const {return solid_data_[thread_index];};

    /*!
      \fn inline cl::Buffer* GetLabelData(GGsize const& thread_index) const
      \param thread_index - index of the thread (= activated device index)
      \brief get buffer to label buffer
      \return data on label infos
    */
    inline cl::Buffer* GetLabelData(GGsize const& thread_index) const {return label_data_[thread_index];};

    /*!
      \fn void SetRotation(GGfloat3 const& rotation_xyz)
      \param rotation_xyz - rotation in X, Y and Z
      \brief set a rotation for solid
    */
    void SetRotation(GGfloat3 const& rotation_xyz);

    /*!
      \fn void SetPosition(GGfloat3 const& position_xyz)
      \param position_xyz - position in X, Y and Z
      \brief set a position for solid
    */
    void SetPosition(GGfloat3 const& position_xyz);

    /*!
      \fn void SetSolidID(GGsize const& solid_id, GGsize const& thread_index)
      \param solid_id - index of the solid
      \param thread_index - index of the thread (= activated device index)
      \brief set the global solid index
    */
    template<typename T>
    void SetSolidID(GGsize const& solid_id, GGsize const& thread_index);

    /*!
      \fn void GetTransformationMatrix(GGsize const& thread_index)
      \param thread_index - index of the thread (= activated device index)
      \brief Get the transformation matrix for solid object
    */
    virtual void GetTransformationMatrix(GGsize const& thread_index) = 0;

    /*!
      \fn void Initialize(std::weak_ptr<GGEMSMaterials> materials)
      \param materials - pointer on GGEMS materials
      \brief Initialize solid for geometric navigation
    */
    virtual void Initialize(GGEMSMaterials* materials) = 0;

    /*!
      \fn void PrintInfos(void) const
      \brief printing infos about solid
    */
    virtual void PrintInfos(void) const = 0;

    /*!
      \fn GGEMSHistogramMode* GetHistogram(void)
      \return pointer on hit
      \brief return the point on histogram
    */
    inline GGEMSHistogramMode* GetHistogram(void) {return &histogram_;};

    /*!
      \fn std::string GetRegisteredDataType(void) const
      \return the type of registered data
      \brief get the type of registered data
    */
    inline std::string GetRegisteredDataType(void) const {return data_reg_type_;};

  protected:
    /*!
      \fn void InitializeKernel(void)
      \brief Initialize kernel for particle solid distance
    */
    virtual void InitializeKernel(void) = 0;

  protected:
    // Solid data infos and label (for voxelized solid)
    cl::Buffer** solid_data_; /*!< Data about solid */
    cl::Buffer** label_data_; /*!< Pointer storing the buffer about label data, useful for voxelized solid only */
    GGsize number_activated_devices_; /*!< Number of activated device */

    // Geometric transformation applyied to solid
    GGEMSGeometryTransformation* geometry_transformation_; /*!< Pointer storing the geometry transformation */

    // OpenCL kernels and options for kernel
    cl::Kernel** kernel_particle_solid_distance_; /*!< OpenCL kernel computing distance between particles and solid */
    cl::Kernel** kernel_project_to_solid_; /*!< OpenCL kernel moving particles to solid */
    cl::Kernel** kernel_track_through_solid_; /*!< OpenCL kernel tracking particles through a solid */
    std::string kernel_option_; /*!< Preprocessor option for kernel */

    // Output data
    std::string data_reg_type_; /*!< Type of registering data */
    GGEMSHistogramMode histogram_; /*!< Storing histogram useful for GGEMSSystem only */
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<typename T>
void GGEMSSolid::SetSolidID(GGsize const& solid_id, GGsize const& thread_index)
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device
  T* solid_data_device = opencl_manager.GetDeviceBuffer<T>(solid_data_[thread_index], sizeof(T), thread_index);

  solid_data_device->solid_id_ = static_cast<GGint>(solid_id);

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_[thread_index], solid_data_device, thread_index);
}

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSSOLID_HH
