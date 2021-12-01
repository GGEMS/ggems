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
class GGEMSOpenGLVolume;

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
      \fn void SetXAngleOpenGL(GLfloat const& angle_x) const
      \param angle_x - angle in X
      \brief set angle of rotation in X
    */
    void SetXAngleOpenGL(GLfloat const& angle_x) const;

    /*!
      \fn void SetYAngleOpenGL(GLfloat const& angle_y) const
      \param angle_y - angle in Y
      \brief set angle of rotation in Y
    */
    void SetYAngleOpenGL(GLfloat const& angle_y) const;

    /*!
      \fn void SetZAngleOpenGL(GLfloat const& angle_z) const
      \param angle_z - angle in Z
      \brief set angle of rotation in Z
    */
    void SetZAngleOpenGL(GLfloat const& angle_z) const;

    /*!
      \fn void SetXUpdateAngleOpenGL(GLfloat const& update_angle_x) const
      \param update_angle_x - angle in X
      \brief set angle of rotation in X (after translation)
    */
    void SetXUpdateAngleOpenGL(GLfloat const& update_angle_x) const;

    /*!
      \fn void SetYUpdateAngleOpenGL(GLfloat const& update_angle_y) const
      \param update_angle_y - angle in Y
      \brief set angle of rotation in Y (after translation)
    */
    void SetYUpdateAngleOpenGL(GLfloat const& update_angle_y) const;

    /*!
      \fn void SetZUpdateAngleOpenGL(GLfloat const& update_angle_z) const
      \param update_angle_z - angle in Z
      \brief set angle of rotation in Z (after translation)
    */
    void SetZUpdateAngleOpenGL(GLfloat const& update_angle_z) const;

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
      \fn void UpdateTransformationMatrix(GGsize const& thread_index)
      \param thread_index - index of the thread (= activated device index)
      \brief Update transformation matrix for solid object
    */
    virtual void UpdateTransformationMatrix(GGsize const& thread_index) = 0;

    /*!
      \fn void Initialize(GGEMSMaterials* materials)
      \param materials - pointer on GGEMS materials
      \brief Initialize solid for geometric navigation
    */
    virtual void Initialize(GGEMSMaterials* materials) = 0;

    /*!
      \fn void EnableScatter(void)
      \brief Activate scatter registration
    */
    virtual void EnableScatter(void) = 0;

    /*!
      \fn void PrintInfos(void) const
      \brief printing infos about solid
    */
    virtual void PrintInfos(void) const = 0;

    /*!
      \fn std::string GetRegisteredDataType(void) const
      \return the type of registered data
      \brief get the type of registered data
    */
    inline std::string GetRegisteredDataType(void) const {return data_reg_type_;};

    /*!
      \fn cl::Kernel* GetKernelParticleSolidDistance(GGsize const& thread_index) const
      \param thread_index - index of activated device (thread index)
      \return pointer to kernel associated to a device
      \brief get the pointer to kernel associated to a device
    */
    inline cl::Kernel* GetKernelParticleSolidDistance(GGsize const& thread_index) const {return kernel_particle_solid_distance_[thread_index];}

    /*!
      \fn cl::Kernel* GetKernelProjectToSolid(GGsize const& thread_index) const
      \param thread_index - index of activated device (thread index)
      \return pointer to kernel associated to a device
      \brief get the pointer to kernel associated to a device
    */
    inline cl::Kernel* GetKernelProjectToSolid(GGsize const& thread_index) const {return kernel_project_to_solid_[thread_index];}

    /*!
      \fn cl::Kernel* GetKernelTrackThroughSolid(GGsize const& thread_index) const
      \param thread_index - index of activated device (thread index)
      \return pointer to kernel associated to a device
      \brief get the pointer to kernel associated to a device
    */
    inline cl::Kernel* GetKernelTrackThroughSolid(GGsize const& thread_index) const {return kernel_track_through_solid_[thread_index];}

    /*!
      \fn GGEMSHistogramMode* GetHistogram(GGsize const& thread_index)
      \param thread_index - index of activated device (thread index)
      \return pointer on histogram
      \brief return the point on histogram
    */
    inline cl::Buffer* GetHistogram(GGsize const& thread_index) const {return histogram_.histogram_[thread_index];}

    /*!
      \fn GGEMSHistogramMode* GetScatterHistogram(GGsize const& thread_index)
      \param thread_index - index of activated device (thread index)
      \return pointer on scatter histogram
      \brief return the point on scatter histogram
    */
    inline cl::Buffer* GetScatterHistogram(GGsize const& thread_index) const {return histogram_.scatter_[thread_index];}

    /*!
      \fn void SetVisible(bool const& is_visible)
      \param is_visible - true if navigator is drawn using OpenGL
      \brief set to true to draw navigator
    */
    void SetVisible(bool const& is_visible);

    /*!
      \fn void SetColorName(std::string const& color) const
      \param color - Color name
      \brief setting color for OpenGL volume
    */
    void SetColorName(std::string const& color) const;

    /*!
      \fn void SetMaterialName(std::string const& material_name) const
      \param material_name - name of the material
      \brief set material name to find color for OpenGL
    */
    void SetMaterialName(std::string const& material_name) const;

    /*!
      \fn void BuildOpenGL(void) const
      \brief building OpenGL volume in GGEMS
    */
    void BuildOpenGL(void) const;

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
    std::size_t number_of_voxels_; /*!< Number of voxel 1 for GGEMSSolidBox */
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
    bool is_scatter_; /*!< boolean storing scatter in solid */

    // OpenGL volume
    #ifdef OPENGL_VISUALIZATION
    GGEMSOpenGLVolume* opengl_solid_; /*!< OpenGL solid */
    #endif
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
  T* solid_data_device = opencl_manager.GetDeviceBuffer<T>(solid_data_[thread_index], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(T), thread_index);

  solid_data_device->solid_id_ = static_cast<GGint>(solid_id);

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_[thread_index], solid_data_device, thread_index);
}

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSSOLID_HH
