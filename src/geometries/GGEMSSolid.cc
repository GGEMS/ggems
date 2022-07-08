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
  \file GGEMSSolid.cc

  \brief GGEMS class for solid. This class store geometry about phantom or detector

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 2, 2020
*/

#include "GGEMS/geometries/GGEMSSolid.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/physics/GGEMSCrossSections.hh"
#include "GGEMS/randoms/GGEMSPseudoRandomGenerator.hh"
#include "GGEMS/maths/GGEMSGeometryTransformation.hh"
#include "GGEMS/geometries/GGEMSSolidBoxData.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolidData.hh"
#include "GGEMS/graphics/GGEMSOpenGLParaGrid.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolid::GGEMSSolid(void)
: number_of_voxels_(0), 
  kernel_option_("")
{
  GGcout("GGEMSSolid", "GGEMSSolid", 3) << "GGEMSSolid creating..." << GGendl;

  // Allocation of geometry transformation
  geometry_transformation_ = new GGEMSGeometryTransformation();
  data_reg_type_ = "";

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  number_activated_devices_ = opencl_manager.GetNumberOfActivatedDevice();

  solid_data_ = new cl::Buffer*[number_activated_devices_];
  label_data_ = new cl::Buffer*[number_activated_devices_];

  // Storing a kernel for each device
  kernel_particle_solid_distance_ = new cl::Kernel*[number_activated_devices_];
  kernel_project_to_solid_ = new cl::Kernel*[number_activated_devices_];
  kernel_track_through_solid_ = new cl::Kernel*[number_activated_devices_];

  is_scatter_ = false;

  #ifdef OPENGL_VISUALIZATION
  opengl_solid_ = nullptr;
  #endif

  GGcout("GGEMSSolid", "GGEMSSolid", 3) << "GGEMSSolid created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolid::~GGEMSSolid(void)
{
  GGcout("GGEMSSolid", "~GGEMSSolid", 3) << "GGEMSSolid erasing..." << GGendl;

  if (kernel_particle_solid_distance_) {
    delete[] kernel_particle_solid_distance_;
    kernel_particle_solid_distance_ = nullptr;
  }

  if (kernel_project_to_solid_) {
    delete[] kernel_project_to_solid_;
    kernel_project_to_solid_ = nullptr;
  }

  if (kernel_track_through_solid_) {
    delete[] kernel_track_through_solid_;
    kernel_track_through_solid_ = nullptr;
  }

  if (geometry_transformation_) {
    delete geometry_transformation_;
    geometry_transformation_ = nullptr;
  }

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  if (label_data_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(label_data_[i], number_of_voxels_*sizeof(GGuchar), i);
    }
    delete[] label_data_;
    label_data_ = nullptr;
  }

  if (solid_data_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(solid_data_[i], sizeof(GGEMSSolidBoxData), i);
    }
    delete[] solid_data_;
    solid_data_ = nullptr;
  }

  GGcout("GGEMSSolid", "~GGEMSSolid", 3) << "GGEMSSolid erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::EnableTracking(void)
{
  kernel_option_ += " -DGGEMS_TRACKING";
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::AddKernelOption(std::string const& option)
{
  kernel_option_ += " -DTLE";
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::SetRotation(GGfloat3 const& rotation_xyz)
{
  geometry_transformation_->SetRotation(rotation_xyz);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifdef OPENGL_VISUALIZATION
void GGEMSSolid::SetXAngleOpenGL(GLfloat const& angle_x) const
{
  if (opengl_solid_)
    opengl_solid_->SetXAngle(angle_x);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::SetYAngleOpenGL(GLfloat const& angle_y) const
{
  if (opengl_solid_)
    opengl_solid_->SetYAngle(angle_y);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::SetZAngleOpenGL(GLfloat const& angle_z) const
{
  if (opengl_solid_)
    opengl_solid_->SetZAngle(angle_z);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::SetXUpdateAngleOpenGL(GLfloat const& update_angle_x) const
{
  if (opengl_solid_)
    opengl_solid_->SetXUpdateAngle(update_angle_x);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::SetYUpdateAngleOpenGL(GLfloat const& update_angle_y) const
{
  if (opengl_solid_)
    opengl_solid_->SetYUpdateAngle(update_angle_y);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::SetZUpdateAngleOpenGL(GLfloat const& update_angle_z) const
{
  if (opengl_solid_)
    opengl_solid_->SetZUpdateAngle(update_angle_z);
}
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::SetPosition(GGfloat3 const& position_xyz)
{
  geometry_transformation_->SetTranslation(position_xyz);

  #ifdef OPENGL_VISUALIZATION
  if (opengl_solid_)
    opengl_solid_->SetPosition(position_xyz.s[0], position_xyz.s[1], position_xyz.s[2]);
  #endif
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::SetVisible(bool const& is_visible)
{
  #ifdef OPENGL_VISUALIZATION
  if (opengl_solid_)
    opengl_solid_->SetVisible(is_visible);
  #endif
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::BuildOpenGL(void) const
{
  #ifdef OPENGL_VISUALIZATION
  if (opengl_solid_)
    opengl_solid_->Build();
  #endif
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::SetMaterialName(std::string const& material_name) const
{
  #ifdef OPENGL_VISUALIZATION
  if (opengl_solid_)
    opengl_solid_->SetMaterial(material_name);
  #endif
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::SetCustomMaterialColor(MaterialRGBColorUMap const& custom_material_rgb)
{
  #ifdef OPENGL_VISUALIZATION
  if (opengl_solid_)
    opengl_solid_->SetCustomMaterialColor(custom_material_rgb);
  #endif
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::SetMaterialVisible(MaterialVisibleUMap const& material_visible)
{
  #ifdef OPENGL_VISUALIZATION
  if (opengl_solid_)
    opengl_solid_->SetMaterialVisible(material_visible);
  #endif
}
