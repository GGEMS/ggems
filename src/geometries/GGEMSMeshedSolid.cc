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
  \file GGEMSMeshedSolid.cc

  \brief GGEMS class for meshed solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday June 22, 2022
*/

#include "GGEMS/geometries/GGEMSMeshedSolid.hh"
#include "GGEMS/maths/GGEMSGeometryTransformation.hh"
#include "GGEMS/io/GGEMSSTLReader.hh"
#include "GGEMS/graphics/GGEMSOpenGLMesh.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMeshedSolid::GGEMSMeshedSolid(std::string const& meshed_phantom_name, std::string const& data_reg_type)
: GGEMSSolid(),
  meshed_phantom_name_(meshed_phantom_name)
{
  GGcout("GGEMSMeshedSolid", "GGEMSMeshedSolid", 3) << "GGEMSMeshedSolid creating..." << GGendl;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Loop over the device
  for (GGsize d = 0; d < number_activated_devices_; ++d) {
    // Allocating memory on OpenCL device
    solid_data_[d] = opencl_manager.Allocate(nullptr, sizeof(GGEMSMeshedSolidData), d, CL_MEM_READ_WRITE, "GGEMSMeshedSolid");
  }

  // Local axis for phantom. Voxelized solid used only for phantom
  geometry_transformation_->SetAxisTransformation(
    {
      {1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f}
    }
  );

  // Checking format registration
  data_reg_type_ = data_reg_type;
  if (!data_reg_type.empty()) {
    if (data_reg_type == "DOSIMETRY") {
      kernel_option_ += " -DDOSIMETRY";
    }
    else {
      std::ostringstream oss(std::ostringstream::out);
      oss << "False registration type name!!!" << std::endl;
      oss << "Registration type is :" << std::endl;
      oss << "    - DOSIMETRY" << std::endl;
      //oss << "    - LISTMODE" << std::endl;
      //oss << "    - HISTOGRAM" << std::endl;
      GGEMSMisc::ThrowException("GGEMSMeshedSolid", "GGEMSMeshedSolid", oss.str());
    }
  }

  GGcout("GGEMSMeshedSolid", "GGEMSMeshedSolid", 3) << "GGEMSMeshedSolid created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMeshedSolid::~GGEMSMeshedSolid(void)
{
  GGcout("GGEMSMeshedSolid", "GGEMSMeshedSolid", 3) << "GGEMSMeshedSolid erasing..." << GGendl;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  if (triangles_) {
    for (std::size_t i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.SVMDeallocate(triangles_[i], number_of_triangles_ * sizeof(GGEMSTriangle3), i, "GGEMSMeshedSolid");
    }
    delete[] triangles_;
    triangles_ = nullptr;
  }

  GGcout("GGEMSMeshedSolid", "GGEMSMeshedSolid", 3) << "GGEMSMeshedSolid erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMeshedSolid::Initialize(GGEMSMaterials*)
{

  GGcout("GGEMSMeshedSolid", "Initialize", 3) << "Initializing meshed solid..." << GGendl;

  // Initializing kernels and loading image
  // InitializeKernel();
  LoadVolumeImage();

  // Building octree for mesh

  // Creating volume for OpenGL
  // Get some infos for grid
  #ifdef OPENGL_VISUALIZATION
  GGEMSOpenGLManager& opengl_manager = GGEMSOpenGLManager::GetInstance();
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  if (opengl_manager.IsOpenGLActivated()) {
    // Mapping triangles
    opencl_manager.GetSVMData(
      triangles_[0],
      sizeof(GGEMSTriangle3) * number_of_triangles_,
      0,
      CL_TRUE,
      CL_MAP_READ
    );

    opengl_solid_ = new GGEMSOpenGLMesh(triangles_[0], number_of_triangles_);

    // Unmapping triangles
    opencl_manager.ReleaseSVMData(triangles_[0], 0);

    //GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
/*
    GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_[0], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSVoxelizedSolidData), 0);

    opengl_solid_ = new GGEMSOpenGLParaGrid(
      static_cast<GGsize>(solid_data_device->number_of_voxels_xyz_.s[0]),
      static_cast<GGsize>(solid_data_device->number_of_voxels_xyz_.s[1]),
      static_cast<GGsize>(solid_data_device->number_of_voxels_xyz_.s[2]),
      solid_data_device->voxel_sizes_xyz_.s[0],
      solid_data_device->voxel_sizes_xyz_.s[1],
      solid_data_device->voxel_sizes_xyz_.s[2],
      true // Draw midplanes
    );*/

    // Release the pointer
    //opencl_manager.ReleaseDeviceBuffer(solid_data_[0], solid_data_device, 0);

    // Loading labels and materials for OpenGL
    // opengl_solid_->SetMaterial(materials, label_data_[0], number_of_voxels_);
  }
  #endif
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMeshedSolid::LoadVolumeImage(void)
{
  GGcout("GGEMSMeshedSolid", "LoadVolumeImage", 3) << "Loading volume image from stl file..." << GGendl;

  // Read STL input file
  GGEMSSTLReader stl_input_phantom;

  // Load triangles
  stl_input_phantom.Read(meshed_phantom_name_);

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Allocating memory for triangles in each engine
  number_of_triangles_ = stl_input_phantom.GetNumberOfTriangles();
  triangles_ = new GGEMSTriangle3*[number_activated_devices_];

  // Load triangles to OpenCL devices
  for (std::size_t i = 0; i < number_activated_devices_; ++i) {
    triangles_[i] = opencl_manager.SVMAllocate<GGEMSTriangle3>(
      number_of_triangles_ * sizeof(GGEMSTriangle3),
      i,
      CL_MEM_READ_WRITE,
      0,
      "GGEMSMeshedSolid"
    );

    // Mapping triangles
    opencl_manager.GetSVMData(
      triangles_[i],
      sizeof(GGEMSTriangle3) * number_of_triangles_,
      i,
      CL_TRUE,
      CL_MAP_WRITE
    );

    // Loading triangles from STL
    stl_input_phantom.LoadTriangles(triangles_[i]);
  
    // Unmapping triangles
    opencl_manager.ReleaseSVMData(triangles_[i], i);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMeshedSolid::PrintInfos(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Loop over the device
  for (GGsize d = 0; d < number_activated_devices_; ++d) {
    // Get pointer on OpenCL device
    GGEMSMeshedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSMeshedSolidData>(solid_data_[d], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSMeshedSolidData), d);

    // Get the index of device
    GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(d);

    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "GGEMSMeshedSolid Infos:" << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "--------------------------" << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "Meshed solid on device: " << opencl_manager.GetDeviceName(device_index) << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "Mesh filename: " << meshed_phantom_name_ << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "Number of triangles: " << number_of_triangles_ << GGendl;
    //GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "* Dimension: " << solid_data_device->number_of_voxels_xyz_.s[0] << " " << solid_data_device->number_of_voxels_xyz_.s[1] << " " << solid_data_device->number_of_voxels_xyz_.s[2] << GGendl;
    //GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "* Number of voxels: " << solid_data_device->number_of_voxels_ << GGendl;
    //GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "* Size of voxels: (" << solid_data_device->voxel_sizes_xyz_.s[0] /mm << "x" << solid_data_device->voxel_sizes_xyz_.s[1]/mm << "x" << solid_data_device->voxel_sizes_xyz_.s[2]/mm << ") mm3" << GGendl;
    //GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "* Oriented bounding box (OBB) in local position:" << GGendl;
    //GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "    - X: " << solid_data_device->obb_geometry_.border_min_xyz_.s[0] << " <-> " << solid_data_device->obb_geometry_.border_max_xyz_.s[0] << GGendl;
    //GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "    - Y: " << solid_data_device->obb_geometry_.border_min_xyz_.s[1] << " <-> " << solid_data_device->obb_geometry_.border_max_xyz_.s[1] << GGendl;
    //GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "    - Z: " << solid_data_device->obb_geometry_.border_min_xyz_.s[2] << " <-> " << //solid_data_device->obb_geometry_.border_max_xyz_.s[2] << GGendl;
    //GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "    - Transformation matrix:" << GGendl;
    //GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "    [" << GGendl;
    //GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m0_[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m0_[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m0_[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m0_[3] << GGendl;
    //GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m1_[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m1_[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m1_[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m1_[3] << GGendl;
    //GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m2_[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m2_[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m2_[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m2_[3] << GGendl;
    //GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m3_[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m3_[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m3_[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m3_[3] << GGendl;
    //GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "    ]" << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "* Solid index: " << solid_data_device->solid_id_ << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << GGendl;

    // Release the pointer
    opencl_manager.ReleaseDeviceBuffer(solid_data_[d], solid_data_device, d);
  }
}
