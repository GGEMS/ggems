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
  \file GGEMSVoxelizedSolid.cc

  \brief GGEMS class for voxelized solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday June 10, 2020
*/

#include "GGEMS/geometries/GGEMSVoxelizedSolid.hh"
#include "GGEMS/geometries/GGEMSGeometryConstants.hh"
#include "GGEMS/io/GGEMSMHDImage.hh"
#include "GGEMS/maths/GGEMSGeometryTransformation.hh"
#include "GGEMS/graphics/GGEMSOpenGLParaGrid.hh"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

// Local representability check only; independent of placement and orientation.
GGfloat ComputeVoxelizedGeometryTolerance(GGEMSVoxelizedSolidData const& data)
{
  auto reject = [](GGint axis, char const* reason) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Invalid local FP32 voxel geometry on axis " << axis << ": " << reason
        << ". Use coarser voxels or a smaller local extent.";
    GGEMSMisc::ThrowException("GGEMSVoxelizedSolid", "LoadVolumeImage", oss.str());
  };

  GGfloat min_voxel_size = std::numeric_limits<GGfloat>::max();
  for (GGint j = 0; j < 3; ++j) {
    GGint const n = data.number_of_voxels_xyz_.s[j];
    GGfloat const h = data.voxel_sizes_xyz_.s[j];
    // Adjacent integer indices must convert exactly to binary32.
    if (n <= 0 || n > 16777216 || !std::isfinite(h) || h <= 0.0f) {
      reject(j, "invalid voxel count or spacing");
    }
    min_voxel_size = std::min(min_voxel_size, h);
  }

  // Spacings and GEOMETRY_TOLERANCE are in GGEMS internal millimeters.
  GGfloat const tolerance = std::min(GEOMETRY_TOLERANCE, 5.0e-04f * min_voxel_size);
  // Do not depend on optional OpenCL support for subnormal arithmetic.
  if (!std::isnormal(tolerance)) reject(-1, "tolerance is zero or subnormal");

  GGfloat const infinity = std::numeric_limits<GGfloat>::infinity();
  for (GGint j = 0; j < 3; ++j) {
    GGint const n = data.number_of_voxels_xyz_.s[j];
    GGfloat const h = data.voxel_sizes_xyz_.s[j];
    GGfloat const lo = data.obb_geometry_.border_min_xyz_.s[j];
    GGfloat const hi = data.obb_geometry_.border_max_xyz_.s[j];
    if (!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi)) {
      reject(j, "nonfinite or collapsed local extent");
    }
    // The kernel's last upper face can differ from hi. Monotonic face
    // construction lets these endpoints bound all faces without a voxel scan.
    GGfloat const last = lo + static_cast<GGfloat>(n - 1) * h;
    GGfloat const last_upper = last + h;
    GGfloat const magnitude = std::max({std::fabs(lo), std::fabs(hi), std::fabs(last_upper)});
    GGfloat const spacing = std::nextafter(magnitude, infinity) - magnitude;
    if (!std::isfinite(last_upper) || !std::isfinite(spacing) || tolerance < spacing ||
        !std::isfinite(magnitude + tolerance) ||
        !(magnitude - tolerance < magnitude && magnitude < magnitude + tolerance)) {
      reject(j, "tolerance cannot resolve the largest local boundary");
    }
    // One local spacing on either side keeps the two tolerance bands distinct.
    // The fixed relative factor also leaves h >= about 2000 local spacings.
    if (!(h > 2.0f * (tolerance + spacing) &&
          lo + tolerance < hi - tolerance && last + tolerance < hi - tolerance)) {
      reject(j, "neighboring boundaries or tolerance bands are not distinct");
    }
  }
  return tolerance;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedSolid::GGEMSVoxelizedSolid(std::string const& volume_header_filename, std::string const& range_filename, std::string const& data_reg_type)
: GGEMSSolid(),
  volume_header_filename_(volume_header_filename),
  range_filename_(range_filename)
{
  GGcout("GGEMSVoxelizedSolid", "GGEMSVoxelizedSolid", 3) << "GGEMSVoxelizedSolid creating..." << GGendl;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Loop over the device
  for (GGsize d = 0; d < number_activated_devices_; ++d) {
    // Allocating memory on OpenCL device
    solid_data_[d] = opencl_manager.Allocate(nullptr, sizeof(GGEMSVoxelizedSolidData), d, CL_MEM_READ_WRITE, "GGEMSVoxelizedSolid");
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
      GGEMSMisc::ThrowException("GGEMSVoxelizedSolid", "GGEMSVoxelizedSolid", oss.str());
    }
  }

  GGcout("GGEMSVoxelizedSolid", "GGEMSVoxelizedSolid", 3) << "GGEMSVoxelizedSolid created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedSolid::~GGEMSVoxelizedSolid(void)
{
  GGcout("GGEMSVoxelizedSolid", "GGEMSVoxelizedSolid", 3) << "GGEMSVoxelizedSolid erasing..." << GGendl;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  if (solid_data_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(solid_data_[i], sizeof(GGEMSVoxelizedSolidData), i);
    }
    delete[] solid_data_;
    solid_data_ = nullptr;
  }

  GGcout("GGEMSVoxelizedSolid", "GGEMSVoxelizedSolid", 3) << "GGEMSVoxelizedSolid erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedSolid::InitializeKernel(void)
{
  GGcout("GGEMSVoxelizedSolid", "InitializeKernel", 3) << "Initializing kernel for voxelized solid..." << GGendl;

  // Getting OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Getting the path to kernel
  std::string openCL_kernel_path = OPENCL_KERNEL_PATH;
  std::string particle_solid_distance_filename = openCL_kernel_path + "/ParticleSolidDistanceGGEMSVoxelizedSolid.cl";
  std::string project_to_filename = openCL_kernel_path + "/ProjectToGGEMSVoxelizedSolid.cl";
  std::string track_through_filename = openCL_kernel_path + "/TrackThroughGGEMSVoxelizedSolid.cl";

  // Compiling the kernels
  opencl_manager.CompileKernel(particle_solid_distance_filename, "particle_solid_distance_ggems_voxelized_solid", kernel_particle_solid_distance_, nullptr, const_cast<char*>(kernel_option_.c_str()));
  opencl_manager.CompileKernel(project_to_filename, "project_to_ggems_voxelized_solid", kernel_project_to_solid_, nullptr, const_cast<char*>(kernel_option_.c_str()));
  opencl_manager.CompileKernel(track_through_filename, "track_through_ggems_voxelized_solid", kernel_track_through_solid_, nullptr, const_cast<char*>(kernel_option_.c_str()));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedSolid::Initialize(GGEMSMaterials* materials)
{
  GGcout("GGEMSVoxelizedSolid", "Initialize", 3) << "Initializing voxelized solid..." << GGendl;

  // Initializing kernels and loading image
  InitializeKernel();
  LoadVolumeImage(materials);

  // Creating volume for OpenGL
  // Get some infos for grid
  #ifdef OPENGL_VISUALIZATION
  GGEMSOpenGLManager& opengl_manager = GGEMSOpenGLManager::GetInstance();

  if (opengl_manager.IsOpenGLActivated()) {
    GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

    GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_[0], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSVoxelizedSolidData), 0);

    opengl_solid_ = new GGEMSOpenGLParaGrid(
      static_cast<GGsize>(solid_data_device->number_of_voxels_xyz_.s[0]),
      static_cast<GGsize>(solid_data_device->number_of_voxels_xyz_.s[1]),
      static_cast<GGsize>(solid_data_device->number_of_voxels_xyz_.s[2]),
      solid_data_device->voxel_sizes_xyz_.s[0],
      solid_data_device->voxel_sizes_xyz_.s[1],
      solid_data_device->voxel_sizes_xyz_.s[2],
      true // Draw midplanes
    );

    // Release the pointer
    opencl_manager.ReleaseDeviceBuffer(solid_data_[0], solid_data_device, 0);

    // Loading labels and materials for OpenGL
    opengl_solid_->SetMaterial(materials, label_data_[0], number_of_voxels_);
  }
  #endif
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedSolid::UpdateTransformationMatrix(GGsize const& thread_index)
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Copy information to OBB
  GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_[thread_index], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSVoxelizedSolidData), thread_index);
  GGfloat44* transformation_matrix_device = opencl_manager.GetDeviceBuffer<GGfloat44>(geometry_transformation_->GetTransformationMatrix(thread_index), CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGfloat44), thread_index);

  for (GGint i = 0; i < 4; ++i) {
    solid_data_device->obb_geometry_.matrix_transformation_.m0_[i] = transformation_matrix_device->m0_[i];
    solid_data_device->obb_geometry_.matrix_transformation_.m1_[i] = transformation_matrix_device->m1_[i];
    solid_data_device->obb_geometry_.matrix_transformation_.m2_[i] = transformation_matrix_device->m2_[i];
    solid_data_device->obb_geometry_.matrix_transformation_.m3_[i] = transformation_matrix_device->m3_[i];
  }

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_[thread_index], solid_data_device, thread_index);
  opencl_manager.ReleaseDeviceBuffer(geometry_transformation_->GetTransformationMatrix(thread_index), transformation_matrix_device, thread_index);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat3 GGEMSVoxelizedSolid::GetVoxelSizes(GGsize const& thread_index) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_[thread_index], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSVoxelizedSolidData), thread_index);

  GGfloat3 voxel_sizes = solid_data_device->voxel_sizes_xyz_;

  opencl_manager.ReleaseDeviceBuffer(solid_data_[thread_index], solid_data_device, thread_index);

  return voxel_sizes;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOBB GGEMSVoxelizedSolid::GetOBBGeometry(GGsize const& thread_index) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_[thread_index], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSVoxelizedSolidData), thread_index);

  GGEMSOBB obb_geometry = solid_data_device->obb_geometry_;

  opencl_manager.ReleaseDeviceBuffer(solid_data_[thread_index], solid_data_device, thread_index);

  return obb_geometry;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedSolid::PrintInfos(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Loop over the device
  for (GGsize d = 0; d < number_activated_devices_; ++d) {
    // Get pointer on OpenCL device
    GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_[d], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSVoxelizedSolidData), d);

    // Get the index of device
    GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(d);

    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "GGEMSVoxelizedSolid Infos:" << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "--------------------------" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "Voxelized solid on device: " << opencl_manager.GetDeviceName(device_index) << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "* Dimension: " << solid_data_device->number_of_voxels_xyz_.s[0] << " " << solid_data_device->number_of_voxels_xyz_.s[1] << " " << solid_data_device->number_of_voxels_xyz_.s[2] << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "* Number of voxels: " << solid_data_device->number_of_voxels_ << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "* Size of voxels: (" << solid_data_device->voxel_sizes_xyz_.s[0] /mm << "x" << solid_data_device->voxel_sizes_xyz_.s[1]/mm << "x" << solid_data_device->voxel_sizes_xyz_.s[2]/mm << ") mm3" << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "* Oriented bounding box (OBB) in local position:" << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "    - X: " << solid_data_device->obb_geometry_.border_min_xyz_.s[0] << " <-> " << solid_data_device->obb_geometry_.border_max_xyz_.s[0] << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "    - Y: " << solid_data_device->obb_geometry_.border_min_xyz_.s[1] << " <-> " << solid_data_device->obb_geometry_.border_max_xyz_.s[1] << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "    - Z: " << solid_data_device->obb_geometry_.border_min_xyz_.s[2] << " <-> " << solid_data_device->obb_geometry_.border_max_xyz_.s[2] << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "    - Transformation matrix:" << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "    [" << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m0_[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m0_[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m0_[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m0_[3] << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m1_[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m1_[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m1_[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m1_[3] << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m2_[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m2_[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m2_[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m2_[3] << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m3_[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m3_[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m3_[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m3_[3] << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "    ]" << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "* Solid index: " << solid_data_device->solid_id_ << GGendl;
    GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << GGendl;

    // Release the pointer
    opencl_manager.ReleaseDeviceBuffer(solid_data_[d], solid_data_device, d);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedSolid::LoadVolumeImage(GGEMSMaterials* materials)
{
  GGcout("GGEMSVoxelizedSolid", "LoadVolumeImage", 3) << "Loading volume image from mhd file..." << GGendl;

  // Read MHD input file
  GGEMSMHDImage mhd_input_phantom;
  // Loop over the device
  for (GGsize d = 0; d < number_activated_devices_; ++d) {
    mhd_input_phantom.Read(volume_header_filename_, solid_data_[d], d);

    // Initialize the per-solid policy before raw loading or any navigation.
    GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
    GGEMSVoxelizedSolidData* data = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_[d], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSVoxelizedSolidData), d);
    try {
      data->geometry_tolerance_ = ComputeVoxelizedGeometryTolerance(*data);
    }
    catch (...) {
      opencl_manager.ReleaseDeviceBuffer(solid_data_[d], data, d);
      throw;
    }
    opencl_manager.ReleaseDeviceBuffer(solid_data_[d], data, d);
  }

  // Get the name of raw file from mhd reader
  std::string output_dir = mhd_input_phantom.GetOutputDirectory();
  std::string raw_filename = output_dir + mhd_input_phantom.GetRawMDHfilename();

  // Get the type
  std::string const kDataType = mhd_input_phantom.GetDataMHDType();

  // Convert raw data to material id data
  if (!kDataType.compare("MET_CHAR")) {
    ConvertImageToLabel<GGchar>(raw_filename, range_filename_, materials);
  }
  else if (!kDataType.compare("MET_UCHAR")) {
    ConvertImageToLabel<GGuchar>(raw_filename, range_filename_, materials);
  }
  else if (!kDataType.compare("MET_SHORT")) {
    ConvertImageToLabel<GGshort>(raw_filename, range_filename_, materials);
  }
  else if (!kDataType.compare("MET_USHORT")) {
    ConvertImageToLabel<GGushort>(raw_filename, range_filename_, materials);
  }
  else if (!kDataType.compare("MET_INT")) {
    ConvertImageToLabel<GGint>(raw_filename, range_filename_, materials);
  }
  else if (!kDataType.compare("MET_UINT")) {
    ConvertImageToLabel<GGuint>(raw_filename, range_filename_, materials);
  }
  else if (!kDataType.compare("MET_FLOAT")) {
    ConvertImageToLabel<GGfloat>(raw_filename, range_filename_, materials);
  }
}
