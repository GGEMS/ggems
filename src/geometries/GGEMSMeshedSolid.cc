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
#include "GGEMS/geometries/GGEMSOctree.hh"

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
    label_data_[d] = nullptr;
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

  octree_ = nullptr;

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

  if (octree_) {
    delete octree_;
  }

  if (solid_data_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(solid_data_[i], sizeof(GGEMSMeshedSolidData), i);
    }
    delete[] solid_data_;
    solid_data_ = nullptr;
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
  InitializeKernel();
  LoadVolumeImage();

  // Creating volume for OpenGL
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
  }
  #endif
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMeshedSolid::InitializeKernel(void)
{
  GGcout("GGEMSMeshedSolid", "InitializeKernel", 3) << "Initializing kernel for mesh solid..." << GGendl;

  // Getting OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Getting the path to kernel
  std::string openCL_kernel_path = OPENCL_KERNEL_PATH;
  std::string particle_solid_distance_filename = openCL_kernel_path + "/ParticleSolidDistanceGGEMSMeshSolid.cl";
  std::string project_to_filename = openCL_kernel_path + "/ProjectToGGEMSMeshSolid.cl";
  std::string track_through_filename = openCL_kernel_path + "/TrackThroughGGEMSMeshSolid.cl";

  // Compiling the kernels
  opencl_manager.CompileKernel(particle_solid_distance_filename, "particle_solid_distance_ggems_meshed_solid", kernel_particle_solid_distance_, nullptr, const_cast<char*>(kernel_option_.c_str()));
  opencl_manager.CompileKernel(project_to_filename, "project_to_ggems_meshed_solid", kernel_project_to_solid_, nullptr, const_cast<char*>(kernel_option_.c_str()));
  opencl_manager.CompileKernel(track_through_filename, "track_through_ggems_meshed_solid", kernel_track_through_solid_, nullptr, const_cast<char*>(kernel_option_.c_str()));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMeshedSolid::UpdateTransformationMatrix(GGsize const& thread_index)
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Copy information to OBB
  GGEMSMeshedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSMeshedSolidData>(solid_data_[thread_index], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSMeshedSolidData), thread_index);

  solid_data_device->obb_geometry_.matrix_transformation_.m0_[0] = 1.0f;
  solid_data_device->obb_geometry_.matrix_transformation_.m1_[1] = 1.0f;
  solid_data_device->obb_geometry_.matrix_transformation_.m2_[2] = 1.0f;
  solid_data_device->obb_geometry_.matrix_transformation_.m3_[3] = 1.0f;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_[thread_index], solid_data_device, thread_index);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMeshedSolid::UpdateTriangles(GGsize const& thread_index)
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Loop over device
  GGfloat44* transformation_matrix_device = opencl_manager.GetDeviceBuffer<GGfloat44>(geometry_transformation_->GetTransformationMatrix(thread_index), CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGfloat44), thread_index);

  // Computing new position of triangles
  // Mapping triangles
  opencl_manager.GetSVMData(
    triangles_[thread_index],
    sizeof(GGEMSTriangle3) * number_of_triangles_,
    0,
    CL_TRUE,
    CL_MAP_READ | CL_MAP_WRITE
  );

  // Translation
  GGEMSPoint3 translation = {
    transformation_matrix_device->m0_[3],
    transformation_matrix_device->m1_[3],
    transformation_matrix_device->m2_[3]
  };

  // Rotation
  GGEMSPoint3 row0 = {
    transformation_matrix_device->m0_[0],
    transformation_matrix_device->m0_[1],
    transformation_matrix_device->m0_[2]
  };

  GGEMSPoint3 row1 = {
    transformation_matrix_device->m1_[0],
    transformation_matrix_device->m1_[1],
    transformation_matrix_device->m1_[2]
  };

  GGEMSPoint3 row2 = {
    transformation_matrix_device->m2_[0],
    transformation_matrix_device->m2_[1],
    transformation_matrix_device->m2_[2]
  };

  // Loop over all triangles
  GGEMSPoint3 rotated_new_point;
  for (GGuint t = 0; t < number_of_triangles_; ++t) {
    // Loop over the 3 points
    for (int p = 0; p < 3; ++p) {
      rotated_new_point.x_ = Dot(row0, triangles_[thread_index][t].pts_[p]);
      rotated_new_point.y_ = Dot(row1, triangles_[thread_index][t].pts_[p]);
      rotated_new_point.z_ = Dot(row2, triangles_[thread_index][t].pts_[p]);

      triangles_[thread_index][t].pts_[p].x_ = rotated_new_point.x_ + translation.x_;
      triangles_[thread_index][t].pts_[p].y_ = rotated_new_point.y_ + translation.y_;
      triangles_[thread_index][t].pts_[p].z_ = rotated_new_point.z_ + translation.z_;
    }
    // Change position of bounding sphere
    rotated_new_point.x_ = Dot(row0, triangles_[thread_index][t].bounding_sphere_.center_);
    rotated_new_point.y_ = Dot(row1, triangles_[thread_index][t].bounding_sphere_.center_);
    rotated_new_point.z_ = Dot(row2, triangles_[thread_index][t].bounding_sphere_.center_);

    triangles_[thread_index][t].bounding_sphere_.center_.x_ = rotated_new_point.x_ + translation.x_;
    triangles_[thread_index][t].bounding_sphere_.center_.y_ = rotated_new_point.y_ + translation.y_;
    triangles_[thread_index][t].bounding_sphere_.center_.z_ = rotated_new_point.z_ + translation.z_;
  }

  // Unmapping triangles
  opencl_manager.ReleaseSVMData(triangles_[thread_index], thread_index);

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(geometry_transformation_->GetTransformationMatrix(thread_index), transformation_matrix_device, thread_index);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMeshedSolid::BuildOctree(GGint const& depth)
{
  // Compute center of octree
  GGEMSPoint3 octree_center = ComputeOctreeCenter();

  // Compute half_width of octree
  GGfloat octree_half_width[3];
  ComputeHalfWidthCenter(octree_half_width);

  // Creating octree and build it
  octree_ = new GGEMSOctree(depth, octree_half_width);
  octree_->Build(octree_center);

  // Inserting triangles
  octree_->InsertTriangles(triangles_, number_of_triangles_);

  GGEMSNode** nodes = octree_->GetNodes();

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Storing border of octree in OBB
  for (GGsize d = 0; d < number_activated_devices_; ++d) {
    GGEMSMeshedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSMeshedSolidData>(solid_data_[d], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSMeshedSolidData), d);

    solid_data_device->obb_geometry_.matrix_transformation_.m0_[0] = 1.0f;
    solid_data_device->obb_geometry_.matrix_transformation_.m1_[1] = 1.0f;
    solid_data_device->obb_geometry_.matrix_transformation_.m2_[2] = 1.0f;
    solid_data_device->obb_geometry_.matrix_transformation_.m3_[3] = 1.0f;

    solid_data_device->obb_geometry_.matrix_transformation_.m0_[3] = 0.0f;
    solid_data_device->obb_geometry_.matrix_transformation_.m1_[3] = 0.0f;
    solid_data_device->obb_geometry_.matrix_transformation_.m2_[3] = 0.0f;

    solid_data_device->obb_geometry_.border_min_xyz_.s[0] = octree_center.x_ - octree_half_width[0];
    solid_data_device->obb_geometry_.border_min_xyz_.s[1] = octree_center.y_ - octree_half_width[1];
    solid_data_device->obb_geometry_.border_min_xyz_.s[2] = octree_center.z_ - octree_half_width[2];

    solid_data_device->obb_geometry_.border_max_xyz_.s[0] = octree_center.x_ + octree_half_width[0];
    solid_data_device->obb_geometry_.border_max_xyz_.s[1] = octree_center.y_ + octree_half_width[1];
    solid_data_device->obb_geometry_.border_max_xyz_.s[2] = octree_center.z_ + octree_half_width[2];

    solid_data_device->nodes_ = nodes[d];
    solid_data_device->total_nodes_ = octree_->GetTotalNodes();


    // Release the pointer
    opencl_manager.ReleaseDeviceBuffer(solid_data_[d], solid_data_device, d);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPoint3 GGEMSMeshedSolid::ComputeOctreeCenter(void) const
{
  // Min and max points
  GGEMSPoint3 lo;
  lo.x_ = FLT_MAX; lo.y_ = FLT_MAX; lo.z_ = FLT_MAX;

  GGEMSPoint3 hi;
  hi.x_ = FLT_MIN; hi.y_ = FLT_MIN; hi.z_ = FLT_MIN;

  for (unsigned int i = 0; i < number_of_triangles_; ++i) {
    for (int j = 0; j < 3; ++j) { // Loop over points
      if (triangles_[0][i].pts_[j].x_ < lo.x_) lo.x_ = triangles_[0][i].pts_[j].x_;
      if (triangles_[0][i].pts_[j].y_ < lo.y_) lo.y_ = triangles_[0][i].pts_[j].y_;
      if (triangles_[0][i].pts_[j].z_ < lo.z_) lo.z_ = triangles_[0][i].pts_[j].z_;
      if (triangles_[0][i].pts_[j].x_ > hi.x_) hi.x_ = triangles_[0][i].pts_[j].x_;
      if (triangles_[0][i].pts_[j].y_ > hi.y_) hi.y_ = triangles_[0][i].pts_[j].y_;
      if (triangles_[0][i].pts_[j].z_ > hi.z_) hi.z_ = triangles_[0][i].pts_[j].z_;
    }
  }

  // Expand it by 2.5% so that all points are well interior
  GGfloat expanding_x = (hi.x_ - lo.x_) * 0.025f;
  GGfloat expanding_y = (hi.y_ - lo.y_) * 0.025f;
  GGfloat expanding_z = (hi.z_ - lo.z_) * 0.025f;

  lo.x_ -= expanding_x;
  hi.x_ += expanding_x;
  lo.y_ -= expanding_y;
  hi.y_ += expanding_y;
  lo.z_ -= expanding_z;
  hi.z_ += expanding_z;

  if (lo.x_ < 0.0f) lo.x_ = std::floor(lo.x_);
  else lo.x_ = std::ceil(lo.x_);

  if (lo.y_ < 0.0f) lo.y_ = std::floor(lo.y_);
  else lo.y_ = std::ceil(lo.y_);

  if (lo.z_ < 0.0f) lo.z_ = std::floor(lo.z_);
  else lo.z_ = std::ceil(lo.z_);

  if (hi.x_ < 0.0f) hi.x_ = std::floor(hi.x_);
  else hi.x_ = std::ceil(hi.x_);

  if (hi.y_ < 0.0f) hi.y_ = std::floor(hi.y_);
  else hi.y_ = std::ceil(hi.y_);

  if (hi.z_ < 0.0f) hi.z_ = std::floor(hi.z_);
  else hi.z_ = std::ceil(hi.z_);

  // Computing the center of octree box
  GGEMSPoint3 center;
  center.x_ = (hi.x_+lo.x_)*0.5f;
  center.y_ = (hi.y_+lo.y_)*0.5f;
  center.z_ = (hi.z_+lo.z_)*0.5f;

  return center;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMeshedSolid::ComputeHalfWidthCenter(GGfloat* half_width) const
{
  // Min and max points
  GGEMSPoint3 lo;
  lo.x_ = FLT_MAX; lo.y_ = FLT_MAX; lo.z_ = FLT_MAX;

  GGEMSPoint3 hi;
  hi.x_ = FLT_MIN; hi.y_ = FLT_MIN; hi.z_ = FLT_MIN;

  for (unsigned int i = 0; i < number_of_triangles_; ++i) {
    for (int j = 0; j < 3; ++j) { // Loop over points
      if (triangles_[0][i].pts_[j].x_ < lo.x_) lo.x_ = triangles_[0][i].pts_[j].x_;
      if (triangles_[0][i].pts_[j].y_ < lo.y_) lo.y_ = triangles_[0][i].pts_[j].y_;
      if (triangles_[0][i].pts_[j].z_ < lo.z_) lo.z_ = triangles_[0][i].pts_[j].z_;
      if (triangles_[0][i].pts_[j].x_ > hi.x_) hi.x_ = triangles_[0][i].pts_[j].x_;
      if (triangles_[0][i].pts_[j].y_ > hi.y_) hi.y_ = triangles_[0][i].pts_[j].y_;
      if (triangles_[0][i].pts_[j].z_ > hi.z_) hi.z_ = triangles_[0][i].pts_[j].z_;
    }
  }

  // Expand it by 2.5% so that all points are well interior
  GGfloat expanding_x = (hi.x_ - lo.x_) * 0.025f;
  GGfloat expanding_y = (hi.y_ - lo.y_) * 0.025f;
  GGfloat expanding_z = (hi.z_ - lo.z_) * 0.025f;

  lo.x_ -= expanding_x;
  hi.x_ += expanding_x;
  lo.y_ -= expanding_y;
  hi.y_ += expanding_y;
  lo.z_ -= expanding_z;
  hi.z_ += expanding_z;

  if (lo.x_ < 0.0f) lo.x_ = std::floor(lo.x_);
  else lo.x_ = std::ceil(lo.x_);

  if (lo.y_ < 0.0f) lo.y_ = std::floor(lo.y_);
  else lo.y_ = std::ceil(lo.y_);

  if (lo.z_ < 0.0f) lo.z_ = std::floor(lo.z_);
  else lo.z_ = std::ceil(lo.z_);

  if (hi.x_ < 0.0f) hi.x_ = std::floor(hi.x_);
  else hi.x_ = std::ceil(hi.x_);

  if (hi.y_ < 0.0f) hi.y_ = std::floor(hi.y_);
  else hi.y_ = std::ceil(hi.y_);

  if (hi.z_ < 0.0f) hi.z_ = std::floor(hi.z_);
  else hi.z_ = std::ceil(hi.z_);

  // Computing the center of octree box
  half_width[0] = (hi.x_-lo.x_)*0.5f;
  half_width[1] = (hi.y_-lo.y_)*0.5f;
  half_width[2] = (hi.z_-lo.z_)*0.5f;
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat3 GGEMSMeshedSolid::GetVoxelSizes(GGsize const& thread_index) const
{
  return {{0.0f, 0.0f, 0.0f}};
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOBB GGEMSMeshedSolid::GetOBBGeometry(GGsize const& thread_index) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  GGEMSMeshedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSMeshedSolidData>(solid_data_[thread_index], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSMeshedSolidData), thread_index);

  GGEMSOBB obb_geometry = solid_data_device->obb_geometry_;

  opencl_manager.ReleaseDeviceBuffer(solid_data_[thread_index], solid_data_device, thread_index);

  return obb_geometry;
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
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "* Dimension: " << solid_data_device->number_of_voxels_xyz_.s[0] << " " << solid_data_device->number_of_voxels_xyz_.s[1] << " " << solid_data_device->number_of_voxels_xyz_.s[2] << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "* Number of voxels: " << solid_data_device->number_of_voxels_ << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "* Size of voxels: (" << solid_data_device->voxel_sizes_xyz_.s[0] /mm << "x" << solid_data_device->voxel_sizes_xyz_.s[1]/mm << "x" << solid_data_device->voxel_sizes_xyz_.s[2]/mm << ") mm3" << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "* Oriented bounding box (OBB) in local position:" << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "    - X: " << solid_data_device->obb_geometry_.border_min_xyz_.s[0] << " <-> " << solid_data_device->obb_geometry_.border_max_xyz_.s[0] << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "    - Y: " << solid_data_device->obb_geometry_.border_min_xyz_.s[1] << " <-> " << solid_data_device->obb_geometry_.border_max_xyz_.s[1] << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "    - Z: " << solid_data_device->obb_geometry_.border_min_xyz_.s[2] << " <-> " << solid_data_device->obb_geometry_.border_max_xyz_.s[2] << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "    - Transformation matrix:" << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "    [" << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m0_[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m0_[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m0_[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m0_[3] << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m1_[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m1_[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m1_[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m1_[3] << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m2_[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m2_[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m2_[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m2_[3] << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m3_[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m3_[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m3_[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m3_[3] << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "    ]" << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << "* Solid index: " << solid_data_device->solid_id_ << GGendl;
    GGcout("GGEMSMeshedSolid", "PrintInfos", 0) << GGendl;

    // Release the pointer
    opencl_manager.ReleaseDeviceBuffer(solid_data_[d], solid_data_device, d);
  }
}
