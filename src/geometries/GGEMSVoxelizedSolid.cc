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
#include "GGEMS/io/GGEMSMHDImage.hh"
#include "GGEMS/maths/GGEMSGeometryTransformation.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedSolid::GGEMSVoxelizedSolid(std::string const& volume_header_filename, std::string const& range_filename)
: GGEMSSolid(),
  volume_header_filename_(volume_header_filename),
  range_filename_(range_filename)
{
  GGcout("GGEMSVoxelizedSolid", "GGEMSVoxelizedSolid", 3) << "Allocation of GGEMSVoxelizedSolid..." << GGendl;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Allocating memory on OpenCL device
  solid_data_cl_ = opencl_manager.Allocate(nullptr, sizeof(GGEMSVoxelizedSolidData), CL_MEM_READ_WRITE);

  // Local axis for phantom. Voxelized solid used only for phantom
  geometry_transformation_->SetAxisTransformation(
    {
      {1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f}
    }
  );
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedSolid::~GGEMSVoxelizedSolid(void)
{
  GGcout("GGEMSVoxelizedSolid", "~GGEMSVoxelizedSolid", 3) << "Deallocation of GGEMSVoxelizedSolid..." << GGendl;
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
  kernel_particle_solid_distance_cl_ = opencl_manager.CompileKernel(particle_solid_distance_filename, "particle_solid_distance_ggems_voxelized_solid", nullptr, const_cast<char*>(kernel_option_.c_str()));
  kernel_project_to_solid_cl_ = opencl_manager.CompileKernel(project_to_filename, "project_to_ggems_voxelized_solid", nullptr, const_cast<char*>(kernel_option_.c_str()));
  kernel_track_through_solid_cl_ = opencl_manager.CompileKernel(track_through_filename, "track_through_ggems_voxelized_solid", nullptr, const_cast<char*>(kernel_option_.c_str()));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedSolid::Initialize(std::weak_ptr<GGEMSMaterials> materials)
{
  GGcout("GGEMSVoxelizedSolid", "Initialize", 3) << "Initializing voxelized solid..." << GGendl;

  // Initializing kernels and loading image
  InitializeKernel();
  LoadVolumeImage(materials);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedSolid::GetTransformationMatrix(void)
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Copy information to OBB
  GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_cl_.get(), sizeof(GGEMSVoxelizedSolidData));
  GGfloat44* transformation_matrix_device = opencl_manager.GetDeviceBuffer<GGfloat44>(geometry_transformation_->GetTransformationMatrix(), sizeof(GGfloat44));

  for (GGint i = 0; i < 4; ++i) {
    solid_data_device->obb_geometry_.matrix_transformation_.m0_[i] = transformation_matrix_device->m0_[i];
    solid_data_device->obb_geometry_.matrix_transformation_.m1_[i] = transformation_matrix_device->m1_[i];
    solid_data_device->obb_geometry_.matrix_transformation_.m2_[i] = transformation_matrix_device->m2_[i];
    solid_data_device->obb_geometry_.matrix_transformation_.m3_[i] = transformation_matrix_device->m3_[i];
  }

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_cl_.get(), solid_data_device);
  opencl_manager.ReleaseDeviceBuffer(geometry_transformation_->GetTransformationMatrix(), transformation_matrix_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat3 GGEMSVoxelizedSolid::GetVoxelSizes(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_cl_.get(), sizeof(GGEMSVoxelizedSolidData));

  GGfloat3 voxel_sizes = {
    solid_data_device->voxel_sizes_xyz_[0],
    solid_data_device->voxel_sizes_xyz_[1],
    solid_data_device->voxel_sizes_xyz_[2]
  };

  opencl_manager.ReleaseDeviceBuffer(solid_data_cl_.get(), solid_data_device);

  return voxel_sizes;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// void GGEMSVoxelizedSolid::TrackThroughSolid(std::weak_ptr<GGEMSCrossSections> cross_sections, std::weak_ptr<GGEMSMaterials> materials)
// {
  // Getting the OpenCL manager
  // GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  // cl::CommandQueue* queue_cl = opencl_manager.GetCommandQueue();
  // cl::Event* event_cl = opencl_manager.GetEvent();

  // // Getting the buffer of primary particles and random from source
  // GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  // GGEMSParticles* particles = source_manager.GetParticles();
  // cl::Buffer* primary_particles_cl = particles->GetPrimaryParticles();
  // cl::Buffer* randoms_cl = source_manager.GetPseudoRandomGenerator()->GetPseudoRandomNumbers();

  // // Getting OpenCL buffer for cross section
  // cl::Buffer* cross_sections_cl = cross_sections.lock()->GetCrossSections();

  // // Getting OpenCL buffer for materials
  // cl::Buffer* materials_cl = materials.lock()->GetMaterialTables().lock().get();

  // // Getting the number of particles
  // GGlong number_of_particles = particles->GetNumberOfParticles();

  // // // Set parameters for kernel
  // std::shared_ptr<cl::Kernel> kernel_cl = kernel_track_through_solid_cl_.lock();
  // kernel_cl->setArg(0, number_of_particles);
  // kernel_cl->setArg(1, *primary_particles_cl);
  // kernel_cl->setArg(2, *randoms_cl);
  // kernel_cl->setArg(3, *solid_data_cl_);
  // kernel_cl->setArg(4, *label_data_cl_);
  // kernel_cl->setArg(5, *cross_sections_cl);
  // kernel_cl->setArg(6, *materials_cl);

  // // Get number of max work group size
  // std::size_t max_work_group_size = opencl_manager.GetWorkGroupSize();

  // // Compute work item number
  // std::size_t number_of_work_items = number_of_particles + (max_work_group_size - number_of_particles%max_work_group_size);

  // cl::NDRange global_wi(number_of_work_items);
  // cl::NDRange offset_wi(0);
  // cl::NDRange local_wi(max_work_group_size);

  // // Launching kernel
  // GGint kernel_status = queue_cl->enqueueNDRangeKernel(*kernel_cl, offset_wi, global_wi, local_wi, nullptr, event_cl);
  // opencl_manager.CheckOpenCLError(kernel_status, "GGEMSSolid", "TrackThroughSolid");
  // queue_cl->finish(); // Wait until the kernel status is finish

  // // Storing elapsed time in kernel
  // kernel_track_through_solid_timer_ += opencl_manager.GetElapsedTimeInKernel();
//}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedSolid::PrintInfos(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device
  GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_cl_.get(), sizeof(GGEMSVoxelizedSolidData));

  GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << GGendl;
  GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "GGEMSVoxelizedSolid Infos:" << GGendl;
  GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "--------------------------" << GGendl;
  GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "* Dimension: " << solid_data_device->number_of_voxels_xyz_[0] << " " << solid_data_device->number_of_voxels_xyz_[1] << " " << solid_data_device->number_of_voxels_xyz_[2] << GGendl;
  GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "* Number of voxels: " << solid_data_device->number_of_voxels_ << GGendl;
  GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "* Size of voxels: (" << solid_data_device->voxel_sizes_xyz_[0]/mm << "x" << solid_data_device->voxel_sizes_xyz_[1]/mm << "x" << solid_data_device->voxel_sizes_xyz_[2]/mm << ") mm3" << GGendl;
  GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "* Oriented bounding box (OBB) in local position:" << GGendl;
  GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "    - X: " << solid_data_device->obb_geometry_.border_min_xyz_[0] << " <-> " << solid_data_device->obb_geometry_.border_max_xyz_[0] << GGendl;
  GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "    - Y: " << solid_data_device->obb_geometry_.border_min_xyz_[1] << " <-> " << solid_data_device->obb_geometry_.border_max_xyz_[1] << GGendl;
  GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "    - Z: " << solid_data_device->obb_geometry_.border_min_xyz_[2] << " <-> " << solid_data_device->obb_geometry_.border_max_xyz_[2] << GGendl;
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
  opencl_manager.ReleaseDeviceBuffer(solid_data_cl_.get(), solid_data_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedSolid::LoadVolumeImage(std::weak_ptr<GGEMSMaterials> materials)
{
  GGcout("GGEMSVoxelizedSolid", "LoadVolumeImage", 3) << "Loading volume image from mhd file..." << GGendl;

  // Read MHD input file
  GGEMSMHDImage mhd_input_phantom;
  mhd_input_phantom.Read(volume_header_filename_, solid_data_cl_);

  // Get the name of raw file from mhd reader
  std::string const kRawFilename = mhd_input_phantom.GetRawMDHfilename();

  // Get the type
  std::string const kDataType = mhd_input_phantom.GetDataMHDType();

  // Convert raw data to material id data
  if (!kDataType.compare("MET_CHAR")) {
    ConvertImageToLabel<char>(kRawFilename, range_filename_, materials);
  }
  else if (!kDataType.compare("MET_UCHAR")) {
    ConvertImageToLabel<unsigned char>(kRawFilename, range_filename_, materials);
  }
  else if (!kDataType.compare("MET_SHORT")) {
    ConvertImageToLabel<GGshort>(kRawFilename, range_filename_, materials);
  }
  else if (!kDataType.compare("MET_USHORT")) {
    ConvertImageToLabel<GGushort>(kRawFilename, range_filename_, materials);
  }
  else if (!kDataType.compare("MET_INT")) {
    ConvertImageToLabel<GGint>(kRawFilename, range_filename_, materials);
  }
  else if (!kDataType.compare("MET_UINT")) {
    ConvertImageToLabel<GGuint>(kRawFilename, range_filename_, materials);
  }
  else if (!kDataType.compare("MET_FLOAT")) {
    ConvertImageToLabel<GGfloat>(kRawFilename, range_filename_, materials);
  }
}
