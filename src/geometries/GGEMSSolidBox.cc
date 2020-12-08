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

#include "GGEMS/geometries/GGEMSSolidBox.hh"
#include "GGEMS/geometries/GGEMSSolidBoxData.hh"

#include "GGEMS/randoms/GGEMSPseudoRandomGenerator.hh"

#include "GGEMS/physics/GGEMSCrossSections.hh"

#include "GGEMS/maths/GGEMSGeometryTransformation.hh"

#include "GGEMS/sources/GGEMSSourceManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolidBox::GGEMSSolidBox(GGint const& virtual_element_number_x, GGint const& virtual_element_number_y, GGint const& virtual_element_number_z, GGfloat const& box_size_x, GGfloat const& box_size_y, GGfloat const& box_size_z, std::string const& data_reg_type)
: GGEMSSolid()
{
  GGcout("GGEMSSolidBox", "GGEMSSolidBox", 3) << "Allocation of GGEMSSolidBox..." << GGendl;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Allocating memory on OpenCL device and getting pointer on it
  solid_data_cl_ = opencl_manager.Allocate(nullptr, sizeof(GGEMSSolidBoxData), CL_MEM_READ_WRITE);
  GGEMSSolidBoxData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSSolidBoxData>(solid_data_cl_.get(), sizeof(GGEMSSolidBoxData));

  solid_data_device->virtual_element_number_xyz_[0] = virtual_element_number_x;
  solid_data_device->virtual_element_number_xyz_[1] = virtual_element_number_y;
  solid_data_device->virtual_element_number_xyz_[2] = virtual_element_number_z;

  solid_data_device->box_size_xyz_[0] = box_size_x;
  solid_data_device->box_size_xyz_[1] = box_size_y;
  solid_data_device->box_size_xyz_[2] = box_size_z;

  solid_data_device->obb_geometry_.border_min_xyz_[0] = -box_size_x*0.5f;
  solid_data_device->obb_geometry_.border_min_xyz_[1] = -box_size_y*0.5f;
  solid_data_device->obb_geometry_.border_min_xyz_[2] = -box_size_z*0.5f;

  solid_data_device->obb_geometry_.border_max_xyz_[0] = box_size_x*0.5f;
  solid_data_device->obb_geometry_.border_max_xyz_[1] = box_size_y*0.5f;
  solid_data_device->obb_geometry_.border_max_xyz_[2] = box_size_z*0.5f;

  // Releasing pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_cl_.get(), solid_data_device);

  // Local axis definition for system
  geometry_transformation_->SetAxisTransformation(
    {
      {0.0f, 0.0f, 1.0f},
      {0.0f, 1.0f, 0.0f},
      {-1.0f, 0.0f, 0.0f}
    }
  );

  // Solid box associated at hit collection
  data_reg_type_ = data_reg_type;
  if (data_reg_type == "HIT") {
    hit_.number_of_elements_ = virtual_element_number_x*virtual_element_number_y*virtual_element_number_z;
    hit_.hit_cl_ = opencl_manager.Allocate(nullptr, hit_.number_of_elements_*sizeof(GGint), CL_MEM_READ_WRITE);

    kernel_option_ += " -DHIT";

    GGint* hit_device = opencl_manager.GetDeviceBuffer<GGint>(hit_.hit_cl_.get(), hit_.number_of_elements_*sizeof(GGint));

    for (GGint i = 0; i < hit_.number_of_elements_; ++i) hit_device[i] = 0;

    opencl_manager.ReleaseDeviceBuffer(hit_.hit_cl_ .get(), hit_device);
  }
  else {
    std::ostringstream oss(std::ostringstream::out);
    oss << "False registration type name!!!" << std::endl;
    oss << "Registration type is :" << std::endl;
    oss << "    - HIT" << std::endl;
    oss << "    - SINGLE" << std::endl;
    oss << "    - DOSE" << std::endl;
    GGEMSMisc::ThrowException("GGEMSSolidBox", "GGEMSSolidBox", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolidBox::~GGEMSSolidBox(void)
{
  GGcout("GGEMSSolidBox", "~GGEMSSolidBox", 3) << "Deallocation of GGEMSSolidBox..." << GGendl;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  opencl_manager.Deallocate(hit_.hit_cl_, hit_.number_of_elements_);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolidBox::InitializeKernel(void)
{
  GGcout("GGEMSSolidBox", "InitializeKernel", 3) << "Initializing kernel for solid box..." << GGendl;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Getting the path to kernel
  std::string openCL_kernel_path = OPENCL_KERNEL_PATH;
  std::string particle_solid_distance_filename = openCL_kernel_path + "/ParticleSolidDistanceGGEMSSolidBox.cl";
  std::string project_to_filename = openCL_kernel_path + "/ProjectToGGEMSSolidBox.cl";
  std::string track_through_filename = openCL_kernel_path + "/TrackThroughGGEMSSolidBox.cl";

  // Compiling the kernels
  kernel_particle_solid_distance_cl_ = opencl_manager.CompileKernel(particle_solid_distance_filename, "particle_solid_distance_ggems_solid_box", nullptr, const_cast<char*>(kernel_option_.c_str()));
  kernel_project_to_solid_cl_ = opencl_manager.CompileKernel(project_to_filename, "project_to_ggems_solid_box", nullptr, const_cast<char*>(kernel_option_.c_str()));
  kernel_track_through_solid_cl_ = opencl_manager.CompileKernel(track_through_filename, "track_through_ggems_solid_box", nullptr, const_cast<char*>(kernel_option_.c_str()));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolidBox::Initialize(std::weak_ptr<GGEMSMaterials> materials)
{
  GGcout("GGEMSSolidBox", "Initialize", 3) << "Initializing voxelized solid..." << GGendl;

  // Initializing kernels
  InitializeKernel();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolidBox::TrackThroughSolid(std::weak_ptr<GGEMSCrossSections> cross_sections, std::weak_ptr<GGEMSMaterials> materials)
{
  // Getting the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue_cl = opencl_manager.GetCommandQueue();
  cl::Event* event_cl = opencl_manager.GetEvent();

  // Getting the buffer of primary particles and random from source
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  GGEMSParticles* particles = source_manager.GetParticles();
  cl::Buffer* primary_particles_cl = particles->GetPrimaryParticles();
  cl::Buffer* randoms_cl = source_manager.GetPseudoRandomGenerator()->GetPseudoRandomNumbers();

  // Getting OpenCL buffer for cross section
  cl::Buffer* cross_sections_cl = cross_sections.lock()->GetCrossSections();

  // Getting OpenCL buffer for materials
  cl::Buffer* materials_cl = materials.lock()->GetMaterialTables().lock().get();

  // Getting the number of particles
  GGlong number_of_particles = particles->GetNumberOfParticles();

  // // Set parameters for kernel
  std::shared_ptr<cl::Kernel> kernel_cl = kernel_track_through_solid_cl_.lock();
  kernel_cl->setArg(0, number_of_particles);
  kernel_cl->setArg(1, *primary_particles_cl);
  kernel_cl->setArg(2, *randoms_cl);
  kernel_cl->setArg(3, *solid_data_cl_);
  kernel_cl->setArg(4, *label_data_cl_);
  kernel_cl->setArg(5, *cross_sections_cl);
  kernel_cl->setArg(6, *materials_cl);
  if (data_reg_type_ == "HIT") kernel_cl->setArg(7, *hit_.hit_cl_);

  // Get number of max work group size
  std::size_t max_work_group_size = opencl_manager.GetWorkGroupSize();

  // Compute work item number
  std::size_t number_of_work_items = number_of_particles + (max_work_group_size - number_of_particles%max_work_group_size);

  cl::NDRange global_wi(number_of_work_items);
  cl::NDRange offset_wi(0);
  cl::NDRange local_wi(max_work_group_size);

  // Launching kernel
  GGint kernel_status = queue_cl->enqueueNDRangeKernel(*kernel_cl, offset_wi, global_wi, local_wi, nullptr, event_cl);
  opencl_manager.CheckOpenCLError(kernel_status, "GGEMSSolid", "TrackThroughSolid");
  queue_cl->finish(); // Wait until the kernel status is finish

  // Storing elapsed time in kernel
  kernel_track_through_solid_timer_ += opencl_manager.GetElapsedTimeInKernel();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolidBox::PrintInfos(void) const
{
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Getting pointer on OpenCL device
  GGEMSSolidBoxData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSSolidBoxData>(solid_data_cl_.get(), sizeof(GGEMSSolidBoxData));

  GGcout("GGEMSSolidBox", "PrintInfos", 0) << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << "GGEMSSolidBox Infos:" << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << "--------------------------" << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << "* Virtual elements: " << solid_data_device->virtual_element_number_xyz_[0] << "x" << solid_data_device->virtual_element_number_xyz_[1] << "x" << solid_data_device->virtual_element_number_xyz_[2] << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << "* Lengths: (" << solid_data_device->box_size_xyz_[0] << "x" << solid_data_device->box_size_xyz_[1] << "x" << solid_data_device->box_size_xyz_[2] << ") mm3" << GGendl;
  GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "* Oriented bounding box (OBB) in local position:" << GGendl;
  GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "    - X: " << solid_data_device->obb_geometry_.border_min_xyz_[0] << " <-> " << solid_data_device->obb_geometry_.border_max_xyz_[0] << GGendl;
  GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "    - Y: " << solid_data_device->obb_geometry_.border_min_xyz_[1] << " <-> " << solid_data_device->obb_geometry_.border_max_xyz_[1] << GGendl;
  GGcout("GGEMSVoxelizedSolid", "PrintInfos", 0) << "    - Z: " << solid_data_device->obb_geometry_.border_min_xyz_[2] << " <-> " << solid_data_device->obb_geometry_.border_max_xyz_[2] << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << "    - Transformation matrix:" << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << "    [" << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m0_[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m0_[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m0_[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m0_[3] << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m1_[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m1_[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m1_[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m1_[3] << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m2_[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m2_[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m2_[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m2_[3] << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m3_[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m3_[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m3_[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m3_[3] << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << "    ]" << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << "* Solid index: " << solid_data_device->solid_id_ << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << GGendl;

  // Releasing the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_cl_.get(), solid_data_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolidBox::GetTransformationMatrix(void)
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Copy information to OBB
  GGEMSSolidBoxData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSSolidBoxData>(solid_data_cl_.get(), sizeof(GGEMSSolidBoxData));
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
