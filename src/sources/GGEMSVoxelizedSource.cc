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
  \file GGEMSVoxelizedSource.cc

  \brief This class defines a voxelized source in GGEMS useful for SPECT/PET simulations

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday September 1, 2025
*/

#include "GGEMS/sources/GGEMSVoxelizedSource.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/maths/GGEMSGeometryTransformation.hh"
#include "GGEMS/global/GGEMSConstants.hh"
#include "GGEMS/tools/GGEMSRAMManager.hh"
#include "GGEMS/randoms/GGEMSPseudoRandomGenerator.hh"
#include "GGEMS/tools/GGEMSProfilerManager.hh"
#include "GGEMS/io/GGEMSMHDImage.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolidData.hh"

#include "GGEMS/maths/GGEMSMathAlgorithms.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedSource::GGEMSVoxelizedSource(std::string const& source_name)
: GGEMSSource(source_name),
  phantom_source_filename_(""),
  number_of_activity_bins_(0)
{
  GGcout("GGEMSVoxelizedSource", "GGEMSVoxelizedSource", 3) << "GGEMSVoxelizedSource creating..." << GGendl;

  // Initialization of local axis for X-ray source
  geometry_transformation_->SetAxisTransformation(
    {
      {1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f}
    }
  );

  // Allocating memory for phantom vox. data
  phantom_vox_data_ = new cl::Buffer*[number_activated_devices_];

  // Allocating memory for cdf and activity index
  activity_index_ = new cl::Buffer*[number_activated_devices_];
  activity_cdf_ = new cl::Buffer*[number_activated_devices_];

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Loop over the device
  for (GGsize d = 0; d < number_activated_devices_; ++d) {
    phantom_vox_data_[d] = opencl_manager.Allocate(nullptr, sizeof(GGEMSVoxelizedSolidData), d, CL_MEM_READ_WRITE, "GGEMSVoxelizedSource");
  }

  GGcout("GGEMSVoxelizedSource", "GGEMSVoxelizedSource", 3) << "GGEMSVoxelizedSource created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedSource::~GGEMSVoxelizedSource(void)
{
  GGcout("GGEMSVoxelizedSource", "~GGEMSVoxelizedSource", 3) << "GGEMSVoxelizedSource erasing..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  if (phantom_vox_data_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(phantom_vox_data_[i], sizeof(GGEMSVoxelizedSolidData), i);
    }
    delete[] phantom_vox_data_;
    phantom_vox_data_ = nullptr;
  }

  if (activity_index_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(activity_index_[i], (number_of_activity_bins_+1)*sizeof(GGint), i);
    }
    delete[] activity_index_;
    activity_index_ = nullptr;
  }

  if (activity_cdf_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(activity_cdf_[i], (number_of_activity_bins_+1)*sizeof(GGfloat), i);
    }
    delete[] activity_cdf_;
    activity_cdf_ = nullptr;
  }

  GGcout("GGEMSVoxelizedSource", "~GGEMSVoxelizedSource", 3) << "GGEMSVoxelizedSource erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedSource::CheckParameters(void) const
{
  GGcout("GGEMSVoxelizedSource", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  // Checking phantom source file
  if (phantom_source_filename_.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a MHD phantom source file for voxelized source!!!";
    GGEMSMisc::ThrowException("GGEMSVoxelizedSource", "CheckParameters", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedSource::InitializeKernel(void)
{
  GGcout("GGEMSVoxelizedSource", "InitializeKernel", 3) << "Initializing kernel..." << GGendl;

  // Getting the path to kernel
  std::string openCL_kernel_path = OPENCL_KERNEL_PATH;
  std::string filename = openCL_kernel_path + "/GetPrimariesGGEMSVoxelizedSource.cl";

  // Compiling the kernel
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Compiling kernel on each device
  opencl_manager.CompileKernel(filename, "get_primaries_ggems_voxelized_source", kernel_get_primaries_, nullptr, const_cast<char*>(tracking_kernel_option_.c_str()));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedSource::GetPrimaries(GGsize const& thread_index, GGsize const& number_of_particles)
{
  // Get command queue and event
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue = opencl_manager.GetCommandQueue(thread_index);

  // Get Device name and storing methode name + device
  GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(thread_index);
  std::string device_name = opencl_manager.GetDeviceName(device_index);
  std::ostringstream oss(std::ostringstream::out);
  oss << "GGEMSVoxelizedSource::GetPrimaries on " << device_name << ", index " << device_index;

  // Get the OpenCL buffers
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  cl::Buffer* particles = source_manager.GetParticles()->GetPrimaryParticles(thread_index);
  cl::Buffer* randoms = source_manager.GetPseudoRandomGenerator()->GetPseudoRandomNumbers(thread_index);
  cl::Buffer* matrix_transformation = geometry_transformation_->GetTransformationMatrix(thread_index);

  // Getting work group size, and work-item number
  GGsize work_group_size = opencl_manager.GetWorkGroupSize();
  GGsize number_of_work_items = opencl_manager.GetBestWorkItem(number_of_particles);

  // Parameters for work-item in kernel
  cl::NDRange global_wi(number_of_work_items);
  cl::NDRange local_wi(work_group_size);

  // Set parameters for kernel
  kernel_get_primaries_[thread_index]->setArg(0, number_of_particles);
  kernel_get_primaries_[thread_index]->setArg(1, *particles);
  kernel_get_primaries_[thread_index]->setArg(2, *randoms);
  kernel_get_primaries_[thread_index]->setArg(3, particle_type_);
  kernel_get_primaries_[thread_index]->setArg(4, *energy_spectrum_[thread_index]);
  kernel_get_primaries_[thread_index]->setArg(5, *energy_cdf_[thread_index]);
  kernel_get_primaries_[thread_index]->setArg(6, static_cast<GGint>(number_of_energy_bins_+1));
  kernel_get_primaries_[thread_index]->setArg(7, is_interp_);
  kernel_get_primaries_[thread_index]->setArg(8, *activity_index_[thread_index]);
  kernel_get_primaries_[thread_index]->setArg(9, *activity_cdf_[thread_index]);
  kernel_get_primaries_[thread_index]->setArg(10, static_cast<GGint>(number_of_activity_bins_+1));
  kernel_get_primaries_[thread_index]->setArg(11, *phantom_vox_data_[thread_index]);
  kernel_get_primaries_[thread_index]->setArg(12, *matrix_transformation);

  // Launching kernel
  cl::Event event;
  GGint kernel_status = queue->enqueueNDRangeKernel(*kernel_get_primaries_[thread_index], 0, global_wi, local_wi, nullptr, &event);
  opencl_manager.CheckOpenCLError(kernel_status, "GGEMSVoxelizedSource", "GetPrimaries");

  // GGEMS Profiling
  GGEMSProfilerManager& profiler_manager = GGEMSProfilerManager::GetInstance();
  profiler_manager.HandleEvent(event, oss.str());
  queue->finish();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedSource::PrintInfos(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Loop over each device
  for (GGsize j = 0; j < number_activated_devices_; ++j) {
    // Get pointer on OpenCL device
    GGfloat44* transformation_matrix_device = opencl_manager.GetDeviceBuffer<GGfloat44>(geometry_transformation_->GetTransformationMatrix(j), CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGfloat44), j);

    // Getting index of the device
    GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(j);

    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << GGendl;
    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << "GGEMSVoxelizedSource Infos: " << GGendl;
    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << "----------------------"  << GGendl;
    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << "* Device: " << opencl_manager.GetDeviceName(device_index) << GGendl;
    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << "* Source name: " << source_name_ << GGendl;
    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << "* Particle type: ";
    if (particle_type_ == PHOTON) {
      std::cout << "Photon" << std::endl;
    }
    else if (particle_type_ == ELECTRON) {
      std::cout << "Electron" << std::endl;
    }
    else if (particle_type_ == POSITRON) {
      std::cout << "Positron" << std::endl;
    }
    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << "* Number of particles: " << number_of_particles_by_device_[j] << GGendl;
    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << "* Number of batches: " << number_of_batchs_[j] << GGendl;
    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << "* Position: " << "(" << geometry_transformation_->GetPosition().s[0]/mm << ", " << geometry_transformation_->GetPosition().s[1]/mm << ", " << geometry_transformation_->GetPosition().s[2]/mm << " ) mm3" << GGendl;
    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << "* Rotation: " << "(" << geometry_transformation_->GetRotation().s[0] << ", " << geometry_transformation_->GetRotation().s[1] << ", " << geometry_transformation_->GetRotation().s[2] << ") degree" << GGendl;
    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << "* Transformation matrix: " << GGendl;
    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << "[" << GGendl;
    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << "    " << transformation_matrix_device->m0_[0] << " " << transformation_matrix_device->m0_[1] << " " << transformation_matrix_device->m0_[2] << " " << transformation_matrix_device->m0_[3] << GGendl;
    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << "    " << transformation_matrix_device->m1_[0] << " " << transformation_matrix_device->m1_[1] << " " << transformation_matrix_device->m1_[2] << " " << transformation_matrix_device->m1_[3] << GGendl;
    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << "    " << transformation_matrix_device->m2_[0] << " " << transformation_matrix_device->m2_[1] << " " << transformation_matrix_device->m2_[2] << " " << transformation_matrix_device->m2_[3] << GGendl;
    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << "    " << transformation_matrix_device->m3_[0] << " " << transformation_matrix_device->m3_[1] << " " << transformation_matrix_device->m3_[2] << " " << transformation_matrix_device->m3_[3] << GGendl;
    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << "]" << GGendl;
    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << GGendl;

    // Release the pointer
    opencl_manager.ReleaseDeviceBuffer(geometry_transformation_->GetTransformationMatrix(j), transformation_matrix_device, j);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedSource::Initialize(bool const& is_tracking)
{
  GGcout("GGEMSVoxelizedSource", "Initialize", 3) << "Initializing the GGEMS Voxelized source..." << GGendl;

  // Initialize GGEMS source
  GGEMSSource::Initialize(is_tracking);

  // Check the mandatory parameters
  CheckParameters();

  // Initializing the kernel for OpenCL
  InitializeKernel();

  // Filling the energy
  FillEnergy();

  // Filling activity for each voxel
  FillVoxelActivity();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedSource::FillVoxelActivity(void)
{
  GGcout("GGEMSVoxelizedSource", "FillVoxelActivity", 3) << "Filling activity for voxelized source..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Read MHD input file
  GGEMSMHDImage mhd_input_phantom_source;

  // Loop over the device
  for (GGsize d = 0; d < number_activated_devices_; ++d) {
    mhd_input_phantom_source.Read(phantom_source_filename_, phantom_vox_data_[d], d);

    // Check the type of data
    std::string data_type = mhd_input_phantom_source.GetDataMHDType();
    if (data_type != "MET_FLOAT") {
      std::ostringstream oss(std::ostringstream::out);
      oss << "For voxelized source, data type is MET_FLOAT only!!!";
      GGEMSMisc::ThrowException("GGEMSVoxelizedSource", "FillVoxelActivity", oss.str());
    }

    // Get the raw filename
    std::string output_dir = mhd_input_phantom_source.GetOutputDirectory();
    std::string phantom_src_raw_filename = output_dir + mhd_input_phantom_source.GetRawMDHfilename();

    // Checking if file exists
    std::ifstream phantom_src_raw_stream(phantom_src_raw_filename, std::ios::in | std::ios::binary);
    GGEMSFileStream::CheckInputStream(phantom_src_raw_stream, phantom_src_raw_filename);

    // Read file into a buffer
    // Number of element for source
    GGEMSVoxelizedSolidData* phantom_vox_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(phantom_vox_data_[d], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSVoxelizedSolidData), d);
    GGint number_of_voxels =  phantom_vox_data_device->number_of_voxels_;
    // Release the pointer
    opencl_manager.ReleaseDeviceBuffer(phantom_vox_data_[d], phantom_vox_data_device, d);

    GGfloat* activity = new GGfloat[number_of_voxels];
    phantom_src_raw_stream.read(reinterpret_cast<char*>(&activity[0]), number_of_voxels * sizeof(GGfloat));

    // Closing file
    phantom_src_raw_stream.close();

    // count nb of non zeros activities
    GGint nb_activity = 0; // Number of non zero activity
    for (GGint i = 0; i < number_of_voxels; ++i) {
      if (activity[i] != 0.0f) ++nb_activity;
    }
    number_of_activity_bins_ = nb_activity;

    // Allocation of buffers
    activity_cdf_[d] = opencl_manager.Allocate(nullptr, (number_of_activity_bins_+1) * sizeof(GGfloat), d, CL_MEM_READ_WRITE, "GGEMSVoxelizedSource");
    activity_index_[d] = opencl_manager.Allocate(nullptr, (number_of_activity_bins_+1) * sizeof(GGint), d, CL_MEM_READ_WRITE, "GGEMSVoxelizedSource");

    // Get the index pointer on OpenCL device
    GGint* activity_index_device = opencl_manager.GetDeviceBuffer<GGint>(activity_index_[d], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, (number_of_activity_bins_+1) * sizeof(GGint), d);

    // Get the cdf pointer on OpenCL device
    GGfloat* activity_cdf_device = opencl_manager.GetDeviceBuffer<GGfloat>(activity_cdf_[d], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, (number_of_activity_bins_+1) * sizeof(GGfloat), d);

    // Storing index activity and computing cdf
    GGint index = 1;
    GGfloat sum_cdf = 0.0f;
    GGfloat* tmp_cdf = new GGfloat[number_of_activity_bins_+1];
    tmp_cdf[0] = 0.0f;
    for (GGint i = 0; i < number_of_voxels; ++i) {
      if (activity[i] != 0.0f) {
        activity_index_device[index] = i;
        tmp_cdf[index] = activity[i];
        sum_cdf += activity[i];
        ++index;
      }
    }
    activity_index_device[number_of_activity_bins_] = activity_index_device[number_of_activity_bins_ - 1];

    // compute cummulative density function
    activity_cdf_device[0] = tmp_cdf[0];

    for (GGint i = 1; i <= number_of_activity_bins_; ++i) {
      tmp_cdf[i] = tmp_cdf[i] + tmp_cdf[i - 1];
      activity_cdf_device[i]= tmp_cdf[i] / sum_cdf;
    }

    // By security, final value of cdf must be 1 !!!
    activity_cdf_device[number_of_activity_bins_] = 1.0f;

    // Release the pointers
    opencl_manager.ReleaseDeviceBuffer(activity_index_[d], activity_index_device, d);
    opencl_manager.ReleaseDeviceBuffer(activity_cdf_[d], activity_cdf_device, d);

    // Freeing memory
    delete[] activity;
    delete[] tmp_cdf;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedSource::SetPhantomSourceFile(std::string const& phantom_source_filename)
{
  phantom_source_filename_ = phantom_source_filename;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedSource* create_ggems_voxelized_source(char const* source_name)
{
  return new(std::nothrow) GGEMSVoxelizedSource(source_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_position_ggems_voxelized_source(GGEMSVoxelizedSource* voxelized_source, GGfloat const pos_x, GGfloat const pos_y, GGfloat const pos_z, char const* unit)
{
  voxelized_source->SetPosition(pos_x, pos_y, pos_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_number_of_particles_voxelized_source(GGEMSVoxelizedSource* voxelized_source, GGsize const number_of_particles)
{
  voxelized_source->SetNumberOfParticles(number_of_particles);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_source_particle_type_ggems_voxelized_source(GGEMSVoxelizedSource* voxelized_source, char const* particle_name)
{
  voxelized_source->SetSourceParticleType(particle_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_monoenergy_ggems_voxelized_source(GGEMSVoxelizedSource* voxelized_source, GGfloat const monoenergy, char const* unit)
{
  voxelized_source->SetMonoenergy(monoenergy, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_polyenergy_ggems_voxelized_source(GGEMSVoxelizedSource* voxelized_source, char const* energy_spectrum)
{
  voxelized_source->SetPolyenergy(energy_spectrum);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_phantom_source_ggems_voxelized_source(GGEMSVoxelizedSource* voxelized_source, char const* phantom_source_file)
{
  voxelized_source->SetPhantomSourceFile(phantom_source_file);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_energy_peak_ggems_voxelized_source(GGEMSVoxelizedSource* voxelized_source, GGfloat const energy, char const* unit, GGfloat const intensity)
{
  voxelized_source->SetEnergyPeak(energy, intensity, unit);
}
