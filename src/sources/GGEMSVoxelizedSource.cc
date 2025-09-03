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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedSource::GGEMSVoxelizedSource(std::string const& source_name)
: GGEMSSource(source_name),
  phantom_source_filename_(""),
  number_of_voxel_activity_(0)
{
  GGcout("GGEMSVoxelizedSource", "GGEMSVoxelizedSource", 3) << "GGEMSVoxelizedSource creating..." << GGendl;

  // Initialization of local axis for X-ray source
  geometry_transformation_->SetAxisTransformation(
    {
      {0.0f, 0.0f, -1.0f},
      {0.0f, 1.0f,  0.0f},
      {1.0f, 0.0f,  0.0f}
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
      opencl_manager.Deallocate(activity_index_[i], number_of_voxel_activity_*sizeof(GGint), i);
    }
    delete[] activity_index_;
    activity_index_ = nullptr;
  }

  if (activity_cdf_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(activity_cdf_[i], number_of_voxel_activity_*sizeof(GGfloat), i);
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

  // Checking the energy
  if (is_monoenergy_mode_) {
    if (monoenergy_ == -1.0f) {
      std::ostringstream oss(std::ostringstream::out);
      oss << "You have to set an energy in monoenergetic mode!!!";
      GGEMSMisc::ThrowException("GGEMSVoxelizedSource", "CheckParameters", oss.str());
    }

    if (monoenergy_ < 0.0f) {
      std::ostringstream oss(std::ostringstream::out);
      oss << "The energy must be a positive value!!!";
      GGEMSMisc::ThrowException("GGEMSVoxelizedSource", "CheckParameters", oss.str());
    }
  }

  if (!is_monoenergy_mode_) {
    if (energy_spectrum_filename_.empty()) {
      std::ostringstream oss(std::ostringstream::out);
      oss << "You have to provide a energy spectrum file in polyenergy mode!!!";
      GGEMSMisc::ThrowException("GGEMSVoxelizedSource", "CheckParameters", oss.str());
    }
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
/*
  // Set parameters for kernel
  kernel_get_primaries_[thread_index]->setArg(0, number_of_particles);
  kernel_get_primaries_[thread_index]->setArg(1, *particles);
  kernel_get_primaries_[thread_index]->setArg(2, *randoms);
  kernel_get_primaries_[thread_index]->setArg(3, particle_type_);
  kernel_get_primaries_[thread_index]->setArg(4, *energy_spectrum_[thread_index]);
  kernel_get_primaries_[thread_index]->setArg(5, *cdf_[thread_index]);
  kernel_get_primaries_[thread_index]->setArg(6, static_cast<GGint>(number_of_energy_bins_));
  kernel_get_primaries_[thread_index]->setArg(7, beam_aperture_);
  kernel_get_primaries_[thread_index]->setArg(8, focal_spot_size_);
  kernel_get_primaries_[thread_index]->setArg(9, *matrix_transformation);

  // Launching kernel
  cl::Event event;
  GGint kernel_status = queue->enqueueNDRangeKernel(*kernel_get_primaries_[thread_index], 0, global_wi, local_wi, nullptr, &event);
  opencl_manager.CheckOpenCLError(kernel_status, "GGEMSXRaySource", "GetPrimaries");

  // GGEMS Profiling
  GGEMSProfilerManager& profiler_manager = GGEMSProfilerManager::GetInstance();
  profiler_manager.HandleEvent(event, oss.str());
  queue->finish();*/
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
    GGcout("GGEMSVoxelizedSource", "PrintInfos", 0) << "* Energy mode: ";
    if (is_monoenergy_mode_) {
      std::cout << "Monoenergy" << std::endl;
    }
    else {
      std::cout << "Polyenergy" << std::endl;
    }
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
    number_of_voxel_activity_ = nb_activity;

    // Allocation of buffers
    activity_cdf_[d] = opencl_manager.Allocate(nullptr, number_of_voxel_activity_ * sizeof(GGfloat), d, CL_MEM_READ_WRITE, "GGEMSVoxelizedSource");
    activity_index_[d] = opencl_manager.Allocate(nullptr, number_of_voxel_activity_ * sizeof(GGint), d, CL_MEM_READ_WRITE, "GGEMSVoxelizedSource");

    // Get the index pointer on OpenCL device
    GGint* activity_index_device = opencl_manager.GetDeviceBuffer<GGint>(activity_index_[d], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, number_of_voxel_activity_ * sizeof(GGint), d);

    // Get the cdf pointer on OpenCL device
    GGfloat* activity_cdf_device = opencl_manager.GetDeviceBuffer<GGfloat>(activity_cdf_[d], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, number_of_voxel_activity_ * sizeof(GGfloat), d);

    // Storing index activity and computing cdf
    GGint index = 0;
    GGfloat sum_cdf = 0.0f;
    GGfloat* tmp_cdf = new GGfloat[number_of_voxel_activity_];
    for (GGint i = 0; i < number_of_voxels; ++i) {
      if (activity[i] != 0.0f) {
        activity_index_device[index] = i;
        tmp_cdf[index] = activity[i];
        sum_cdf += activity[i];
        ++index;
      }
    }

    // compute cummulative density function
    activity_cdf_device[0] = tmp_cdf[0] / sum_cdf;

    for (GGint i = 0; i < number_of_voxel_activity_; ++i) {
      tmp_cdf[i] = tmp_cdf[i] + tmp_cdf[i - 1];
      activity_cdf_device[i]= tmp_cdf[i] / sum_cdf;
    }

    // By security, final value of cdf must be 1 !!!
    activity_cdf_device[number_of_voxel_activity_ - 1] = 1.0f;

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

/*

// VoxelizedSource
class VoxelizedSource {
    public:
        VoxelizedSource();

        void set_position(f32 vpx, f32 vpy, f32 vpz);
        void set_energy(f32 venergy);
        void set_histpoint(f32 venergy, f32 vpart);
        void set_source_type(std::string vtype);
        void set_seed(ui32 vseed);
        void set_in_geometry(ui32 vgeometry_id);
        void set_source_name(std::string vsource_name);

        void load_from_mhd(std::string filename);
        void compute_cdf();

        ui32 seed, geometry_id;
        std::string source_name, source_type;
        f32 px, py, pz;
        f32 energy;

        ui16 nb_vox_x, nb_vox_y, nb_vox_z;
        ui32 number_of_voxels;
        f32 spacing_x, spacing_y, spacing_z;

        // Activities
        f32 *activity_volume;
        f32 tot_activity;
        // CDF
        f32 *activity_cdf;
        f32 *activity_index;
        ui32 activity_size;
        
        std::vector<f32> energy_hist, partpdec;

    private:
        // For mhd
        void skip_comment(std::istream &);
        std::string remove_white_space(std::string);
        std::string read_mhd_key(std::string);
        std::string read_mhd_string_arg(std::string);
        i32 read_mhd_int(std::string);
        i32 read_mhd_int_atpos(std::string, i32);
        f32 read_mhd_f32_atpos(std::string, i32);

};

__host__ __device__ void voxelized_source_primary_mono_generator(ParticleStack particles, ui32 id,
                                                            f32 *cdf_index, f32 *cdf_act, ui32 nb_acts,
                                                            f32 px, f32 py, f32 pz,
                                                            ui32 nb_vox_x, ui32 nb_vox_y, ui32 nb_vox_z,
                                                            f32 sx, f32 sy, f32 sz,
                                                            f32 energy, ui8 type, ui32 geom_id) {

    f32 jump = (f32)(nb_vox_x*nb_vox_y);
    f32 ind, x, y, z;

    // use cdf to find the next emission spot
    f32 rnd = JKISS32(particles, id);
    ui32 pos = binary_search(rnd, cdf_act, nb_acts);

    // convert position index to emitted position
    ind = cdf_index[pos];
    z = floor(ind / jump);
    ind -= (z*jump);
    y = floor(ind / (f32)nb_vox_x);
    x = ind - y*nb_vox_x;
    
    // random positon within the voxel
    x += JKISS32(particles, id);
    y += JKISS32(particles, id);
    z += JKISS32(particles, id);

    // Due to float operation aproximation: 1+(1-Epsilon) = 2
    // we need to check that x, y, z are not equal to the size of the vox source
    // x, y, z must be in [0, size[
    if (x == nb_vox_x) x -= EPSILON3;
    if (y == nb_vox_y) y -= EPSILON3;
    if (z == nb_vox_z) z -= EPSILON3;

    // convert in mm
    x *= sx;
    y *= sy;
    z *= sz;

    // shift according to center of phantom and translation
    x = x - nb_vox_x*sx*0.5 + px;
    y = y - nb_vox_y*sy*0.5 + py;
    z = z - nb_vox_z*sz*0.5 + pz;

    // random orientation
    f32 phi = JKISS32(particles, id);
    f32 theta = JKISS32(particles, id);
    phi *= gpu_twopi;
    theta = acosf(1.0f - 2.0f*theta);

    // compute direction vector
    f32 dx = cos(phi)*sin(theta);
    f32 dy = sin(phi)*sin(theta);
    f32 dz = cos(theta);

    // set particle stack 1
    particles.E[id] = energy;
    particles.dx[id] = dx;
    particles.dy[id] = dy;
    particles.dz[id] = dz;
    particles.px[id] = x;
    particles.py[id] = y;
    particles.pz[id] = z;
    particles.tof[id] = 0.0;
    particles.endsimu[id] = PARTICLE_ALIVE;
    particles.level[id] = PRIMARY;
    particles.pname[id] = type;
    particles.geometry_id[id] = geom_id;
}

VoxelizedSource::VoxelizedSource() {
    // Default values
    seed=10;
    geometry_id=0;
    source_name="VoxSrc01";
    source_type="back2back";
    px=0.0; py=0.0; pz=0.0;
    energy=511*keV;

    // Init pointer
    activity_volume = NULL;
    activity_cdf = NULL;
    activity_index = NULL;
}

void VoxelizedSource::set_position(f32 vpx, f32 vpy, f32 vpz) {
    px = vpx; py = vpy; pz = vpz;
}

void VoxelizedSource::set_energy(f32 venergy) {
    energy = venergy;
}

void VoxelizedSource::set_histpoint(f32 venergy, f32 vpart) {
      energy_hist.push_back(venergy);
      partpdec.push_back(vpart);
}  

void VoxelizedSource::set_source_type(std::string vtype) {
    source_type = vtype;
}

void VoxelizedSource::set_seed(ui32 vseed) {
    seed = vseed;
}

void VoxelizedSource::set_in_geometry(ui32 vgeometry_id) {
    geometry_id = vgeometry_id;
}

void VoxelizedSource::set_source_name(std::string vsource_name) {
    source_name = vsource_name;
}

//// MHD //////////////////////////////////////////////////////:


// Load activities from mhd file (only f32 data)
void VoxelizedSource::load_from_mhd(std::string filename) {

    /////////////// First read the MHD file //////////////////////

    std::string line, key;
    nb_vox_x=0, nb_vox_y=0, nb_vox_z=0;
    spacing_x=0, spacing_y=0, spacing_z=0;

    // Watchdog
    std::string ObjectType="", BinaryData="", BinaryDataByteOrderMSB="", CompressedData="",
                ElementType="", ElementDataFile="";
    i32 NDims=0;

    // Read range file
    std::ifstream file(filename.c_str());
    if(!file) { printf("Error, file %s not found \n", filename.c_str()); exit(EXIT_FAILURE);}
    while (file) {
        skip_comment(file);
        std::getline(file, line);

        if (file) {
            key = read_mhd_key(line);
            if (key=="ObjectType")              ObjectType = read_mhd_string_arg(line);
            if (key=="NDims")                   NDims = read_mhd_int(line);
            if (key=="BinaryData")              BinaryData = read_mhd_string_arg(line);
            if (key=="BinaryDataByteOrderMSB")  BinaryDataByteOrderMSB=read_mhd_string_arg(line);
            if (key=="CompressedData")          CompressedData = read_mhd_string_arg(line);
            //if (key=="TransformMatrix") printf("Matrix\n");
            //if (key=="Offset")  printf("Offset\n");
            //if (key=="CenterOfRotation") printf("CoR\n");
            if (key=="ElementSpacing") {
                                                spacing_x=read_mhd_f32_atpos(line, 0);
                                                spacing_y=read_mhd_f32_atpos(line, 1);
                                                spacing_z=read_mhd_f32_atpos(line, 2);
            }
            if (key=="DimSize") {
                                                nb_vox_x=read_mhd_int_atpos(line, 0);
                                                nb_vox_y=read_mhd_int_atpos(line, 1);
                                                nb_vox_z=read_mhd_int_atpos(line, 2);
            }

            //if (key=="AnatomicalOrientation") printf("Anato\n");
            if (key=="ElementType")             ElementType = read_mhd_string_arg(line);
            if (key=="ElementDataFile")         ElementDataFile = read_mhd_string_arg(line);
        }

    } // read file

    if (nb_vox_x == 0 || nb_vox_y == 0 || nb_vox_z == 0 ||
            spacing_x == 0 || spacing_y == 0 || spacing_z == 0) {
        printf("Error when loading mhd file (unknown dimension and spacing)\n");
        printf("   => dim %i %i %i - spacing %f %f %f\n", nb_vox_x, nb_vox_y, nb_vox_z,
                                                          spacing_x, spacing_y, spacing_z);
        exit(EXIT_FAILURE);
    }
    // Read data
    FILE *pfile = fopen(ElementDataFile.c_str(), "rb");
    if (!pfile) {
        std::string nameWithRelativePath = filename;
        i32 lastindex = nameWithRelativePath.find_last_of(".");
        nameWithRelativePath = nameWithRelativePath.substr(0, lastindex);
        nameWithRelativePath+=".raw";
        pfile = fopen(nameWithRelativePath.c_str(), "rb");
        if (!pfile) {
            printf("Error when loading mhd file: %s\n", ElementDataFile.c_str());
            exit(EXIT_FAILURE);
        }
    }

    number_of_voxels = nb_vox_x*nb_vox_y*nb_vox_z;

    activity_volume = (f32*)malloc(sizeof(f32) * number_of_voxels);
    fread(activity_volume, sizeof(f32), number_of_voxels, pfile);
    fclose(pfile);

    // Compute the associated CDF of the activities
    compute_cdf();

}

// Compute the CDF of the activities
void VoxelizedSource::compute_cdf() {

    // count nb of non zeros activities
    ui32 nb=0;
    ui32 i=0; while (i<number_of_voxels) {
        if (activity_volume[i] != 0.0f) ++nb;
        ++i;
    }
    activity_size = nb;

    // mem allocation
    activity_index = (f32*)malloc(nb*sizeof(f32));
    activity_cdf = (f32*)malloc(nb*sizeof(f32));

    // Buffer
    f32* cdf = new f32[nb];

    // fill array with non zeros values activity
    ui32 index = 0;
    f32 val;
    f32 sum = 0.0; // for the cdf
    i=0; while (i<number_of_voxels) {
        val = activity_volume[i];
        if (val != 0.0f) {
            activity_index[index] = i;
            cdf[index] = val;
            sum += val;
            //printf("cdf i %d val %lf \n", index, cdf[index]);
            ++index;
        }
        ++i;
    }
    tot_activity = sum;
    printf("tot_activity %lf \n", sum);
    
    // compute cummulative density function
    cdf[0] /= sum;
    activity_cdf[0] = cdf[0];
      
    i = 1; while (i<nb) {
       // printf("i %d test div %4.12lf \n", i, (cdf[i]/sum));
        cdf[i] = (cdf[i]/sum) + cdf[i-1];
        activity_cdf[i]= (f32) cdf[i];
       // printf("i %d test div %4.12lf \n", i, cdf[i]);
        ++i;
    }

    delete cdf;

}
*/