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
  \file GGEMSXRaySource.cc

  \brief This class define a XRay source in GGEMS useful for CT/CBCT simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 22, 2019
*/

#include "GGEMS/sources/GGEMSXRaySource.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/maths/GGEMSGeometryTransformation.hh"
#include "GGEMS/global/GGEMSConstants.hh"
#include "GGEMS/tools/GGEMSRAMManager.hh"
#include "GGEMS/randoms/GGEMSPseudoRandomGenerator.hh"
#include "GGEMS/tools/GGEMSProfilerManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSXRaySource::GGEMSXRaySource(std::string const& source_name)
: GGEMSSource(source_name),
  beam_aperture_(std::numeric_limits<float>::min())
{
  GGcout("GGEMSXRaySource", "GGEMSXRaySource", 3) << "GGEMSXRaySource creating..." << GGendl;

  // Initialization of local axis for X-ray source
  geometry_transformation_->SetAxisTransformation(
    {
      {0.0f, 0.0f, -1.0f},
      {0.0f, 1.0f,  0.0f},
      {1.0f, 0.0f,  0.0f}
    }
  );

  // Initialization of parameters
  focal_spot_size_.s[0] = std::numeric_limits<float>::min();
  focal_spot_size_.s[1] = std::numeric_limits<float>::min();
  focal_spot_size_.s[2] = std::numeric_limits<float>::min();

  GGcout("GGEMSXRaySource", "GGEMSXRaySource", 3) << "GGEMSXRaySource created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSXRaySource::~GGEMSXRaySource(void)
{
  GGcout("GGEMSXRaySource", "~GGEMSXRaySource", 3) << "GGEMSXRaySource erasing..." << GGendl;

  GGcout("GGEMSXRaySource", "~GGEMSXRaySource", 3) << "GGEMSXRaySource erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSXRaySource::InitializeKernel(void)
{
  GGcout("GGEMSXRaySource", "InitializeKernel", 3) << "Initializing kernel..." << GGendl;

  // Getting the path to kernel
  std::string openCL_kernel_path = OPENCL_KERNEL_PATH;
  std::string filename = openCL_kernel_path + "/GetPrimariesGGEMSXRaySource.cl";

  // Compiling the kernel
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Compiling kernel on each device
  opencl_manager.CompileKernel(filename, "get_primaries_ggems_xray_source", kernel_get_primaries_, nullptr, const_cast<char*>(tracking_kernel_option_.c_str()));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSXRaySource::GetPrimaries(GGsize const& thread_index, GGsize const& number_of_particles)
{
  // Get command queue and event
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue = opencl_manager.GetCommandQueue(thread_index);

  // Get Device name and storing methode name + device
  GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(thread_index);
  std::string device_name = opencl_manager.GetDeviceName(device_index);
  std::ostringstream oss(std::ostringstream::out);
  oss << "GGEMSXRaySource::GetPrimaries on " << device_name << ", index " << device_index;

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
  kernel_get_primaries_[thread_index]->setArg(6, static_cast<GGint>(number_of_energy_bins_));
  kernel_get_primaries_[thread_index]->setArg(7, is_interp_);
  kernel_get_primaries_[thread_index]->setArg(8, beam_aperture_);
  kernel_get_primaries_[thread_index]->setArg(9, focal_spot_size_);
  kernel_get_primaries_[thread_index]->setArg(10, *matrix_transformation);

  // Launching kernel
  cl::Event event;
  GGint kernel_status = queue->enqueueNDRangeKernel(*kernel_get_primaries_[thread_index], 0, global_wi, local_wi, nullptr, &event);
  opencl_manager.CheckOpenCLError(kernel_status, "GGEMSXRaySource", "GetPrimaries");

  // GGEMS Profiling
  GGEMSProfilerManager& profiler_manager = GGEMSProfilerManager::GetInstance();
  profiler_manager.HandleEvent(event, oss.str());
  queue->finish();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSXRaySource::PrintInfos(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Loop over each device
  for (GGsize j = 0; j < number_activated_devices_; ++j) {
    // Get pointer on OpenCL device
    GGfloat44* transformation_matrix_device = opencl_manager.GetDeviceBuffer<GGfloat44>(geometry_transformation_->GetTransformationMatrix(j), CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGfloat44), j);

    // Getting index of the device
    GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(j);

    GGcout("GGEMSXRaySource", "PrintInfos", 0) << GGendl;
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << "GGEMSXRaySource Infos: " << GGendl;
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << "----------------------"  << GGendl;
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << "* Device: " << opencl_manager.GetDeviceName(device_index) << GGendl;
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << "* Source name: " << source_name_ << GGendl;
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << "* Particle type: ";
    if (particle_type_ == PHOTON) {
      std::cout << "Photon" << std::endl;
    }
    else if (particle_type_ == ELECTRON) {
      std::cout << "Electron" << std::endl;
    }
    else if (particle_type_ == POSITRON) {
      std::cout << "Positron" << std::endl;
    }
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << "* Number of particles: " << number_of_particles_by_device_[j] << GGendl;
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << "* Number of batches: " << number_of_batchs_[j] << GGendl;
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << "* Position: " << "(" << geometry_transformation_->GetPosition().s[0]/mm << ", " << geometry_transformation_->GetPosition().s[1]/mm << ", " << geometry_transformation_->GetPosition().s[2]/mm << " ) mm3" << GGendl;
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << "* Rotation: " << "(" << geometry_transformation_->GetRotation().s[0] << ", " << geometry_transformation_->GetRotation().s[1] << ", " << geometry_transformation_->GetRotation().s[2] << ") degree" << GGendl;
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << "* Beam aperture: " << beam_aperture_/deg << " degrees" << GGendl;
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << "* Focal spot size: " << "(" << focal_spot_size_.s[0]/mm << ", " << focal_spot_size_.s[1]/mm << ", " << focal_spot_size_.s[2]/mm << ") mm3" << GGendl;
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << "* Transformation matrix: " << GGendl;
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << "[" << GGendl;
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << "    " << transformation_matrix_device->m0_[0] << " " << transformation_matrix_device->m0_[1] << " " << transformation_matrix_device->m0_[2] << " " << transformation_matrix_device->m0_[3] << GGendl;
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << "    " << transformation_matrix_device->m1_[0] << " " << transformation_matrix_device->m1_[1] << " " << transformation_matrix_device->m1_[2] << " " << transformation_matrix_device->m1_[3] << GGendl;
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << "    " << transformation_matrix_device->m2_[0] << " " << transformation_matrix_device->m2_[1] << " " << transformation_matrix_device->m2_[2] << " " << transformation_matrix_device->m2_[3] << GGendl;
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << "    " << transformation_matrix_device->m3_[0] << " " << transformation_matrix_device->m3_[1] << " " << transformation_matrix_device->m3_[2] << " " << transformation_matrix_device->m3_[3] << GGendl;
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << "]" << GGendl;
    GGcout("GGEMSXRaySource", "PrintInfos", 0) << GGendl;

    // Release the pointer
    opencl_manager.ReleaseDeviceBuffer(geometry_transformation_->GetTransformationMatrix(j), transformation_matrix_device, j);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSXRaySource::CheckParameters(void) const
{
  GGcout("GGEMSXRaySource", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  // Checking the beam aperture
  if (beam_aperture_ == std::numeric_limits<float>::min()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a beam aperture for the source!!!";
    GGEMSMisc::ThrowException("GGEMSXRaySource", "CheckParameters", oss.str());
  }
  else if (beam_aperture_ < 0.0f) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "The beam aperture must be >= 0!!!";
    GGEMSMisc::ThrowException("GGEMSXRaySource", "CheckParameters", oss.str());
  }

  // Checking the focal spot size
  if (focal_spot_size_.s[0] == std::numeric_limits<float>::min() || focal_spot_size_.s[1] == std::numeric_limits<float>::min() || focal_spot_size_.s[2] == std::numeric_limits<float>::min()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a focal spot size!!!";
    GGEMSMisc::ThrowException("GGEMSXRaySource", "CheckParameters", oss.str());
  }

  // Focal spot size must be a positive value
  if (focal_spot_size_.s[0] < 0.0f || focal_spot_size_.s[1] < 0.0f || focal_spot_size_.s[2] < 0.0f) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "The focal spot size is a posivite value!!!";
    GGEMSMisc::ThrowException("GGEMSXRaySource", "CheckParameters", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSXRaySource::Initialize(bool const& is_tracking)
{
  GGcout("GGEMSXRaySource", "Initialize", 3) << "Initializing the GGEMS X-Ray source..." << GGendl;

  // Initialize GGEMS source
  GGEMSSource::Initialize(is_tracking);

  // Check the mandatory parameters
  CheckParameters();

  // Initializing the kernel for OpenCL
  InitializeKernel();

  // Filling the energy
  FillEnergy();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSXRaySource::SetBeamAperture(GGfloat const& beam_aperture, std::string const& unit)
{
  beam_aperture_ = AngleUnit(beam_aperture, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSXRaySource::SetFocalSpotSize(GGfloat const& width, GGfloat const& height, GGfloat const& depth, std::string const& unit)
{
  focal_spot_size_.s[0] = DistanceUnit(width, unit);
  focal_spot_size_.s[1] = DistanceUnit(height, unit);
  focal_spot_size_.s[2] = DistanceUnit(depth, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSXRaySource* create_ggems_xray_source(char const* source_name)
{
  return new(std::nothrow) GGEMSXRaySource(source_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_position_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const pos_x, GGfloat const pos_y, GGfloat const pos_z, char const* unit)
{
  xray_source->SetPosition(pos_x, pos_y, pos_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_number_of_particles_xray_source(GGEMSXRaySource* xray_source, GGsize const number_of_particles)
{
  xray_source->SetNumberOfParticles(number_of_particles);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_source_particle_type_ggems_xray_source(GGEMSXRaySource* xray_source, char const* particle_name)
{
  xray_source->SetSourceParticleType(particle_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_beam_aperture_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const beam_aperture, char const* unit)
{
  xray_source->SetBeamAperture(beam_aperture, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_focal_spot_size_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const width, GGfloat const height, GGfloat const depth, char const* unit)
{
  xray_source->SetFocalSpotSize(width, height, depth, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_rotation_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const rx, GGfloat const ry, GGfloat const rz, char const* unit)
{
  xray_source->SetRotation(rx, ry, rz, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_monoenergy_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const monoenergy, char const* unit)
{
  xray_source->SetMonoenergy(monoenergy, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_polyenergy_ggems_xray_source(GGEMSXRaySource* xray_source, char const* energy_spectrum)
{
  xray_source->SetPolyenergy(energy_spectrum);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_energy_peak_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const energy, char const* unit, GGfloat const intensity)
{
  xray_source->SetEnergyPeak(energy, intensity, unit);
}
