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
  \file GGEMSSource.cc

  \brief GGEMS mother class for the source

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 15, 2019
*/

#include "GGEMS/global/GGEMSConstants.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/maths/GGEMSGeometryTransformation.hh"
#include "GGEMS/physics/GGEMSPrimaryParticles.hh"
#include "GGEMS/randoms/GGEMSRandom.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSource::GGEMSSource(std::string const& source_name)
: source_name_(source_name),
  number_of_particles_(0),
  number_of_particles_by_device_(nullptr),
  number_of_particles_in_batch_(nullptr),
  number_of_batchs_(nullptr),
  is_interp_(TRUE),
  number_of_energy_bins_(0),
  energy_spectrum_(nullptr),
  energy_cdf_(nullptr),
  particle_type_(99),
  tracking_kernel_option_("")
{
  GGcout("GGEMSSource", "GGEMSSource", 3) << "GGEMSSource creating..." << GGendl;

  // Allocation of geometry transformation
  geometry_transformation_ = new GGEMSGeometryTransformation();

  // Allocating memory for cdf and energy spectrum
  energy_spectrum_ = new cl::Buffer*[number_activated_devices_];
  energy_cdf_ = new cl::Buffer*[number_activated_devices_];

  // Get the number of activated device
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  number_activated_devices_ = opencl_manager.GetNumberOfActivatedDevice();

  // Storing a kernel for each device
  kernel_get_primaries_ = new cl::Kernel*[number_activated_devices_];

  if (!energy_mappings_.empty()) energy_mappings_.clear();

  // Store the source in source manager
  GGEMSSourceManager::GetInstance().Store(this);

  GGcout("GGEMSSource", "GGEMSSource", 3) << "GGEMSSource created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSource::~GGEMSSource(void)
{
  GGcout("GGEMSSource", "~GGEMSSource", 3) << "GGEMSSource erasing..." << GGendl;

  if (geometry_transformation_) {
    delete geometry_transformation_;
    geometry_transformation_ = nullptr;
  }

  if (number_of_batchs_) {
    delete number_of_batchs_;
    number_of_batchs_ = nullptr;
  }

  if (number_of_particles_by_device_) {
    delete number_of_particles_by_device_;
    number_of_particles_by_device_ = nullptr;
  }

  if (kernel_get_primaries_) {
    delete[] kernel_get_primaries_;
    kernel_get_primaries_ = nullptr;
  }

  if (number_of_particles_in_batch_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      delete number_of_particles_in_batch_[i];
      number_of_particles_in_batch_[i] = nullptr;
    }
    delete[] number_of_particles_in_batch_;
    number_of_particles_in_batch_ = nullptr;
  }

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  if (energy_spectrum_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(energy_spectrum_[i], (number_of_energy_bins_+1)*sizeof(GGfloat), i);
    }
    delete[] energy_spectrum_;
    energy_spectrum_ = nullptr;
  }

  if (energy_cdf_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(energy_cdf_[i], (number_of_energy_bins_+1)*sizeof(GGfloat), i);
    }
    delete[] energy_cdf_;
    energy_cdf_ = nullptr;
  }

  GGcout("GGEMSSource", "~GGEMSSource", 3) << "GGEMSSource erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::EnableTracking(void)
{
  tracking_kernel_option_ = " -DGGEMS_TRACKING";
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::SetPosition(GGfloat const& pos_x, GGfloat const& pos_y, GGfloat const& pos_z, std::string const& unit)
{
  GGfloat3 translation;
  translation.s[0] = DistanceUnit(pos_x, unit);
  translation.s[1] = DistanceUnit(pos_y, unit);
  translation.s[2] = DistanceUnit(pos_z, unit);
  geometry_transformation_->SetTranslation(translation);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::SetNumberOfParticles(GGsize const& number_of_particles)
{
  number_of_particles_ = number_of_particles;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::SetSourceParticleType(std::string const& particle_type)
{
  if (particle_type == "gamma") {
    particle_type_ = PHOTON;
  }
  else if (particle_type == "e-") {
    particle_type_ = ELECTRON;
  }
  else if (particle_type == "e+") {
    particle_type_ = POSITRON;
  }
  else
  {
    GGEMSMisc::ThrowException("GGEMSSourceManager", "SetParticleType", "Unknown particle!!!");
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::SetRotation(GGfloat const& rx, GGfloat const& ry, GGfloat const& rz, std::string const& unit)
{
  GGfloat3 rotation;
  rotation.s[0] = AngleUnit(rx, unit);
  rotation.s[1] = AngleUnit(ry, unit);
  rotation.s[2] = AngleUnit(rz, unit);
  geometry_transformation_->SetRotation(rotation);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::SetMonoenergy(GGfloat const& monoenergy, std::string const& unit)
{
  if (!energy_mappings_.empty()) energy_mappings_.clear();

  energy_mappings_.push_back({ EnergyUnit(monoenergy, unit), 1.0 });

  number_of_energy_bins_ = energy_mappings_.size();
  is_interp_ = FALSE;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::SetPolyenergy(std::string const& energy_spectrum_filename)
{
  if (!energy_mappings_.empty()) energy_mappings_.clear();

  // Read a first time the spectrum file counting the number of lines
  std::ifstream spectrum_stream(energy_spectrum_filename, std::ios::in);
  GGEMSFileStream::CheckInputStream(spectrum_stream, energy_spectrum_filename);

  // Compute number of energy bins
  std::string line;
  while (std::getline(spectrum_stream, line)) ++number_of_energy_bins_;

  // Returning to beginning of the file to read it again
  spectrum_stream.clear();
  spectrum_stream.seekg(0, std::ios::beg);

  // Storing spectrum from file
  while (std::getline(spectrum_stream, line)) {
    std::istringstream iss(line);
    GGEMSEnergyMapping energy_mapping;
    iss >> energy_mapping.energy_ >> energy_mapping.intensity_;
    energy_mappings_.push_back(energy_mapping);
  }

  // Closing file
  spectrum_stream.close();

  number_of_energy_bins_ = energy_mappings_.size();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::SetEnergyPeak(GGfloat const& energy, GGfloat const& intensity, std::string const& unit)
{
  energy_mappings_.push_back({ EnergyUnit(energy, unit), intensity });
  number_of_energy_bins_ += 1;
  is_interp_ = FALSE;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::FillEnergy(void)
{
  GGcout("GGEMSSource", "FillEnergy", 3) << "Filling energy..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  for (GGsize j = 0; j < number_activated_devices_; ++j) {
    // Allocation of memory on OpenCL device
    // Energy
    energy_spectrum_[j] = opencl_manager.Allocate(nullptr, (number_of_energy_bins_+1)*sizeof(GGfloat), j, CL_MEM_READ_WRITE, "GGEMSSource");

    // Cumulative distribution function
    energy_cdf_[j] = opencl_manager.Allocate(nullptr, (number_of_energy_bins_+1)*sizeof(GGfloat), j, CL_MEM_READ_WRITE, "GGEMSSource");

    // Get the energy pointer on OpenCL device
    GGfloat* energy_spectrum_device = opencl_manager.GetDeviceBuffer<GGfloat>(energy_spectrum_[j], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, (number_of_energy_bins_+1)*sizeof(GGfloat), j);

    // Get the cdf pointer on OpenCL device
    GGfloat* cdf_device = opencl_manager.GetDeviceBuffer<GGfloat>(energy_cdf_[j], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, (number_of_energy_bins_+1)*sizeof(GGfloat), j);

    // Computing sum of intensities
    GGdouble sum_cdf = 0.0;
    for (auto const& e_map : energy_mappings_) {
      sum_cdf += e_map.intensity_;
    }

    // Filling a tmp buffer
    GGdouble* tmp_cdf = new GGdouble[number_of_energy_bins_+1];
    tmp_cdf[0] = 0.0f;
    for (GGsize i = 0; i < number_of_energy_bins_; ++i) {
      tmp_cdf[i+1] = energy_mappings_[i].intensity_;
      energy_spectrum_device[i] = energy_mappings_[i].energy_;
    }
    energy_spectrum_device[number_of_energy_bins_] = energy_spectrum_device[number_of_energy_bins_ - 1]; // copy second time the last value

    // Compute CDF and normalized it
    cdf_device[0] = tmp_cdf[0];
    //cdf_device[0] = static_cast<GGfloat>(tmp_cdf[0] / sum_cdf);
    for (GGsize i = 1; i <= number_of_energy_bins_; ++i) {
      tmp_cdf[i] = tmp_cdf[i] + tmp_cdf[i - 1];
      cdf_device[i] = static_cast<GGfloat>(tmp_cdf[i] / sum_cdf);
    }

    // By security, final value of cdf must be 1 !!!
    cdf_device[number_of_energy_bins_] = 1.0f;

    // Release the pointers
    opencl_manager.ReleaseDeviceBuffer(energy_spectrum_[j], energy_spectrum_device, j);
    opencl_manager.ReleaseDeviceBuffer(energy_cdf_[j], cdf_device, j);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::CheckParameters(void) const
{
  GGcout("GGEMSSource", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  // Checking the type of particles
  if (particle_type_ == 99) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a particle type for the source:" << std::endl;
    oss << "    - Photon" << std::endl;
    oss << "    - Electron" << std::endl;
    oss << "    - Positron" << std::endl;
    GGEMSMisc::ThrowException("GGEMSSource", "CheckParameters", oss.str());
  }

  // Checking name of the source
  if (source_name_.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a name for the source!!!";
    GGEMSMisc::ThrowException("GGEMSSource", "CheckParameters", oss.str());
  }

  // Checking the number of particles
  if (number_of_particles_ == 0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a number of particles > 0!!!";
    GGEMSMisc::ThrowException("GGEMSSource", "CheckParameters", oss.str());
  }

  // Checking the position of particles
  GGfloat3 const kPosition = geometry_transformation_->GetPosition();
  if (kPosition.s[0] == std::numeric_limits<float>::min() || kPosition.s[1] == std::numeric_limits<float>::min() || kPosition.s[2] == std::numeric_limits<float>::min()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a position for the source!!!";
    GGEMSMisc::ThrowException("GGEMSSource", "CheckParameters", oss.str());
  }

  // Checking the rotation of particles
  GGfloat3 const kRotation = geometry_transformation_->GetRotation();
  if (kRotation.s[0] == std::numeric_limits<float>::min() || kRotation.s[1] == std::numeric_limits<float>::min() || kRotation.s[2] == std::numeric_limits<float>::min()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a rotation for the source!!!";
    GGEMSMisc::ThrowException("GGEMSSource", "CheckParameters", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::OrganizeParticlesInBatch(void)
{
  GGcout("GGEMSSource", "OrganizeParticlesInBatch", 3) << "Organizing the number of particles in batch..." << GGendl;

  // Getting OpenCL singleton
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Computing number of particles to simulate for each device
  number_of_particles_by_device_ = new GGsize[number_activated_devices_];
  if (opencl_manager.GetNumberDeviceBalancing() == 0) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      number_of_particles_by_device_[i] = number_of_particles_ / number_activated_devices_;
    }

    // Adding the remaing particles
    for (GGsize i = 0; i < number_of_particles_ % number_activated_devices_; ++i) {
      number_of_particles_by_device_[i]++;
    }
  }
  else {
    GGsize tmp_number_of_particles = 0;
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      number_of_particles_by_device_[i] = static_cast<GGsize>(static_cast<GGfloat>(number_of_particles_) * opencl_manager.GetDeviceBalancing(i));
      tmp_number_of_particles += number_of_particles_by_device_[i];
    }

    // Checking number of particle
    if (tmp_number_of_particles != number_of_particles_) {
      number_of_particles_by_device_[0] += static_cast<GGsize>(abs(static_cast<GGlong>(number_of_particles_ - tmp_number_of_particles)));
    }
  }

  // Computing number of batch for each device
  number_of_particles_in_batch_ = new GGsize*[number_activated_devices_];
  number_of_batchs_ = new GGsize[number_activated_devices_];
  for (GGsize i = 0; i < number_activated_devices_; ++i) {
    number_of_batchs_[i] = static_cast<GGsize>(std::ceil(static_cast<GGfloat>(number_of_particles_by_device_[i]) / MAXIMUM_PARTICLES));

    number_of_particles_in_batch_[i] = new GGsize[number_of_batchs_[i]];

    // Computing the number of simulated particles in batch
    if (number_of_batchs_[i] == 1) {
      number_of_particles_in_batch_[i][0] = number_of_particles_by_device_[i];
    }
    else {
      for (GGsize j = 0; j < number_of_batchs_[i]; ++j) {
        number_of_particles_in_batch_[i][j] = number_of_particles_by_device_[i] / number_of_batchs_[i];
      }

      // Adding the remaing particles
      for (GGsize j = 0; j < number_of_particles_by_device_[i] % number_of_batchs_[i]; ++j) {
        number_of_particles_in_batch_[i][j]++;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::Initialize(bool const& is_tracking)
{
  GGcout("GGEMSSource", "Initialize", 3) << "Initializing the a GGEMS source..." << GGendl;

  // Checking the parameters of Source
  CheckParameters();

  // Organize the particles in batch
  OrganizeParticlesInBatch();

  // Enable tracking
  if (is_tracking) EnableTracking();

  GGcout("GGEMSSource", "Initialize", 0) << "Particles arranged in batch OK" << GGendl;
}
