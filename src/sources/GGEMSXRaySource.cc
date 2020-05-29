/*!
  \file GGEMSXRaySource.cc

  \brief This class define a XRay source in GGEMS useful for CT/CBCT simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 22, 2019
*/

#include <sstream>

#include "GGEMS/sources/GGEMSXRaySource.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/maths/GGEMSGeometryTransformation.hh"
#include "GGEMS/global/GGEMSConstants.hh"
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/physics/GGEMSParticles.hh"
#include "GGEMS/randoms/GGEMSPseudoRandomGenerator.hh"
#include "GGEMS/tools/GGEMSRAMManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSXRaySource::GGEMSXRaySource(void)
: GGEMSSource(this),
  beam_aperture_(std::numeric_limits<float>::min()),
  is_monoenergy_mode_(false),
  monoenergy_(-1.0f),
  energy_spectrum_filename_(""),
  number_of_energy_bins_(0),
  energy_spectrum_(nullptr),
  cdf_(nullptr)
{
  GGcout("GGEMSXRaySource", "GGEMSXRaySource", 3) << "Allocation of GGEMSXRaySource..." << GGendl;

  // Initialization of local axis for X-ray source
  geometry_transformation_->SetAxisTransformation(
    0.0f, 0.0f, -1.0f,
    0.0f, 1.0f, 0.0f,
    1.0f, 0.0f, 0.0f
  );

  // Initialization of parameters
  focal_spot_size_ = MakeFloat3(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min());
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSXRaySource::~GGEMSXRaySource(void)
{
  GGcout("GGEMSXRaySource", "~GGEMSXRaySource", 3) << "Deallocation of GGEMSXRaySource..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSXRaySource::InitializeKernel(void)
{
  GGcout("GGEMSXRaySource", "InitializeKernel", 3) << "Initializing kernel..." << GGendl;

  // Getting the path to kernel
  std::string const kOpenCLKernelPath = OPENCL_KERNEL_PATH;
  std::string const kFilename = kOpenCLKernelPath + "/GetPrimariesGGEMSXRaySource.cl";

  // Compiling the kernel
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  kernel_get_primaries_ = opencl_manager.CompileKernel(kFilename, "get_primaries_ggems_xray_source");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSXRaySource::GetPrimaries(GGulong const& number_of_particles)
{
  GGcout("GGEMSXRaySource", "GetPrimaries", 3) << "Generating " << number_of_particles << " new particles..." << GGendl;

  // Get command queue and event
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue = opencl_manager.GetCommandQueue();
  cl::Event* event = opencl_manager.GetEvent();

  // Get the OpenCL buffers
  GGEMSSourceManager& p_source_manager = GGEMSSourceManager::GetInstance();
  cl::Buffer* particles = p_source_manager.GetParticles()->GetPrimaryParticles();
  cl::Buffer* randoms = p_source_manager.GetPseudoRandomGenerator()->GetPseudoRandomNumbers();
  cl::Buffer* matrix_transformation = geometry_transformation_->GetTransformationMatrix();

  // Set parameters for kernel
  kernel_get_primaries_->setArg(0, *particles);
  kernel_get_primaries_->setArg(1, *randoms);
  kernel_get_primaries_->setArg(2, particle_type_);
  kernel_get_primaries_->setArg(3, *energy_spectrum_);
  kernel_get_primaries_->setArg(4, *cdf_);
  kernel_get_primaries_->setArg(5, number_of_energy_bins_);
  kernel_get_primaries_->setArg(6, beam_aperture_);
  kernel_get_primaries_->setArg(7, focal_spot_size_);
  kernel_get_primaries_->setArg(8, *matrix_transformation);

  // Define the number of work-item to launch
  cl::NDRange global(number_of_particles);
  cl::NDRange offset(0);

  // Launching kernel
  GGint kernel_status = queue->enqueueNDRangeKernel(*kernel_get_primaries_, offset, global, cl::NullRange, nullptr, event);
  opencl_manager.CheckOpenCLError(kernel_status, "GGEMSXRaySource", "GetPrimaries");
  queue->finish(); // Wait until the kernel status is finish
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSXRaySource::PrintInfos(void) const
{
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << GGendl;
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << "GGEMSXRaySource Infos: " << GGendl;
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << "----------------------"  << GGendl;
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << "*Source name: " << source_name_ << GGendl;
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << "*Particle type: ";
  if (particle_type_ == GGEMSParticle::PHOTON) {
    std::cout << "Photon" << std::endl;
  }
  else if (particle_type_ == GGEMSParticle::ELECTRON) {
    std::cout << "Electron" << std::endl;
  }
  else if (particle_type_ == GGEMSParticle::POSITRON) {
    std::cout << "Positron" << std::endl;
  }
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << "*Number of particles: " << number_of_particles_ << GGendl;
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << "*Number of batches: " << GetNumberOfBatchs() << GGendl;
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << "*Energy mode: ";
  if (is_monoenergy_mode_) {
    std::cout << "Monoenergy" << std::endl;
  }
  else {
    std::cout << "Polyenergy" << std::endl;
  }
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << "*Position: " << "(" << geometry_transformation_->GetPosition().s[0]/GGEMSUnits::mm << ", " << geometry_transformation_->GetPosition().s[1]/GGEMSUnits::mm << ", " << geometry_transformation_->GetPosition().s[2]/GGEMSUnits::mm << " ) mm3" << GGendl;
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << "*Rotation: " << "(" << geometry_transformation_->GetRotation().s[0] << ", " << geometry_transformation_->GetRotation().s[1] << ", " << geometry_transformation_->GetRotation().s[2] << ") degree" << GGendl;
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << "*Beam aperture: " << beam_aperture_/GGEMSUnits::deg << " degrees" << GGendl;
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << "*Focal spot size: " << "(" << focal_spot_size_.s[0]/GGEMSUnits::mm << ", " << focal_spot_size_.s[1]/GGEMSUnits::mm << ", " << focal_spot_size_.s[2]/GGEMSUnits::mm << ") mm3" << GGendl;
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << "*Local axis: " << GGendl;
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << "[" << GGendl;
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << "    " << geometry_transformation_->GetLocalAxis().m00_ << " " << geometry_transformation_->GetLocalAxis().m01_ << " " << geometry_transformation_->GetLocalAxis().m02_ << GGendl;
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << "    " << geometry_transformation_->GetLocalAxis().m10_ << " " << geometry_transformation_->GetLocalAxis().m11_ << " " << geometry_transformation_->GetLocalAxis().m12_ << GGendl;
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << "    " << geometry_transformation_->GetLocalAxis().m20_ << " " << geometry_transformation_->GetLocalAxis().m21_ << " " << geometry_transformation_->GetLocalAxis().m22_ << GGendl;
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << "]" << GGendl;
  GGcout("GGEMSXRaySource", "PrintInfos", 0) << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSXRaySource::SetMonoenergy(GGfloat const& monoenergy, std::string const& unit)
{
  monoenergy_ = GGEMSUnits::EnergyUnit(monoenergy, unit);
  is_monoenergy_mode_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSXRaySource::SetPolyenergy(std::string const& energy_spectrum_filename)
{
  energy_spectrum_filename_ = energy_spectrum_filename;
  is_monoenergy_mode_ = false;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSXRaySource::CheckParameters(void) const
{
  GGcout("GGEMSXRaySource", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  // Checking the beam aperture
  if (GGEMSMisc::IsEqual(beam_aperture_, std::numeric_limits<float>::min())) {
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
  if (GGEMSMisc::IsEqual(focal_spot_size_.s[0], std::numeric_limits<float>::min()) || GGEMSMisc::IsEqual(focal_spot_size_.s[1], std::numeric_limits<float>::min()) || GGEMSMisc::IsEqual(focal_spot_size_.s[2], std::numeric_limits<float>::min())) {
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

  // Checking the energy
  if (is_monoenergy_mode_) {
    if (GGEMSMisc::IsEqual(monoenergy_, -1.0f)) {
      std::ostringstream oss(std::ostringstream::out);
      oss << "You have to set an energy in monoenergetic mode!!!";
      GGEMSMisc::ThrowException("GGEMSXRaySource", "CheckParameters", oss.str());
    }

    if (monoenergy_ < 0.0f) {
      std::ostringstream oss(std::ostringstream::out);
      oss << "The energy must be a positive value!!!";
      GGEMSMisc::ThrowException("GGEMSXRaySource", "CheckParameters", oss.str());
    }
  }

  if (!is_monoenergy_mode_) {
    if (energy_spectrum_filename_.empty()) {
      std::ostringstream oss(std::ostringstream::out);
      oss << "You have to provide a energy spectrum file in polyenergy mode!!!";
      GGEMSMisc::ThrowException("GGEMSXRaySource", "CheckParameters", oss.str());
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSXRaySource::FillEnergy(void)
{
  GGcout("GGEMSXRaySource", "FillEnergy", 3) << "Filling energy..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get the RAM manager
  GGEMSRAMManager& ram_manager = GGEMSRAMManager::GetInstance();

  // Monoenergy mode
  if (is_monoenergy_mode_) {
    number_of_energy_bins_ = 2;

    // Allocation of memory on OpenCL device
    // Energy
    energy_spectrum_ = opencl_manager.Allocate(nullptr, 2 * sizeof(GGfloat), CL_MEM_READ_WRITE);
    ram_manager.AddSourceRAMMemory(2 * sizeof(GGfloat));

    // Cumulative distribution function
    cdf_ = opencl_manager.Allocate(nullptr, 2 * sizeof(GGfloat), CL_MEM_READ_WRITE);
    ram_manager.AddSourceRAMMemory(2 * sizeof(GGfloat));

    // Get the energy pointer on OpenCL device
    GGfloat* energy_spectrum_device = opencl_manager.GetDeviceBuffer<GGfloat>(energy_spectrum_, 2 * sizeof(GGfloat));

    // Get the cdf pointer on OpenCL device
    GGfloat* cdf_device = opencl_manager.GetDeviceBuffer<GGfloat>(cdf_, 2 * sizeof(GGfloat));

    energy_spectrum_device[0] = monoenergy_;
    energy_spectrum_device[1] = monoenergy_;

    cdf_device[0] = 1.0;
    cdf_device[1] = 1.0;

    // Release the pointers
    opencl_manager.ReleaseDeviceBuffer(energy_spectrum_, energy_spectrum_device);
    opencl_manager.ReleaseDeviceBuffer(cdf_, cdf_device);
  }
  else { // Polyenergy mode 
    // Read a first time the spectrum file counting the number of lines
    std::ifstream spectrum_stream(energy_spectrum_filename_, std::ios::in);
    GGEMSFileStream::CheckInputStream(spectrum_stream, energy_spectrum_filename_);

    // Compute number of energy bins
    std::string line;
    while (std::getline(spectrum_stream, line)) ++number_of_energy_bins_;

    // Returning to beginning of the file to read it again
    spectrum_stream.clear();
    spectrum_stream.seekg(0, std::ios::beg);

    // Allocation of memory on OpenCL device
    // Energy
    energy_spectrum_ = opencl_manager.Allocate(nullptr, number_of_energy_bins_ * sizeof(GGfloat), CL_MEM_READ_WRITE);
    ram_manager.AddSourceRAMMemory(number_of_energy_bins_ * sizeof(GGfloat));

    // Cumulative distribution function
    cdf_ = opencl_manager.Allocate(nullptr, number_of_energy_bins_ * sizeof(GGfloat), CL_MEM_READ_WRITE);
    ram_manager.AddSourceRAMMemory(number_of_energy_bins_ * sizeof(GGfloat));

    // Get the energy pointer on OpenCL device
    GGfloat* energy_spectrum_device = opencl_manager.GetDeviceBuffer<GGfloat>(energy_spectrum_, number_of_energy_bins_ * sizeof(GGfloat));

    // Get the cdf pointer on OpenCL device
    GGfloat* cdf_device = opencl_manager.GetDeviceBuffer<GGfloat>(cdf_, number_of_energy_bins_ * sizeof(GGfloat));

    // Read the input spectrum and computing the sum for the cdf
    GGint line_index = 0;
    GGfloat sum_cdf = 0.0;
    while (std::getline(spectrum_stream, line)) {
      std::istringstream iss(line);
      iss >> energy_spectrum_device[line_index] >> cdf_device[line_index];
      sum_cdf += cdf_device[line_index];
      ++line_index;
    }

    // Compute CDF and normalized it
    cdf_device[0] /= sum_cdf;
    for (GGuint i = 1; i < number_of_energy_bins_; ++i) {
      cdf_device[i] = cdf_device[i]/sum_cdf + cdf_device[i-1];
    }

    // By security, final value of cdf must be 1 !!!
    cdf_device[number_of_energy_bins_-1] = 1.0;

    // Release the pointers
    opencl_manager.ReleaseDeviceBuffer(energy_spectrum_, energy_spectrum_device);
    opencl_manager.ReleaseDeviceBuffer(cdf_, cdf_device);

    // Closing file
    spectrum_stream.close();
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSXRaySource::Initialize(void)
{
  GGcout("GGEMSXRaySource", "Initialize", 3) << "Initializing the GGEMS X-Ray source..." << GGendl;

  // Initialize GGEMS source
  GGEMSSource::Initialize();

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
  beam_aperture_ = GGEMSUnits::AngleUnit(beam_aperture, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSXRaySource::SetFocalSpotSize(GGfloat const& width, GGfloat const& height, GGfloat const& depth, std::string const& unit)
{
  focal_spot_size_ = MakeFloat3(GGEMSUnits::DistanceUnit(width, unit), GGEMSUnits::DistanceUnit(height, unit), GGEMSUnits::DistanceUnit(depth, unit));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSXRaySource* create_ggems_xray_source(void)
{
  return new(std::nothrow) GGEMSXRaySource;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void initialize_ggems_xray_source(GGEMSXRaySource* xray_source)
{
  xray_source->Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_source_name_ggems_xray_source(GGEMSXRaySource* xray_source, char const* source_name)
{
  xray_source->SetSourceName(source_name);
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

void set_number_of_particles_xray_source(GGEMSXRaySource* xray_source, GGulong const number_of_particles)
{
  xray_source->SetNumberOfParticles(number_of_particles);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_source_particle_type_ggems_xray_source( GGEMSXRaySource* xray_source, char const* particle_name)
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

void set_local_axis_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const m00, GGfloat const m01, GGfloat const m02,GGfloat const m10, GGfloat const m11, GGfloat const m12, GGfloat const m20, GGfloat const m21, GGfloat const m22)
{
  xray_source->SetLocalAxis(m00, m01, m02, m10, m11, m12, m20, m21, m22);
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
