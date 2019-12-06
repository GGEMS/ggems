/*!
  \file xray_source.cc

  \brief This class define a XRay source in GGEMS useful for CT/CBCT simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 22, 2019
*/

#include <sstream>

#include "GGEMS/sources/xray_source.hh"
#include "GGEMS/tools/print.hh"
#include "GGEMS/global/ggems_constants.hh"
#include "GGEMS/tools/functions.hh"
#include "GGEMS/tools/matrix.hh"
#include "GGEMS/processes/particles.hh"

#ifdef _WIN32
#ifdef min
#undef min
#endif
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

XRaySource::XRaySource(void)
: GGEMSSourceManager(),
  beam_aperture_(std::numeric_limits<float>::min()),
  is_monoenergy_mode_(false),
  monoenergy_(-1.0f),
  energy_spectrum_filename_(""),
  number_of_energy_bins_(0),
  p_energy_spectrum_(nullptr),
  p_cdf_(nullptr)
{
  GGEMScout("XRaySource", "XRaySource", 1)
    << "Allocation of XRaySource..." << GGEMSendl;

  // Initialization of parameters
  focal_spot_size_ = Matrix::MakeFloatXYZ(
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::min()
  );

  // Initialization of local axis
  p_geometry_transformation_->SetAxisTransformation(
    0.0f, 0.0f, -1.0f,
    0.0f, 1.0f, 0.0f,
    1.0f, 0.0f, 0.0f
  );
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

XRaySource::~XRaySource(void)
{
  // Get the pointer on OpenCL singleton
  OpenCLManager& opencl_manager = OpenCLManager::GetInstance();

  // Freeing the device buffers
  if (p_energy_spectrum_) {
    opencl_manager.Deallocate(p_energy_spectrum_,
      number_of_energy_bins_ * sizeof(cl_double));
    p_energy_spectrum_ = nullptr;
  }

  if (p_cdf_) {
    opencl_manager.Deallocate(p_cdf_,
      number_of_energy_bins_ * sizeof(cl_double));
    p_cdf_ = nullptr;
  }

    opencl_manager.Deallocate(p_debug_,
      number_of_energy_bins_ * sizeof(cl_float));
    p_debug_ = nullptr;

  GGEMScout("XRaySource", "~XRaySource", 1)
    << "Deallocation of XRaySource..." << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void XRaySource::InitializeKernel(void)
{
  GGEMScout("XRaySource", "InitializeKernel", 1)
    << "Initializing kernel..." << GGEMSendl;

  // Getting the pointer on OpenCL manager
  OpenCLManager& opencl_manager = OpenCLManager::GetInstance();

  // Getting the path to kernel
  std::string const kOpenCLKernelPath = OPENCL_KERNEL_PATH;
  std::string const kFilename = kOpenCLKernelPath
    + "/get_primaries_xray_source.cl";

  // Compiling the kernel
  p_kernel_get_primaries_ = opencl_manager.CompileKernel(kFilename,
    "get_primaries_xray_source");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void XRaySource::GetPrimaries(Particle* p_particle)
{
  GGEMScout("XRaySource", "GetPrimaries", 1)
    << "Getting primaries..." << GGEMSendl;

  // Getting the opencl manager for event and command queue
  OpenCLManager& opencl_manager = OpenCLManager::GetInstance();

  // Command queue
  cl::CommandQueue* p_queue = opencl_manager.GetCommandQueue();

  // Event
  cl::Event* p_event = opencl_manager.GetEvent();

  // Get the number of particles
  cl_ulong const kNumberOfParticles = p_particle->GetNumberOfParticles();

  GGEMScout("XRaySource", "GetPrimaries", 0) << "Generating "
    << kNumberOfParticles << " new particles..." << GGEMSendl;

  // Set parameters for kernel
  //p_kernel_get_primaries_->setArg(0, *p_cdf_);
  //p_kernel_get_primaries_->setArg(1, *p_energy_spectrum_);
  p_kernel_get_primaries_->setArg(0, *p_debug_);
  p_kernel_get_primaries_->setArg(1, number_of_energy_bins_);

  // Define the number of work-item to launch
  cl::NDRange global(kNumberOfParticles);
  cl::NDRange offset(0);

  // Launching kernel
  cl_int kernel_status = p_queue->enqueueNDRangeKernel(*p_kernel_get_primaries_,
    offset, global, cl::NullRange, nullptr, p_event);
  opencl_manager.CheckOpenCLError(kernel_status, "XRaySource", "GetPrimaries");
  p_queue->finish(); // Wait until the kernel status is finish

  // Displaying time in kernel
  opencl_manager.DisplayElapsedTimeInKernel("GetPrimaries");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void XRaySource::PrintInfos(void) const
{
  GGEMScout("XRaySource", "PrintInfos", 0) << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "XRaySource Infos:" << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "-----------------" << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "*Particle type: ";
  if (particle_type_ == ParticleName::PHOTON) {
    std::cout << "Photon" << std::endl;
  }
  if (particle_type_ == ParticleName::ELECTRON) {
    std::cout << "Electron" << std::endl;
  }
  GGEMScout("XRaySource", "PrintInfos", 0) << "*Energy mode: ";
  if (is_monoenergy_mode_) std::cout << "Monoenergy" << std::endl;
  else std::cout << "Polyenergy" << std::endl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "*Position: " << "("
    << p_geometry_transformation_->GetPosition().s[0] << ", "
    << p_geometry_transformation_->GetPosition().s[1] << ", "
    << p_geometry_transformation_->GetPosition().s[2] << " ) m3" << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "*Rotation: " << "("
    << p_geometry_transformation_->GetRotation().s[0] << ", "
    << p_geometry_transformation_->GetRotation().s[1] << ", "
    << p_geometry_transformation_->GetRotation().s[2] << ") degree"
    << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "*Beam aperture: "
    << beam_aperture_ << " degrees" << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "*Focal spot size: " << "("
    << focal_spot_size_.s[0] << ", "
    << focal_spot_size_.s[1] << ", "
    << focal_spot_size_.s[2] << ") mm3" << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "*Local axis: " << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "[" << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "    "
    << p_geometry_transformation_->GetLocalAxis().m00_ << " "
    << p_geometry_transformation_->GetLocalAxis().m01_ << " "
    << p_geometry_transformation_->GetLocalAxis().m02_ << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "    "
    << p_geometry_transformation_->GetLocalAxis().m10_ << " "
    << p_geometry_transformation_->GetLocalAxis().m11_ << " "
    << p_geometry_transformation_->GetLocalAxis().m12_ << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "    "
    << p_geometry_transformation_->GetLocalAxis().m20_ << " "
    << p_geometry_transformation_->GetLocalAxis().m21_ << " "
    << p_geometry_transformation_->GetLocalAxis().m22_ << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "]" << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void XRaySource::SetMonoenergy(float const& monoenergy)
{
  monoenergy_ = monoenergy;
  is_monoenergy_mode_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void XRaySource::SetPolyenergy(char const* energy_spectrum_filename)
{
  energy_spectrum_filename_ = energy_spectrum_filename;
  is_monoenergy_mode_ = false;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void XRaySource::CheckParameters(void) const
{
  GGEMScout("XRaySource", "CheckParameters", 1)
    << "Checking the mandatory parameters..." << GGEMSendl;

  // Checking the parameters of Source Manager
  GGEMSSourceManager::CheckParameters();

  // Checking the beam aperture
  if (Misc::IsEqual(beam_aperture_, std::numeric_limits<float>::min())) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a beam aperture for the source!!!";
    Misc::ThrowException("XRaySource", "CheckParameters", oss.str());
  }
  else if (beam_aperture_ < 0.0f) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "The beam aperture must be >= 0!!!";
    Misc::ThrowException("XRaySource", "CheckParameters", oss.str());
  }

  // Checking the focal spot size
  if (Misc::IsEqual(focal_spot_size_.s[0], std::numeric_limits<float>::min()) ||
      Misc::IsEqual(focal_spot_size_.s[1], std::numeric_limits<float>::min()) ||
      Misc::IsEqual(focal_spot_size_.s[2], std::numeric_limits<float>::min())) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a focal spot size!!!";
    Misc::ThrowException("XRaySource", "CheckParameters", oss.str());
  }

  // Focal spot size must be a positive value
  if (focal_spot_size_.s[0] < 0.0f ||
      focal_spot_size_.s[1] < 0.0f ||
      focal_spot_size_.s[2] < 0.0f) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "The focal spot size is a posivite value!!!";
    Misc::ThrowException("XRaySource", "CheckParameters", oss.str());
  }

  // Checking the energy
  if (is_monoenergy_mode_) {
    if (Misc::IsEqual(monoenergy_, -1.0f)) {
      std::ostringstream oss(std::ostringstream::out);
      oss << "You have to set an energy in monoenergetic mode!!!";
      Misc::ThrowException("XRaySource", "CheckParameters", oss.str());
    }

    if (monoenergy_ < 0.0f) {
      std::ostringstream oss(std::ostringstream::out);
      oss << "The energy must be a positive value!!!";
      Misc::ThrowException("XRaySource", "CheckParameters", oss.str());
    }
  }

  if (!is_monoenergy_mode_) {
    if (energy_spectrum_filename_.empty()) {
      std::ostringstream oss(std::ostringstream::out);
      oss << "You have to provide a energy spectrum file in polyenergy mode!!!";
      Misc::ThrowException("XRaySource", "CheckParameters", oss.str());
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void XRaySource::FillEnergy(void)
{
  GGEMScout("XRaySource", "FillEnergy", 1) << "Filling energy..." << GGEMSendl;

  // Get the pointer on OpenCL Manager
  OpenCLManager& opencl_manager = OpenCLManager::GetInstance();

  // Monoenergy mode
  if (is_monoenergy_mode_) {
    number_of_energy_bins_ = 1;

    // Allocation of memory on OpenCL device
    // Energy
    p_energy_spectrum_ = opencl_manager.Allocate(nullptr, 1 * sizeof(cl_double),
      CL_MEM_READ_WRITE);

    // Cumulative distribution function
    p_cdf_ = opencl_manager.Allocate(nullptr, 1 * sizeof(cl_double),
      CL_MEM_READ_WRITE);
  }
  else { // Polyenergy mode 
    // Read a first time the spectrum file counting the number of lines
    std::ifstream spectrum_stream(energy_spectrum_filename_, std::ios::in);
    Stream::CheckInputStream(spectrum_stream, energy_spectrum_filename_);

    // Compute number of energy bins
    std::string line;
    while (std::getline(spectrum_stream, line)) ++number_of_energy_bins_;

    // Returning to beginning of the file to read it again
    spectrum_stream.clear();
    spectrum_stream.seekg(0, std::ios::beg);

    p_debug_ = opencl_manager.Allocate(nullptr,
      number_of_energy_bins_ * sizeof(cl_float), CL_MEM_READ_WRITE);

    cl_float* p_debug =
      opencl_manager.GetDeviceBuffer<cl_float>(p_debug_);

    for (int i = 0; i < number_of_energy_bins_; ++i) {
      p_debug[i] = 10.0f;
    }
    // Allocation of memory on OpenCL device
    // Energy
//    p_energy_spectrum_ = opencl_manager.Allocate(nullptr,
 //     number_of_energy_bins_ * sizeof(cl_double), CL_MEM_READ_WRITE);

    // Cumulative distribution function
    //p_cdf_ = opencl_manager.Allocate(nullptr,
      //number_of_energy_bins_ * sizeof(cl_double), CL_MEM_READ_WRITE);

    // Creating 2 temporary buffers for energy and cdf on host memory
    //double* p_tmp_energy = new double[number_of_energy_bins_];
    //double* p_tmp_cdf = new double[number_of_energy_bins_];

    // Get the energy pointer on OpenCL device
   // cl_double* p_energy_spectrum =
   //   opencl_manager.GetDeviceBuffer<cl_double>(p_energy_spectrum_);

    // Get the cdf pointer on OpenCL device
    //cl_double* p_cdf = opencl_manager.GetDeviceBufferWrite<cl_double>(p_cdf_);

    // Read the input spectrum and computing the sum for the cdf
   /* double test = 0.0;
    for (int i = 0; i < 10; ++i) {
      test = 10.0 + i;
      p_energy_spectrum[i] = test;
    }*/
    /*int line_index = 0;
    double sum_cdf = 0.0;
    while (std::getline(spectrum_stream, line)) {
      std::istringstream iss(line);
      *p_energy_spectrum = 2.0;
      //iss >> *p_energy_spectrum++;// >> p_cdf[line_index];
      //sum_cdf += p_cdf[line_index];
      std::cout << *p_energy_spectrum << std::endl;
      ++p_energy_spectrum;
      ++line_index;
    }*/

    // Compute CDF and normalized in same time by security
  /*  p_cdf[0] /= sum_cdf;
    for (cl_uint i = 1; i < number_of_energy_bins_; ++i) {
      p_cdf[i] = p_cdf[i]/sum_cdf + p_cdf[i-1];
    }

    // By security, final value of cdf must be 1 !!!
    p_cdf[number_of_energy_bins_ - 1] = 1.0;*/

    // Release the pointers
   // opencl_manager.ReleaseDeviceBuffer(p_energy_spectrum_, p_energy_spectrum);
    //opencl_manager.ReleaseDeviceBuffer(p_cdf_, p_cdf);
    opencl_manager.ReleaseDeviceBuffer(p_debug_, p_debug);

    // Closing file
    spectrum_stream.close();
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void XRaySource::Initialize(void)
{
  GGEMScout("XRaySource", "Initialize", 1)
    << "Initializing the X-Ray source..." << GGEMSendl;

  // Check the mandatory parameters
  CheckParameters();

  // Initializing the kernel for OpenCL
  InitializeKernel();

  // Filling the energy
  FillEnergy();

  // The source is initialized
  is_initialized_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void XRaySource::SetBeamAperture(float const& beam_aperture)
{
  beam_aperture_ = beam_aperture;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void XRaySource::SetFocalSpotSize(float const& width, float const& height,
  float const& depth)
{
  focal_spot_size_ = Matrix::MakeFloatXYZ(width, height, depth);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

XRaySource* create_ggems_xray_source(void)
{
  return XRaySource::GetInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void delete_ggems_xray_source(void)
{
  GGEMSSourceManager::DeleteInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void initialize_xray_source(XRaySource* p_source_manager)
{
  p_source_manager->Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_position_xray_source(XRaySource* p_source_manager, float const pos_x,
  float const pos_y, float const pos_z)
{
  p_source_manager->SetPosition(pos_x, pos_y, pos_z);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_infos_xray_source(XRaySource* p_source_manager)
{
  p_source_manager->PrintInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_particle_type_xray_source(XRaySource* p_source_manager,
  char const* particle_name)
{
  p_source_manager->SetParticleType(particle_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_beam_aperture_xray_source(XRaySource* p_source_manager,
  float const beam_aperture)
{
  p_source_manager->SetBeamAperture(beam_aperture);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_focal_spot_size_xray_source(XRaySource* p_source_manager,
  float const width, float const height, float const depth)
{
  p_source_manager->SetFocalSpotSize(width, height, depth);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_local_axis_xray_source(XRaySource* p_source_manager,
  float const m00, float const m01, float const m02,
  float const m10, float const m11, float const m12,
  float const m20, float const m21, float const m22)
{
  p_source_manager->SetLocalAxis(
    m00, m01, m02,
    m10, m11, m12,
    m20, m21, m22
  );
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_rotation_xray_source(XRaySource* p_source_manager, float const rx,
  float const ry, float const rz)
{
  p_source_manager->SetRotation(rx, ry, rz);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void update_rotation_xray_source(XRaySource* p_source_manager, float const rx,
  float const ry, float const rz)
{
  p_source_manager->UpdateRotation(rx, ry, rz);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_monoenergy_xray_source(XRaySource* p_source_manager,
  float const monoenergy)
{
  p_source_manager->SetMonoenergy(monoenergy);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_polyenergy_xray_source(XRaySource* p_source_manager,
  char const* energy_spectrum)
{
  p_source_manager->SetPolyenergy(energy_spectrum);
}
