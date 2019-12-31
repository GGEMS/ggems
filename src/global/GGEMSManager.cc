/*!
  \file GGEMSManager.cc

  \brief GGEMS class managing the GGEMS simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday September 30, 2019
*/

#include <algorithm>
#include <fcntl.h>

#ifdef _WIN32
#ifdef _MSC_VER
#define NOMINMAX
#endif
#include <windows.h>
#include <wincrypt.h>
#else
#include <unistd.h>
#endif

#include "GGEMS/sources/ggems_source_manager.hh"

#include "GGEMS/tools/GGEMSSystemOfUnits.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/tools/GGEMSChrono.hh"
#include "GGEMS/tools/GGEMSMemoryAllocation.hh"
#include "GGEMS/tools/GGEMSTools.hh"

#include "GGEMS/global/GGEMSManager.hh"
#include "GGEMS/global/GGEMSConstants.hh"

#include "GGEMS/processes/particles.hh"
#include "GGEMS/processes/primary_particles.hh"

#include "GGEMS/randoms/pseudo_random_generator.hh"
#include "GGEMS/randoms/random.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSManager::GGEMSManager(void)
: seed_(0),
  version_("1.0"),
  number_of_particles_(0),
  v_number_of_particles_in_batch_(0),
  v_physics_list_(0),
  v_secondaries_list_(0),
  photon_distance_cut_(Units::um),
  electron_distance_cut_(Units::um),
  geometry_tolerance_(Tolerance::GEOMETRY),
  photon_level_secondaries_(0),
  electron_level_secondaries_(0),
  cross_section_table_number_of_bins_(Limit::CROSS_SECTION_TABLE_NUMBER_BINS),
  cross_section_table_energy_min_(Limit::CROSS_SECTION_TABLE_ENERGY_MIN),
  cross_section_table_energy_max_(Limit::CROSS_SECTION_TABLE_ENERGY_MAX),
  p_particle_(nullptr),
  p_random_generator_(nullptr),
  source_manager_(GGEMSSourceManager::GetInstance()),
  opencl_manager_(OpenCLManager::GetInstance())
{
  GGEMScout("GGEMSManager", "GGEMSManager", 3)
    << "Allocation of GGEMS Manager singleton..." << GGEMSendl;

  // Allocation of the memory for the physics list
  v_physics_list_.resize(ProcessName::NUMBER_PROCESSES, false);

  // Allocation of the memory for the secondaries list
  v_secondaries_list_.resize(ProcessName::NUMBER_PARTICLES, false);

  // Allocation of particle
  p_particle_ = new Particle();

  // Allocation of pseudo random generator
  p_random_generator_ = new RandomGenerator();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSManager::~GGEMSManager(void)
{
  // Freeing memory
  if (p_particle_) {
    delete p_particle_;
    p_particle_ = nullptr;
  }

  if (p_random_generator_) {
    delete p_random_generator_;
    p_random_generator_ = nullptr;
  }

  GGEMScout("GGEMSManager", "~GGEMSManager", 3)
    << "Deallocation of GGEMS Manager singleton..." << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetSeed(uint32_t const& seed)
{
  seed_ = seed;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGuint GGEMSManager::GenerateSeed() const
{
  #ifdef _WIN32
  HCRYPTPROV seedWin32;
  if (CryptAcquireContext(&seedWin32, NULL, NULL, PROV_RSA_FULL,
    CRYPT_VERIFYCONTEXT ) == FALSE) {
    std::ostringstream oss(std::ostringstream::out);
    char buffer_error[256];
    oss << "Error finding a seed: " <<
      strerror_s(buffer_error, 256, errno) << std::endl;
    Misc::ThrowException("GGEMSManager", "GenerateSeed",
      oss.str());
  }
  return static_cast<uint32_t>(seedWin32);
  #else
  // Open a system random file
  GGint file_descriptor = ::open("/dev/urandom", O_RDONLY | O_NONBLOCK);
  if (file_descriptor < 0) {
    std::ostringstream oss( std::ostringstream::out );
    oss << "Error opening the file '/dev/urandom': " << strerror(errno)
      << std::endl;
    Misc::ThrowException("GGEMSManager", "GenerateSeed",
      oss.str());
  }

  // Buffer storing 8 characters
  char seedArray[sizeof(GGuint)];
  ::read(file_descriptor, reinterpret_cast<GGuint*>(seedArray),
     sizeof(GGuint));
  ::close(file_descriptor);
  GGuint *seedUInt = reinterpret_cast<GGuint*>(seedArray);
  return *seedUInt;
  #endif
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetNumberOfParticles(GGulong const& number_of_particles)
{
  number_of_particles_ = number_of_particles;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::CheckMemoryForParticles(void) const
{
  // By security the particle allocation by batch should not exceed 10% of
  // RAM memory

  // Compute the RAM memory percentage allocated for primary particles
  GGdouble const kRAMParticles =
    static_cast<GGdouble>(sizeof(PrimaryParticles))
    + static_cast<GGdouble>(sizeof(Random));

  // Getting the RAM memory on activated device
  GGdouble const kMaxRAMDevice = static_cast<GGdouble>(
    opencl_manager_.GetMaxRAMMemoryOnActivatedDevice());

  // Computing the ratio of used RAM memory on device
  GGdouble const kMaxRatioUsedRAM = kRAMParticles / kMaxRAMDevice;

  // Computing a theoric max. number of particles depending on activated
  // device and advice this number to the user. 10% of RAM memory for particles
  GGulong const kTheoricMaxNumberOfParticles = static_cast<GGulong>(
    0.1 * kMaxRAMDevice / (kRAMParticles/MAXIMUM_PARTICLES));

  if (kMaxRatioUsedRAM > 0.1) { // Printing warning
    GGEMSwarn("GGEMSManager", "CheckMemoryForParticles", 0)
      << "Warning!!! The number of particles in a batch defined during GGEMS "
      << "compilation is maybe to high. We recommand to not use more than 10% "
      << "of RAM memory for particles allocation. Your theoric number of "
      << "particles is " << kTheoricMaxNumberOfParticles << ". Recompile GGEMS "
      << "with this number of particles is recommended." << GGEMSendl;
  }
  else { // Printing theoric number of particle
    GGEMScout("GGEMSManager", "CheckMemoryForParticles", 0)
      << "The number of particles in a batch defined during the compilation is "
      << "correct. We recommend to not used more than 10% of memory for "
      << "particle allocation. Your theoric number of particles is "
      << kTheoricMaxNumberOfParticles << ". Recompile GGEMS with this number "
      << " of particles is recommended" << GGEMSendl;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::OrganizeParticlesInBatch(void)
{
  GGEMScout("GGEMSManager", "OrganizeParticlesInBatch", 3)
    << "Organizing the number of particles in batch..." << GGEMSendl;

  // Computing the number of batch depending on the number of simulated
  // particles and the maximum simulated particles defined during GGEMS
  // compilation
  std::size_t const kNumberOfBatchs =
    number_of_particles_ / MAXIMUM_PARTICLES + 1;

  // Resizing vector storing the number of particles in batch
  v_number_of_particles_in_batch_.resize(kNumberOfBatchs, 0);

  // Computing the number of simulated particles in batch
  if (kNumberOfBatchs == 1) {
    v_number_of_particles_in_batch_[0] = number_of_particles_;
  }
  else {
    for (auto&& i : v_number_of_particles_in_batch_) {
      i = number_of_particles_ / kNumberOfBatchs;
    }

    // Adding the remaing particles
    for (std::size_t i = 0; i < number_of_particles_ % kNumberOfBatchs; ++i) {
      v_number_of_particles_in_batch_[i]++;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetProcess(char const* process_name)
{
  // Convert the process name in string
  std::string process_name_str(process_name);

  // Transform the string to lower character
  std::transform(process_name_str.begin(), process_name_str.end(),
    process_name_str.begin(), ::tolower);

  // Activate process
  if (!process_name_str.compare("compton")) {
    v_physics_list_.at(ProcessName::PHOTON_COMPTON) = true;
  }
  else if (!process_name_str.compare("photoelectric")) {
    v_physics_list_.at(ProcessName::PHOTON_PHOTOELECTRIC) = true;
  }
  else if (!process_name_str.compare("rayleigh")) {
    v_physics_list_.at(ProcessName::PHOTON_RAYLEIGH) = true;
  }
  else if (!process_name_str.compare("eionisation")) {
    v_physics_list_.at(ProcessName::ELECTRON_IONISATION) = true;
  }
  else if (!process_name_str.compare("ebremsstrahlung")) {
    v_physics_list_.at(ProcessName::ELECTRON_BREMSSTRAHLUNG) = true;
  }
  else if (!process_name_str.compare("emultiplescattering")) {
    v_physics_list_.at(ProcessName::ELECTRON_MSC) = true;
  }
  else {
    Misc::ThrowException("GGEMSManager", "SetProcess", "Unknown process!!!");
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetParticleCut(char const* particle_name,
  GGdouble const& distance)
{
  // Convert the particle name in string
  std::string particle_name_str(particle_name);

  // Transform the string to lower character
  std::transform(particle_name_str.begin(), particle_name_str.end(),
    particle_name_str.begin(), ::tolower);

  if (!particle_name_str.compare("photon")) {
    photon_distance_cut_ = distance;
  }
  else if (!particle_name_str.compare("electron")) {
    electron_distance_cut_ = distance;
  }
  else {
    Misc::ThrowException("GGEMSManager", "SetParticleCut",
      "Unknown particle!!!");
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetParticleSecondaryAndLevel(char const* particle_name,
  GGuint const& level)
{
  // Convert the particle name in string
  std::string particle_name_str(particle_name);

  // Transform the string to lower character
  std::transform(particle_name_str.begin(), particle_name_str.end(),
    particle_name_str.begin(), ::tolower);

  if (!particle_name_str.compare("photon")) {
    GGEMScout("GGEMSManager", "SetParticleSecondaryAndLevel",0)
      << "Warning!!! Photon as secondary is not available yet!!!" << GGEMSendl;
    v_secondaries_list_.at(ParticleName::PHOTON) = false;
    photon_level_secondaries_ = 0;
  }
  else if (!particle_name_str.compare("electron")) {
    v_secondaries_list_.at(ParticleName::ELECTRON) = true;
    electron_level_secondaries_ = level;
  }
  else {
    Misc::ThrowException("GGEMSManager", "SetParticleSecondaryAndLevel",
      "Unknown particle!!!");
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetGeometryTolerance(GGdouble const& distance)
{
  // Geometry tolerance distance in the range [1mm;1nm]
  geometry_tolerance_ = fmax(Units::nm, fmin(Units::mm, distance));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetCrossSectionTableNumberOfBins(
  GGuint const& number_of_bins)
{
  cross_section_table_number_of_bins_ = number_of_bins;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetCrossSectionTableEnergyMin(GGdouble const& min_energy)
{
  cross_section_table_energy_min_ = min_energy;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetCrossSectionTableEnergyMax(GGdouble const& max_energy)
{
  cross_section_table_energy_max_ = max_energy;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::CheckParameters()
{
  GGEMScout("GGEMSManager", "CheckParameters", 1)
    << "Checking the mandatory parameters..." << GGEMSendl;

  // Checking the seed of the random generator
  if (seed_ == 0) seed_ = GenerateSeed();

  // Checking the number of particles
  if (number_of_particles_ == 0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a number of particles > 0!!!";
    Misc::ThrowException("GGEMSManager", "CheckParameters", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::Initialize()
{
  GGEMScout("GGEMSManager", "Initialize", 1)
    << "Initialization of GGEMS Manager singleton..." << GGEMSendl;

  // Printing the banner with the GGEMS version
  PrintBanner();

  // Checking the mandatory parameters
  CheckParameters();
  GGEMScout("GGEMSManager", "Initialize", 0) << "Parameters OK" << GGEMSendl;

  // Initialize the pseudo random number generator
  srand(seed_);
  GGEMScout("GGEMSManager", "Initialize", 0)
    << "C++ Pseudo-random number generator seeded OK" << GGEMSendl;

  // Checking the RAM memory for particle and propose a new MAXIMUM_PARTICLE
  // number
  CheckMemoryForParticles();

  // Organize the particles in batch
  OrganizeParticlesInBatch();
  GGEMScout("GGEMSManager", "Initialize", 0)
    << "Particles arranged in batch OK" << GGEMSendl;

  // Initialization of the particles
  p_particle_->Initialize();
  GGEMScout("GGEMSManager", "Initialize", 0)
    << "Initialization of particles OK" << GGEMSendl;

  // Initialization of GGEMS pseudo random generator
  p_random_generator_->Initialize();
  GGEMScout("GGEMSManager", "Initialize", 0)
    << "Initialization of GGEMS pseudo random generator OK" << GGEMSendl;

  // Give particle and random to source
  source_manager_.SetParticle(p_particle_);
  source_manager_.SetRandomGenerator(p_random_generator_);

  // Initialization of the source
  source_manager_.Initialize();
  // Checking if the source is defined by the user
  if (!source_manager_.IsReady()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Problem during the source initialization!!!";
    Misc::ThrowException("GGEMSManager", "Initialize", oss.str());
  }

  // Printing informations about the simulation
  PrintInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::Run()
{
  GGEMScout("GGEMSManager", "Run", 0)
    << "GGEMS simulation is running..." << GGEMSendl;

  // Get the start time
  ChronoTime start_time = Chrono::Now();

  // Loop over the number of batch
  for (std::size_t i = 0; i < v_number_of_particles_in_batch_.size(); ++i) {
    GGEMScout("GGEMSManager", "Run", 0) << "----> Launching batch " << i+1
      << "/" << v_number_of_particles_in_batch_.size() << GGEMSendl;

    // Generating particles
    GGEMScout("GGEMSManager", "Run", 0) << "      + Generating "
      << v_number_of_particles_in_batch_[i] << " particles..." << GGEMSendl;
    source_manager_.GetPrimaries(v_number_of_particles_in_batch_[i]);
  }

  // Get the end time
  ChronoTime end_time = Chrono::Now();

  GGEMScout("GGEMSManager", "Run", 0)
    << "GGEMS is finished!" << GGEMSendl;

  // Display the elapsed time in GGEMS
  Chrono::DisplayTime(end_time - start_time, "GGEMS simulation");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::PrintInfos(void) const
{
  GGEMScout("GGEMSManager", "PrintInfos", 0) << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "++++++++++++++++" << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "*Seed: " << seed_ << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "*Number of particles: "
    << number_of_particles_ << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "*Number of batchs: "
    << v_number_of_particles_in_batch_.size() << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "*Physics list:" << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "  --> Photon: "
    << (v_physics_list_.at(0) ? "Compton " : "")
    << (v_physics_list_.at(1) ? "Photoelectric " : "")
    << (v_physics_list_.at(2) ? "Rayleigh" : "")
    << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "  --> Electron: "
    << (v_physics_list_.at(4) ? "eIonisation " : "")
    << (v_physics_list_.at(5) ? "eMultipleScattering " : "")
    << (v_physics_list_.at(6) ? "eBremsstrahlung" : "")
    << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "  --> Tables Min: "
    << cross_section_table_energy_min_/Units::MeV << " MeV, Max: "
    << cross_section_table_energy_max_/Units::MeV << " MeV, energy bins: "
    << cross_section_table_number_of_bins_ << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "  --> Range cuts Photon: "
    << photon_distance_cut_/Units::mm << " mm, Electron: "
    << electron_distance_cut_/Units::mm << " mm" << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "*Secondary particles:"
    << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "  --> Photon level: "
    << photon_level_secondaries_  << " NOT ACTIVATED!!!" << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "  --> Electron level: "
    << electron_level_secondaries_ << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "*Geometry tolerance:"
    << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "  --> Range: "
    << geometry_tolerance_/Units::mm << " mm" << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "++++++++++++++++" << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << GGEMSendl;
  ;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::PrintBanner(void) const
{
  GGcout("GGEMSManager", "PrintBanner", 0) << "      ____                  "
    << GGendl;
  GGcout("GGEMSManager", "PrintBanner", 0) << ".--. /\\__/\\ .--.            "
    << GGendl;
  GGcout("GGEMSManager", "PrintBanner", 0) << "`O  / /  \\ \\  .`     GGEMS "
    << version_ << GGendl;
  GGcout("GGEMSManager", "PrintBanner", 0) << "  `-| |  | |O`              "
    << GGendl;
  GGcout("GGEMSManager", "PrintBanner", 0) << "   -|`|..|`|-        "
    << GGendl;
  GGcout("GGEMSManager", "PrintBanner", 0) << " .` \\.\\__/./ `.    "
    << GGendl;
  GGcout("GGEMSManager", "PrintBanner", 0) << "'.-` \\/__\\/ `-.'   "
    << GGendl;
  GGcout("GGEMSManager", "PrintBanner", 0) << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSManager* get_instance_ggems_manager(void)
{
  return &GGEMSManager::GetInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_seed_ggems_manager(GGEMSManager* p_ggems_manager, GGuint const seed)
{
  p_ggems_manager->SetSeed(seed);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void initialize_ggems_manager(GGEMSManager* p_ggems_manager)
{
  p_ggems_manager->Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_number_of_particles_ggems_manager(GGEMSManager* p_ggems_manager,
  GGulong const number_of_particles)
{
  p_ggems_manager->SetNumberOfParticles(number_of_particles);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_process_ggems_manager(GGEMSManager* p_ggems_manager,
  char const* process_name)
{
  p_ggems_manager->SetProcess(process_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_particle_cut_ggems_manager(GGEMSManager* p_ggems_manager,
  char const* particle_name, GGdouble const distance)
{
  p_ggems_manager->SetParticleCut(particle_name, distance);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_geometry_tolerance_ggems_manager(GGEMSManager* p_ggems_manager,
  GGdouble const distance)
{
  p_ggems_manager->SetGeometryTolerance(distance);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_secondary_particle_and_level_ggems_manager(
  GGEMSManager* p_ggems_manager, char const* particle_name, GGuint const level)
{
  p_ggems_manager->SetParticleSecondaryAndLevel(particle_name, level);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_cross_section_table_number_of_bins_ggems_manager(
  GGEMSManager* p_ggems_manager, GGuint const number_of_bins)
{
  p_ggems_manager->SetCrossSectionTableNumberOfBins(number_of_bins);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_cross_section_table_energy_min_ggems_manager(
  GGEMSManager* p_ggems_manager, GGdouble const min_energy)
{
  p_ggems_manager->SetCrossSectionTableEnergyMin(min_energy);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_cross_section_table_energy_max_ggems_manager(
  GGEMSManager* p_ggems_manager, GGdouble const max_energy)
{
  p_ggems_manager->SetCrossSectionTableEnergyMax(max_energy);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void run_ggems_manager(GGEMSManager* p_ggems_manager)
{
  p_ggems_manager->Run();
}
