/*!
  \file ggems_manager.cc

  \brief GGEMS class managing the complete simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday September 30, 2019
*/

#include <algorithm>

#include <fcntl.h>
#ifdef _WIN32
#include <windows.h>
#include <wincrypt.h>
#else
#include <unistd.h>
#endif

#include "GGEMS/tools/system_of_units.hh"
#include "GGEMS/tools/print.hh"
#include "GGEMS/global/ggems_manager.hh"
#include "GGEMS/tools/functions.hh"
#include "GGEMS/global/ggems_constants.hh"
#include "GGEMS/tools/memory.hh"
#include "GGEMS/processes/particles.hh"

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
  p_particle_(nullptr)
{
  GGEMScout("GGEMSManager", "GGEMSManager", 1)
    << "Allocation of GGEMS Manager singleton..." << GGEMSendl;

  // Allocation of the memory for the physics list
  v_physics_list_.resize(ProcessName::NUMBER_PROCESSES, false);

  // Allocation of the memory for the secondaries list
  v_secondaries_list_.resize(ProcessName::NUMBER_PARTICLES, false);

  // Allocation of particle object
  p_particle_ = new Particle();
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

  GGEMScout("GGEMSManager", "~GGEMSManager", 1)
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

uint32_t GGEMSManager::GenerateSeed() const
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
  return static_cast<uint64_t>(seedWin32);
  #else
  // Open a system random file
  int file_descriptor = ::open("/dev/urandom", O_RDONLY | O_NONBLOCK);
  if (file_descriptor < 0) {
    std::ostringstream oss( std::ostringstream::out );
    oss << "Error opening the file '/dev/urandom': " << strerror(errno)
      << std::endl;
    Misc::ThrowException("GGEMSManager", "GenerateSeed",
      oss.str());
  }

  // Buffer storing 8 characters
  char seedArray[sizeof(uint32_t)];
  ::read(file_descriptor, reinterpret_cast<uint32_t*>(seedArray),
     sizeof(uint32_t));
  ::close(file_descriptor);
  uint32_t *seedUInt = reinterpret_cast<uint32_t*>(seedArray);
  return *seedUInt;
  #endif
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetNumberOfParticles(uint64_t const& number_of_particles)
{
  number_of_particles_ = number_of_particles;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetNumberOfBatchs(uint32_t const& number_of_batchs)
{
  v_number_of_particles_in_batch_.resize(number_of_batchs, 0);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::ComputeNumberOfParticlesInBatch()
{
  // Get the number of batchs
  size_t const kNumberOfBatchs = v_number_of_particles_in_batch_.size();

  // Distribute the number of particles in batch
  for (auto&& i : v_number_of_particles_in_batch_) {
    i = number_of_particles_ / kNumberOfBatchs;
  }

  // Add the remaining particles
  for (uint32_t i = 0; i < number_of_particles_ % kNumberOfBatchs; ++i)
    v_number_of_particles_in_batch_[i]++;
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
  double const& distance)
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
  uint32_t const& level)
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

void GGEMSManager::SetGeometryTolerance(double const& distance)
{
  // Geometry tolerance distance in the range [1mm;1nm]
  geometry_tolerance_ = std::max(Units::nm, std::min(Units::mm, distance));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetCrossSectionTableNumberOfBins(
  uint32_t const& number_of_bins)
{
  cross_section_table_number_of_bins_ = number_of_bins;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetCrossSectionTableEnergyMin(double const& min_energy)
{
  cross_section_table_energy_min_ = min_energy;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetCrossSectionTableEnergyMax(double const& max_energy)
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

  if (v_number_of_particles_in_batch_.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a number of batch > 0";
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
  GGEMSTools::PrintBanner();

  // Checking the mandatory parameters
  CheckParameters();
  GGEMScout("GGEMSManager", "Initialize", 0) << "Parameters OK" << GGEMSendl;

  // Initialize the pseudo random number generator
  srand(seed_);
  GGEMScout("GGEMSManager", "Initialize", 0)
    << "Pseudo-random number generator seeded OK" << GGEMSendl;

  // Compute the number of particles in batch
  ComputeNumberOfParticlesInBatch();
  GGEMScout("GGEMSManager", "Initialize", 0)
    << "Particles arranged in batch OK" << GGEMSendl;

  // Initialization of the particles
  p_particle_->Initialize();
  GGEMScout("GGEMSManager", "Initialize", 0)
    << "Initialization of particles OK" << GGEMSendl;

  // Printing informations about the simulation
  PrintInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::PrintInfos() const
{
  GGEMScout("GGEMSManager", "PrintInfos", 0) << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "++++++++++++++++" << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "*Seed: " << seed_ << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "*Number of particles: "
    << number_of_particles_ << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 0) << "*Number of batchs: "
    << v_number_of_particles_in_batch_.size() << GGEMSendl;
  GGEMScout("GGEMSManager", "PrintInfos", 1) << "*Number of particles in batch:"
    << GGEMSendl;
  for (auto&& i : v_number_of_particles_in_batch_) {
    GGEMScout("GGEMSManager", "PrintInfos", 1) << "  --> " << i << GGEMSendl;
  }
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

GGEMSManager* get_instance_ggems_manager(void)
{
  return &GGEMSManager::GetInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_seed(GGEMSManager* ggems_manager, uint32_t const seed)
{
  ggems_manager->SetSeed(seed);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void initialize_simulation(GGEMSManager* ggems_manager)
{
  ggems_manager->Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_number_of_particles(GGEMSManager* ggems_manager,
  uint64_t const number_of_particles)
{
  ggems_manager->SetNumberOfParticles(number_of_particles);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_number_of_batchs(GGEMSManager* ggems_manager,
  uint32_t const number_of_batchs)
{
  ggems_manager->SetNumberOfBatchs(number_of_batchs);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_process(GGEMSManager* ggems_manager, char const* process_name)
{
  ggems_manager->SetProcess(process_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_particle_cut(GGEMSManager* ggems_manager, char const* particle_name,
  double const distance)
{
  ggems_manager->SetParticleCut(particle_name, distance);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_geometry_tolerance(GGEMSManager* ggems_manager, double const distance)
{
  ggems_manager->SetGeometryTolerance(distance);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_secondary_particle_and_level(GGEMSManager* ggems_manager,
  char const* particle_name, uint32_t const level)
{
  ggems_manager->SetParticleSecondaryAndLevel(particle_name, level);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_cross_section_table_number_of_bins(GGEMSManager* ggems_manager,
  uint32_t const number_of_bins)
{
  ggems_manager->SetCrossSectionTableNumberOfBins(number_of_bins);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_cross_section_table_energy_min(GGEMSManager* ggems_manager,
  double const min_energy)
{
  ggems_manager->SetCrossSectionTableEnergyMin(min_energy);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_cross_section_table_energy_max(GGEMSManager* ggems_manager,
  double const max_energy)
{
  ggems_manager->SetCrossSectionTableEnergyMax(max_energy);
}
