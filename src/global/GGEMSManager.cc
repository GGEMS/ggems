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
#include <cmath>
#include <sstream>

#ifdef _WIN32
#ifdef _MSC_VER
#define NOMINMAX
#endif
#include <windows.h>
#include <wincrypt.h>
#else
#include <unistd.h>
#endif

#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/physics/GGEMSMaterialsManager.hh"
#include "GGEMS/physics/GGEMSProcessesManager.hh"
#include "GGEMS/physics/GGEMSRangeCutsManager.hh"
#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/global/GGEMSManager.hh"
#include "GGEMS/global/GGEMSConstants.hh"
#include "GGEMS/geometries/GGEMSPhantomNavigatorManager.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"
#include "GGEMS/tools/GGEMSChrono.hh"
#include "GGEMS/tools/GGEMSTools.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSManager::GGEMSManager(void)
: seed_(0),
  version_("1.0"),
  is_opencl_verbose_(false),
  is_material_database_verbose_(false),
  is_source_verbose_(false),
  is_phantom_verbose_(false),
  is_memory_ram_verbose_(false),
  is_processes_verbose_(false),
  is_range_cuts_verbose_(false),
  is_random_verbose_(false),
  source_manager_(GGEMSSourceManager::GetInstance()),
  opencl_manager_(GGEMSOpenCLManager::GetInstance()),
  material_manager_(GGEMSMaterialsManager::GetInstance()),
  phantom_navigator_manager_(GGEMSPhantomNavigatorManager::GetInstance()),
  range_cuts_manager_(GGEMSRangeCutsManager::GetInstance()),
  processes_manager_(GGEMSProcessesManager::GetInstance())
  //physics_list_(0),
  //secondaries_list_(0),
  //photon_distance_cut_(GGEMSUnits::um),
  //electron_distance_cut_(GGEMSUnits::um),
  //photon_level_secondaries_(0),
  //electron_level_secondaries_(0)
{
  GGcout("GGEMSManager", "GGEMSManager", 3) << "Allocation of GGEMS Manager..." << GGendl;

  // Allocation of the memory for the physics list
  //physics_list_.resize(GGEMSProcessName::NUMBER_PROCESSES, false);

  // Allocation of the memory for the secondaries list
  //secondaries_list_.resize(GGEMSProcessName::NUMBER_PARTICLES, false);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSManager::~GGEMSManager(void)
{
  GGcout("GGEMSManager", "~GGEMSManager", 3) << "Deallocation of GGEMS Manager..." << GGendl;
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

GGuint GGEMSManager::GenerateSeed(void) const
{
  #ifdef _WIN32
  HCRYPTPROV seedWin32;
  if (CryptAcquireContext(&seedWin32, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT ) == FALSE) {
    std::ostringstream oss(std::ostringstream::out);
    char buffer_error[256];
    oss << "Error finding a seed: " << strerror_s(buffer_error, 256, errno) << std::endl;
    GGEMSMisc::ThrowException("GGEMSManager", "GenerateSeed", oss.str());
  }
  return static_cast<uint32_t>(seedWin32);
  #else
  // Open a system random file
  GGint file_descriptor = ::open("/dev/urandom", O_RDONLY | O_NONBLOCK);
  if (file_descriptor < 0) {
    std::ostringstream oss( std::ostringstream::out );
    oss << "Error opening the file '/dev/urandom': " << strerror(errno) << std::endl;
    GGEMSMisc::ThrowException("GGEMSManager", "GenerateSeed", oss.str());
  }

  // Buffer storing 8 characters
  char seedArray[sizeof(GGuint)];
  ::read(file_descriptor, reinterpret_cast<GGuint*>(seedArray), sizeof(GGuint));
  ::close(file_descriptor);
  GGuint *seedUInt = reinterpret_cast<GGuint*>(seedArray);
  return *seedUInt;
  #endif
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetOpenCLVerbose(bool const& is_opencl_verbose)
{
  is_opencl_verbose_ = is_opencl_verbose;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetMaterialDatabaseVerbose(bool const& is_material_database_verbose)
{
  is_material_database_verbose_ = is_material_database_verbose;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetSourceVerbose(bool const& is_source_verbose)
{
  is_source_verbose_ = is_source_verbose;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetPhantomVerbose(bool const& is_phantom_verbose)
{
  is_phantom_verbose_ = is_phantom_verbose;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetMemoryRAMVerbose(bool const& is_memory_ram_verbose)
{
  is_memory_ram_verbose_ = is_memory_ram_verbose;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetProcessesVerbose(bool const& is_processes_verbose)
{
  is_processes_verbose_ = is_processes_verbose;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetRangeCutsVerbose(bool const& is_range_cuts_verbose)
{
  is_range_cuts_verbose_ = is_range_cuts_verbose;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::SetRandomVerbose(bool const& is_random_verbose)
{
  is_random_verbose_ = is_random_verbose;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*void GGEMSManager::SetProcess(char const* process_name)
{
  // Convert the process name in string
  std::string process_name_str(process_name);

  // Transform the string to lower character
  std::transform(process_name_str.begin(), process_name_str.end(), process_name_str.begin(), ::tolower);

  // Activate process
  if (!process_name_str.compare("compton")) {
    physics_list_.at(GGEMSProcessName::PHOTON_COMPTON) = 1;
  }
  else if (!process_name_str.compare("photoelectric")) {
    physics_list_.at(GGEMSProcessName::PHOTON_PHOTOELECTRIC) = 1;
  }
  else if (!process_name_str.compare("rayleigh")) {
    physics_list_.at(GGEMSProcessName::PHOTON_RAYLEIGH) = 1;
  }
  else if (!process_name_str.compare("eionisation")) {
    physics_list_.at(GGEMSProcessName::ELECTRON_IONISATION) = 1;
  }
  else if (!process_name_str.compare("ebremsstrahlung")) {
    physics_list_.at(GGEMSProcessName::ELECTRON_BREMSSTRAHLUNG) = 1;
  }
  else if (!process_name_str.compare("emultiplescattering")) {
    physics_list_.at(GGEMSProcessName::ELECTRON_MSC) = 1;
  }
  else {
    GGEMSMisc::ThrowException("GGEMSManager", "SetProcess", "Unknown process!!!");
  }
}*/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*void GGEMSManager::SetParticleCut(char const* particle_name, GGdouble const& distance)
{
  // Convert the particle name in string
  std::string particle_name_str(particle_name);

  // Transform the string to lower character
  std::transform(particle_name_str.begin(), particle_name_str.end(), particle_name_str.begin(), ::tolower);

  if (!particle_name_str.compare("photon")) {
    photon_distance_cut_ = distance;
  }
  else if (!particle_name_str.compare("electron")) {
    electron_distance_cut_ = distance;
  }
  else {
    GGEMSMisc::ThrowException("GGEMSManager", "SetParticleCut", "Unknown particle!!!");
  }
}*/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*void GGEMSManager::SetParticleSecondaryAndLevel(char const* particle_name, GGuint const& level)
{
  // Convert the particle name in string
  std::string particle_name_str(particle_name);

  // Transform the string to lower character
  std::transform(particle_name_str.begin(), particle_name_str.end(), particle_name_str.begin(), ::tolower);

  if (!particle_name_str.compare("photon")) {
    GGcout("GGEMSManager", "SetParticleSecondaryAndLevel",0) << "Warning!!! Photon as secondary is not available yet!!!" << GGendl;
    secondaries_list_.at(GGEMSParticleName::PHOTON) = false;
    photon_level_secondaries_ = 0;
  }
  else if (!particle_name_str.compare("electron")) {
    secondaries_list_.at(GGEMSParticleName::ELECTRON) = true;
    electron_level_secondaries_ = level;
  }
  else {
    GGEMSMisc::ThrowException("GGEMSManager", "SetParticleSecondaryAndLevel", "Unknown particle!!!");
  }
}*/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::CheckParameters(void)
{
  GGcout("GGEMSManager", "CheckParameters", 1) << "Checking the mandatory parameters..." << GGendl;

  // Checking the seed of the random generator
  if (seed_ == 0) seed_ = GenerateSeed();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::Initialize(void)
{
  GGcout("GGEMSManager", "Initialize", 1) << "Initialization of GGEMS Manager singleton..." << GGendl;

  // Get the start time
  ChronoTime start_time = GGEMSChrono::Now();

  // Printing the banner with the GGEMS version
  PrintBanner();

  // Checking the mandatory parameters
  CheckParameters();
  GGcout("GGEMSManager", "Initialize", 0) << "Parameters OK" << GGendl;

  // Initialize the pseudo random number generator
  srand(seed_);
  GGcout("GGEMSManager", "Initialize", 0) << "C++ Pseudo-random number generator seeded OK" << GGendl;

  // Checking if material manager is ready
  if (!material_manager_.IsReady()) GGEMSMisc::ThrowException("GGEMSManager", "Initialize", "Materials are not loaded in GGEMS!!!");

  // Initialization of the source
  source_manager_.Initialize();

  // Initialization of the phantom(s)
  phantom_navigator_manager_.Initialize();

  // Printing infos about OpenCL
  if (is_opencl_verbose_) {
    opencl_manager_.PrintPlatformInfos();
    opencl_manager_.PrintDeviceInfos();
    opencl_manager_.PrintContextInfos();
    opencl_manager_.PrintCommandQueueInfos();
    opencl_manager_.PrintActivatedContextInfos();
    opencl_manager_.PrintBuildOptions();
  }

  // Printing infos about material database
  if (is_material_database_verbose_) material_manager_.PrintAvailableMaterials();

  // Printing infos about source(s)
  if (is_source_verbose_) source_manager_.PrintInfos();

  // Printing infos about phantom(s)
  if (is_phantom_verbose_) phantom_navigator_manager_.PrintInfos();

  // Printing infos about processe(s)
  if (is_processes_verbose_) {
    processes_manager_.PrintAvailableProcesses();
    processes_manager_.PrintInfos();
  }

  // Printing infos about range cuts
  if (is_range_cuts_verbose_) range_cuts_manager_.PrintInfos();

  // Printing infos about random in GGEMS
  if (is_random_verbose_) GGcout("GGEMSManager", "Initialize", 0) << "Seed: " << seed_ << GGendl;

  // Printing infos about RAM
  if (is_memory_ram_verbose_) opencl_manager_.PrintRAMStatus();

  // Get the end time
  ChronoTime end_time = GGEMSChrono::Now();

  GGcout("GGEMSManager", "Initialize", 0) << "GGEMS initialization succeeded" << GGendl;

  // Display the elapsed time in GGEMS
  GGEMSChrono::DisplayTime(end_time - start_time, "GGEMS initialization");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::Run()
{
  GGcout("GGEMSManager", "Run", 0) << "GGEMS simulation started" << GGendl;

  // Get the start time
  ChronoTime start_time = GGEMSChrono::Now();

  // Loop over the number of batch
/*  for (std::size_t i = 0; i < source_manager_.GetNumberOfBatchs(); ++i) {
    GGcout("GGEMSManager", "Run", 0) << "----> Launching batch " << i+1
      << "/" << source_manager_.GetNumberOfBatchs() << GGendl;

    GGulong const kNumberParticles =
      source_manager_.GetNumberOfParticlesInBatch(i);

    // Generating primary particles
    GGcout("GGEMSManager", "Run", 0) << "      + Generating "
      << kNumberParticles << " particles..." << GGendl;
    source_manager_.GetPrimaries(kNumberParticles);
  }*/

  // Get the end time
  ChronoTime end_time = GGEMSChrono::Now();

  GGcout("GGEMSManager", "Run", 0) << "GGEMS simulation succeeded" << GGendl;

  // Display the elapsed time in GGEMS
  GGEMSChrono::DisplayTime(end_time - start_time, "GGEMS simulation");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::PrintInfos(void) const
{
  /*GGcout("GGEMSManager", "PrintInfos", 0) << GGendl;
  GGcout("GGEMSManager", "PrintInfos", 0) << "++++++++++++++++" << GGendl;
  GGcout("GGEMSManager", "PrintInfos", 0) << "*Seed: " << seed_ << GGendl;
  GGcout("GGEMSManager", "PrintInfos", 0) << "*Physics list:" << GGendl;
  GGcout("GGEMSManager", "PrintInfos", 0) << "  --> Photon: " << (physics_list_.at(0) ? "Compton " : "") << (physics_list_.at(1) ? "Photoelectric " : "") << (physics_list_.at(2) ? "Rayleigh" : "") << GGendl;
  GGcout("GGEMSManager", "PrintInfos", 0) << "  --> Electron: " << (physics_list_.at(4) ? "eIonisation " : "") << (physics_list_.at(5) ? "eMultipleScattering " : "") << (physics_list_.at(6) ? "eBremsstrahlung" : "") << GGendl;
  GGcout("GGEMSManager", "PrintInfos", 0) << "  --> Tables Min: " << cross_section_table_energy_min_/GGEMSUnits::MeV << " MeV, Max: " << cross_section_table_energy_max_/GGEMSUnits::MeV << " MeV, energy bins: " << cross_section_table_number_of_bins_ << GGendl;
  GGcout("GGEMSManager", "PrintInfos", 0) << "  --> Range cuts Photon: " << photon_distance_cut_/GGEMSUnits::mm << " mm, Electron: " << electron_distance_cut_/GGEMSUnits::mm << " mm" << GGendl;
  GGcout("GGEMSManager", "PrintInfos", 0) << "*Secondary particles:" << GGendl;
  GGcout("GGEMSManager", "PrintInfos", 0) << "  --> Photon level: " << photon_level_secondaries_  << " NOT ACTIVATED!!!" << GGendl;
  GGcout("GGEMSManager", "PrintInfos", 0) << "  --> Electron level: " << electron_level_secondaries_ << GGendl;
  GGcout("GGEMSManager", "PrintInfos", 0) << "*Geometry tolerance:" << GGendl;
  GGcout("GGEMSManager", "PrintInfos", 0) << "++++++++++++++++" << GGendl;
  GGcout("GGEMSManager", "PrintInfos", 0) << GGendl;*/
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSManager::PrintBanner(void) const
{
  std::cout << std::endl;
  #ifdef _WIN32
  CONSOLE_SCREEN_BUFFER_INFO info;
  GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info);
  HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
  FlushConsoleInputBuffer(hConsole);
  SetConsoleTextAttribute(hConsole, 0x02);
  std::cout << "      ____" << std::endl;
  SetConsoleTextAttribute(hConsole, 0x01);
  std::cout << ".--. ";
  SetConsoleTextAttribute(hConsole, 0x02);
  std::cout << "/\\__/\\ ";
  SetConsoleTextAttribute(hConsole, 0x01);
  std::cout << ".--." << std::endl;
  SetConsoleTextAttribute(hConsole, 0x01);
  std::cout << "`";
  SetConsoleTextAttribute(hConsole, 0x06);
  std::cout << "O  ";
  SetConsoleTextAttribute(hConsole, 0x02);
  std::cout << "/ /  \\ \\  ";
  SetConsoleTextAttribute(hConsole, 0x01);
  std::cout << ".`     ";
  SetConsoleTextAttribute(hConsole, info.wAttributes);
  std::cout << "GGEMS ";
  SetConsoleTextAttribute(hConsole, 0x04);
  std::cout << version_ << std::endl;
  SetConsoleTextAttribute(hConsole, 0x01);
  std::cout << "  `-";
  SetConsoleTextAttribute(hConsole, 0x02);
  std::cout << "| |  | |";
  SetConsoleTextAttribute(hConsole, 0x06);
  std::cout << "O";
  SetConsoleTextAttribute(hConsole, 0x01);
  std::cout << "`" << std::endl;
  SetConsoleTextAttribute(hConsole, 0x01);
  std::cout << "   -";
  SetConsoleTextAttribute(hConsole, 0x02);
  std::cout << "|";
  SetConsoleTextAttribute(hConsole, 0x01);
  std::cout << "`";
  SetConsoleTextAttribute(hConsole, 0x02);
  std::cout << "|";
  SetConsoleTextAttribute(hConsole, 0x01);
  std::cout << "..";
  SetConsoleTextAttribute(hConsole, 0x02);
  std::cout << "|";
  SetConsoleTextAttribute(hConsole, 0x01);
  std::cout << "`";
  SetConsoleTextAttribute(hConsole, 0x02);
  std::cout << "|";
  SetConsoleTextAttribute(hConsole, 0x01);
  std::cout << "-" << std::endl;
  SetConsoleTextAttribute(hConsole, 0x01);
  std::cout << " .` ";
  SetConsoleTextAttribute(hConsole, 0x02);
  std::cout << "\\";
  SetConsoleTextAttribute(hConsole, 0x01);
  std::cout << ".";
  SetConsoleTextAttribute(hConsole, 0x02);
  std::cout << "\\__/";
  SetConsoleTextAttribute(hConsole, 0x01);
  std::cout << ".";
  SetConsoleTextAttribute(hConsole, 0x02);
  std::cout << "/ ";
  SetConsoleTextAttribute(hConsole, 0x01);
  std::cout << "`." << std::endl;
  SetConsoleTextAttribute(hConsole, 0x01);
  std::cout << "'.-` ";
  SetConsoleTextAttribute(hConsole, 0x02);
  std::cout << "\\/__\\/ ";
  SetConsoleTextAttribute(hConsole, 0x01);
  std::cout << "`-.'" << std::endl;
  SetConsoleTextAttribute(hConsole, info.wAttributes);
  #else
  std::cout << "      \033[32m____\033[0m" << std::endl;
  std::cout << "\033[34m.--.\033[0m \033[32m/\\__/\\\033[0m ";
  std::cout << "\033[34m.--.\033[0m" << std::endl;
  std::cout << "\033[34m`\033[0m\033[33mO\033[0m  \033[32m/ /  \\ \\\033[0m  ";
  std::cout << "\033[34m.`\033[0m     GGEMS \033[31m" << version_ << "\033[0m" << std::endl;
  std::cout << "  \033[34m`-\033[0m\033[32m| |  | |\033[0m\033[33mO\033[0m";
  std::cout << "\033[34m`\033[0m" << std::endl;
  std::cout << "   \033[34m-\033[0m\033[32m|\033[0m\033[34m`\033[0m";
  std::cout << "\033[32m|\033[0m\033[34m..\033[0m\033[32m|\033[0m";
  std::cout << "\033[34m`\033[0m\033[32m|\033[0m\033[34m-\033[0m" << std::endl;
  std::cout << " \033[34m.`\033[0m \033[32m\\\033[0m\033[34m.\033[0m";
  std::cout << "\033[32m\\__/\033[0m\033[34m.\033[0m\033[32m/\033[0m ";
  std::cout << "\033[34m`.\033[0m" << std::endl;
  std::cout << "\033[34m'.-`\033[0m \033[32m\\/__\\/\033[0m ";
  std::cout << "\033[34m`-.'\033[0m" << std::endl;
  #endif
  std::cout << std::endl;
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

void set_seed_ggems_manager(GGEMSManager* ggems_manager, GGuint const seed)
{
  ggems_manager->SetSeed(seed);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void initialize_ggems_manager(GGEMSManager* ggems_manager)
{
  ggems_manager->Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_opencl_verbose_ggems_manager(GGEMSManager* ggems_manager, bool const is_opencl_verbose)
{
  ggems_manager->SetOpenCLVerbose(is_opencl_verbose);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_material_database_verbose_ggems_manager(GGEMSManager* ggems_manager, bool const is_material_database_verbose)
{
  ggems_manager->SetMaterialDatabaseVerbose(is_material_database_verbose);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_source_ggems_manager(GGEMSManager* ggems_manager, bool const is_source_verbose)
{
  ggems_manager->SetSourceVerbose(is_source_verbose);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_phantom_ggems_manager(GGEMSManager* ggems_manager, bool const is_phantom_verbose)
{
  ggems_manager->SetPhantomVerbose(is_phantom_verbose);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_memory_ram_ggems_manager(GGEMSManager* ggems_manager, bool const is_memory_ram_verbose)
{
  ggems_manager->SetMemoryRAMVerbose(is_memory_ram_verbose);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_processes_ggems_manager(GGEMSManager* ggems_manager, bool const is_processes_verbose)
{
  ggems_manager->SetProcessesVerbose(is_processes_verbose);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_range_cuts_ggems_manager(GGEMSManager* ggems_manager, bool const is_range_cuts_verbose)
{
  ggems_manager->SetRangeCutsVerbose(is_range_cuts_verbose);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_random_ggems_manager(GGEMSManager* ggems_manager, bool const is_random_verbose)
{
  ggems_manager->SetRandomVerbose(is_random_verbose);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void run_ggems_manager(GGEMSManager* ggems_manager)
{
  ggems_manager->Run();
}
