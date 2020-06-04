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

#include "GGEMS/global/GGEMSManager.hh"
#include "GGEMS/physics/GGEMSProcessesManager.hh"
#include "GGEMS/physics/GGEMSRangeCutsManager.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/navigators/GGEMSNavigatorManager.hh"
#include "GGEMS/tools/GGEMSChrono.hh"
#include "GGEMS/tools/GGEMSRAMManager.hh"

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
  is_random_verbose_(false)
{
  GGcout("GGEMSManager", "GGEMSManager", 3) << "Allocation of GGEMS Manager..." << GGendl;
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

  // Getting the GGEMS singletons
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGEMSMaterialsDatabaseManager& material_database_manager = GGEMSMaterialsDatabaseManager::GetInstance();
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  GGEMSNavigatorManager& navigator_manager = GGEMSNavigatorManager::GetInstance();
  GGEMSProcessesManager& processes_manager = GGEMSProcessesManager::GetInstance();
  GGEMSRangeCutsManager& range_cuts_manager = GGEMSRangeCutsManager::GetInstance();
  GGEMSRAMManager& ram_manager = GGEMSRAMManager::GetInstance();

  // Checking if a context is activated
  if (!opencl_manager.IsReady()) {
    GGEMSMisc::ThrowException("GGEMSManager", "Initialize", "OpenCL Manager is not ready, you have to choose a context!!!");
  }

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
  if (!material_database_manager.IsReady()) GGEMSMisc::ThrowException("GGEMSManager", "Initialize", "Materials are not loaded in GGEMS!!!");

  // Initialization of the source
  source_manager.Initialize();

  // Initialization of the phantom(s)
  navigator_manager.Initialize();

  // Printing infos about OpenCL
  if (is_opencl_verbose_) {
    opencl_manager.PrintPlatformInfos();
    opencl_manager.PrintDeviceInfos();
    opencl_manager.PrintContextInfos();
    opencl_manager.PrintCommandQueueInfos();
    opencl_manager.PrintActivatedContextInfos();
    opencl_manager.PrintBuildOptions();
  }

  // Printing infos about material database
  if (is_material_database_verbose_) material_database_manager.PrintAvailableMaterials();

  // Printing infos about source(s)
  if (is_source_verbose_) source_manager.PrintInfos();

  // Printing infos about navigator(s)
  if (is_phantom_verbose_) navigator_manager.PrintInfos();

  // Printing infos about processe(s)
  if (is_processes_verbose_) {
    processes_manager.PrintAvailableProcesses();
    processes_manager.PrintInfos();
  }

  // Printing infos about range cuts
  if (is_range_cuts_verbose_) range_cuts_manager.PrintInfos();

  // Printing infos about random in GGEMS
  if (is_random_verbose_) GGcout("GGEMSManager", "Initialize", 0) << "GGEMS Seed: " << seed_ << GGendl;

  // Printing infos about RAM
  if (is_memory_ram_verbose_) ram_manager.PrintRAMStatus();

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

  // Get singletons
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  GGEMSNavigatorManager& navigator_manager = GGEMSNavigatorManager::GetInstance();

  // Get the start time
  ChronoTime start_time = GGEMSChrono::Now();

  // Loop over the number of sources
  for (std::size_t j = 0; j < source_manager.GetNumberOfSources(); ++j) {
    // Loop over the number of batch for each sources
    for (std::size_t i = 0; i < source_manager.GetNumberOfBatchs(j); ++i) {
      GGcout("GGEMSManager", "Run", 1) << "----> Launching batch " << i+1 << "/" << source_manager.GetNumberOfBatchs(j) << GGendl;

      // Getting the number of particles
      GGulong const kNumberOfParticles = source_manager.GetNumberOfParticlesInBatch(j, i);
      // Step 1: Generating primaries from source
      GGcout("GGEMSManager", "Run", 1) << "      + Generating " << kNumberOfParticles << " particles..." << GGendl;
      source_manager.GetPrimaries(j, kNumberOfParticles);

      // Step 2: Find closest navigator (phantom and detector) before track to in operation
      GGcout("GGEMSManager", "Run", 1) << "      + Finding closest navigator..." << GGendl;
      navigator_manager.FindClosestNavigator();

      // Step 3: Track to in step, particles are projected to navigator
      GGcout("GGEMSManager", "Run", 1) << "      + Moving particles to navigator..." << GGendl;
      navigator_manager.TrackToIn();

      // Step X: Checking if all particles are dead
    }
  }

  // Get the end time
  ChronoTime end_time = GGEMSChrono::Now();

  GGcout("GGEMSManager", "Run", 0) << "GGEMS simulation succeeded" << GGendl;

  // Display the elapsed time in GGEMS
  GGEMSChrono::DisplayTime(end_time - start_time, "GGEMS simulation");
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
