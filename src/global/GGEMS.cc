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
  \file GGEMS.cc

  \brief GGEMS class managing the GGEMS simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday September 30, 2019
*/

#include <fcntl.h>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <wincrypt.h>
#else
#include <unistd.h>
#endif
#include <mutex>

/*!
  \brief empty namespace storing mutex
*/
namespace {
  std::mutex mutex; /*!< Mutex variable */
}

#include "GGEMS/global/GGEMS.hh"
#include "GGEMS/physics/GGEMSProcessesManager.hh"
#include "GGEMS/physics/GGEMSRangeCutsManager.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/navigators/GGEMSNavigatorManager.hh"
#include "GGEMS/tools/GGEMSRAMManager.hh"
#include "GGEMS/randoms/GGEMSPseudoRandomGenerator.hh"
#include "GGEMS/tools/GGEMSProfilerManager.hh"
#include "GGEMS/tools/GGEMSProgressBar.hh"

#ifdef OPENGL_VISUALIZATION
#include "GGEMS/graphics/GGEMSOpenGLManager.hh"
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMS::GGEMS(void)
: is_opencl_verbose_(false),
  is_material_database_verbose_(false),
  is_source_verbose_(false),
  is_navigator_verbose_(false),
  is_memory_ram_verbose_(false),
  is_process_verbose_(false),
  is_range_cuts_verbose_(false),
  is_random_verbose_(false),
  is_tracking_verbose_(false),
  is_profiling_verbose_(false),
  particle_tracking_id_(0)
{
  GGcout("GGEMS", "GGEMS", 3) << "GGEMS creating..." << GGendl;

  GGcout("GGEMS", "GGEMS", 3) << "GGEMS created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMS::~GGEMS(void)
{
  GGcout("GGEMS", "~GGEMS", 3) << "GGEMS erasing..." << GGendl;

  GGcout("GGEMS", "~GGEMS", 3) << "GGEMS erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMS::SetOpenCLVerbose(bool const& is_opencl_verbose)
{
  is_opencl_verbose_ = is_opencl_verbose;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMS::SetMaterialDatabaseVerbose(bool const& is_material_database_verbose)
{
  is_material_database_verbose_ = is_material_database_verbose;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMS::SetSourceVerbose(bool const& is_source_verbose)
{
  is_source_verbose_ = is_source_verbose;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMS::SetNavigatorVerbose(bool const& is_navigator_verbose)
{
  is_navigator_verbose_ = is_navigator_verbose;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMS::SetMemoryRAMVerbose(bool const& is_memory_ram_verbose)
{
  is_memory_ram_verbose_ = is_memory_ram_verbose;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMS::SetProcessVerbose(bool const& is_process_verbose)
{
  is_process_verbose_ = is_process_verbose;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMS::SetProfilingVerbose(bool const& is_profiling_verbose)
{
  is_profiling_verbose_ = is_profiling_verbose;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMS::SetRangeCutsVerbose(bool const& is_range_cuts_verbose)
{
  is_range_cuts_verbose_ = is_range_cuts_verbose;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMS::SetRandomVerbose(bool const& is_random_verbose)
{
  is_random_verbose_ = is_random_verbose;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMS::SetTrackingVerbose(bool const& is_tracking_verbose, GGint const& particle_tracking_id)
{
  is_tracking_verbose_ = is_tracking_verbose;
  particle_tracking_id_ = particle_tracking_id;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMS::Initialize(GGuint const& seed)
{
  GGcout("GGEMS", "Initialize", 1) << "Initialization of GGEMS Manager singleton..." << GGendl;

  // Getting the GGEMS singletons
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGEMSMaterialsDatabaseManager& material_database_manager = GGEMSMaterialsDatabaseManager::GetInstance();
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  GGEMSNavigatorManager& navigator_manager = GGEMSNavigatorManager::GetInstance();
  GGEMSProcessesManager& processes_manager = GGEMSProcessesManager::GetInstance();
  GGEMSRangeCutsManager& range_cuts_manager = GGEMSRangeCutsManager::GetInstance();
  GGEMSRAMManager& ram_manager = GGEMSRAMManager::GetInstance();

  // Get the start time
  ChronoTime start_time = GGEMSChrono::Now();

  // Printing the banner with the GGEMS version
  PrintBanner();

  // Checking if material manager is ready
  if (!material_database_manager.IsReady()) GGEMSMisc::ThrowException("GGEMS", "Initialize", "Materials are not loaded in GGEMS!!!");

  // Initialization of the source
  source_manager.Initialize(seed, is_tracking_verbose_, particle_tracking_id_);

  // Initialization of the navigators (phantom + system)
  navigator_manager.Initialize(is_tracking_verbose_);

  // Printing infos about OpenCL
  if (is_opencl_verbose_) {
    opencl_manager.PrintPlatformInfos();
    opencl_manager.PrintDeviceInfos();
    opencl_manager.PrintActivatedDevices();
    opencl_manager.PrintBuildOptions();
  }

  // Printing infos about material database
  if (is_material_database_verbose_) material_database_manager.PrintAvailableMaterials();

  // Printing infos about source(s)
  if (is_source_verbose_) source_manager.PrintInfos();

  // Printing infos about navigator(s)
  if (is_navigator_verbose_) navigator_manager.PrintInfos();

  // Printing infos about processe(s)
  if (is_process_verbose_) {
    processes_manager.PrintAvailableProcesses();
    processes_manager.PrintInfos();
  }

  // Printing infos about range cuts
  if (is_range_cuts_verbose_) range_cuts_manager.PrintInfos();

  // Printing infos about random in GGEMS
  if (is_random_verbose_) source_manager.GetPseudoRandomGenerator()->PrintInfos();

  // Printing infos about RAM
  if (is_memory_ram_verbose_) ram_manager.PrintRAMStatus();

  // Get the end time
  ChronoTime end_time = GGEMSChrono::Now();

  GGcout("GGEMS", "Initialize", 0) << "GGEMS initialization succeeded" << GGendl;

  // Initializing OpenGL and create window
  // #ifdef OPENGL_VISUALIZATION
  // if (is_visugl_) {
  //   GGEMSOpenGLManager& opengl_manager = GGEMSOpenGLManager::GetInstance();
  //   //opengl_manager.Initialize();
  //   opengl_manager.Display();
  // }
  // #endif

  // Display the elapsed time in GGEMS
  GGEMSChrono::DisplayTime(end_time - start_time, "GGEMS initialization");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMS::RunOnDevice(GGsize const& thread_index)
{
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  GGEMSNavigatorManager& navigator_manager = GGEMSNavigatorManager::GetInstance();

  // Printing progress bar
  mutex.lock();
  static GGEMSProgressBar progress_bar(source_manager.GetTotalNumberOfBatchs());
  mutex.unlock();

  // Loop over sources
  for (GGsize i = 0; i < source_manager.GetNumberOfSources(); ++i) {
    // Number of batch for a source
    GGsize number_of_batchs = source_manager.GetNumberOfBatchs(i, thread_index);

    // Loop over batch
    for (GGsize j = 0; j < number_of_batchs; ++j) {
      GGsize number_of_particles = source_manager.GetNumberOfParticlesInBatch(i, thread_index, j);

      // Generating particles
      source_manager.GetPrimaries(i, thread_index, number_of_particles);

      // Loop until ALL particles are dead
      GGint loop_counter = 0, max_loop = 100; // Prevent infinite loop
      do {
        // Step 2: Find closest navigator (phantom, detector) before projection and track operation
        navigator_manager.FindSolid(thread_index);

        // Optional step: World tracking
        navigator_manager.WorldTracking(thread_index);

        // Step 3: Project particles to solid
        navigator_manager.ProjectToSolid(thread_index);

        // Step 4: Track through step, particles are tracked in selected solid
        navigator_manager.TrackThroughSolid(thread_index);

        loop_counter++;
      } while (source_manager.IsAlive(thread_index) || loop_counter == max_loop); // Step 5: Checking if all particles are dead, otherwize go back to step 2
    
      // Incrementing progress bar
      mutex.lock();
      ++progress_bar;
      mutex.unlock();
    }
  }

  // Computing dose
  navigator_manager.ComputeDose(thread_index);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMS::Run()
{
  GGcout("GGEMS", "Run", 0) << "GGEMS simulation started" << GGendl;

  // Checking number of source, if 0 stop run
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  if (source_manager.GetNumberOfSources() == 0) {
    GGwarn("GGEMS", "Run", 0) << "No source defined. Run can not be executed!!!" << GGendl;
    return;
  }

  ChronoTime start_time = GGEMSChrono::Now();

  // Creating a thread for each OpenCL device
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGsize number_of_activated_devices = opencl_manager.GetNumberOfActivatedDevice();
  std::thread* thread_device = new std::thread[number_of_activated_devices];

  for (GGsize i = 0; i < number_of_activated_devices; ++i) {
    thread_device[i] = std::thread(&GGEMS::RunOnDevice, this, i);
  }

  for (GGsize i = 0; i < number_of_activated_devices; ++i) thread_device[i].join();

  // Deleting threads
  delete[] thread_device;

  // End of simulation, storing output
  GGcout("GGEMS", "Run", 1) << "Saving results..." << GGendl;
  GGEMSNavigatorManager& navigator_manager = GGEMSNavigatorManager::GetInstance();
  navigator_manager.SaveResults();

  // Printing elapsed time in kernels
  if (is_profiling_verbose_) {
    GGEMSProfilerManager& profiler_manager = GGEMSProfilerManager::GetInstance();
    profiler_manager.PrintSummaryProfile();
  }

  ChronoTime end_time = GGEMSChrono::Now();

  GGcout("GGEMS", "Run", 0) << "GGEMS simulation succeeded" << GGendl;

  GGEMSChrono::DisplayTime(end_time - start_time, "GGEMS simulation");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMS::PrintBanner(void) const
{
  std::cout << std::endl;
  std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
  std::cout << "$  ___   ___   ___  __ __  ___          _  $" << std::endl;
  std::cout << "$ /  _> /  _> | __>|  \\  \\/ __>    _ _ / | $" << std::endl;
  std::cout << "$ | <_/\\| <_/\\| _> |     |\\__ \\   | | || | $" << std::endl;
  std::cout << "$ `____/`____/|___>|_|_|_|<___/   |__/ |_| $" << std::endl;
  std::cout << "$                                          $" << std::endl;
  std::cout << "$ Welcome to GGEMS v1.2   https://ggems.fr $" << std::endl;
  std::cout << "$ Copyright (c) GGEMS Team 2021            $" << std::endl;
  std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
  std::cout << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMS* create_ggems(void)
{
  return new(std::nothrow) GGEMS;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void delete_ggems(GGEMS* ggems)
{
  if (ggems) {
    delete ggems;
    ggems = nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void initialize_ggems(GGEMS* ggems, GGuint const seed)
{
  ggems->Initialize(seed);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_opencl_verbose_ggems(GGEMS* ggems, bool const is_opencl_verbose)
{
  ggems->SetOpenCLVerbose(is_opencl_verbose);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_material_database_verbose_ggems(GGEMS* ggems, bool const is_material_database_verbose)
{
  ggems->SetMaterialDatabaseVerbose(is_material_database_verbose);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_source_ggems(GGEMS* ggems, bool const is_source_verbose)
{
  ggems->SetSourceVerbose(is_source_verbose);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_navigator_ggems(GGEMS* ggems, bool const is_navigator_verbose)
{
  ggems->SetNavigatorVerbose(is_navigator_verbose);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_memory_ram_ggems(GGEMS* ggems, bool const is_memory_ram_verbose)
{
  ggems->SetMemoryRAMVerbose(is_memory_ram_verbose);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_process_ggems(GGEMS* ggems, bool const is_process_verbose)
{
  ggems->SetProcessVerbose(is_process_verbose);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_range_cuts_ggems(GGEMS* ggems, bool const is_range_cuts_verbose)
{
  ggems->SetRangeCutsVerbose(is_range_cuts_verbose);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_random_ggems(GGEMS* ggems, bool const is_random_verbose)
{
  ggems->SetRandomVerbose(is_random_verbose);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_profiling_ggems(GGEMS* ggems, bool const is_profiling_verbose)
{
  ggems->SetProfilingVerbose(is_profiling_verbose);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_tracking_ggems(GGEMS* ggems, bool const is_tracking_verbose, GGint const particle_id_tracking)
{
  ggems->SetTrackingVerbose(is_tracking_verbose, particle_id_tracking);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void run_ggems(GGEMS* ggems)
{
  ggems->Run();
}
