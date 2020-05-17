/*!
  \file GGEMSSourceManager.cc

  \brief GGEMS class handling the source(s)

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thursday January 16, 2020
*/

#include <algorithm>
#include <sstream>

#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/physics/GGEMSParticles.hh"
#include "GGEMS/randoms/GGEMSPseudoRandomGenerator.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSourceManager::GGEMSSourceManager(void)
: sources_(0),
  particles_(nullptr),
  pseudo_random_generator_(nullptr)
{
  GGcout("GGEMSSourceManager", "GGEMSSourceManager", 3) << "Allocation of GGEMSSourceManager..." << GGendl;

  // Allocation of Particle object
  particles_.reset(new GGEMSParticles());

  // Allocation of pseudo random generator object
  pseudo_random_generator_.reset(new GGEMSPseudoRandomGenerator());
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSourceManager::~GGEMSSourceManager(void)
{
  // Deleting source
  sources_.clear();

  GGcout("GGEMSSourceManager", "~GGEMSSourceManager", 3) << "Deallocation of GGEMSSourceManager..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::Store(GGEMSSource* source)
{
  GGcout("GGEMSSourceManager", "Store", 3) << "Storing new source in GGEMS source manager..." << GGendl;
  sources_.emplace_back(source);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::PrintInfos(void) const
{
  GGcout("GGEMSSourceManager", "PrintInfos", 0) << "Printing infos about sources" << GGendl;
  GGcout("GGEMSSourceManager", "PrintInfos", 0) << "Number of source(s): " << sources_.size() << GGendl;

  // Printing infos about each navigator
  for (auto&& i : sources_) i->PrintInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::Initialize(void) const
{
  GGcout("GGEMSSourceManager", "Initialize", 3) << "Initializing the GGEMS source(s)..." << GGendl;

  // Checking number of source, if 0 kill the simulation
  if (sources_.empty()) {
    GGEMSMisc::ThrowException("GGEMSSourceManager", "Initialize", "You have to define a source before to run GGEMS!!!");
  }

  // Initialization of particle stack and random stack
  particles_->Initialize();
  GGcout("GGEMSSourceManager", "Initialize", 0) << "Initialization of particles OK" << GGendl;

  pseudo_random_generator_->Initialize();
  GGcout("GGEMSSourceManager", "Initialize", 0) << "Initialization of GGEMS pseudo random generator OK" << GGendl;

  // Initialization of sources
  for (auto&& i : sources_) i->Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSourceManager* get_instance_ggems_source_manager(void)
{
  return &GGEMSSourceManager::GetInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_infos_ggems_source_manager(GGEMSSourceManager* source_manager)
{
  source_manager->PrintInfos();
}
