/*!
  \file GGEMSSourceManager.hh

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
#include "GGEMS/sources/GGEMSSource.hh"
#include "GGEMS/sources/GGEMSXRaySource.hh"
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/physics/GGEMSParticles.hh"
#include "GGEMS/randoms/GGEMSPseudoRandomGenerator.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSourceManager::GGEMSSourceManager(void)
: p_sources_(nullptr),
  number_of_sources_(0),
  p_particles_(nullptr),
  p_pseudo_random_generator_(nullptr)
{
  GGcout("GGEMSSourceManager", "GGEMSSourceManager", 3)
    << "Allocation of GGEMSSourceManager..." << GGendl;

  // Allocation of Particle object
  p_particles_ = new GGEMSParticles();

  // Allocation of pseudo random generator object
  p_pseudo_random_generator_ = new GGEMSPseudoRandomGenerator();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSourceManager::~GGEMSSourceManager(void)
{
  // Deleting source
  if (p_sources_) {
    for (GGuint i = 0; i < number_of_sources_; ++i) {
      delete p_sources_[i];
      p_sources_[i] = nullptr;
    }
    delete p_sources_;
    p_sources_ = nullptr;
  }

  // Freeing memory
  if (p_particles_) {
    delete p_particles_;
    p_particles_ = nullptr;
  }

  if (p_pseudo_random_generator_) {
    delete p_pseudo_random_generator_;
    p_pseudo_random_generator_ = nullptr;
  }

  GGcout("GGEMSSourceManager", "~GGEMSSourceManager", 3)
    << "Deallocation of GGEMSSourceManager..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::Store(GGEMSSource* p_source)
{
  GGcout("GGEMSSourceManager", "Store", 3)
    << "Storing new source in GGEMS source manager..." << GGendl;

  // Compute new size of buffer
  number_of_sources_ += 1;

  // If number of source == 1, not need to resize
  if (number_of_sources_ == 1) {
    // Allocating 1 source
    p_sources_ = new GGEMSSource*[1];

    // Storing the source
    p_sources_[0] = p_source;
  }
  else {
    // Creating new buffer
    GGEMSSource** p_new_buffer = new GGEMSSource*[number_of_sources_];

    // Saving old pointer in new buffer and freeing old buffer
    for (GGuint i = 0; i < number_of_sources_ - 1; ++i) {
      p_new_buffer[i] = p_sources_[i];
    }
    delete p_sources_;
    p_sources_ = nullptr;

    // Give new buffer
    p_new_buffer[number_of_sources_ - 1] = p_source;
    p_sources_ = p_new_buffer;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::PrintInfos(void) const
{
  GGcout("GGEMSSourceManager", "PrintInfos", 0)
    << "Printing infos about sources" << GGendl;
  GGcout("GGEMSSourceManager", "PrintInfos", 0)
    << "Number of source(s): " << number_of_sources_ << GGendl;

  // Printing infos about each navigator
  for (GGuint i = 0; i < number_of_sources_; ++i) {
    p_sources_[i]->PrintInfos();
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::Initialize(void) const
{
  GGcout("GGEMSSourceManager", "Initialize", 3)
    << "Initializing the GGEMS source..." << GGendl;

  // Initialization of particle stack and random stack
  p_particles_->Initialize();
  GGcout("GGEMSSourceManager", "Initialize", 0)
    << "Initialization of particles OK" << GGendl;

  p_pseudo_random_generator_->Initialize();
  GGcout("GGEMSSourceManager", "Initialize", 0)
    << "Initialization of GGEMS pseudo random generator OK" << GGendl;

  // Initialization of sources
  for (GGuint i = 0; i < number_of_sources_; ++i) {
    p_sources_[i]->Initialize();
  }
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

void print_infos_ggems_source_manager(GGEMSSourceManager* p_source_manager)
{
  p_source_manager->PrintInfos();
}
