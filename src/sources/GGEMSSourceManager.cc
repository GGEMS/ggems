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

#include "GGEMS/processes/GGEMSParticles.hh"
//#include "GGEMS/processes/GGEMSPrimaryParticlesStack.hh"

#include "GGEMS/randoms/GGEMSPseudoRandomGenerator.hh"
//#include "GGEMS/randoms/GGEMSRandomStack.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSourceManager::GGEMSSourceManager(void)
: p_sources_(nullptr),
  number_of_sources_(0),
  p_particle_(nullptr),
  p_pseudo_random_generator_(nullptr)
{
  GGcout("GGEMSSourceManager", "GGEMSSourceManager", 3)
    << "Allocation of GGEMSSourceManager..." << GGendl;

  // Allocation of Particle object
  p_particle_ = new GGEMSParticles();

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
    delete p_sources_;
    p_sources_ = nullptr;
  }

  // Freeing memory
  if (p_particle_) {
    delete p_particle_;
    p_particle_ = nullptr;
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
    p_sources_ = p_source;
  }
  else {
    number_of_sources_ = 1;
    GGEMSMisc::ThrowException("GGEMSSourceManager", "Store",
      "Multiple defined sources in not allowed by GGEMS!!!");
  }
}
