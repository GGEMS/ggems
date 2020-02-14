/*!
  \file GGEMSPhantomNavigatorManager.cc

  \brief GGEMS class handling the phantom navigators in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday February 11, 2020
*/

#include "GGEMS/geometries/GGEMSPhantomNavigator.hh"
#include "GGEMS/geometries/GGEMSPhantomNavigatorManager.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhantomNavigatorManager::GGEMSPhantomNavigatorManager(void)
: p_phantom_navigators_(nullptr),
  number_of_phantom_navigators_(0)
{
  GGcout("GGEMSPhantomNavigatorManager", "GGEMSPhantomNavigatorManager", 3)
    << "Allocation of GGEMS phantom navigator manager..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhantomNavigatorManager::~GGEMSPhantomNavigatorManager(void)
{
  // Freeing memory
  if (p_phantom_navigators_) {
    for (GGuint i = 0; i < number_of_phantom_navigators_; ++i) {
      delete p_phantom_navigators_[i];
      p_phantom_navigators_[i] = nullptr;
    }
    delete p_phantom_navigators_;
    p_phantom_navigators_ = nullptr;
  }

  GGcout("GGEMSPhantomNavigatorManager", "~GGEMSPhantomNavigatorManager", 3)
    << "Deallocation of GGEMS phantom navigator manager..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantomNavigatorManager::Store(
  GGEMSPhantomNavigator* p_phantom_navigator)
{
  GGcout("GGEMSPhantomNavigatorManager", "Store", 3)
    << "Storing new phantom navigator in GGEMS..." << GGendl;

  // Compute new size of buffer
  number_of_phantom_navigators_ += 1;

  // If number of phantom navigators == 1, not need to resize
  if (number_of_phantom_navigators_ == 1) {
    // Allocating 1 phantom navigator
    p_phantom_navigators_ = new GGEMSPhantomNavigator*[1];

    // Store the navigator
    p_phantom_navigators_[0] = p_phantom_navigator;
  }
  else {
    // Creating new buffer
    GGEMSPhantomNavigator** p_new_buffer =
      new GGEMSPhantomNavigator*[number_of_phantom_navigators_];

    // Saving old pointer in new buffer and freeing old buffer
    for (GGuint i = 0; i < number_of_phantom_navigators_ - 1; ++i) {
      p_new_buffer[i] = p_phantom_navigators_[i];
    }
    delete p_phantom_navigators_;
    p_phantom_navigators_ = nullptr;

    // Give new buffer
    p_new_buffer[number_of_phantom_navigators_ - 1] = p_phantom_navigator;
    p_phantom_navigators_ = p_new_buffer;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantomNavigatorManager::PrintInfos(void) const
{
  GGcout("GGEMSPhantomNavigatorManager", "PrintInfos", 0)
    << "Printing infos about phantom navigators" << GGendl;
  GGcout("GGEMSPhantomNavigatorManager", "PrintInfos", 0)
    << "Number of phantom navigator(s): " << number_of_phantom_navigators_
    << GGendl;

  // Printing infos about each navigator
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhantomNavigatorManager*
  get_instance_ggems_phantom_navigator_manager(void)
{
  return &GGEMSPhantomNavigatorManager::GetInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_infos_ggems_phantom_navigator_manager(
  GGEMSPhantomNavigatorManager* p_phantom_navigator_manager)
{
  p_phantom_navigator_manager->PrintInfos();
}
