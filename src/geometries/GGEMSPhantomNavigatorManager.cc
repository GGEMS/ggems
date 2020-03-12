/*!
  \file GGEMSPhantomNavigatorManager.cc

  \brief GGEMS class handling the phantom navigators in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday February 11, 2020
*/

#include <sstream>

#include "GGEMS/physics/GGEMSRangeCutsManager.hh"
#include "GGEMS/geometries/GGEMSPhantomNavigator.hh"
#include "GGEMS/geometries/GGEMSSolidPhantom.hh"
#include "GGEMS/geometries/GGEMSSolidPhantomStack.hh"
#include "GGEMS/geometries/GGEMSPhantomNavigatorManager.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/tools/GGEMSTools.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhantomNavigatorManager::GGEMSPhantomNavigatorManager(void)
: phantom_navigators_(0),
  opencl_manager_(GGEMSOpenCLManager::GetInstance())
{
  GGcout("GGEMSPhantomNavigatorManager", "GGEMSPhantomNavigatorManager", 3) << "Allocation of GGEMS phantom navigator manager..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhantomNavigatorManager::~GGEMSPhantomNavigatorManager(void)
{
  // Freeing memory
  phantom_navigators_.clear();

  GGcout("GGEMSPhantomNavigatorManager", "~GGEMSPhantomNavigatorManager", 3) << "Deallocation of GGEMS phantom navigator manager..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantomNavigatorManager::Store(GGEMSPhantomNavigator* phantom_navigator)
{
  GGcout("GGEMSPhantomNavigatorManager", "Store", 3) << "Storing new phantom navigator in GGEMS..." << GGendl;
  phantom_navigators_.emplace_back(phantom_navigator);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantomNavigatorManager::Initialize(void) const
{
  GGcout("GGEMSPhantomNavigatorManager", "Initialize", 3) << "Initializing the GGEMS phantom(s)..." << GGendl;

  // Checking cuts
  GGEMSRangeCutsManager::GetInstance().CheckRangeCuts();

  // Initialization of phantoms
  for (auto&& i : phantom_navigators_) i->Initialize();

  // Checking overlap between phantoms
  for (std::size_t i = 0; i < phantom_navigators_.size(); ++i) {
    for (std::size_t j = i + 1; j < phantom_navigators_.size(); ++j) {
      if (CheckOverlap(phantom_navigators_[i], phantom_navigators_[j])) {
        std::ostringstream oss(std::ostringstream::out);
        oss << "There is an overlap between the phantom '" << phantom_navigators_[i]->GetPhantomName() << "' and '" << phantom_navigators_[j]->GetPhantomName() << "'!!! Please check your simulation parameters about phantom.";
        GGEMSMisc::ThrowException("GGEMSPhantomNavigatorManager", "Initialize", oss.str());
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

bool GGEMSPhantomNavigatorManager::CheckOverlap(std::shared_ptr<GGEMSPhantomNavigator> phantom_a, std::shared_ptr<GGEMSPhantomNavigator> phantom_b) const
{

  // Get OpenCL buffer on phantom A and B
  std::shared_ptr<cl::Buffer> solid_phantom_data_a = phantom_a->GetSolidPhantom()->GetSolidPhantomData();
  std::shared_ptr<cl::Buffer> solid_phantom_data_b = phantom_b->GetSolidPhantom()->GetSolidPhantomData();

  // Get data on OpenCL device for phantom A and B
  GGEMSSolidPhantomData* header_data_a = opencl_manager_.GetDeviceBuffer<GGEMSSolidPhantomData>(solid_phantom_data_a, sizeof(GGEMSSolidPhantomData));
  GGEMSSolidPhantomData* header_data_b = opencl_manager_.GetDeviceBuffer<GGEMSSolidPhantomData>(solid_phantom_data_b, sizeof(GGEMSSolidPhantomData));

  // Variable checking overlap
  bool is_overlap(false);

  // Get bounding boxes for A and B
  GGdouble const x_min_a = header_data_a->border_min_xyz_.s[0];
  GGdouble const x_max_a = header_data_a->border_max_xyz_.s[0];
  GGdouble const x_min_b = header_data_b->border_min_xyz_.s[0];
  GGdouble const x_max_b = header_data_b->border_max_xyz_.s[0];

  GGdouble const y_min_a = header_data_a->border_min_xyz_.s[1];
  GGdouble const y_max_a = header_data_a->border_max_xyz_.s[1];
  GGdouble const y_min_b = header_data_b->border_min_xyz_.s[1];
  GGdouble const y_max_b = header_data_b->border_max_xyz_.s[1];

  GGdouble const z_min_a = header_data_a->border_min_xyz_.s[2];
  GGdouble const z_max_a = header_data_a->border_max_xyz_.s[2];
  GGdouble const z_min_b = header_data_b->border_min_xyz_.s[2];
  GGdouble const z_max_b = header_data_b->border_max_xyz_.s[2];

  if (x_max_a > x_min_b && x_min_a < x_max_b && y_max_a > y_min_b && y_min_a < y_max_b && z_max_a > z_min_b && z_min_a < z_max_b) is_overlap = true;

  // Release the pointers
  opencl_manager_.ReleaseDeviceBuffer(solid_phantom_data_a, header_data_a);
  opencl_manager_.ReleaseDeviceBuffer(solid_phantom_data_b, header_data_b);

  return is_overlap;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantomNavigatorManager::PrintInfos(void) const
{
  GGcout("GGEMSPhantomNavigatorManager", "PrintInfos", 0) << "Printing infos about phantom navigators" << GGendl;
  GGcout("GGEMSPhantomNavigatorManager", "PrintInfos", 0) << "Number of phantom navigator(s): " << phantom_navigators_.size() << GGendl;

  // Printing infos about each navigator
  for (auto&&i : phantom_navigators_) i->PrintInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhantomNavigatorManager* get_instance_ggems_phantom_navigator_manager(void)
{
  return &GGEMSPhantomNavigatorManager::GetInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_infos_ggems_phantom_navigator_manager(GGEMSPhantomNavigatorManager* phantom_navigator_manager)
{
  phantom_navigator_manager->PrintInfos();
}
