/*!
  \file GGEMSNavigatorManager.cc

  \brief GGEMS class handling the navigators (detector + phantom) in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday February 11, 2020
*/

#include "GGEMS/physics/GGEMSRangeCutsManager.hh"
#include "GGEMS/geometries/GGEMSSolid.hh"
#include "GGEMS/physics/GGEMSParticles.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSNavigatorManager::GGEMSNavigatorManager(void)
: navigators_(0)
{
  GGcout("GGEMSNavigatorManager", "GGEMSNavigatorManager", 3) << "Allocation of GGEMS navigator manager..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSNavigatorManager::~GGEMSNavigatorManager(void)
{
  // Freeing memory
  navigators_.clear();

  GGcout("GGEMSNavigatorManager", "~GGEMSNavigatorManager", 3) << "Deallocation of GGEMS navigator manager..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::Store(GGEMSNavigator* navigator)
{
  GGcout("GGEMSNavigatorManager", "Store", 3) << "Storing new navigator in GGEMS..." << GGendl;

  // Set index of navigator and store the pointer
  navigator->SetNavigatorID(navigators_.size());
  navigators_.emplace_back(navigator);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::Initialize(void) const
{
  GGcout("GGEMSNavigatorManager", "Initialize", 3) << "Initializing the GGEMS navigator(s)..." << GGendl;

  // Initialization of phantoms
  for (auto&& i : navigators_) i->Initialize();

  // Checking overlap between phantoms
  for (std::size_t i = 0; i < navigators_.size(); ++i) {
    for (std::size_t j = i + 1; j < navigators_.size(); ++j) {
      if (CheckOverlap(navigators_[i], navigators_[j])) {
        std::ostringstream oss(std::ostringstream::out);
        oss << "There is an overlap between the navigator '" << navigators_[i]->GetNavigatorName() << "' and '" << navigators_[j]->GetNavigatorName() << "'!!! Please check your simulation parameters about navigator.";
        GGEMSMisc::ThrowException("GGEMSNavigatorManager", "Initialize", oss.str());
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

bool GGEMSNavigatorManager::CheckOverlap(std::shared_ptr<GGEMSNavigator> navigator_a, std::shared_ptr<GGEMSNavigator> navigator_b) const
{
  // Get OpenCL singleton
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get OpenCL buffer on navigator A and B
  std::shared_ptr<cl::Buffer> solid_phantom_data_a = navigator_a->GetSolid()->GetSolidData();
  std::shared_ptr<cl::Buffer> solid_phantom_data_b = navigator_b->GetSolid()->GetSolidData();

  // Get data on OpenCL device for navigator A and B
  GGEMSVoxelizedSolidData* header_data_a_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_phantom_data_a, sizeof(GGEMSVoxelizedSolidData));
  GGEMSVoxelizedSolidData* header_data_b_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_phantom_data_b, sizeof(GGEMSVoxelizedSolidData));

  // Variable checking overlap
  bool is_overlap(false);

  // Get bounding boxes for A and B
  GGdouble const x_min_a = header_data_a_device->border_min_xyz_.s[0];
  GGdouble const x_max_a = header_data_a_device->border_max_xyz_.s[0];
  GGdouble const x_min_b = header_data_b_device->border_min_xyz_.s[0];
  GGdouble const x_max_b = header_data_b_device->border_max_xyz_.s[0];

  GGdouble const y_min_a = header_data_a_device->border_min_xyz_.s[1];
  GGdouble const y_max_a = header_data_a_device->border_max_xyz_.s[1];
  GGdouble const y_min_b = header_data_b_device->border_min_xyz_.s[1];
  GGdouble const y_max_b = header_data_b_device->border_max_xyz_.s[1];

  GGdouble const z_min_a = header_data_a_device->border_min_xyz_.s[2];
  GGdouble const z_max_a = header_data_a_device->border_max_xyz_.s[2];
  GGdouble const z_min_b = header_data_b_device->border_min_xyz_.s[2];
  GGdouble const z_max_b = header_data_b_device->border_max_xyz_.s[2];

  if (x_max_a > x_min_b && x_min_a < x_max_b && y_max_a > y_min_b && y_min_a < y_max_b && z_max_a > z_min_b && z_min_a < z_max_b) is_overlap = true;

  // Release the pointers
  opencl_manager.ReleaseDeviceBuffer(solid_phantom_data_a, header_data_a_device);
  opencl_manager.ReleaseDeviceBuffer(solid_phantom_data_b, header_data_b_device);

  return is_overlap;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::PrintInfos(void) const
{
  GGcout("GGEMSNavigatorManager", "PrintInfos", 0) << "Printing infos about phantom navigators" << GGendl;
  GGcout("GGEMSNavigatorManager", "PrintInfos", 0) << "Number of navigator(s): " << navigators_.size() << GGendl;

  // Printing infos about each navigator
  for (auto&&i : navigators_) i->PrintInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::FindClosestNavigator(void) const
{
  // Loop over all declared navigators and compute distance particle / navigator
  for (auto&& i : navigators_) i->ComputeParticleNavigatorDistance();
}
