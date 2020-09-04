/*!
  \file GGEMSSolid.cc

  \brief GGEMS class for solid. This class store geometry about phantom or detector

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 2, 2020
*/

#include "GGEMS/geometries/GGEMSSolid.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolid::GGEMSSolid(void)
: solid_data_cl_(nullptr),
  label_data_cl_(nullptr),
  tracking_kernel_option_("")
{
  GGcout("GGEMSSolid", "GGEMSSolid", 3) << "Allocation of GGEMSSolid..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolid::~GGEMSSolid(void)
{
  GGcout("GGEMSSolid", "~GGEMSSolid", 3) << "Deallocation of GGEMSSolid..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::EnableTracking(void)
{
  tracking_kernel_option_ = "-DGGEMS_TRACKING_VERBOSE";
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::SetGeometryTolerance(GGfloat const& tolerance)
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device
  GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_cl_.get(), sizeof(GGEMSVoxelizedSolidData));

  // Storing the geometry tolerance
  solid_data_device->tolerance_ = tolerance;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_cl_.get(), solid_data_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::SetNavigatorID(std::size_t const& navigator_id)
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device
  GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_cl_.get(), sizeof(GGEMSVoxelizedSolidData));

  // Storing the geometry tolerance
  solid_data_device->navigator_id_ = static_cast<GGuchar>(navigator_id);

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_cl_.get(), solid_data_device);
}
