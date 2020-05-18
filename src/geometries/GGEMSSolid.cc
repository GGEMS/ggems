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
#include "GGEMS/io/GGEMSMHDImage.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolid::GGEMSSolid(void)
: solid_data_(nullptr),
  label_data_(nullptr),
  kernel_particle_solid_distance_(nullptr)
{
  GGcout("GGEMSSolid", "GGEMSSolid", 3) << "Allocation of GGEMSSolid..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get the RAM manager
  GGEMSRAMManager& ram_manager = GGEMSRAMManager::GetInstance();

  // Allocation of memory on OpenCL device for header data
  solid_data_ = opencl_manager.Allocate(nullptr, sizeof(GGEMSVoxelizedSolidData), CL_MEM_READ_WRITE);
  ram_manager.AddGeometryRAMMemory(sizeof(GGEMSVoxelizedSolidData));
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

void GGEMSSolid::LoadPhantomImage(std::string const& phantom_filename, std::string const& range_data_filename, std::shared_ptr<GGEMSMaterials> materials)
{
  GGcout("GGEMSSolid", "LoadPhantomImage", 3) << "Loading image phantom from mhd file..." << GGendl;

  // Read MHD input file
  GGEMSMHDImage mhd_input_phantom;
  mhd_input_phantom.Read(phantom_filename, solid_data_);

  // Get the name of raw file from mhd reader
  std::string const kRawFilename = mhd_input_phantom.GetRawMDHfilename();

  // Get the type
  std::string const kDataType = mhd_input_phantom.GetDataMHDType();

  // Convert raw data to material id data
  if (!kDataType.compare("MET_CHAR")) {
    ConvertImageToLabel<char>(kRawFilename, range_data_filename, materials);
  }
  else if (!kDataType.compare("MET_UCHAR")) {
    ConvertImageToLabel<unsigned char>(kRawFilename, range_data_filename, materials);
  }
  else if (!kDataType.compare("MET_SHORT")) {
    ConvertImageToLabel<GGshort>(kRawFilename, range_data_filename, materials);
  }
  else if (!kDataType.compare("MET_USHORT")) {
    ConvertImageToLabel<GGushort>(kRawFilename, range_data_filename, materials);
  }
  else if (!kDataType.compare("MET_INT")) {
    ConvertImageToLabel<GGint>(kRawFilename, range_data_filename, materials);
  }
  else if (!kDataType.compare("MET_UINT")) {
    ConvertImageToLabel<GGuint>(kRawFilename, range_data_filename, materials);
  }
  else if (!kDataType.compare("MET_FLOAT")) {
    ConvertImageToLabel<GGfloat>(kRawFilename, range_data_filename, materials);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::ApplyOffset(GGfloat3 const& offset_xyz)
{
  GGcout("GGEMSSolid", "ApplyOffset", 3) << "Applyng the offset defined by the user..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device
  GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_, sizeof(GGEMSVoxelizedSolidData));

  for (GGuint i = 0; i < 3; ++i ) {
    // Offset
    solid_data_device->offsets_xyz_.s[i] = offset_xyz.s[i];

    // Bounding box
    solid_data_device->border_min_xyz_.s[i] = -solid_data_device->offsets_xyz_.s[i];
    solid_data_device->border_max_xyz_.s[i] = solid_data_device->border_min_xyz_.s[i] + solid_data_device->number_of_voxels_xyz_.s[i] * solid_data_device->voxel_sizes_xyz_.s[i];
  }

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_, solid_data_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::PrintInfos(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device
  GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_, sizeof(GGEMSVoxelizedSolidData));

  GGcout("GGEMSSolid", "PrintInfos", 0) << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "GGEMSSolid Infos: " << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "--------------------------------------------" << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "*Dimension: " << solid_data_device->number_of_voxels_xyz_.s[0] << " " << solid_data_device->number_of_voxels_xyz_.s[1] << " " << solid_data_device->number_of_voxels_xyz_.s[2] << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "*Number of voxels: " << solid_data_device->number_of_voxels_ << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "*Size of voxels: (" << solid_data_device->voxel_sizes_xyz_.s[0] << "x" << solid_data_device->voxel_sizes_xyz_.s[1] << "x" << solid_data_device->voxel_sizes_xyz_.s[2] << ") mm3" << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "*Offset: (" << solid_data_device->offsets_xyz_.s[0] << "x" << solid_data_device->offsets_xyz_.s[1] << "x" << solid_data_device->offsets_xyz_.s[2] << ") mm3" << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "*Bounding box:" << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "    - X: " << solid_data_device->border_min_xyz_.s[0] << " <-> " << solid_data_device->border_max_xyz_.s[0] << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "    - Y: " << solid_data_device->border_min_xyz_.s[1] << " <-> " << solid_data_device->border_max_xyz_.s[1] << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "    - Z: " << solid_data_device->border_min_xyz_.s[2] << " <-> " << solid_data_device->border_max_xyz_.s[2] << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << GGendl;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_, solid_data_device);
}
