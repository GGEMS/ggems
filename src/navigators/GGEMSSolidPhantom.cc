
/*!
  \file GGEMSSolidPhantom.cc

  \brief GGEMS class for solid phantom informations

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 2, 2020
*/

#include "GGEMS/navigators/GGEMSSolidPhantomStack.hh"
#include "GGEMS/navigators/GGEMSSolidPhantom.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/io/GGEMSMHDImage.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolidPhantom::GGEMSSolidPhantom()
: opencl_manager_(GGEMSOpenCLManager::GetInstance())
{
  GGcout("GGEMSSolidPhantom", "GGEMSSolidPhantom", 3) << "Allocation of GGEMSSolidPhantom..." << GGendl;

  // Allocation of memory on OpenCL device for header data
  solid_phantom_data_ = opencl_manager_.Allocate(nullptr, sizeof(GGEMSSolidPhantomData), CL_MEM_READ_WRITE);
  opencl_manager_.AddRAMMemory(sizeof(GGEMSSolidPhantomData));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolidPhantom::~GGEMSSolidPhantom(void)
{
  GGcout("GGEMSSolidPhantom", "~GGEMSSolidPhantom", 3) << "Deallocation of GGEMSSolidPhantom..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolidPhantom::LoadPhantomImage(std::string const& phantom_filename, std::string const& range_data_filename, std::shared_ptr<GGEMSMaterials> materials)
{
  GGcout("GGEMSSolidPhantom", "LoadPhantomImage", 3) << "Loading image phantom from mhd file..." << GGendl;

  // Read MHD input file
  GGEMSMHDImage mhd_input_phantom;
  mhd_input_phantom.Read(phantom_filename, solid_phantom_data_);

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

void GGEMSSolidPhantom::ApplyOffset(GGfloat3 const& offset_xyz)
{
  GGcout("GGEMSSolidPhantom", "ApplyOffset", 3) << "Applyng the offset defined by the user..." << GGendl;

  // Get pointer on OpenCL device
  GGEMSSolidPhantomData* solid_data = opencl_manager_.GetDeviceBuffer<GGEMSSolidPhantomData>(solid_phantom_data_, sizeof(GGEMSSolidPhantomData));

  for (GGuint i = 0; i < 3; ++i ) {
    // Offset
    solid_data->offsets_xyz_.s[i] = offset_xyz.s[i];

    // Bounding box
    solid_data->border_min_xyz_.s[i] = -solid_data->offsets_xyz_.s[i];
    solid_data->border_max_xyz_.s[i] = solid_data->border_min_xyz_.s[i] + solid_data->number_of_voxels_xyz_.s[i] * solid_data->voxel_sizes_xyz_.s[i];
  }

  // Release the pointer
  opencl_manager_.ReleaseDeviceBuffer(solid_phantom_data_, solid_data);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolidPhantom::PrintInfos(void) const
{
  // Get pointer on OpenCL device
  GGEMSSolidPhantomData* solid_data = opencl_manager_.GetDeviceBuffer<GGEMSSolidPhantomData>(solid_phantom_data_, sizeof(GGEMSSolidPhantomData));

  GGcout("GGEMSSolidPhantom", "PrintInfos", 0) << GGendl;
  GGcout("GGEMSSolidPhantom", "PrintInfos", 0) << "GGEMSSolidPhantom Infos: " << GGendl;
  GGcout("GGEMSSolidPhantom", "PrintInfos", 0) << "--------------------------------------------" << GGendl;
  GGcout("GGEMSSolidPhantom", "PrintInfos", 0) << "*Dimension: " << solid_data->number_of_voxels_xyz_.s[0] << " " << solid_data->number_of_voxels_xyz_.s[1] << " " << solid_data->number_of_voxels_xyz_.s[2] << GGendl;
  GGcout("GGEMSSolidPhantom", "PrintInfos", 0) << "*Number of voxels: " << solid_data->number_of_voxels_ << GGendl;
  GGcout("GGEMSSolidPhantom", "PrintInfos", 0) << "*Size of voxels: (" << solid_data->voxel_sizes_xyz_.s[0] << "x" << solid_data->voxel_sizes_xyz_.s[1] << "x" << solid_data->voxel_sizes_xyz_.s[2] << ") mm3" << GGendl;
  GGcout("GGEMSSolidPhantom", "PrintInfos", 0) << "*Offset: (" << solid_data->offsets_xyz_.s[0] << "x" << solid_data->offsets_xyz_.s[1] << "x" << solid_data->offsets_xyz_.s[2] << ") mm3" << GGendl;
  GGcout("GGEMSSolidPhantom", "PrintInfos", 0) << "*Bounding box:" << GGendl;
  GGcout("GGEMSSolidPhantom", "PrintInfos", 0) << "    - X: " << solid_data->border_min_xyz_.s[0] << " <-> " << solid_data->border_max_xyz_.s[0] << GGendl;
  GGcout("GGEMSSolidPhantom", "PrintInfos", 0) << "    - Y: " << solid_data->border_min_xyz_.s[1] << " <-> " << solid_data->border_max_xyz_.s[1] << GGendl;
  GGcout("GGEMSSolidPhantom", "PrintInfos", 0) << "    - Z: " << solid_data->border_min_xyz_.s[2] << " <-> " << solid_data->border_max_xyz_.s[2] << GGendl;
  GGcout("GGEMSSolidPhantom", "PrintInfos", 0) << GGendl;

  // Release the pointer
  opencl_manager_.ReleaseDeviceBuffer(solid_phantom_data_, solid_data);
}
