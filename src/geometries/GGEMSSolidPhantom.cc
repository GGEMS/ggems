
/*!
  \file GGEMSSolidPhantom.cc

  \brief GGEMS class for solid phantom informations

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 2, 2020
*/

#include "GGEMS/geometries/GGEMSSolidPhantomStack.hh"
#include "GGEMS/geometries/GGEMSSolidPhantom.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/io/GGEMSMHDImage.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolidPhantom::GGEMSSolidPhantom(void)
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

void GGEMSSolidPhantom::LoadPhantomImage(std::string const& phantom_filename, std::string const& range_data_filename)
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
    ConvertImageToLabel<char>(kRawFilename);
  }
  else if (!kDataType.compare("MET_UCHAR")) {
    ConvertImageToLabel<unsigned char>(kRawFilename);
  }
  else if (!kDataType.compare("MET_SHORT")) {
    ConvertImageToLabel<GGshort>(kRawFilename);
  }
  else if (!kDataType.compare("MET_USHORT")) {
    ConvertImageToLabel<GGushort>(kRawFilename);
  }
  else if (!kDataType.compare("MET_INT")) {
    ConvertImageToLabel<GGint>(kRawFilename);
  }
  else if (!kDataType.compare("MET_UINT")) {
    ConvertImageToLabel<GGuint>(kRawFilename);
  }
  else if (!kDataType.compare("MET_FLOAT")) {
    ConvertImageToLabel<GGfloat>(kRawFilename);
  }

/*  // Read data
    FILE *pfile = fopen(ElementDataFile.c_str(), "rb");

    if(ElementType == "MET_FLOAT") {
      ui32 mem_size = sizeof(f32) * h_volume->number_of_voxels;

      f32 *raw_data = (f32*)malloc(mem_size);
      fread(raw_data, sizeof(f32), h_volume->number_of_voxels, pfile);
      fclose(pfile);
      
      /////////////// Then, convert the raw data into material id //////////////////////
      m_define_materials_from_range(raw_data, range_name);

      // Free memory
      free(raw_data);
    }
    
    if(ElementType == "MET_USHORT") {
      ui32 mem_size = sizeof(ui16) * h_volume->number_of_voxels;

      ui16 *raw_data = (ui16*)malloc(mem_size);
      fread(raw_data, sizeof(ui16), h_volume->number_of_voxels, pfile);
      fclose(pfile);
      /////////////// Then, convert the raw data into material id //////////////////////
      m_define_materials_from_range(raw_data, range_name);

      // Free memory
      free(raw_data);
    }  

    if(ElementType == "MET_SHORT") {
      ui32 mem_size = sizeof(i16) * h_volume->number_of_voxels;

      i16 *raw_data = (i16*)malloc(mem_size);
      fread(raw_data, sizeof(i16), h_volume->number_of_voxels, pfile);
      fclose(pfile);
      /////////////// Then, convert the raw data into material id //////////////////////
      m_define_materials_from_range(raw_data, range_name);

      // Free memory
      free(raw_data);
    }

    if(ElementType == "MET_UCHAR") {
      ui32 mem_size = sizeof(ui8) * h_volume->number_of_voxels;

      ui8 *raw_data = (ui8*)malloc(mem_size);
      fread(raw_data, sizeof(ui8), h_volume->number_of_voxels, pfile);
      fclose(pfile);
      /////////////// Then, convert the raw data into material id //////////////////////
      m_define_materials_from_range(raw_data, range_name);

      // Free memory
      free(raw_data);
    }

    if(ElementType == "MET_UINT") {
      ui32 mem_size = sizeof(ui32) * h_volume->number_of_voxels;

      ui32 *raw_data = (ui32*)malloc(mem_size);
      fread(raw_data, sizeof(ui32), h_volume->number_of_voxels, pfile);
      fclose(pfile);
      /////////////// Then, convert the raw data into material id //////////////////////
      m_define_materials_from_range(raw_data, range_name);

      // Free memory
      free(raw_data);
    }
*/
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolidPhantom::ApplyOffset(GGdouble3 const& offset_xyz)
{
  GGcout("GGEMSSolidPhantom", "ApplyOffset", 3) << "Applyng the offset defined by the user..." << GGendl;

  // Get pointer on OpenCL device
  GGEMSSolidPhantomData* solid_data = opencl_manager_.GetDeviceBuffer<GGEMSSolidPhantomData>(solid_phantom_data_, sizeof(GGEMSSolidPhantomData));

  solid_data->offsets_xyz_.s[0] = offset_xyz.s[0];
  solid_data->offsets_xyz_.s[1] = offset_xyz.s[1];
  solid_data->offsets_xyz_.s[2] = offset_xyz.s[2];

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
