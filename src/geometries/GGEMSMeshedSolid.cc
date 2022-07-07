// ************************************************************************
// * This file is part of GGEMS.                                          *
// *                                                                      *
// * GGEMS is free software: you can redistribute it and/or modify        *
// * it under the terms of the GNU General Public License as published by *
// * the Free Software Foundation, either version 3 of the License, or    *
// * (at your option) any later version.                                  *
// *                                                                      *
// * GGEMS is distributed in the hope that it will be useful,             *
// * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
// * GNU General Public License for more details.                         *
// *                                                                      *
// * You should have received a copy of the GNU General Public License    *
// * along with GGEMS.  If not, see <https://www.gnu.org/licenses/>.      *
// *                                                                      *
// ************************************************************************

/*!
  \file GGEMSMeshedSolid.cc

  \brief GGEMS class for meshed solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday June 22, 2022
*/

#include "GGEMS/geometries/GGEMSMeshedSolid.hh"
#include "GGEMS/maths/GGEMSGeometryTransformation.hh"
#include "GGEMS/io/GGEMSSTLReader.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMeshedSolid::GGEMSMeshedSolid(std::string const& meshed_phantom_name, std::string const& data_reg_type)
: GGEMSSolid(),
  meshed_phantom_name_(meshed_phantom_name)
{
  GGcout("GGEMSMeshedSolid", "GGEMSMeshedSolid", 3) << "GGEMSMeshedSolid creating..." << GGendl;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Loop over the device
  for (GGsize d = 0; d < number_activated_devices_; ++d) {
    // Allocating memory on OpenCL device
    solid_data_[d] = opencl_manager.Allocate(nullptr, sizeof(GGEMSMeshedSolidData), d, CL_MEM_READ_WRITE, "GGEMSMeshedSolid");
  }

  // Local axis for phantom. Voxelized solid used only for phantom
  geometry_transformation_->SetAxisTransformation(
    {
      {1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f}
    }
  );

  // Checking format registration
  data_reg_type_ = data_reg_type;
  if (!data_reg_type.empty()) {
    if (data_reg_type == "DOSIMETRY") {
      kernel_option_ += " -DDOSIMETRY";
    }
    else {
      std::ostringstream oss(std::ostringstream::out);
      oss << "False registration type name!!!" << std::endl;
      oss << "Registration type is :" << std::endl;
      oss << "    - DOSIMETRY" << std::endl;
      //oss << "    - LISTMODE" << std::endl;
      //oss << "    - HISTOGRAM" << std::endl;
      GGEMSMisc::ThrowException("GGEMSMeshedSolid", "GGEMSMeshedSolid", oss.str());
    }
  }

  GGcout("GGEMSMeshedSolid", "GGEMSMeshedSolid", 3) << "GGEMSMeshedSolid created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMeshedSolid::~GGEMSMeshedSolid(void)
{
  GGcout("GGEMSMeshedSolid", "GGEMSMeshedSolid", 3) << "GGEMSMeshedSolid erasing..." << GGendl;

  GGcout("GGEMSMeshedSolid", "GGEMSMeshedSolid", 3) << "GGEMSMeshedSolid erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMeshedSolid::Initialize(GGEMSMaterials*)
{

  GGcout("GGEMSMeshedSolid", "Initialize", 3) << "Initializing meshed solid..." << GGendl;

  // Initializing kernels and loading image
 // InitializeKernel();
  LoadVolumeImage();
/*
  // Creating volume for OpenGL
  // Get some infos for grid
  #ifdef OPENGL_VISUALIZATION
  GGEMSOpenGLManager& opengl_manager = GGEMSOpenGLManager::GetInstance();

  if (opengl_manager.IsOpenGLActivated()) {
    GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

    GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_[0], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSVoxelizedSolidData), 0);

    opengl_solid_ = new GGEMSOpenGLParaGrid(
      static_cast<GGsize>(solid_data_device->number_of_voxels_xyz_.s[0]),
      static_cast<GGsize>(solid_data_device->number_of_voxels_xyz_.s[1]),
      static_cast<GGsize>(solid_data_device->number_of_voxels_xyz_.s[2]),
      solid_data_device->voxel_sizes_xyz_.s[0],
      solid_data_device->voxel_sizes_xyz_.s[1],
      solid_data_device->voxel_sizes_xyz_.s[2],
      true // Draw midplanes
    );

    // Release the pointer
    opencl_manager.ReleaseDeviceBuffer(solid_data_[0], solid_data_device, 0);

    // Loading labels and materials for OpenGL
    opengl_solid_->SetMaterial(materials, label_data_[0], number_of_voxels_);
  }
  #endif
*/
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMeshedSolid::LoadVolumeImage(void)
{
  GGcout("GGEMSMeshedSolid", "LoadVolumeImage", 3) << "Loading volume image from stl file..." << GGendl;

  // Read STL input file
  GGEMSSTLReader stl_input_phantom;

 /* GGEMSMHDImage mhd_input_phantom;
  // Loop over the device
  for (GGsize d = 0; d < number_activated_devices_; ++d) {
    mhd_input_phantom.Read(volume_header_filename_, solid_data_[d], d);
  }

  // Get the name of raw file from mhd reader
  std::string output_dir = mhd_input_phantom.GetOutputDirectory();
  std::string raw_filename = output_dir + mhd_input_phantom.GetRawMDHfilename();

  // Get the type
  std::string const kDataType = mhd_input_phantom.GetDataMHDType();

  // Convert raw data to material id data
  if (!kDataType.compare("MET_CHAR")) {
    ConvertImageToLabel<GGchar>(raw_filename, range_filename_, materials);
  }
  else if (!kDataType.compare("MET_UCHAR")) {
    ConvertImageToLabel<GGuchar>(raw_filename, range_filename_, materials);
  }
  else if (!kDataType.compare("MET_SHORT")) {
    ConvertImageToLabel<GGshort>(raw_filename, range_filename_, materials);
  }
  else if (!kDataType.compare("MET_USHORT")) {
    ConvertImageToLabel<GGushort>(raw_filename, range_filename_, materials);
  }
  else if (!kDataType.compare("MET_INT")) {
    ConvertImageToLabel<GGint>(raw_filename, range_filename_, materials);
  }
  else if (!kDataType.compare("MET_UINT")) {
    ConvertImageToLabel<GGuint>(raw_filename, range_filename_, materials);
  }
  else if (!kDataType.compare("MET_FLOAT")) {
    ConvertImageToLabel<GGfloat>(raw_filename, range_filename_, materials);
  }*/
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
