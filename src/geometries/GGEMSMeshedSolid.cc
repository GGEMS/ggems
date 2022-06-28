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
