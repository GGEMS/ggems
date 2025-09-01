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
  \file GGEMSVoxelizedSource.cc

  \brief This class defines a voxelized source in GGEMS useful for SPECT/PET simulations

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday September 15, 2025
*/

#include "GGEMS/sources/GGEMSVoxelizedSource.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/maths/GGEMSGeometryTransformation.hh"
#include "GGEMS/global/GGEMSConstants.hh"
#include "GGEMS/tools/GGEMSRAMManager.hh"
#include "GGEMS/randoms/GGEMSPseudoRandomGenerator.hh"
#include "GGEMS/tools/GGEMSProfilerManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedSource::GGEMSVoxelizedSource(std::string const& source_name)
: GGEMSSource(source_name),
  is_monoenergy_mode_(false),
  monoenergy_(-1.0f),
  energy_spectrum_filename_(""),
  number_of_energy_bins_(0),
  energy_spectrum_(nullptr),
  energy_cdf_(nullptr)
{
  GGcout("GGEMSVoxelizedSource", "GGEMSVoxelizedSource", 3) << "GGEMSVoxelizedSource creating..." << GGendl;

  // Initialization of local axis for X-ray source
  geometry_transformation_->SetAxisTransformation(
    {
      {0.0f, 0.0f, -1.0f},
      {0.0f, 1.0f,  0.0f},
      {1.0f, 0.0f,  0.0f}
    }
  );

  // Allocating memory for cdf and energy spectrum
  energy_spectrum_ = new cl::Buffer*[number_activated_devices_];
  energy_cdf_ = new cl::Buffer*[number_activated_devices_];

  GGcout("GGEMSVoxelizedSource", "GGEMSVoxelizedSource", 3) << "GGEMSVoxelizedSource created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedSource::~GGEMSVoxelizedSource(void)
{
  GGcout("GGEMSVoxelizedSource", "~GGEMSVoxelizedSource", 3) << "GGEMSVoxelizedSource erasing..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  if (energy_spectrum_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      if (is_monoenergy_mode_) {
        opencl_manager.Deallocate(energy_spectrum_[i], 2*sizeof(GGfloat), i);
      }
      else {
        opencl_manager.Deallocate(energy_spectrum_[i], number_of_energy_bins_*sizeof(GGfloat), i);
      }
    }
    delete[] energy_spectrum_;
    energy_spectrum_ = nullptr;
  }

  if (energy_cdf_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      if (is_monoenergy_mode_) {
        opencl_manager.Deallocate(energy_cdf_[i], 2*sizeof(GGfloat), i);
      }
      else {
        opencl_manager.Deallocate(energy_cdf_[i], number_of_energy_bins_*sizeof(GGfloat), i);
      }
    }
    delete[] energy_cdf_;
    energy_cdf_ = nullptr;
  }

  GGcout("GGEMSVoxelizedSource", "~GGEMSVoxelizedSource", 3) << "GGEMSVoxelizedSource erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedSource::CheckParameters(void) const
{
  GGcout("GGEMSVoxelizedSource", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  // Checking the energy
  if (is_monoenergy_mode_) {
    if (monoenergy_ == -1.0f) {
      std::ostringstream oss(std::ostringstream::out);
      oss << "You have to set an energy in monoenergetic mode!!!";
      GGEMSMisc::ThrowException("GGEMSXRaySource", "CheckParameters", oss.str());
    }

    if (monoenergy_ < 0.0f) {
      std::ostringstream oss(std::ostringstream::out);
      oss << "The energy must be a positive value!!!";
      GGEMSMisc::ThrowException("GGEMSXRaySource", "CheckParameters", oss.str());
    }
  }

  if (!is_monoenergy_mode_) {
    if (energy_spectrum_filename_.empty()) {
      std::ostringstream oss(std::ostringstream::out);
      oss << "You have to provide a energy spectrum file in polyenergy mode!!!";
      GGEMSMisc::ThrowException("GGEMSXRaySource", "CheckParameters", oss.str());
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedSource::Initialize(bool const& is_tracking)
{
  GGcout("GGEMSVoxelizedSource", "Initialize", 3) << "Initializing the GGEMS X-Ray source..." << GGendl;

  // Initialize GGEMS source
  GGEMSSource::Initialize(is_tracking);

  // Check the mandatory parameters
  CheckParameters();

  // Initializing the kernel for OpenCL
  //InitializeKernel();

  // Filling the energy
  //FillEnergy();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedSource* create_ggems_voxelized_source(char const* source_name)
{
  return new(std::nothrow) GGEMSVoxelizedSource(source_name);
}
