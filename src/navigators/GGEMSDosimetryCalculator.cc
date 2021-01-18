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
  \file GGEMSDosimetryCalculator.cc

  \brief Class providing tools storing and computing dose in phantom

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Wednesday January 13, 2021
*/

#include <sstream>

#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/navigators/GGEMSDosimetryCalculator.hh"
#include "GGEMS/navigators/GGEMSDoseParams.hh"
#include "GGEMS/navigators/GGEMSNavigatorManager.hh"
#include "GGEMS/navigators/GGEMSNavigator.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolid.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/tools/GGEMSTools.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSDosimetryCalculator::GGEMSDosimetryCalculator(void)
: dosel_sizes_({-1.0f, -1.0f, -1.0f}),
  dosimetry_output_filename("dosi"),
  navigator_(nullptr)
{
  GGcout("GGEMSDosimetryCalculator", "GGEMSDosimetryCalculator", 3) << "Allocation of GGEMSDosimetryCalculator..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSDosimetryCalculator::~GGEMSDosimetryCalculator(void)
{
  GGcout("GGEMSDosimetryCalculator", "~GGEMSDosimetryCalculator", 3) << "Deallocation of GGEMSDosimetryCalculator..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SetDoselSizes(GGfloat3 const& dosel_sizes)
{
  dosel_sizes_ = dosel_sizes;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SetOutputDosimetryFilename(std::string const& output_filename)
{
  dosimetry_output_filename = output_filename;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::CheckParameters(void) const
{
  if (!navigator_) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "A navigator has to be associated to GGEMSDosimetryCalculator!!!";
    GGEMSMisc::ThrowException("GGEMSDosimetryCalculator", "CheckParameters", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::SetNavigator(std::string const& navigator_name)
{
  GGEMSNavigatorManager& navigator_manager = GGEMSNavigatorManager::GetInstance();
  navigator_ = navigator_manager.GetNavigator(navigator_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::Initialize(void)
{
  GGcout("GGEMSDosimetryCalculator", "Initialize", 3) << "Initializing dosimetry calculator..." << GGendl;

  CheckParameters();

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Allocate dosemetry params on OpenCL device
  dose_params_ = opencl_manager.Allocate(nullptr, sizeof(GGEMSDoseParams), CL_MEM_READ_WRITE);

  // Get pointer on OpenCL device for dose parameters
  GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_params_.get(), sizeof(GGEMSDoseParams));

  if (dosel_sizes_.x < 0.0f && dosel_sizes_.y < 0.0f && dosel_sizes_.z < 0.0f) { // Custom dosel size
    GGfloat3 voxel_sizes = dynamic_cast<GGEMSVoxelizedSolid*>(navigator_->GetSolids().at(0).get())->GetVoxelSizes();
    dose_params_device->size_of_dosels_ = voxel_sizes;
    dose_params_device->inv_size_of_dosels_ = {
      1.0f / voxel_sizes.x,
      1.0f / voxel_sizes.y,
      1.0f / voxel_sizes.z
    };
  }
  else { // Dosel size = voxel size
    dose_params_device->size_of_dosels_ = dosel_sizes_;
    dose_params_device->inv_size_of_dosels_ = {
      1.0f / dosel_sizes_.x,
      1.0f / dosel_sizes_.y,
      1.0f / dosel_sizes_.z
    };
  }

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(dose_params_.get(), dose_params_device);

  //std::cout << dosel_sizes_.x << " " << dosel_sizes_.y << " " << dosel_sizes_.z << std::endl;

/*
    /// Compute dosemap parameters /////////////////////////////

    // Select a doxel size
    if ( m_dosel_size.x > 0.0 && m_dosel_size.y > 0.0 && m_dosel_size.z > 0.0 )
    {
        h_dose->dosel_size = m_dosel_size;
        h_dose->inv_dosel_size = fxyz_inv( m_dosel_size );
    }
    else
    {
        h_dose->dosel_size = make_f32xyz( m_phantom.h_volume->spacing_x,
                                       m_phantom.h_volume->spacing_y,
                                       m_phantom.h_volume->spacing_z );
        h_dose->inv_dosel_size = fxyz_inv( h_dose->dosel_size );
    }*/

/*
    // Compute min-max volume of interest
    f32xyz phan_size = make_f32xyz( m_phantom.h_volume->nb_vox_x * m_phantom.h_volume->spacing_x,
                                    m_phantom.h_volume->nb_vox_y * m_phantom.h_volume->spacing_y,
                                    m_phantom.h_volume->nb_vox_z * m_phantom.h_volume->spacing_z );
    f32xyz half_phan_size = fxyz_scale( phan_size, 0.5f );
    f32 phan_xmin = -half_phan_size.x; f32 phan_xmax = half_phan_size.x;
    f32 phan_ymin = -half_phan_size.y; f32 phan_ymax = half_phan_size.y;
    f32 phan_zmin = -half_phan_size.z; f32 phan_zmax = half_phan_size.z;
    */

    // Select a min-max VOI
 /*   if ( !m_xmin && !m_xmax && !m_ymin && !m_ymax && !m_zmin && !m_zmax )
    {
        h_dose->xmin = m_phantom.h_volume->xmin;
        h_dose->xmax = m_phantom.h_volume->xmax;
        h_dose->ymin = m_phantom.h_volume->ymin;
        h_dose->ymax = m_phantom.h_volume->ymax;
        h_dose->zmin = m_phantom.h_volume->zmin;
        h_dose->zmax = m_phantom.h_volume->zmax;
    }
    else
    {
        h_dose->xmin = m_xmin;
        h_dose->xmax = m_xmax;
        h_dose->ymin = m_ymin;
        h_dose->ymax = m_ymax;
        h_dose->zmin = m_zmin;
        h_dose->zmax = m_zmax;
    }

    // Get the current dimension of the dose map
    f32xyz cur_dose_size = make_f32xyz( h_dose->xmax - h_dose->xmin,
                                        h_dose->ymax - h_dose->ymin,
                                        h_dose->zmax - h_dose->zmin );

    // New nb of voxels
    h_dose->nb_dosels.x = floor( cur_dose_size.x / h_dose->dosel_size.x );
    h_dose->nb_dosels.y = floor( cur_dose_size.y / h_dose->dosel_size.y );
    h_dose->nb_dosels.z = floor( cur_dose_size.z / h_dose->dosel_size.z );
    h_dose->slice_nb_dosels = h_dose->nb_dosels.x * h_dose->nb_dosels.y;
    h_dose->tot_nb_dosels = h_dose->slice_nb_dosels * h_dose->nb_dosels.z;

    // Compute the new size (due to integer nb of doxels)
    f32xyz new_dose_size = fxyz_mul( h_dose->dosel_size, cast_ui32xyz_to_f32xyz( h_dose->nb_dosels ) );

    if ( new_dose_size.x <= 0.0 || new_dose_size.y <= 0.0 || new_dose_size.z <= 0.0 )
    {
        GGcerr << "Dosemap dimension: "
               << new_dose_size.x << " "
               << new_dose_size.y << " "
               << new_dose_size.z << GGendl;
        exit_simulation();
    }

    // Compute new min and max after voxel alignment // TODO: Check here, offset is not considered? - JB
    f32xyz half_delta_size = fxyz_scale( fxyz_sub( cur_dose_size, new_dose_size ), 0.5f );

    h_dose->xmin += half_delta_size.x;
    h_dose->xmax -= half_delta_size.x;

    h_dose->ymin += half_delta_size.y;
    h_dose->ymax -= half_delta_size.y;

    h_dose->zmin += half_delta_size.z;
    h_dose->zmax -= half_delta_size.z;

    // Get the new offset
    h_dose->offset.x = m_phantom.h_volume->off_x - ( h_dose->xmin - m_phantom.h_volume->xmin );
    h_dose->offset.y = m_phantom.h_volume->off_y - ( h_dose->ymin - m_phantom.h_volume->ymin );
    h_dose->offset.z = m_phantom.h_volume->off_z - ( h_dose->zmin - m_phantom.h_volume->zmin );

    // Init dose map
    h_dose->edep = (f64*)malloc( h_dose->tot_nb_dosels*sizeof(f64) );
    h_dose->edep_squared = (f64*)malloc( h_dose->tot_nb_dosels*sizeof(f64) );
    h_dose->number_of_hits = (ui32*)malloc( h_dose->tot_nb_dosels*sizeof(ui32) );
    ui32 i=0; while (i < h_dose->tot_nb_dosels)
    {
        h_dose->edep[i] = 0.0;
        h_dose->edep_squared[i] = 0.0;
        h_dose->number_of_hits[i] = 0;
        ++i;
    }

    //////////////////////////////////////////////////////////

    // Host allocation
    m_dose_values = (f32*)malloc( h_dose->tot_nb_dosels * sizeof(f32) );
    m_uncertainty_values = (f32*)malloc( h_dose->tot_nb_dosels * sizeof(f32) );

    // Device allocation and copy
    m_copy_dosemap_to_gpu();
*/
}
