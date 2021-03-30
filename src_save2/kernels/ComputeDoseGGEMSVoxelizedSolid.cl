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
  \file ComputeDoseGGEMSVoxelizedSolid.cl

  \brief OpenCL kernel compute dose in voxelized solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday January 13, 2021
*/

#include "GGEMS/navigators/GGEMSDoseParams.hh"
#include "GGEMS/materials/GGEMSMaterialTables.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolidData.hh"

/*!
  \fn kernel void compute_dose_ggems_voxelized_solid(GGsize const dosel_id_limit, global GGEMSDoseParams const* dose_params, global GGDosiType const* edep, global GGint const* hit, global GGDosiType const* edep_squared, global GGEMSVoxelizedSolidData const* voxelized_solid_data, global GGshort const* label_data, global GGEMSMaterialTables const* materials, global GGfloat* dose, global GGfloat* uncertainty, GGfloat const scale_factor, GGchar const is_water_reference, GGfloat const minimum_density)
  \param dosel_id_limit - number total of dosels
  \param dose_params - params about dosemap
  \param edep - buffer storing energy deposit
  \param hit - buffer storing hit
  \param edep_squared - buffer storing edep squared
  \param voxelized_solid_data - pointer to voxelized solid data
  \param label_data - label data associated to voxelized phantom
  \param materials - registered material in voxelized phantom
  \param dose - output buffer storing dose in gray (Gy)
  \param uncertainty - output buffer storing dose uncertainty
  \param scale_factor - scale factor apply to dose
  \param is_water_reference - water reference mode
  \param minimum_density - minimum density threshold
  \brief computing dose for voxelized solid
*/
kernel void compute_dose_ggems_voxelized_solid(
  GGsize const dosel_id_limit,
  global GGEMSDoseParams const* dose_params,
  global GGDosiType const* edep,
  global GGint const* hit,
  global GGDosiType const* edep_squared,
  global GGEMSVoxelizedSolidData const* voxelized_solid_data,
  global GGuchar const* label_data,
  global GGEMSMaterialTables const* materials,
  global GGfloat* dose,
  global GGfloat* uncertainty,
  GGfloat const scale_factor,
  GGchar const is_water_reference,
  GGfloat const minimum_density
)
{
  // Getting index of thread
  GGint global_id = get_global_id(0);

  // Return if index > to particle limit
  if (global_id >= dosel_id_limit) return;

  GGint3 dosel_id;
  dosel_id.z = global_id/dose_params->slice_number_of_dosels_;
  dosel_id.x = (global_id - dosel_id.z*dose_params->slice_number_of_dosels_)%dose_params->number_of_dosels_.x;
  dosel_id.y = (global_id - dosel_id.z*dose_params->slice_number_of_dosels_)/dose_params->number_of_dosels_.x;

  // Convert doxel_id into position
  GGfloat3 dosel_pos = convert_float3(dosel_id) * dose_params->size_of_dosels_ + convert_float3((dose_params->number_of_dosels_-1)) * (-0.5f*dose_params->size_of_dosels_);

  // Get index of voxelized phantom, x, y, z
  GGint3 voxel_id = convert_int3((dosel_pos - voxelized_solid_data->obb_geometry_.border_min_xyz_) / voxelized_solid_data->voxel_sizes_xyz_);

  // Get the material that compose this volume
  GGuchar material_id = label_data[
    voxel_id.x +
    voxel_id.y * voxelized_solid_data->number_of_voxels_xyz_.x +
    voxel_id.z * voxelized_solid_data->number_of_voxels_xyz_.x * voxelized_solid_data->number_of_voxels_xyz_.y
  ];

  // Compute volume of dosel
  GGfloat dosel_vol = dose_params->size_of_dosels_.x * dose_params->size_of_dosels_.y * dose_params->size_of_dosels_.z;

  // Get density
  GGfloat density = is_water_reference ? 1.0f * (g/cm3) : materials->density_of_material_[material_id];

  // Apply threshold on density and computing dose
  dose[global_id] = density < minimum_density ? 0.0f : scale_factor * edep[global_id] / density / dosel_vol / Gy;

  // Relative statistical uncertainty (from Ma et al. PMB 47 2002 p1671)
  //              /                                    \ ^1/2
  //              |    N*Sum(Edep^2) - Sum(Edep)^2     |
  //  relError =  | __________________________________ |
  //              |                                    |
  //              \         (N-1)*Sum(Edep)^2          /
  //
  //   where Edep represents the energy deposit in one hit and N the number of energy deposits (hits)

  // Computing uncertainty
  if (hit[global_id] > 1 && edep[global_id] != 0.0) {
    GGDosiType sum_edep_2 = edep[global_id] * edep[global_id];
    uncertainty[global_id] = sqrt((hit[global_id]*edep_squared[global_id] - sum_edep_2) / ((hit[global_id]-1) * sum_edep_2));
  }
  else {
    uncertainty[global_id] = 1.0f;
  }
}
