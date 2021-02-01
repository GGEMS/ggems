#include "GGEMS/tools/GGEMSTypes.hh"
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
  global GGshort const* label_data,
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
  // GGfloat dosel_pos_x = dosel_id_x * dose_params->size_of_dosels_.x + (dose_params->number_of_dosels_.x-1) * (-0.5f*dose_params->size_of_dosels_.x);
  // GGfloat dosel_pos_y = dosel_id_y * dose_params->size_of_dosels_.y + (dose_params->number_of_dosels_.y-1) * (-0.5f*dose_params->size_of_dosels_.y);
  // GGfloat dosel_pos_z = dosel_id_z * dose_params->size_of_dosels_.z + (dose_params->number_of_dosels_.z-1) * (-0.5f*dose_params->size_of_dosels_.z);

  // Get index of voxelized phantom, x, y, z
  GGint3 voxel_id = convert_int3((dosel_pos - voxelized_solid_data->obb_geometry_.border_min_xyz_) / voxelized_solid_data->voxel_sizes_xyz_);
  // GGint voxel_id_x = (dosel_pos_x - voxelized_solid_data->obb_geometry_.border_min_xyz_.x) / voxelized_solid_data->voxel_sizes_xyz_.x;
  // GGint voxel_id_y = (dosel_pos_y - voxelized_solid_data->obb_geometry_.border_min_xyz_.y) / voxelized_solid_data->voxel_sizes_xyz_.y;
  // GGint voxel_id_z = (dosel_pos_z - voxelized_solid_data->obb_geometry_.border_min_xyz_.z) / voxelized_solid_data->voxel_sizes_xyz_.z;

  // Get the material that compose this volume
  GGshort material_id = label_data[
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
