#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/navigators/GGEMSDoseParams.hh"
#include "GGEMS/materials/GGEMSMaterialTables.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolidData.hh"

/*!
  \fn kernel void compute_dose_ggems_voxelized_solid(GGint const dosel_id_limit, global GGEMSDoseParams const* dose_params, global GGDosiType const* edep, global GGEMSVoxelizedSolidData const* voxelized_solid_data, global GGshort const* label_data, global GGEMSMaterialTables const* materials, global GGfloat* dose)
  \param dosel_id_limit - number total of dosels
  \param dose_params - params about dosemap
  \param edep - buffer storing energy deposit
  \param voxelized_solid_data - pointer to voxelized solid data
  \param label_data - label data associated to voxelized phantom
  \param materials - registered material in voxelized phantom
  \param dose - output buffer storing dose in gray (Gy)
  \brief computing dose for voxelized solid
*/
kernel void compute_dose_ggems_voxelized_solid(GGint const dosel_id_limit, global GGEMSDoseParams const* dose_params, global GGDosiType const* edep, global GGEMSVoxelizedSolidData const* voxelized_solid_data, global GGshort const* label_data, global GGEMSMaterialTables const* materials, global GGfloat* dose)
{
  // Getting index of thread
  GGint global_id = get_global_id(0);

  // Return if index > to particle limit
  if (global_id >= dosel_id_limit) return;

  GGint dosel_id_z = global_id/dose_params->slice_number_of_dosels_;
  GGint dosel_id_x = (global_id - dosel_id_z*dose_params->slice_number_of_dosels_)%dose_params->number_of_dosels_.x;
  GGint dosel_id_y = (global_id - dosel_id_z*dose_params->slice_number_of_dosels_)/dose_params->number_of_dosels_.x;

  // Convert doxel_id into position
  GGfloat dosel_pos_x = dosel_id_x * dose_params->size_of_dosels_.x + (dose_params->number_of_dosels_.x-1) * (-0.5f*dose_params->size_of_dosels_.x);
  GGfloat dosel_pos_y = dosel_id_y * dose_params->size_of_dosels_.y + (dose_params->number_of_dosels_.y-1) * (-0.5f*dose_params->size_of_dosels_.y);
  GGfloat dosel_pos_z = dosel_id_z * dose_params->size_of_dosels_.z + (dose_params->number_of_dosels_.z-1) * (-0.5f*dose_params->size_of_dosels_.z);

  // Get index of voxelized phantom, x, y, z
  GGint voxel_id_x = (dosel_pos_x - voxelized_solid_data->obb_geometry_.border_min_xyz_[0]) / voxelized_solid_data->voxel_sizes_xyz_[0];
  GGint voxel_id_y = (dosel_pos_y - voxelized_solid_data->obb_geometry_.border_min_xyz_[1]) / voxelized_solid_data->voxel_sizes_xyz_[1];
  GGint voxel_id_z = (dosel_pos_z - voxelized_solid_data->obb_geometry_.border_min_xyz_[2]) / voxelized_solid_data->voxel_sizes_xyz_[2];

  // Get the material that compose this volume
  GGshort material_id = label_data[
    voxel_id_x +
    voxel_id_y * voxelized_solid_data->number_of_voxels_xyz_[0] +
    voxel_id_z * voxelized_solid_data->number_of_voxels_xyz_[0] * voxelized_solid_data->number_of_voxels_xyz_[1]
  ];

  // Compute volume of dosel
  GGfloat dosel_vol = dose_params->size_of_dosels_.x * dose_params->size_of_dosels_.y * dose_params->size_of_dosels_.z;

  // Get density
  GGfloat density = materials->density_of_material_[material_id];

  // Computing dose
  dose[global_id] = edep[global_id] / density / dosel_vol / Gy;
}
