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
  \file GGEMSEMProcess.cc

  \brief GGEMS mother class for electromagnectic process

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday February 11, 2020
*/

#include "GGEMS/physics/GGEMSProcessesManager.hh"
#include "GGEMS/physics/GGEMSEMProcess.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSEMProcess::GGEMSEMProcess()
: process_id_(0),
  process_name_(""),
  primary_particle_(""),
  secondary_particle_(""),
  is_secondaries_(false)
{
  GGcout("GGEMSEMProcess", "GGEMSEMProcess", 3) << "Allocation of GGEMSEMProcess..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSEMProcess::~GGEMSEMProcess(void)
{
  GGcout("GGEMSEMProcess", "~GGEMSEMProcess", 3) << "Deallocation of GGEMSEMProcess..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSEMProcess::BuildCrossSectionTables(std::weak_ptr<cl::Buffer> particle_cross_sections_cl, std::weak_ptr<cl::Buffer> material_tables_cl)
{
  GGcout("GGEMSEMProcess", "BuildCrossSectionTables", 3) << "Building cross section table for process " << process_name_ << "..." << GGendl;

  // Getting OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Set missing information in cross section table
  GGEMSParticleCrossSections* cross_section_device = opencl_manager.GetDeviceBuffer<GGEMSParticleCrossSections>(particle_cross_sections_cl.lock().get(), sizeof(GGEMSParticleCrossSections));

  // Store index of activated process
  cross_section_device->photon_cs_id_[cross_section_device->number_of_activated_photon_processes_] = process_id_;

  // Increment number of activated photon process
  cross_section_device->number_of_activated_photon_processes_ += 1;

  // Get the material tables
  GGEMSMaterialTables* materials_device = opencl_manager.GetDeviceBuffer<GGEMSMaterialTables>(material_tables_cl.lock().get(), sizeof(GGEMSMaterialTables));

  // Compute Compton cross section par material
  GGsize number_of_bins = cross_section_device->number_of_bins_;
  // Loop over the materials
  for (GGsize j = 0; j < materials_device->number_of_materials_; ++j) {
    // Loop over the number of bins
    for (GGsize i = 0; i < number_of_bins; ++i) {
      cross_section_device->photon_cross_sections_[process_id_][i + j*number_of_bins] = ComputeCrossSectionPerMaterial(cross_section_device, materials_device, j, i);
    }
  }

  // If flag activate print tables
  GGEMSProcessesManager& process_manager = GGEMSProcessesManager::GetInstance();
  if (process_manager.IsPrintPhysicTables()) {
    GGcout("GGEMSEMProcess", "BuildCrossSectionTables", 0) << "* PROCESS " << process_name_ << GGendl;

    // Loop over material
    for (GGsize j = 0; j < materials_device->number_of_materials_; ++j) {
      GGsize id_elt = materials_device->index_of_chemical_elements_[j];
      GGcout("GGEMSEMProcess", "BuildCrossSectionTables", 0) << "    - Material: " << cross_section_device->material_names_[j]
        << ", density: " << materials_device->density_of_material_[j]/(g/cm3) << " g.cm-3" << GGendl;
      // Loop over number of bins (energy)
      for (GGsize i = 0; i < number_of_bins; ++i) {
        GGcout("GGEMSEMProcess", "BuildCrossSectionTables", 0) << "        + Energy: " << cross_section_device->energy_bins_[i]/keV << " keV, cross section: "
          << (cross_section_device->photon_cross_sections_[process_id_][i + j*number_of_bins]/materials_device->density_of_material_[j])/(cm2/g) << " cm2.g-1" << GGendl;
        // Loop over elements
        for (GGsize k = 0; k < materials_device->number_of_chemical_elements_[j]; ++k) {
          GGuchar atomic_number = materials_device->atomic_number_Z_[k+id_elt];
          GGcout("GGEMSEMProcess", "BuildCrossSectionTables", 0) << "            # Element (Z): " << atomic_number
            << ", atomic number density: " << materials_device->atomic_number_density_[k+id_elt]/(1/cm3) << " atom/cm3, cross section per atom: "
            << cross_section_device->photon_cross_sections_per_atom_[process_id_][i + atomic_number*number_of_bins]/(cm2)<< " cm2" << GGendl;
        }
      }
    }
  }

  // Release pointer
  opencl_manager.ReleaseDeviceBuffer(material_tables_cl.lock().get(), materials_device);
  opencl_manager.ReleaseDeviceBuffer(particle_cross_sections_cl.lock().get(), cross_section_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSEMProcess::ComputeCrossSectionPerMaterial(GGEMSParticleCrossSections* cross_section_device, GGEMSMaterialTables const* material_tables, GGsize const& material_index, GGsize const& energy_index)
{
  GGfloat energy = cross_section_device->energy_bins_[energy_index];
  GGfloat cross_section_material = 0.0f;
  GGsize index_of_offset = material_tables->index_of_chemical_elements_[material_index];

  // Loop over all the chemical elements
  for (GGsize i = 0; i < material_tables->number_of_chemical_elements_[material_index]; ++i) {
    GGuchar atomic_number = material_tables->atomic_number_Z_[i+index_of_offset];
    GGfloat cross_section_per_atom = ComputeCrossSectionPerAtom(energy, atomic_number);
    cross_section_device->photon_cross_sections_per_atom_[process_id_][energy_index + atomic_number*cross_section_device->number_of_bins_] = cross_section_per_atom;
    cross_section_material += material_tables->atomic_number_density_[i+index_of_offset] * cross_section_per_atom;
  }
  return cross_section_material;
}
