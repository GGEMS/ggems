/*!
  \file GGEMSEMProcess.cc

  \brief GGEMS mother class for electromagnectic process

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday February 11, 2020
*/

#include "GGEMS/physics/GGEMSEMProcess.hh"
#include "GGEMS/physics/GGEMSParticleCrossSectionsStack.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSEMProcess::GGEMSEMProcess()
: process_id_(0),
  process_name_(""),
  primary_particle_(""),
  secondary_particle_(""),
  is_secondaries_(false),
  opencl_manager_(GGEMSOpenCLManager::GetInstance())
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

void GGEMSEMProcess::BuildCrossSectionTables(std::shared_ptr<cl::Buffer> particle_cross_sections, std::shared_ptr<cl::Buffer> material_tables)
{
  GGcout("GGEMSEMProcess", "BuildCrossSectionTables", 3) << "Building cross section table..." << GGendl;

  // Set missing information in cross section table
  GGEMSParticleCrossSections* cross_section_device = opencl_manager_.GetDeviceBuffer<GGEMSParticleCrossSections>(particle_cross_sections, sizeof(GGEMSParticleCrossSections));

  // Store index of activated process
  cross_section_device->index_photon_cs[cross_section_device->number_of_activated_photon_processes_] = process_id_;

  // Increment number of activated photon process
  cross_section_device->number_of_activated_photon_processes_ += 1;

  // Get the material tables
  GGEMSMaterialTables* materials_device = opencl_manager_.GetDeviceBuffer<GGEMSMaterialTables>(material_tables, sizeof(GGEMSMaterialTables));

  // Compute Compton cross section par material
  GGushort const kNumberOfBins = cross_section_device->number_of_bins_;
  // Loop over the materials
  for (GGuchar j = 0; j < materials_device->number_of_materials_; ++j) {
    // Loop over the number of bins
    for (GGushort i = 0; i < kNumberOfBins; ++i) {
      cross_section_device->photon_cross_sections_[process_id_][i + j*kNumberOfBins] =
        ComputeCrossSectionPerMaterial(materials_device, j, cross_section_device->energy_bins_[i]);
    }
  }

  // Release pointer
  opencl_manager_.ReleaseDeviceBuffer(material_tables, materials_device);
  opencl_manager_.ReleaseDeviceBuffer(particle_cross_sections, cross_section_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSEMProcess::ComputeCrossSectionPerMaterial(GGEMSMaterialTables const* material_tables, GGushort const& material_index, GGfloat const& energy)
{
  GGfloat cross_section_material = 0.0f;
  GGushort const kIndexOffset = material_tables->index_of_chemical_elements_[material_index];
  // Loop over all the chemical elements
  for (GGuchar i = 0; i < material_tables->number_of_chemical_elements_[material_index]; ++i) {
    cross_section_material += material_tables->atomic_number_density_[i+kIndexOffset] * ComputeCrossSectionPerAtom(energy, material_tables->atomic_number_Z_[i+kIndexOffset]);
  }
  return cross_section_material;
}