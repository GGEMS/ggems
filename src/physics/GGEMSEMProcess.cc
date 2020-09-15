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
  cross_section_device->index_photon_cs_[cross_section_device->number_of_activated_photon_processes_] = process_id_;

  // Increment number of activated photon process
  cross_section_device->number_of_activated_photon_processes_ += 1;

  // Get the material tables
  GGEMSMaterialTables* materials_device = opencl_manager.GetDeviceBuffer<GGEMSMaterialTables>(material_tables_cl.lock().get(), sizeof(GGEMSMaterialTables));

  // Compute Compton cross section par material
  GGushort const kNumberOfBins = cross_section_device->number_of_bins_;
  // Loop over the materials
  for (GGuchar j = 0; j < materials_device->number_of_materials_; ++j) {
    // Loop over the number of bins
    for (GGuint i = 0; i < kNumberOfBins; ++i) {
      cross_section_device->photon_cross_sections_[process_id_][i + j*kNumberOfBins] = ComputeCrossSectionPerMaterial(cross_section_device, materials_device, j, i);
    }
  }

  // If flag activate print tables
  GGEMSProcessesManager& process_manager = GGEMSProcessesManager::GetInstance();
  if (process_manager.IsPrintPhysicTables()) {
    GGcout("GGEMSEMProcess", "BuildCrossSectionTables", 0) << "* PROCESS " << process_name_ << GGendl;

    // Loop over material
    for (GGuchar j = 0; j < materials_device->number_of_materials_; ++j) {
      GGushort id_elt = materials_device->index_of_chemical_elements_[j];
      GGcout("GGEMSEMProcess", "BuildCrossSectionTables", 0) << "    - Material: " << cross_section_device->material_names_[j]
        << ", density: " << materials_device->density_of_material_[j]/(g/cm3) << " g.cm-3" << GGendl;
      // Loop over number of bins (energy)
      for (GGushort i = 0; i < kNumberOfBins; ++i) {
        GGcout("GGEMSEMProcess", "BuildCrossSectionTables", 0) << "        + Energy: " << cross_section_device->energy_bins_[i]/keV << " keV, cross section: "
          << (cross_section_device->photon_cross_sections_[process_id_][i + j*kNumberOfBins]/materials_device->density_of_material_[j])/(cm2/g) << " cm2.g-1" << GGendl;
        // Loop over elements
        for (GGushort k = 0; k < materials_device->number_of_chemical_elements_[j]; ++k) {
          GGuint const kAtomicNumber = materials_device->atomic_number_Z_[k+id_elt];
          GGcout("GGEMSEMProcess", "BuildCrossSectionTables", 0) << "            # Element (Z): " << kAtomicNumber
            << ", atomic number density: " << materials_device->atomic_number_density_[k+id_elt]/(1/cm3) << " atom/cm3, cross section per atom: "
            << cross_section_device->photon_cross_sections_per_atom_[process_id_][i + kAtomicNumber*kNumberOfBins]/(cm2)<< " cm2" << GGendl;
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

GGfloat GGEMSEMProcess::ComputeCrossSectionPerMaterial(GGEMSParticleCrossSections* cross_section_device, GGEMSMaterialTables const* material_tables, GGushort const& material_index, GGuint const& energy_index)
{
  GGfloat const kEnergy = cross_section_device->energy_bins_[energy_index];
  GGfloat cross_section_material = 0.0f;
  GGushort const kIndexOffset = material_tables->index_of_chemical_elements_[material_index];

  // Loop over all the chemical elements
  for (GGuchar i = 0; i < material_tables->number_of_chemical_elements_[material_index]; ++i) {
    GGuint const kAtomicNumber = material_tables->atomic_number_Z_[i+kIndexOffset];
    GGfloat const kCrossSectionPerAtom = ComputeCrossSectionPerAtom(kEnergy, kAtomicNumber);
    cross_section_device->photon_cross_sections_per_atom_[process_id_][energy_index + kAtomicNumber*cross_section_device->number_of_bins_] = kCrossSectionPerAtom;
    cross_section_material += material_tables->atomic_number_density_[i+kIndexOffset] * kCrossSectionPerAtom;
  }
  return cross_section_material;
}
