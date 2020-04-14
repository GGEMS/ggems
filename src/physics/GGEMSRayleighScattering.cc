/*!
  \file GGEMSRayleighScattering.cc

  \brief Rayleigh scattering process using Livermore model

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday April 14, 2020
*/

#include "GGEMS/materials/GGEMSMaterials.hh"
#include "GGEMS/maths/GGEMSMathAlgorithms.hh"
#include "GGEMS/physics/GGEMSRayleighScattering.hh"
#include "GGEMS/physics/GGEMSParticleCrossSectionsStack.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSRayleighScattering::GGEMSRayleighScattering(std::string const& primary_particle, bool const& is_secondary)
: GGEMSEMProcess()
{
  GGcout("GGEMSRayleighScattering", "GGEMSRayleighScattering", 3) << "Allocation of GGEMSRayleighScattering..." << GGendl;

  process_name_ = "Rayleigh";

  // Check type of primary particle
  if (primary_particle != "gamma") {
    std::ostringstream oss(std::ostringstream::out);
    oss << "For Rayleigh scattering, incident particle has to be a 'gamma'";
    GGEMSMisc::ThrowException("GGEMSRayleighScattering", "GGEMSRayleighScattering", oss.str());
  }

  // Checking secondaries
  if (is_secondary == true) {
    GGwarn("GGEMSRayleighScattering", "GGEMSRayleighScattering", 0) << "There is no secondary during Rayleigh process!!! Secondary flag set to false" << GGendl;
  }

  primary_particle_ = "gamma";
  is_secondaries_ = false;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSRayleighScattering::~GGEMSRayleighScattering(void)
{
  GGcout("GGEMSRayleighScattering", "~GGEMSRayleighScattering", 3) << "Deallocation of GGEMSRayleighScattering..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRayleighScattering::BuildCrossSectionTables(std::shared_ptr<cl::Buffer> particle_cross_sections, std::shared_ptr<cl::Buffer> material_tables)
{
  GGcout("GGEMSRayleighScattering", "BuildCrossSectionTables", 3) << "Building cross section table for Rayleigh scattering..." << GGendl;

  // Set missing information in cross section table
  GGEMSParticleCrossSections* cross_section_device = opencl_manager_.GetDeviceBuffer<GGEMSParticleCrossSections>(particle_cross_sections, sizeof(GGEMSParticleCrossSections));

  // Store index of activated process
  cross_section_device->index_photon_cs[cross_section_device->number_of_activated_photon_processes_] = GGEMSProcess::RAYLEIGH_SCATTERING;

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
      cross_section_device->photon_cross_sections_[GGEMSProcess::RAYLEIGH_SCATTERING][i + j*kNumberOfBins] =
        ComputeCrossSectionPerMaterial(materials_device, j, cross_section_device->energy_bins_[i]);
    }
  }

  // Computing scatter factors
  // Compute cross section per atom
  /*for (GGuchar k = 0; k < materials_device->number_of_materials_; ++k) {
    GGushort const kIndexOffset = materials_device->index_of_chemical_elements_[k];
    // Loop over the chemical elements
    for (GGuchar j = 0; j < materials_device->number_of_chemical_elements_[k]; ++j) {
      GGfloat const kAtomicNumberDensity = materials_device->atomic_number_density_[j + kIndexOffset];
      GGuchar const kAtomicNumber = materials_device->atomic_number_Z_[j + kIndexOffset];
      // Loop over energy bins
      for (GGushort i = 0; i < kNumberOfBins; ++i) {
        cross_section_device->photon_cross_sections_per_atom_[GGEMSProcess::PHOTOELECTRIC_EFFECT][i + kAtomicNumber*kNumberOfBins] =
          kAtomicNumberDensity * ComputeCrossSectionPerAtom(cross_section_device->energy_bins_[i], kAtomicNumber);
      }
    }
  }*/

  // Release pointer
  opencl_manager_.ReleaseDeviceBuffer(material_tables, materials_device);
  opencl_manager_.ReleaseDeviceBuffer(particle_cross_sections, cross_section_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSRayleighScattering::ComputeCrossSectionPerMaterial(GGEMSMaterialTables const* material_tables, GGushort const& material_index, GGfloat const& energy)
{
  GGfloat cross_section_material = 0.0f;
  GGushort const kIndexOffset = material_tables->index_of_chemical_elements_[material_index];
  // Loop over all the chemical elements
  for (GGuchar i = 0; i < material_tables->number_of_chemical_elements_[material_index]; ++i) {
    cross_section_material += material_tables->atomic_number_density_[i+kIndexOffset] * ComputeCrossSectionPerAtom(energy, material_tables->atomic_number_Z_[i+kIndexOffset]);
  }
  return cross_section_material;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSRayleighScattering::ComputeCrossSectionPerAtom(GGfloat const& energy, GGuchar const& atomic_number)
{
  // Energy in range [250 eV; 100 GeV]
  if (energy < 250e-6f || energy > 100e3f) return 0.0f;

  GGuint const kStart = GGEMSRayleighTable::kCrossSectionCumulativeIntervals[atomic_number];
  GGuint const kStop = kStart + 2 * (GGEMSRayleighTable::kCrossSectionNumberOfIntervals[atomic_number]-1);

  GGuint pos = kStart;
  for (; pos < kStop; pos += 2) {
    if (GGEMSRayleighTable::kCrossSection[pos] >= static_cast<GGfloat>(energy)) break;
  }

  if (energy < 1e3f) { // 1 GeV
    return static_cast<GGfloat>(1.0e-22 * LogLogInterpolation(
      energy,
      static_cast<GGfloat>(GGEMSRayleighTable::kCrossSection[pos-2]), static_cast<GGfloat>(GGEMSRayleighTable::kCrossSection[pos-1]),
      static_cast<GGfloat>(GGEMSRayleighTable::kCrossSection[pos]), static_cast<GGfloat>(GGEMSRayleighTable::kCrossSection[pos+1])));
  }
  else {
    return static_cast<GGfloat>(1.0e-22 * GGEMSRayleighTable::kCrossSection[pos-1]);
  }
}
