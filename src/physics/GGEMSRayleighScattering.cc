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

  process_id_ = RAYLEIGH_SCATTERING;
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

void GGEMSRayleighScattering::BuildCrossSectionTables(std::weak_ptr<cl::Buffer> particle_cross_sections_cl, std::weak_ptr<cl::Buffer> material_tables_cl)
{
  GGcout("GGEMSRayleighScattering", "BuildCrossSectionTables", 3) << "Building cross section table for Rayleigh scattering..." << GGendl;

  // Call mother
  GGEMSEMProcess::BuildCrossSectionTables(particle_cross_sections_cl, material_tables_cl);

  // Get OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Set missing information in cross section table
  GGEMSParticleCrossSections* cross_section_device = opencl_manager.GetDeviceBuffer<GGEMSParticleCrossSections>(particle_cross_sections_cl.lock().get(), sizeof(GGEMSParticleCrossSections));

  // Get the material tables
  GGEMSMaterialTables* materials_device = opencl_manager.GetDeviceBuffer<GGEMSMaterialTables>(material_tables_cl.lock().get(), sizeof(GGEMSMaterialTables));

  // Compute Compton cross section par material
  GGushort const kNumberOfBins = cross_section_device->number_of_bins_;

  // Compute cross section per atom
  for (GGuchar k = 0; k < materials_device->number_of_materials_; ++k) {
    GGushort const kIndexOffset = materials_device->index_of_chemical_elements_[k];
    // Loop over the chemical elements
    for (GGuchar j = 0; j < materials_device->number_of_chemical_elements_[k]; ++j) {
      GGfloat const kAtomicNumberDensity = materials_device->atomic_number_density_[j + kIndexOffset];
      GGuchar const kAtomicNumber = materials_device->atomic_number_Z_[j + kIndexOffset];
      // Loop over energy bins
      for (GGushort i = 0; i < kNumberOfBins; ++i) {
        // Scatter factor
        cross_section_device->rayleigh_scatter_factor_[i + kAtomicNumber*kNumberOfBins] = ComputeScatterFactor(cross_section_device->energy_bins_[i], kAtomicNumber);
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

GGfloat GGEMSRayleighScattering::ComputeCrossSectionPerAtom(GGfloat const& energy, GGuchar const& atomic_number) const
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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSRayleighScattering::ComputeScatterFactor(GGfloat const& energy, GGuchar const& atomic_number) const
{
  GGuint const kStart = GGEMSRayleighTable::kScatterFactorCumulativeIntervals[atomic_number];
  GGuint const kStop = kStart + 2 * (GGEMSRayleighTable::kScatterFactorNumberOfIntervals[atomic_number]-1);

  // Checking energy
  if (energy == 0.0f) return static_cast<GGfloat>(GGEMSRayleighTable::kScatterFactor[kStart+1]);

  GGuint pos = kStart;
  for (; pos < kStop; pos += 2) {
    if (GGEMSRayleighTable::kScatterFactor[pos]*eV >= energy) break; // SF data are in eV
  }

  // If the min bin of energy is equal to 0, loglog is not possible (return Inf)
  if (GGEMSRayleighTable::kScatterFactor[pos-2] == 0.0) {
    return static_cast<GGfloat>(GGEMSRayleighTable::kScatterFactor[pos-1]);
  }
  else {
    return LogLogInterpolation(energy,
      static_cast<GGfloat>(GGEMSRayleighTable::kScatterFactor[pos-2])*eV, static_cast<GGfloat>(GGEMSRayleighTable::kScatterFactor[pos-1]),
      static_cast<GGfloat>(GGEMSRayleighTable::kScatterFactor[pos])*eV, static_cast<GGfloat>(GGEMSRayleighTable::kScatterFactor[pos+1]));
  }
}
