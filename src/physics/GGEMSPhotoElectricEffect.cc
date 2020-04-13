/*!
  \file GGEMSPhotoElectricEffect.cc

  \brief Photoelectric Effect process using Sandia table

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday April 13, 2020
*/

#include "GGEMS/materials/GGEMSMaterials.hh"
#include "GGEMS/physics/GGEMSPhotoElectricEffect.hh"
#include "GGEMS/physics/GGEMSParticleCrossSectionsStack.hh"
#include "GGEMS/physics/GGEMSSandiaTable.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhotoElectricEffect::GGEMSPhotoElectricEffect(std::string const& primary_particle, bool const& is_secondary)
: GGEMSEMProcess()
{
  GGcout("GGEMSPhotoElectricEffect", "GGEMSPhotoElectricEffect", 3) << "Allocation of GGEMSPhotoElectricEffect..." << GGendl;

  process_name_ = "Photoelectric";

  // Check type of primary particle
  if (primary_particle != "gamma") {
    std::ostringstream oss(std::ostringstream::out);
    oss << "For PhotoElectric effect, incident particle has to be a 'gamma'";
    GGEMSMisc::ThrowException("GGEMSPhotoElectricEffect", "GGEMSPhotoElectricEffect", oss.str());
  }

  primary_particle_ = "gamma";
  secondary_particle_ = "e-";
  is_secondaries_ = is_secondary;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhotoElectricEffect::~GGEMSPhotoElectricEffect(void)
{
  GGcout("GGEMSPhotoElectricEffect", "~GGEMSPhotoElectricEffect", 3) << "Deallocation of GGEMSPhotoElectricEffect..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhotoElectricEffect::BuildCrossSectionTables(std::shared_ptr<cl::Buffer> particle_cross_sections, std::shared_ptr<cl::Buffer> material_tables)
{
  GGcout("GGEMSPhotoElectricEffect", "BuildCrossSectionTables", 3) << "Building cross section table for PhotoElectric effect..." << GGendl;

  // Set missing information in cross section table
  GGEMSParticleCrossSections* photoelectric_cs_device = opencl_manager_.GetDeviceBuffer<GGEMSParticleCrossSections>(particle_cross_sections, sizeof(GGEMSParticleCrossSections));

  // Store index of activated process
  photoelectric_cs_device->index_photon_cs[photoelectric_cs_device->number_of_activated_photon_processes_] = GGEMSProcess::PHOTOELECTRIC_EFFECT;

  // Increment number of activated photon process
  photoelectric_cs_device->number_of_activated_photon_processes_ += 1;

  // Get the material tables
  GGEMSMaterialTables* materials_device = opencl_manager_.GetDeviceBuffer<GGEMSMaterialTables>(material_tables, sizeof(GGEMSMaterialTables));

  // Compute Compton cross section par material
  GGuchar const kNumberOfMaterials = photoelectric_cs_device->number_of_materials_;
  GGushort const kNumberOfBins = photoelectric_cs_device->number_of_bins_;
  // Loop over the materials
  for (GGuchar j = 0; j < kNumberOfMaterials; ++j) {
    // Loop over the number of bins
    for (GGushort i = 0; i < kNumberOfBins; ++i) {
      photoelectric_cs_device->photon_cross_sections_[GGEMSProcess::PHOTOELECTRIC_EFFECT][i + j*kNumberOfBins] =
        ComputeCrossSectionPerMaterial(materials_device, j, photoelectric_cs_device->energy_bins_[i]);
    }
  }

  // Release pointer
  opencl_manager_.ReleaseDeviceBuffer(material_tables, materials_device);
  opencl_manager_.ReleaseDeviceBuffer(particle_cross_sections, photoelectric_cs_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSPhotoElectricEffect::ComputeCrossSectionPerMaterial(GGEMSMaterialTables const* material_tables, GGushort const& material_index, GGfloat const& energy)
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

GGfloat GGEMSPhotoElectricEffect::ComputeCrossSectionPerAtom(GGfloat const& energy, GGuchar const& atomic_number)
{
  // Threshold at 10 eV
  GGdouble const kEmin = fmax(GGEMSSandiaTable::kIonizationPotentials[atomic_number], 10.0)*GGEMSUnits::eV;
  if (energy < kEmin) return 0.0f;

  GGushort const kStart = GGEMSSandiaTable::kCumulativeIntervals[atomic_number-1];
  GGushort const kStop = kStart + GGEMSSandiaTable::kNumberOfIntervals[atomic_number];

  GGushort pos = kStop;
  while (energy < static_cast<GGfloat>(GGEMSSandiaTable::kSandiaTable[pos][0])*GGEMSUnits::keV) --pos;

  GGdouble const kAoverAvo = GGEMSPhysicalConstant::ATOMIC_MASS_UNIT * static_cast<GGdouble>(atomic_number) / GGEMSSandiaTable::kZtoARatio[atomic_number];

  GGdouble const kREnergy = 1.0 / energy;
  GGdouble const kREnergy2 = kREnergy * kREnergy;

  return static_cast<GGfloat>(
    kREnergy * GGEMSSandiaTable::kSandiaTable[pos][1] * kAoverAvo * 0.160217648e-22 +
    kREnergy2 * GGEMSSandiaTable::kSandiaTable[pos][2] * kAoverAvo * 0.160217648e-25 +
    kREnergy * kREnergy2 * GGEMSSandiaTable::kSandiaTable[pos][3] * kAoverAvo * 0.160217648e-28 +
    kREnergy2 * kREnergy2 * GGEMSSandiaTable::kSandiaTable[pos][4] * kAoverAvo * 0.160217648e-31);
}
