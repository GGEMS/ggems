/*!
  \file GGEMSMaterials.cc

  \brief GGEMS class handling material(s) for a specific navigator

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 4, 2020
*/

#include <limits>

#include "GGEMS/navigators/GGEMSPhantomNavigatorManager.hh"
#include "GGEMS/materials/GGEMSIonizationParamsMaterial.hh"
#include "GGEMS/physics/GGEMSRangeCuts.hh"
#include "GGEMS/tools/GGEMSTools.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMaterials::GGEMSMaterials(void)
: opencl_manager_(GGEMSOpenCLManager::GetInstance()),
  material_manager_(GGEMSMaterialsDatabaseManager::GetInstance())
{
  GGcout("GGEMSMaterials", "GGEMSMaterials", 3) << "Allocation of GGEMSMaterials..." << GGendl;

  // Allocation of cuts
  range_cuts_.reset(new GGEMSRangeCuts());
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMaterials::~GGEMSMaterials(void)
{
  GGcout("GGEMSMaterials", "~GGEMSMaterials", 3) << "Deallocation of GGEMSMaterials..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterials::AddMaterial(std::string const& material_name)
{
  // Checking the number of material (maximum is 255)
  if (materials_.size() == 256) {
    GGEMSMisc::ThrowException("GGEMSMaterials", "AddMaterial", "Limit of material reached. The limit is 256 materials!!!");
  }

  // Add material and check if the material already exists
  materials_.push_back(material_name);
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterials::SetDistanceCut(std::string const& particle_name, GGfloat const& value, std::string const& unit)
{
  GGfloat const kCut = GGEMSUnits::DistanceUnit(value, unit.c_str());

  if (particle_name == "gamma") {
    range_cuts_->SetPhotonDistanceCut(kCut);
  }
  else if (particle_name == "e+") {
    range_cuts_->SetPositronDistanceCut(kCut);
  }
  else if (particle_name == "e-") {
    range_cuts_->SetElectronDistanceCut(kCut);
  }
  else {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Particle name " << particle_name << " unknown!!! The particles are:" << std::endl;
    oss << "    - gamma" << std::endl;
    oss << "    - e-" << std::endl;
    oss << "    - e+";
    GGEMSMisc::ThrowException("GGEMSMaterials", "SetCut", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterials::PrintInfos(void) const
{
  // Getting the OpenCL pointer on material tables
  GGEMSMaterialTables* material_table_device = opencl_manager_.GetDeviceBuffer<GGEMSMaterialTables>(material_tables_, sizeof(GGEMSMaterialTables));

  // Getting list of activated materials
  GGcout("GGEMSMaterials", "PrintInfos", 0) << GGendl;
  GGcout("GGEMSMaterials", "PrintInfos", 0) << "Number of materials: " << static_cast<GGuint>(material_table_device->number_of_materials_) << GGendl;
  GGcout("GGEMSMaterials", "PrintInfos", 0) << "Total number of chemical elements: " << material_table_device->total_number_of_chemical_elements_ << GGendl;
  GGcout("GGEMSMaterials", "PrintInfos", 0) << "Activated Materials: " << GGendl;
  GGcout("GGEMSMaterials", "PrintInfos", 0) << "-----------------------------------" << GGendl;
  for (std::size_t i = 0; i < material_table_device->number_of_materials_; ++i) {
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "* " << materials_.at(i) << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Number of chemical elements: " << static_cast<GGushort>(material_table_device->number_of_chemical_elements_[i]) << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Density: " << material_table_device->density_of_material_[i]/(GGEMSUnits::g/GGEMSUnits::cm3) << " g/cm3" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Photon cut: " << material_table_device->photon_energy_cut_[i]/GGEMSUnits::keV << " keV" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Electron cut: " << material_table_device->electron_energy_cut_[i]/GGEMSUnits::keV << " keV" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Positron cut: " << material_table_device->positron_energy_cut_[i]/GGEMSUnits::keV << " keV" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Radiation length: " << material_table_device->radiation_length_[i]/(GGEMSUnits::cm) << " cm" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Total atomic density: " << material_table_device->number_of_atoms_by_volume_[i]/(GGEMSUnits::mol/GGEMSUnits::cm3) << " atom/cm3" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Total electron density: " << material_table_device->number_of_electrons_by_volume_[i]/(GGEMSUnits::mol/GGEMSUnits::cm3) << " e-/cm3" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Chemical Elements:" << GGendl;
    for (GGuchar j = 0; j < material_table_device->number_of_chemical_elements_[i]; ++j) {
      GGushort const kIndexChemicalElement = material_table_device->index_of_chemical_elements_[i];
      GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + Z = " << static_cast<GGushort>(material_table_device->atomic_number_Z_[j+kIndexChemicalElement]) << GGendl;
      GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + fraction of chemical element = " << material_table_device->mass_fraction_[j+kIndexChemicalElement]/GGEMSUnits::PERCENT << " %" << GGendl;
      GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + Atomic number density = " << material_table_device->atomic_number_density_[j+kIndexChemicalElement]/(GGEMSUnits::mol/GGEMSUnits::cm3) << " atom/cm3" << GGendl;
      GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + Element abundance = " << 100.0*material_table_device->atomic_number_density_[j+kIndexChemicalElement]/material_table_device->number_of_atoms_by_volume_[i] << " %" << GGendl;
    }
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Energy loss fluctuation data:" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + Mean electron excitation energy: " << material_table_device->mean_excitation_energy_[i]/GGEMSUnits::eV << " eV" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + Log mean electron excitation energy: " << material_table_device->log_mean_excitation_energy_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + f1: " << material_table_device->f1_fluct_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + f2: " << material_table_device->f2_fluct_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + energy0: " << material_table_device->energy0_fluct_[i]/GGEMSUnits::eV << " eV" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + energy1: " << material_table_device->energy1_fluct_[i]/GGEMSUnits::eV << " eV" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + energy2: " << material_table_device->energy2_fluct_[i]/GGEMSUnits::eV << " eV" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + log energy 1: " << material_table_device->log_energy1_fluct_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + log energy 2: " << material_table_device->log_energy2_fluct_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Density correction data:" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + x0 = " << material_table_device->x0_density_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + x1 = " << material_table_device->x1_density_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + d0 = " << material_table_device->d0_density_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + -C = " << material_table_device->c_density_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + a = " << material_table_device->a_density_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + m = " << material_table_device->m_density_[i] << GGendl;
  }
  GGcout("GGEMSMaterials", "PrintInfos", 0) << GGendl;

  // Release the pointer, mandatory step!!!
  opencl_manager_.ReleaseDeviceBuffer(material_tables_, material_table_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterials::BuildMaterialTables(void)
{
  GGcout("GGEMSMaterials", "BuildMaterialTables", 3) << "Building the material tables..." << GGendl;

  // Allocating memory for material tables in OpenCL device
  material_tables_ = opencl_manager_.Allocate(nullptr, sizeof(GGEMSMaterialTables), CL_MEM_READ_WRITE);
  opencl_manager_.AddRAMMemory(sizeof(GGEMSMaterialTables));
  GGEMSPhantomNavigatorManager::GetInstance().AddPhantomNavigatorRAM(sizeof(GGEMSMaterialTables));

  // Getting the OpenCL pointer on material tables
  GGEMSMaterialTables* material_table_device = opencl_manager_.GetDeviceBuffer<GGEMSMaterialTables>(material_tables_, sizeof(GGEMSMaterialTables));

  // Get the number of activated materials
  material_table_device->number_of_materials_ = static_cast<GGuchar>(materials_.size());

  // Loop over the materials
  GGushort index_to_chemical_element = 0;
  for (std::size_t i = 0; i < materials_.size(); ++i) {
    // Getting the material infos from database
    GGEMSSingleMaterial const& kSingleMaterial = material_manager_.GetMaterial(materials_.at(i));

    // Storing infos about material
    material_table_device->number_of_chemical_elements_[i] = kSingleMaterial.nb_elements_;
    material_table_device->density_of_material_[i] = kSingleMaterial.density_;

    // Initialize some counters
    material_table_device->number_of_atoms_by_volume_[i] = 0.0f;
    material_table_device->number_of_electrons_by_volume_[i] = 0.0f;

    // Loop over the chemical elements by material
    for (GGuchar j = 0; j < kSingleMaterial.nb_elements_; ++j) {
      // Getting the chemical element
      GGEMSChemicalElement const& kChemicalElement = material_manager_.GetChemicalElement(kSingleMaterial.chemical_element_name_[j]);

      // Atomic number Z
      material_table_device->atomic_number_Z_[j+index_to_chemical_element] = kChemicalElement.atomic_number_Z_;

      // Mass fraction of element by material
      material_table_device->mass_fraction_[j+index_to_chemical_element] = kSingleMaterial.mixture_f_[j];

      // Atomic number density
      material_table_device->atomic_number_density_[j+index_to_chemical_element] = material_manager_.GetAtomicNumberDensity(materials_.at(i), j);

      // Increment density of atoms and electrons
      material_table_device->number_of_atoms_by_volume_[i] += material_table_device->atomic_number_density_[j+index_to_chemical_element];
      material_table_device->number_of_electrons_by_volume_[i] += material_table_device->atomic_number_density_[j+index_to_chemical_element] * kChemicalElement.atomic_number_Z_;
    }

    // Computing ionization params for a material
    GGEMSIonizationParamsMaterial ionization_params(&kSingleMaterial);
    material_table_device->mean_excitation_energy_[i] = ionization_params.GetMeanExcitationEnergy();
    material_table_device->log_mean_excitation_energy_[i] = ionization_params.GetLogMeanExcitationEnergy();
    material_table_device->x0_density_[i] = ionization_params.GetX0Density();
    material_table_device->x1_density_[i] = ionization_params.GetX1Density();
    material_table_device->d0_density_[i] = ionization_params.GetD0Density();
    material_table_device->c_density_[i] = ionization_params.GetCDensity();
    material_table_device->a_density_[i] = ionization_params.GetADensity();
    material_table_device->m_density_[i] = ionization_params.GetMDensity();

    // Energy fluctuation parameters
    material_table_device->f1_fluct_[i] = ionization_params.GetF1Fluct();
    material_table_device->f2_fluct_[i] = ionization_params.GetF2Fluct();
    material_table_device->energy0_fluct_[i] = ionization_params.GetEnergy0Fluct();
    material_table_device->energy1_fluct_[i] = ionization_params.GetEnergy1Fluct();
    material_table_device->energy2_fluct_[i] = ionization_params.GetEnergy2Fluct();
    material_table_device->log_energy1_fluct_[i] = ionization_params.GetLogEnergy1Fluct();
    material_table_device->log_energy2_fluct_[i] = ionization_params.GetLogEnergy2Fluct();

    // Radiation length
    material_table_device->radiation_length_[i] = material_manager_.GetRadiationLength(materials_.at(i));

    // Computing the access to chemical element by material
    material_table_device->index_of_chemical_elements_[i] = index_to_chemical_element;
    index_to_chemical_element += static_cast<GGushort>(material_table_device->number_of_chemical_elements_[i]);
  }

  // Storing number total of chemical elements
  material_table_device->total_number_of_chemical_elements_ = index_to_chemical_element;

  // Release the pointer, mandatory step!!!
  opencl_manager_.ReleaseDeviceBuffer(material_tables_, material_table_device);

  // Converting length cut to energy cut
  range_cuts_->ConvertCutsFromDistanceToEnergy(this);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSMaterials::GetDensity(std::string const& material_name) const
{
  // Getting the OpenCL pointer on material tables
  GGEMSMaterialTables* material_table_device = opencl_manager_.GetDeviceBuffer<GGEMSMaterialTables>(material_tables_, sizeof(GGEMSMaterialTables));

  // Get index of material
  std::vector<std::string>::const_iterator iter_mat = std::find(materials_.begin(), materials_.end(), material_name);
  if (iter_mat == materials_.end()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Material '" << material_name << "' not found!!!" << std::endl;
    GGEMSMisc::ThrowException("GGEMSMaterials", "GetDensity", oss.str());
  }
  ptrdiff_t const kIndex = std::distance(materials_.begin(), iter_mat);

  GGfloat const kDensity = material_table_device->density_of_material_[kIndex];

  opencl_manager_.ReleaseDeviceBuffer(material_tables_, material_table_device);

  return kDensity / GGEMSUnits::g * GGEMSUnits::cm3;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSMaterials::GetAtomicNumberDensity(std::string const& material_name) const
{
  // Getting the OpenCL pointer on material tables
  GGEMSMaterialTables* material_table_device = opencl_manager_.GetDeviceBuffer<GGEMSMaterialTables>(material_tables_, sizeof(GGEMSMaterialTables));

  // Get index of material
  ptrdiff_t const kIndex = GetMaterialIndex(material_name);

  GGfloat const kAtomicNumberDensity = material_table_device->atomic_number_density_[kIndex];

  opencl_manager_.ReleaseDeviceBuffer(material_tables_, material_table_device);

  return kAtomicNumberDensity /(GGEMSUnits::mol/GGEMSUnits::cm3);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSMaterials::GetEnergyCut(std::string const& material_name, std::string const& particle_type, GGfloat const& distance, std::string const& unit)
{
  // Set distance cut
  SetDistanceCut(particle_type, distance, unit);

  // Convert cut
  range_cuts_->ConvertCutsFromDistanceToEnergy(this);

  // Getting the OpenCL pointer on material tables
  GGEMSMaterialTables* material_table_device = opencl_manager_.GetDeviceBuffer<GGEMSMaterialTables>(material_tables_, sizeof(GGEMSMaterialTables));

  // Get index of material
  ptrdiff_t const kIndex = GetMaterialIndex(material_name);

  GGfloat energy_cut = 0.0f;
  if (particle_type == "gamma") {
    energy_cut = material_table_device->photon_energy_cut_[kIndex];
  }
  else if (particle_type == "e+") {
    energy_cut = material_table_device->positron_energy_cut_[kIndex];
  }
  else if (particle_type == "e-") {
    energy_cut = material_table_device->electron_energy_cut_[kIndex];
  }

  opencl_manager_.ReleaseDeviceBuffer(material_tables_, material_table_device);

  return energy_cut / GGEMSUnits::keV;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterials::Initialize(void)
{
  GGcout("GGEMSMaterials", "Initialize", 3) << "Initializing the materials..." << GGendl;

  // Build material table depending on physics and cuts
  BuildMaterialTables();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMaterials* create_ggems_materials(void)
{
  return new(std::nothrow) GGEMSMaterials;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void add_material_ggems_materials(GGEMSMaterials* materials, char const* material_name)
{
  materials->AddMaterial(material_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void initialize_ggems_materials(GGEMSMaterials* materials)
{
  materials->Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_material_properties_ggems_materials(GGEMSMaterials* materials)
{
  materials->PrintInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat get_density_ggems_materials(GGEMSMaterials* materials, char const* material_name)
{
  return materials->GetDensity(material_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat get_energy_cut_ggems_materials(GGEMSMaterials* materials, char const* material_name, char const* particle_type, GGfloat const distance, char const* unit)
{
  return materials->GetEnergyCut(material_name, particle_type, distance, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat get_atomic_number_density_ggems_materials(GGEMSMaterials* materials, char const* material_name)
{
  return materials->GetAtomicNumberDensity(material_name);
}
