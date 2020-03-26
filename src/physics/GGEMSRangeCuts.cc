/*!
  \file GGEMSRangeCuts.cc

  \brief GGEMS class storing and converting the cut in energy cut

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday March 18, 2020
*/

#include "GGEMS/physics/GGEMSRangeCuts.hh"

#include "GGEMS/physics/GGEMSRangeCutsManager.hh"
#include "GGEMS/physics/GGEMSLogEnergyTable.hh"
#include "GGEMS/physics/GGEMSProcessesManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSRangeCuts::GGEMSRangeCuts(void)
: min_energy_(0.0f),
  max_energy_(10.0f*GGEMSUnits::GeV),
  number_of_bins_(300),
  length_cut_photon_(GGEMSDefaultParams::PHOTON_CUT),
  length_cut_electron_(GGEMSDefaultParams::ELECTRON_CUT),
  length_cut_positron_(GGEMSDefaultParams::POSITRON_CUT)
{
  GGcout("GGEMSRangeCuts", "GGEMSRangeCuts", 3) << "Allocation of GGEMSRangeCuts..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSRangeCuts::~GGEMSRangeCuts(void)
{
  GGcout("GGEMSRangeCuts", "~GGEMSRangeCuts", 3) << "Deallocation of GGEMSRangeCuts..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCuts::SetPhotonLengthCut(GGfloat const& cut)
{
  length_cut_photon_ = cut;
  if (length_cut_photon_ < GGEMSDefaultParams::PHOTON_CUT) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Cut length for photon " << cut << " mm is too small!!! Minimum value is " << GGEMSDefaultParams::PHOTON_CUT << " mm!!!";
    GGEMSMisc::ThrowException("GGEMSRangeCuts", "SetPhotonLengthCut", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCuts::SetElectronLengthCut(GGfloat const& cut)
{
  length_cut_electron_ = cut;
  if (length_cut_electron_ < GGEMSDefaultParams::ELECTRON_CUT) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Cut length for electron " << cut << " mm is too small!!! Minimum value is " << GGEMSDefaultParams::ELECTRON_CUT << " mm!!!";
    GGEMSMisc::ThrowException("GGEMSRangeCuts", "SetElectronLengthCut", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCuts::SetPositronLengthCut(GGfloat const& cut)
{
  length_cut_positron_ = cut;
  if (length_cut_positron_ < GGEMSDefaultParams::POSITRON_CUT) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Cut length for positron " << cut << " mm is too small!!! Minimum value is " << GGEMSDefaultParams::POSITRON_CUT << " mm!!!";
    GGEMSMisc::ThrowException("GGEMSRangeCuts", "SetPositronLengthCut", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSRangeCuts::ConvertToEnergy(GGEMSMaterialTables* material_table, GGuchar const& index_mat, std::string const& particle_name)
{
  // Checking the particle_name
  if (particle_name != "gamma" && particle_name != "e+" && particle_name != "e-") {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Wrong particle name, the particle can be only:" << std::endl;
    oss << "    - gamma" << std::endl;
    oss << "    - e+" << std::endl;
    oss << "    - e-" << std::endl;
    GGEMSMisc::ThrowException("GGEMSRangeCuts", "ConvertToEnergy", oss.str());
  }

  // Set cut depending on particle
  GGfloat cut = 0.0f;
  if (particle_name == "gamma") {
    cut = length_cut_photon_;
  }
  else if (particle_name == "e-") {
    cut = length_cut_electron_;
  }
  else if (particle_name == "e+") {
    cut = length_cut_positron_;
  }

  // Reset tables
  loss_table_dedx_table_elements_.clear();

  // init vars
  GGfloat kinetic_energy_cut = 0.0f;

  // Build dE/dX loss table for each elements
  BuildLossTableElements(material_table, index_mat, particle_name);

  // Absorption table for photon and loss table for electron/positron
  if (particle_name == "gamma") {
    BuildAbsorptionLengthTable(material_table, index_mat);
  }
  else if (particle_name == "e+" || particle_name == "e-") {
    BuildMaterialLossTable(material_table, index_mat);
  }

  // Convert Range Cut ro Kinetic Energy Cut
  kinetic_energy_cut = ConvertLengthToEnergyCut(range_table_material_, cut);

  if (particle_name == "e-" || particle_name == "e+" ) {
    GGfloat constexpr kTune = 0.025f * GGEMSUnits::mm * GGEMSUnits::g / GGEMSUnits::cm3;
    GGfloat constexpr kLowEnergy = 30.f * GGEMSUnits::keV;
    if (kinetic_energy_cut < kLowEnergy) {
      kinetic_energy_cut /= ( 1.0f + (1.0f - kinetic_energy_cut/kLowEnergy) * kTune / (cut*material_table->density_of_material_[index_mat]));
    }
  }

  if (kinetic_energy_cut < min_energy_) {
    kinetic_energy_cut = min_energy_;
  }
  else if (kinetic_energy_cut > max_energy_) {
    kinetic_energy_cut = max_energy_;
  }

  // Storing cut
  if (particle_name == "gamma") {
    material_table->photon_energy_cut_[index_mat] = kinetic_energy_cut;
  }
  else if (particle_name == "e-") {
    material_table->electron_energy_cut_[index_mat] = kinetic_energy_cut;
  }
  else if (particle_name == "e+") {
    material_table->positron_energy_cut_[index_mat] = kinetic_energy_cut;
  }

  return kinetic_energy_cut;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCuts::BuildAbsorptionLengthTable(GGEMSMaterialTables* material_table, GGuchar const& index_mat)
{
  // Allocation buffer for absorption length
  range_table_material_.reset(new GGEMSLogEnergyTable(min_energy_, max_energy_, number_of_bins_));

  // Get the number of elements in material
  GGuchar const kNumberOfElements = material_table->number_of_chemical_elements_[index_mat];

  // Get index offset to element
  GGushort const kIndexOffset = material_table->index_of_chemical_elements_[index_mat];

  // Loop over the bins in the table
  for (GGushort i = 0; i < number_of_bins_; ++i) {
    GGfloat sigma = 0.0f;

    // Loop over the number of elements in material
    for (GGuchar j = 0; j < kNumberOfElements; ++j) {
      sigma += material_table->atomic_number_density_[j+kIndexOffset] * loss_table_dedx_table_elements_.at(j)->GetLossTableData(i);
    }

    // Storing value
    range_table_material_->SetValue(i, 5.0f/sigma);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCuts::BuildMaterialLossTable(GGEMSMaterialTables* material_table, GGuchar const& index_mat)
{
  // calculate parameters of the low energy part first
  std::size_t i = 0;
  std::vector<GGfloat> loss;

  // Allocation buffer for absorption length
  range_table_material_.reset(new GGEMSLogEnergyTable(min_energy_, max_energy_, number_of_bins_));

  // Get the number of elements in material
  GGuchar const kNumberOfElements = material_table->number_of_chemical_elements_[index_mat];

  // Get index offset to element
  GGushort const kIndexOffset = material_table->index_of_chemical_elements_[index_mat];

  for (GGushort i = 0; i <= number_of_bins_; ++i) {
    GGfloat value = 0.0f;

    for (GGuchar j = 0; j < kNumberOfElements; ++j) {
      value += material_table->atomic_number_density_[j+kIndexOffset] * loss_table_dedx_table_elements_.at(j)->GetLossTableData(i);
    }
    loss.push_back(value);
  }

  // Integrate with Simpson formula with logarithmic binning
  GGfloat dltau = 1.0f;
  if (min_energy_ > 0.f) {
    GGfloat ltt = log(max_energy_/min_energy_);
    dltau = ltt/number_of_bins_;
  }

  GGfloat s0 = 0.0f;
  GGfloat value = 0.0f;
  for (GGushort i = 0; i <= number_of_bins_; ++i) {
    GGfloat t = range_table_material_->GetLowEdgeEnergy(i);
    GGfloat q = t / loss.at(i);

    if (i == 0) {
      s0 += 0.5f*q;
    }
    else {
      s0 += q;
    }

    if (i == 0) {
      value = (s0 + 0.5f*q) * dltau;
    }
    else {
      value = (s0 - 0.5f*q) * dltau;
    }
    range_table_material_->SetValue(i, value);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCuts::BuildLossTableElements(GGEMSMaterialTables* material_table, GGuchar const& index_mat, std::string const& particle_name)
{
  // Getting number of elements in material
  GGuchar const kNumberOfElements = material_table->number_of_chemical_elements_[index_mat];

  // Building cross section tables for each elements in material
  loss_table_dedx_table_elements_.reserve(kNumberOfElements);

  // Get index offset to element
  GGushort const kIndexOffset = material_table->index_of_chemical_elements_[index_mat];

  // Filling cross section table
  for (GGuchar i = 0; i < kNumberOfElements; ++i) {
    GGfloat value = 0.0f;
    std::shared_ptr<GGEMSLogEnergyTable> log_energy_table_element(new GGEMSLogEnergyTable(min_energy_, max_energy_, number_of_bins_));

    // Getting atomic number
    GGuchar const kZ = material_table->atomic_number_Z_[i+kIndexOffset];

    for (GGushort j = 0; j < number_of_bins_; ++j) {
      if (particle_name == "gamma") {
        value = ComputePhotonCrossSection(kZ, log_energy_table_element->GetEnergy(j));
      }
      else if (particle_name == "e-") {
        value = ComputeLossElectron(kZ, log_energy_table_element->GetEnergy(j));
      }
      else if (particle_name == "e+") {
        value = ComputeLossPositron(kZ, log_energy_table_element->GetEnergy(j));
      }
      log_energy_table_element->SetValue(j, value);
    }

    // Storing the log table
    loss_table_dedx_table_elements_.push_back(log_energy_table_element);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSRangeCuts::ComputePhotonCrossSection(GGuchar const& atomic_number, GGfloat const& energy) const
{
  // Compute the "absorption" cross section of the photon "absorption"
  // cross section means here the sum of the cross sections of the
  // pair production, Compton scattering and photoelectric processes
  constexpr GGfloat kT1keV = 1.0f*GGEMSUnits::keV;
  constexpr GGfloat kT200keV = 200.f*GGEMSUnits::keV;
  constexpr GGfloat kT100MeV = 100.f*GGEMSUnits::MeV;

  GGfloat gZ = -1.0f;
  GGfloat s200keV = 0.0f;
  GGfloat s1keV = 0.0f;
  GGfloat tmin = 0.0f;
  GGfloat tlow = 0.0f;
  GGfloat smin = 0.0f;
  GGfloat slow = 0.0f;
  GGfloat cmin = 0.0f;
  GGfloat clow = 0.0f;
  GGfloat chigh = 0.0f;

  // Compute Z dependent quantities in the case of a new AtomicNumber
  if (std::abs(atomic_number - gZ) > 0.1f) {
    gZ = atomic_number;
    GGfloat const kZsquare = gZ*gZ;
    GGfloat const kZlog = log(gZ);
    GGfloat const kZlogsquare = kZlog*kZlog;

    s200keV = (0.2651f - 0.1501f*kZlog + 0.02283f*kZlogsquare) * kZsquare;
    tmin = (0.552f + 218.5f/gZ + 557.17f/kZsquare) * GGEMSUnits::MeV;
    smin = (0.01239f + 0.005585f*kZlog - 0.000923f*kZlogsquare) * exp(1.5f*kZlog);
    cmin = log(s200keV/smin) / (log(tmin/kT200keV) * log(tmin/kT200keV));
    tlow = 0.2f * exp(-7.355f/sqrt(gZ)) * GGEMSUnits::MeV;
    slow = s200keV * exp(0.042f*gZ*log(kT200keV/tlow)*log(kT200keV/tlow));
    s1keV = 300.0f*kZsquare;
    clow = log(s1keV/slow) / log(tlow/kT1keV);

    chigh = (7.55e-5f - 0.0542e-5f*gZ ) * kZsquare * gZ/log(kT100MeV/tmin);
  }

  // Calculate the cross section (using an approximate empirical formula)
  GGfloat xs = 0.0f;
  if (energy < tlow){
    if (energy < kT1keV) {
      xs = slow * exp(clow*log(tlow/kT1keV));
    }
    else {
      xs = slow * exp(clow*log(tlow/energy));
    }
  }
  else if (energy < kT200keV) {
    xs = s200keV * exp(0.042f*gZ*log(kT200keV/energy)*log(kT200keV/energy));
  }
  else if(energy < tmin) {
    xs = smin * exp(cmin*log(tmin/energy)*log(tmin/energy));
  }
  else {
    xs = smin + chigh*log(energy/tmin);
  }

  return xs * GGEMSUnits::b;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSRangeCuts::ComputeLossElectron(GGuchar const& atomic_number, GGfloat const& energy) const
{
  return 0.0f;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSRangeCuts::ComputeLossPositron(GGuchar const& atomic_number, GGfloat const& energy) const
{
  return 0.0f;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSRangeCuts::ConvertLengthToEnergyCut(std::shared_ptr<GGEMSLogEnergyTable> range_table, GGfloat const& length_cut) const
{
  GGfloat const kEpsilon = 0.01f;

  // Find max. range and the corresponding energy (rmax,Tmax)
  GGfloat rmax = -1.e10f * GGEMSUnits::mm;
  GGfloat t1 = min_energy_;
  GGfloat r1 = range_table->GetLossTableData(0);
  GGfloat t2 = max_energy_;

  // Check length_cut < r1
  if (length_cut < r1) return t1;

  // scan range vector to find nearest bin
  // suppose that r(ti) > r(tj) if ti >tj
  for (GGushort i = 0; i < number_of_bins_; ++i) {
    GGfloat t = range_table->GetLowEdgeEnergy(i);
    GGfloat r = range_table->GetLossTableData(i);

    if (r > rmax) rmax = r;
    if (r < length_cut) {
      t1 = t;
      r1 = r;
    }
    else if (r > length_cut) {
      t2 = t;
      break;
    }
  }

  // check cut in length is smaller than range max
  if (length_cut >= rmax) return max_energy_;

  // convert range to energy
  GGfloat t3 = sqrt(t1*t2);
  GGfloat r3 = range_table->GetLossTableValue(t3);

  while (fabs(1.0f - r3/length_cut) > kEpsilon) {
    if (length_cut <= r3) {
      t2 = t3;
    }
    else {
      t1 = t3;
    }

    t3 = sqrt(t1*t2);
    r3 = range_table->GetLossTableValue(t3);
  }

  return t3;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCuts::ConvertCutsFromLengthToEnergy(std::shared_ptr<GGEMSMaterials> materials)
{
  // Getting the minimum for the loss/cross section table in process manager
  GGEMSProcessesManager& process_manager = GGEMSProcessesManager::GetInstance();
  min_energy_ = process_manager.GetCrossSectionTableMinEnergy();

  // Get data from OpenCL device
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  std::shared_ptr<cl::Buffer> material_table_device = materials->GetMaterialTables();
  GGEMSMaterialTables* material_table_host = opencl_manager.GetDeviceBuffer<GGEMSMaterialTables>(material_table_device, sizeof(GGEMSMaterialTables));

  // Loop over materials
  for (GGuchar i = 0; i < material_table_host->number_of_materials_; ++i) {
    // Get the name of material

    // Convert photon cuts
    GGfloat energy_cut_photon = ConvertToEnergy(material_table_host, i, "gamma");

    // Convert electron cuts
    //ConvertElectron(material_table_host, i);

    // Storing the cuts in map
    energy_cuts_photon_.insert(std::make_pair(materials->GetMaterialName(i), energy_cut_photon));
  }

  // Release pointer
  opencl_manager.ReleaseDeviceBuffer(material_table_device, material_table_host);
}
