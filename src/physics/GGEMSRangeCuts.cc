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
  max_energy_(10.0f*GeV),
  number_of_bins_(300),
  distance_cut_photon_(PHOTON_DISTANCE_CUT),
  distance_cut_electron_(ELECTRON_DISTANCE_CUT),
  distance_cut_positron_(POSITRON_DISTANCE_CUT),
  number_of_tables_(0)
{
  GGcout("GGEMSRangeCuts", "GGEMSRangeCuts", 3) << "GGEMSRangeCuts creating..." << GGendl;

  range_table_material_ = nullptr;
  loss_table_dedx_table_elements_ = nullptr;

  GGcout("GGEMSRangeCuts", "GGEMSRangeCuts", 3) << "GGEMSRangeCuts created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSRangeCuts::~GGEMSRangeCuts(void)
{
  GGcout("GGEMSRangeCuts", "~GGEMSRangeCuts", 3) << "GGEMSRangeCuts erasing..." << GGendl;

  if (range_table_material_) {
    delete range_table_material_;
    range_table_material_ = nullptr;
  }

  if (loss_table_dedx_table_elements_) {
    for (GGsize i = 0; i < number_of_tables_; ++i) {
      delete loss_table_dedx_table_elements_[i];
      loss_table_dedx_table_elements_[i] = nullptr;
    }
    delete[] loss_table_dedx_table_elements_;
    loss_table_dedx_table_elements_ = nullptr;
  }

  GGcout("GGEMSRangeCuts", "~GGEMSRangeCuts", 3) << "GGEMSRangeCuts erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCuts::SetPhotonDistanceCut(GGfloat const& cut)
{
  distance_cut_photon_ = cut;
  if (distance_cut_photon_ < PHOTON_DISTANCE_CUT) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Distance cut for photon " << cut << " mm is too small!!! Minimum value is " << PHOTON_DISTANCE_CUT << " mm!!!";
    GGEMSMisc::ThrowException("GGEMSRangeCuts", "SetPhotonDistanceCut", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCuts::SetElectronDistanceCut(GGfloat const& cut)
{
  distance_cut_electron_ = cut;
  if (distance_cut_electron_ < ELECTRON_DISTANCE_CUT) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Distance cut for electron " << cut << " mm is too small!!! Minimum value is " << ELECTRON_DISTANCE_CUT << " mm!!!";
    GGEMSMisc::ThrowException("GGEMSRangeCuts", "SetElectronLengthCut", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCuts::SetPositronDistanceCut(GGfloat const& cut)
{
  distance_cut_positron_ = cut;
  if (distance_cut_positron_ < POSITRON_DISTANCE_CUT) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Cut distance for positron " << cut << " mm is too small!!! Minimum value is " << POSITRON_DISTANCE_CUT << " mm!!!";
    GGEMSMisc::ThrowException("GGEMSRangeCuts", "SetPositronLengthCut", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSRangeCuts::ConvertToEnergy(GGEMSMaterialTables* material_table, GGushort const& index_mat, std::string const& particle_name)
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
    cut = distance_cut_photon_;
  }
  else if (particle_name == "e-") {
    cut = distance_cut_electron_;
  }
  else if (particle_name == "e+") {
    cut = distance_cut_positron_;
  }

  // Reset tables
  if (loss_table_dedx_table_elements_) {
    for (GGsize i = 0; i < number_of_tables_; ++i) {
      delete loss_table_dedx_table_elements_[i];
      loss_table_dedx_table_elements_[i] = nullptr;
    }
    delete[] loss_table_dedx_table_elements_;
    loss_table_dedx_table_elements_ = nullptr;
  }

  // init vars
  GGfloat kinetic_energy_cut = 0.0f;

  // Build dE/dX loss table for each elements
  BuildElementsLossTable(material_table, index_mat, particle_name);

  // Absorption table for photon and loss table for electron/positron
  if (particle_name == "gamma") {
    BuildAbsorptionLengthTable(material_table, index_mat);
  }
  else if (particle_name == "e+" || particle_name == "e-") {
    BuildMaterialLossTable(material_table, index_mat);
  }

  // Convert Range Cut ro Kinetic Energy Cut
  //kinetic_energy_cut = ConvertLengthToEnergyCut(range_table_material_, cut);
  kinetic_energy_cut = ConvertLengthToEnergyCut(cut);

  if (particle_name == "e-" || particle_name == "e+" ) {
    GGfloat constexpr kTune = 0.025f * mm * g / cm3;
    GGfloat constexpr kLowEnergy = 30.f * keV;
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

void GGEMSRangeCuts::BuildAbsorptionLengthTable(GGEMSMaterialTables* material_table, GGushort const& index_mat)
{
  // Allocation buffer for absorption length
  if (range_table_material_) delete range_table_material_;
  range_table_material_ = new GGEMSLogEnergyTable(min_energy_, max_energy_, number_of_bins_);

  // Get the number of elements in material
  GGsize number_of_elements = material_table->number_of_chemical_elements_[index_mat];

  // Get index offset to element
  GGsize index_of_offset = material_table->index_of_chemical_elements_[index_mat];

  // Loop over the bins in the table
  for (GGsize i = 0; i < number_of_bins_; ++i) {
    GGfloat sigma = 0.0f;

    // Loop over the number of elements in material
    for (GGsize j = 0; j < number_of_elements; ++j) {
      sigma += material_table->atomic_number_density_[j+index_of_offset] * loss_table_dedx_table_elements_[j]->GetLossTableData(i);
    }

    // Storing value
    range_table_material_->SetValue(i, 5.0f/sigma);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCuts::BuildMaterialLossTable(GGEMSMaterialTables* material_table, GGushort const& index_mat)
{
  // calculate parameters of the low energy part first
  std::vector<GGfloat> loss;

  // Allocation buffer for absorption length
  if (range_table_material_) delete range_table_material_;
  range_table_material_ = new GGEMSLogEnergyTable(min_energy_, max_energy_, number_of_bins_);

  // Get the number of elements in material
  GGsize number_of_elements = static_cast<GGsize>(material_table->number_of_chemical_elements_[index_mat]);

  // Get index offset to element
  GGsize index_of_offset = material_table->index_of_chemical_elements_[index_mat];

  for (GGsize i = 0; i <= number_of_bins_; ++i) {
    GGfloat value = 0.0f;

    for (GGsize j = 0; j < number_of_elements; ++j) {
      value += material_table->atomic_number_density_[j+index_of_offset] * loss_table_dedx_table_elements_[j]->GetLossTableData(i);
    }
    loss.push_back(value);
  }

  // Integrate with Simpson formula with logarithmic binning
  GGfloat dltau = 1.0f;
  if (min_energy_ > 0.f) {
    GGfloat ltt = logf(max_energy_/min_energy_);
    dltau = ltt/static_cast<GGfloat>(number_of_bins_);
  }

  GGfloat s0 = 0.0f;
  GGfloat value = 0.0f;
  for (GGsize i = 0; i <= number_of_bins_; ++i) {
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

void GGEMSRangeCuts::BuildElementsLossTable(GGEMSMaterialTables* material_table, GGushort const& index_mat, std::string const& particle_name)
{
  // Getting number of elements in material
  number_of_tables_ = static_cast<GGsize>(material_table->number_of_chemical_elements_[index_mat]);

  // Building cross section tables for each elements in material
  loss_table_dedx_table_elements_ = new GGEMSLogEnergyTable*[number_of_tables_];

  // Get index offset to element
  GGsize index_of_offset = material_table->index_of_chemical_elements_[index_mat];

  // Filling cross section table
  for (GGsize i = 0; i < number_of_tables_; ++i) {
    GGfloat value = 0.0f;
    GGEMSLogEnergyTable* log_energy_table_element = new GGEMSLogEnergyTable(min_energy_, max_energy_, number_of_bins_);

    // Getting atomic number
    GGuchar const kZ = material_table->atomic_number_Z_[i+index_of_offset];

    for (GGsize j = 0; j < number_of_bins_; ++j) {
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

    // Storing the logf table
    loss_table_dedx_table_elements_[i] = log_energy_table_element;
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
  GGfloat t1keV = 1.0f*keV;
  GGfloat t200keV = 200.f*keV;
  GGfloat t100MeV = 100.f*MeV;

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
    GGfloat const kZlog = logf(gZ);
    GGfloat const kZlogsquare = kZlog*kZlog;

    s200keV = (0.2651f - 0.1501f*kZlog + 0.02283f*kZlogsquare) * kZsquare;
    tmin = (0.552f + 218.5f/gZ + 557.17f/kZsquare) * MeV;
    smin = (0.01239f + 0.005585f*kZlog - 0.000923f*kZlogsquare) * expf(1.5f*kZlog);
    cmin = logf(s200keV/smin) / (logf(tmin/t200keV) * logf(tmin/t200keV));
    tlow = 0.2f * expf(-7.355f/sqrtf(gZ)) * MeV;
    slow = s200keV * expf(0.042f*gZ*logf(t200keV/tlow)*logf(t200keV/tlow));
    s1keV = 300.0f*kZsquare;
    clow = logf(s1keV/slow) / logf(tlow/t1keV);

    chigh = (7.55e-5f - 0.0542e-5f*gZ ) * kZsquare * gZ/logf(t100MeV/tmin);
  }

  // Calculate the cross section (using an approximate empirical formula)
  GGfloat xs = 0.0f;
  if (energy < tlow){
    if (energy < t1keV) {
      xs = slow * expf(clow*logf(tlow/t1keV));
    }
    else {
      xs = slow * expf(clow*logf(tlow/energy));
    }
  }
  else if (energy < t200keV) {
    xs = s200keV * expf(0.042f*gZ*logf(t200keV/energy)*logf(t200keV/energy));
  }
  else if(energy < tmin) {
    xs = smin * expf(cmin*logf(tmin/energy)*logf(tmin/energy));
  }
  else {
    xs = smin + chigh*logf(energy/tmin);
  }

  return xs * b;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSRangeCuts::ComputeLossElectron(GGuchar const& atomic_number, GGfloat const& energy) const
{
  GGfloat cbr1 = 0.02f;
  GGfloat cbr2 = -5.7e-5f;
  GGfloat cbr3 = 1.f;
  GGfloat cbr4 = 0.072f;

  GGfloat Tlow = 10.f*keV;
  GGfloat Thigh = 1.0f*GeV;

  GGfloat Mass = ELECTRON_MASS_C2;
  GGfloat bremfactor = 0.1f;

  GGfloat eZ = -1.f;
  GGfloat taul = 0.0f;
  GGfloat ionpot = 0.0f;
  GGfloat ionpotlog = -1.0e-10f;

  //  calculate dE/dx for electrons
  if (fabsf(atomic_number - eZ) > 0.1f) {
    eZ = atomic_number;
    taul = Tlow/Mass;
    ionpot = 1.6e-5f*MeV * expf(0.9f*logf(eZ)) / Mass;
    ionpotlog = logf(ionpot);
  }

  GGfloat tau = energy / Mass;
  GGfloat dEdx = 0.0f;

  if (tau < taul) {
    GGfloat t1 = taul + 1.0f;
    GGfloat t2 = taul + 2.0f;
    GGfloat tsq = taul*taul;
    GGfloat beta2 = taul*t2 / (t1*t1);
    GGfloat f = 1.f - beta2 + logf(tsq/2.0f) + (0.5f + 0.25f*tsq + (1.f+2.f*taul) * logf(0.5f)) / (t1*t1);
    dEdx = (logf(2.f*taul + 4.f) - 2.f*ionpotlog + f) / beta2;
    dEdx = TWO_PI_MC2_RCL2 * eZ * dEdx;
    GGfloat clow = dEdx * sqrtf(taul);
    dEdx = clow / sqrtf(energy / Mass);
  }
  else {
    GGfloat t1 = tau + 1.f;
    GGfloat t2 = tau + 2.f;
    GGfloat tsq = tau*tau;
    GGfloat beta2 = tau*t2 / ( t1*t1 );
    GGfloat f = 1.f - beta2 + logf(tsq/2.f) + (0.5f + 0.25f*tsq + (1.f + 2.f*tau)*logf(0.5f)) / (t1*t1);
    dEdx = (logf(2.f*tau + 4.f) - 2.f*ionpotlog + f) / beta2;
    dEdx = TWO_PI_MC2_RCL2 * eZ * dEdx;

    // loss from bremsstrahlung follows
    GGfloat cbrem = (cbr1 + cbr2*eZ) * (cbr3 + cbr4*logf(energy/Thigh));
    cbrem = eZ * (eZ+1.f) * cbrem * tau / beta2;
    cbrem *= bremfactor;
    dEdx += TWO_PI_MC2_RCL2 * cbrem;
  }

  return dEdx;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSRangeCuts::ComputeLossPositron(GGuchar const& atomic_number, GGfloat const& energy) const
{
  GGfloat cbr1 = 0.02f;
  GGfloat cbr2 = -5.7e-5f;
  GGfloat cbr3 = 1.f;
  GGfloat cbr4 = 0.072f;

  GGfloat Tlow = 10.f*keV;
  GGfloat Thigh = 1.0f*GeV;

  GGfloat Mass = POSITRON_MASS_C2;
  GGfloat bremfactor = 0.1f;

  GGfloat Z = -1.f;
  GGfloat taul = 0.0f;
  GGfloat ionpot = 0.0f;
  GGfloat ionpotlog = -1.0e-10f;

  //  calculate dE/dx for electrons
  if (fabsf(atomic_number-Z) > 0.1f) {
    Z = atomic_number;
    taul = Tlow/Mass;
    ionpot = 1.6e-5f*MeV * expf(0.9f*logf(Z))/Mass;
    ionpotlog = logf(ionpot);
  } 

  GGfloat tau = energy / Mass;
  GGfloat dEdx;

  if (tau < taul) {
    GGfloat t1 = taul + 1.f;
    GGfloat t2 = taul + 2.f;
    GGfloat tsq = taul*taul;
    GGfloat beta2 = taul*t2 / (t1*t1);
    GGfloat f = 2.f * logf(taul) -(6.f*taul+1.5f*tsq-taul*(1.f-tsq/3.f)/t2-tsq*(0.5f-tsq/12.f)/(t2*t2))/(t1*t1);
    dEdx = (logf(2.f*taul+4.f)-2.f*ionpotlog+f)/beta2;
    dEdx = TWO_PI_MC2_RCL2 * Z * dEdx;
    GGfloat clow = dEdx * sqrtf(taul);
    dEdx = clow/sqrtf(energy/Mass);
  }
  else {
    GGfloat t1 = tau + 1.f;
    GGfloat t2 = tau + 2.f;
    GGfloat tsq = tau*tau;
    GGfloat beta2 = tau*t2/(t1*t1);
    GGfloat f = 2.f*logf(tau)-(6.f*tau+1.5f*tsq-tau*(1.f-tsq/3.f)/t2-tsq*(0.5f-tsq/12.f)/(t2*t2))/(t1*t1);
    dEdx = (logf(2.f*tau+4.f)-2.f*ionpotlog+f)/beta2;
    dEdx = TWO_PI_MC2_RCL2 * Z * dEdx;

    // loss from bremsstrahlung follows
    GGfloat cbrem = (cbr1+cbr2*Z)*(cbr3+cbr4*logf(energy/Thigh));
    cbrem = Z*(Z+1.f)*cbrem*tau/beta2;
    cbrem *= bremfactor;
    dEdx += TWO_PI_MC2_RCL2 * cbrem;
  }
  return dEdx;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSRangeCuts::ConvertLengthToEnergyCut(GGfloat const& length_cut) const
{
  GGfloat epsilon = 0.01f;

  // Find max. range and the corresponding energy (rmax,Tmax)
  GGfloat rmax = -1.e10f * mm;
  GGfloat t1 = min_energy_;
  GGfloat r1 = range_table_material_->GetLossTableData(0);
  GGfloat t2 = max_energy_;

  // Check length_cut < r1
  if (length_cut < r1) return t1;

  // scan range vector to find nearest bin
  // suppose that r(ti) > r(tj) if ti >tj
  for (GGushort i = 0; i < number_of_bins_; ++i) {
    GGfloat t = range_table_material_->GetLowEdgeEnergy(i);
    GGfloat r = range_table_material_->GetLossTableData(i);

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
  GGfloat t3 = sqrtf(t1*t2);
  GGfloat r3 = range_table_material_->GetLossTableValue(t3);

  while (fabsf(1.0f - r3/length_cut) > epsilon) {
    if (length_cut <= r3) {
      t2 = t3;
    }
    else {
      t1 = t3;
    }

    t3 = sqrtf(t1*t2);
    r3 = range_table_material_->GetLossTableValue(t3);
  }

  return t3;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCuts::ConvertCutsFromDistanceToEnergy(GGEMSMaterials* materials)
{
  // Getting the minimum for the loss/cross section table in process manager
  GGEMSProcessesManager& process_manager = GGEMSProcessesManager::GetInstance();
  min_energy_ = process_manager.GetCrossSectionTableMinEnergy();

  // Get data from OpenCL device
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get number of activated device
  GGsize number_activated_devices = opencl_manager.GetNumberOfActivatedDevice();

  for (GGsize j = 0; j < number_activated_devices; ++j) {
    cl::Buffer* material_table = materials->GetMaterialTables(j);
    GGEMSMaterialTables* material_table_device = opencl_manager.GetDeviceBuffer<GGEMSMaterialTables>(material_table, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSMaterialTables), j);

    // Loop over materials
    for (GGushort i = 0; i < material_table_device->number_of_materials_; ++i) {
      // Get the name of material

      // Convert photon cuts
      GGfloat energy_cut_photon = ConvertToEnergy(material_table_device, i, "gamma");

      // Convert electron cuts
      GGfloat energy_cut_electron = ConvertToEnergy(material_table_device, i, "e-");

      // Convert electron cuts
      GGfloat energy_cut_positron = ConvertToEnergy(material_table_device, i, "e+");

      // Storing the cuts in map
      energy_cuts_photon_.insert(std::make_pair(materials->GetMaterialName(i), energy_cut_photon));
      energy_cuts_electron_.insert(std::make_pair(materials->GetMaterialName(i), energy_cut_electron));
      energy_cuts_positron_.insert(std::make_pair(materials->GetMaterialName(i), energy_cut_positron));
    }

    // Release pointer
    opencl_manager.ReleaseDeviceBuffer(material_table, material_table_device, j);
  }
}
