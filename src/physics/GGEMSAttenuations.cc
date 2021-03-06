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
  \file GGEMSAttenuations.cc

  \brief Class computing and storing attenuation coefficient

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author Mateo VILLA <ingmatvillaa@gmail.com>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Tuesday January 18, 2022
*/

#include "GGEMS/physics/GGEMSAttenuations.hh"
#include "GGEMS/physics/GGEMSMuDataConstants.hh"
#include "GGEMS/materials/GGEMSMaterials.hh"
#include "GGEMS/physics/GGEMSCrossSections.hh"
#include "GGEMS/maths/GGEMSMathAlgorithms.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSAttenuations::GGEMSAttenuations(GGEMSMaterials* materials, GGEMSCrossSections* cross_sections)
{
  GGcout("GGEMSAttenuations", "GGEMSAttenuations", 3) << "GGEMSAttenuations creating..." << GGendl;

  // Get the number of activated device
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  number_activated_devices_ = opencl_manager.GetNumberOfActivatedDevice();

  // Allocating buffers
  energies_ = new GGfloat[GGEMSMuDataConstants::kMuNbEnergies];
  mu_ = new GGfloat[GGEMSMuDataConstants::kMuNbEnergies];
  mu_en_ = new GGfloat[GGEMSMuDataConstants::kMuNbEnergies];
  mu_index_ = new GGint[GGEMSMuDataConstants::kMuNbElements];

  materials_ = materials;
  cross_sections_ = cross_sections;

  attenuations_host_ = new GGEMSMuMuEnData();
  mu_tables_ = nullptr;

  GGint index_table = 0;
  GGint index_data = 0;

  // Loading all attenuations values
  for (GGint i = 0; i <= GGEMSMuDataConstants::kMuNbElements; ++i) {
    GGint nb_energies = GGEMSMuDataConstants::kMuNbEnergyBins[i];
    mu_index_[i] = index_table;

    for (GGint j = 0; j < nb_energies; ++j) {
      energies_[index_table] = GGEMSMuDataConstants::kMuData[index_data++];
      mu_[index_table] = GGEMSMuDataConstants::kMuData[index_data++];
      mu_en_[index_table] = GGEMSMuDataConstants::kMuData[index_data++];

      index_table++;
    }
  }

  GGcout("GGEMSAttenuations", "GGEMSAttenuations", 3) << "GGEMSAttenuations created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSAttenuations::~GGEMSAttenuations(void)
{
  GGcout("GGEMSAttenuations", "~GGEMSAttenuations", 3) << "GGEMSAttenuations erasing..." << GGendl;

  if (energies_) {
    delete[] energies_;
    energies_ = nullptr;
  }

  if (mu_) {
    delete[] mu_;
    mu_ = nullptr;
  }

  if (mu_en_) {
    delete[] mu_en_;
    mu_en_ = nullptr;
  }

  if (mu_index_) {
    delete[] mu_index_;
    mu_index_ = nullptr;
  }

  if (attenuations_host_) {
    delete attenuations_host_;
    attenuations_host_ = nullptr;
  }

  GGcout("GGEMSAttenuations", "~GGEMSAttenuations", 3) << "GGEMSAttenuations erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSAttenuations::Clean(void)
{
  GGcout("GGEMSAttenuations", "Clean", 3) << "GGEMSAttenuations cleaning..." << GGendl;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  if (mu_tables_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(mu_tables_[i], sizeof(GGEMSMuMuEnData), i);
    }
    delete[] mu_tables_;
    mu_tables_ = nullptr;
  }

  GGcout("GGEMSAttenuations", "Clean", 3) << "GGEMSAttenuations cleaned!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSAttenuations::Initialize(void)
{
  GGcout("GGEMSAttenuations", "Initialize", 1) << "Initializing attenuation tables..." << GGendl;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Loop over the device and storing value for each materials
  mu_tables_ = new cl::Buffer*[number_activated_devices_];
  for (GGsize d = 0; d < number_activated_devices_; ++d) {
    // Allocating memory on OpenCL device
    mu_tables_[d] = opencl_manager.Allocate(nullptr, sizeof(GGEMSMuMuEnData), d, CL_MEM_READ_WRITE, "GGEMSAttenuations");

    // Getting the OpenCL pointer on Mu tables
    GGEMSMuMuEnData* mu_table_device = opencl_manager.GetDeviceBuffer<GGEMSMuMuEnData>(mu_tables_[d], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSMuMuEnData), d);

    cl::Buffer* particle_cs = cross_sections_->GetCrossSections(d);
    GGEMSParticleCrossSections* particle_cs_device =  opencl_manager.GetDeviceBuffer<GGEMSParticleCrossSections>(particle_cs, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSParticleCrossSections), d);

    mu_table_device->number_of_materials_ = static_cast<GGint>(particle_cs_device->number_of_materials_);
    mu_table_device->energy_max_ = ATTENUATION_ENERGY_MAX;
    mu_table_device->energy_min_ = ATTENUATION_ENERGY_MIN;
    mu_table_device->number_of_bins_ = ATTENUATION_TABLE_NUMBER_BINS;

    opencl_manager.ReleaseDeviceBuffer(particle_cs, particle_cs_device, d);

    // Fill energy table with log scale
    GGfloat slope = logf(mu_table_device->energy_max_ / mu_table_device->energy_min_);
    GGint i = 0;
    while (i < mu_table_device->number_of_bins_) {
      mu_table_device->energy_bins_[i] = mu_table_device->energy_min_ * expf(slope * (static_cast<GGfloat>(i) / (static_cast<GGfloat>(mu_table_device->number_of_bins_)-1.0f)))*MeV;
      ++i;
    }

    GGEMSMaterialTables* materials_device =  opencl_manager.GetDeviceBuffer<GGEMSMaterialTables>(materials_->GetMaterialTables(d), CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSMaterialTables), d);

    // For each material and energy bin compute mu and muen
    GGint imat = 0;
    GGint abs_index, E_index, mu_index_E;
    GGsize iZ, Z;
    GGfloat energy, mu_over_rho, mu_en_over_rho, frac;
    while (imat < mu_table_device->number_of_materials_) {
      // for each energy bin
      i=0;
      while (i < mu_table_device->number_of_bins_) {
        // absolute index to store data within the table
        abs_index = imat*mu_table_device->number_of_bins_ + i;

        // Energy value
        energy = mu_table_device->energy_bins_[i];

        // For each element of the material
        mu_over_rho = 0.0f; mu_en_over_rho = 0.0f;
        iZ=0;
        while (iZ < materials_device->number_of_chemical_elements_[imat]) {
          // Get Z and mass fraction
          Z = materials_device->atomic_number_Z_[materials_device->index_of_chemical_elements_[imat] + iZ];
          frac = materials_device->mass_fraction_[materials_device->index_of_chemical_elements_[imat] + iZ];

          // Get energy index
          mu_index_E = GGEMSMuDataConstants::kMuIndexEnergy[Z];
          E_index = BinarySearchLeft(energy, energies_, mu_index_E+GGEMSMuDataConstants::kMuNbEnergyBins[Z], 0, mu_index_E);

          // Get mu an mu_en from interpolation
          if ( E_index == mu_index_E ) {
            mu_over_rho += mu_[E_index];
            mu_en_over_rho += mu_en_[E_index];
          }
          else
          {
            mu_over_rho += frac * LinearInterpolation(energies_[E_index-1], mu_[E_index-1], energies_[E_index], mu_[E_index], energy);
            mu_en_over_rho += frac * LinearInterpolation(energies_[E_index-1], mu_en_[E_index-1], energies_[E_index], mu_en_[E_index], energy);
          }
          ++iZ;
        }

        // Store values
        mu_table_device->mu_[abs_index] = mu_over_rho * materials_device->density_of_material_[imat] / (g/cm3);
        mu_table_device->mu_en_[abs_index] = mu_en_over_rho * materials_device->density_of_material_[imat] / (g/cm3);

        ++i;
      } // E bin
      ++imat;
    }
    opencl_manager.ReleaseDeviceBuffer(mu_tables_[d], mu_table_device, d);
    opencl_manager.ReleaseDeviceBuffer(materials_->GetMaterialTables(d), materials_device, d);
  }

  // Copy data from device to RAM memory (optimization for python users)
  LoadAttenuationsOnHost();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSAttenuations::LoadAttenuationsOnHost(void)
{
  GGcout("GGEMSAttenuations", "LoadAttenuationsOnHost", 1) << "Loading attenuations coefficient from OpenCL device to host (RAM)..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer to data on OpenCL device
  GGEMSMuMuEnData* attenuations_device = opencl_manager.GetDeviceBuffer<GGEMSMuMuEnData>(mu_tables_[0], CL_TRUE, CL_MAP_READ, sizeof(GGEMSMuMuEnData), 0);

  attenuations_host_->number_of_bins_ = attenuations_device->number_of_bins_;
  attenuations_host_->energy_min_ = attenuations_device->energy_min_;
  attenuations_host_->energy_max_ = attenuations_device->energy_max_;
  attenuations_host_->number_of_materials_ = attenuations_device->number_of_materials_;

  for(GGint i = 0; i < ATTENUATION_TABLE_NUMBER_BINS; ++i) {
    attenuations_host_->energy_bins_[i] = attenuations_device->energy_bins_[i];
  }

  for(GGint i = 0; i < 256*ATTENUATION_TABLE_NUMBER_BINS; ++i) {
    attenuations_host_->mu_[i] = attenuations_device->mu_[i];
    attenuations_host_->mu_en_[i] = attenuations_device->mu_en_[i];
  }

  // Release pointer
  opencl_manager.ReleaseDeviceBuffer(mu_tables_[0], attenuations_device, 0);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSAttenuations::GetAttenuation(std::string const& material_name, GGfloat const& energy, std::string const& unit) const
{
  // Get min and max energy in the table, and number of bins
  GGfloat min_energy = attenuations_host_->energy_min_;
  GGfloat max_energy = attenuations_host_->energy_max_;
  GGint number_of_bins = attenuations_host_->number_of_bins_;

  // Converting energy
  GGfloat e_MeV = EnergyUnit(energy, unit);

  if (e_MeV < min_energy || e_MeV > max_energy) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Problem energy: " << e_MeV << " " << unit << " is not in the range [" << min_energy << ", " << max_energy << "] MeV!!!" << std::endl;
    GGEMSMisc::ThrowException("GGEMSAttenuations", "GetAttenuation", oss.str());
  }

  // Get id of material
  ptrdiff_t material_id = materials_->GetMaterialIndex(material_name);

  // Computing the energy bin
  GGsize energy_bin = static_cast<GGsize>(BinarySearchLeft(e_MeV, attenuations_host_->energy_bins_, static_cast<GGint>(number_of_bins), 0, 0));

  // Computing attenuation
  GGfloat energy_a = attenuations_host_->energy_bins_[energy_bin];
  GGfloat energy_b = attenuations_host_->energy_bins_[energy_bin+1];
  GGfloat attenuation_a = attenuations_host_->mu_[energy_bin + static_cast<GGsize>(number_of_bins*material_id)];
  GGfloat attenuation_b = attenuations_host_->mu_[energy_bin+1 + static_cast<GGsize>(number_of_bins*material_id)];

  GGfloat attenuation = LinearInterpolation(energy_a, attenuation_a, energy_b, attenuation_b, e_MeV);

  return attenuation;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSAttenuations::GetEnergyAttenuation(std::string const& material_name, GGfloat const& energy, std::string const& unit) const
{
  // Get min and max energy in the table, and number of bins
  GGfloat min_energy = attenuations_host_->energy_min_;
  GGfloat max_energy = attenuations_host_->energy_max_;
  GGint number_of_bins = attenuations_host_->number_of_bins_;

  // Converting energy
  GGfloat e_MeV = EnergyUnit(energy, unit);

  if (e_MeV < min_energy || e_MeV > max_energy) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Problem energy: " << e_MeV << " " << unit << " is not in the range [" << min_energy << ", " << max_energy << "] MeV!!!" << std::endl;
    GGEMSMisc::ThrowException("GGEMSAttenuations", "GetEnergyAttenuation", oss.str());
  }

  // Get id of material
  ptrdiff_t material_id = materials_->GetMaterialIndex(material_name);

  // Computing the energy bin
  GGsize energy_bin = static_cast<GGsize>(BinarySearchLeft(e_MeV, attenuations_host_->energy_bins_, static_cast<GGint>(number_of_bins), 0, 0));

  // Computing attenuation
  GGfloat energy_a = attenuations_host_->energy_bins_[energy_bin];
  GGfloat energy_b = attenuations_host_->energy_bins_[energy_bin+1];
  GGfloat energy_attenuation_a = attenuations_host_->mu_en_[energy_bin + static_cast<GGsize>(number_of_bins*material_id)];
  GGfloat energy_attenuation_b = attenuations_host_->mu_en_[energy_bin+1 + static_cast<GGsize>(number_of_bins*material_id)];

  GGfloat energy_attenuation = LinearInterpolation(energy_a, energy_attenuation_a, energy_b, energy_attenuation_b, e_MeV);

  return energy_attenuation;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSAttenuations* create_ggems_attenuations(GGEMSMaterials* materials, GGEMSCrossSections* cross_sections)
{
  return new(std::nothrow) GGEMSAttenuations(materials, cross_sections);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void initialize_ggems_attenuations(GGEMSAttenuations* attenuations)
{
  attenuations->Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat get_mu_ggems_attenuations(GGEMSAttenuations* attenuations, char const* material_name, GGfloat const energy, char const* unit)
{
  return attenuations->GetAttenuation(material_name, energy, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat get_mu_en_ggems_attenuations(GGEMSAttenuations* attenuations, char const* material_name, GGfloat const energy, char const* unit)
{
  return attenuations->GetEnergyAttenuation(material_name, energy, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void clean_ggems_attenuations(GGEMSAttenuations* attenuations)
{
  attenuations->Clean();
}
