/*!
  \file GGEMSMaterials.cc

  \brief GGEMS class handling material(s) for a specific navigator

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 4, 2020
*/

#include <sstream>
#include <limits>

#include "GGEMS/materials/GGEMSMaterials.hh"

#include "GGEMS/materials/GGEMSIonizationParamsMaterial.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMaterials::GGEMSMaterials(void)
: opencl_manager_(GGEMSOpenCLManager::GetInstance()),
  material_manager_(GGEMSMaterialsDatabaseManager::GetInstance())
{
  GGcout("GGEMSMaterials", "GGEMSMaterials", 3) << "Allocation of GGEMSMaterials..." << GGendl;
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

bool GGEMSMaterials::AddMaterial(std::string const& material)
{
  // Checking the number of material (maximum is 255)
  if (materials_.size() == 255) {
    GGEMSMisc::ThrowException("GGEMSMaterials", "AddMaterial", "Limit of material reached. The limit is 255 materials!!!");
  }

  // Add material and check if the material already exists
  return materials_.insert(material).second;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterials::PrintInfos(void) const
{
  // Getting the OpenCL pointer on material tables
  GGEMSMaterialTables* material_table = opencl_manager_.GetDeviceBuffer<GGEMSMaterialTables>(material_tables_, sizeof(GGEMSMaterialTables));

  // Getting list of activated materials
  std::set<std::string>::iterator iter_material = materials_.begin();

  GGcout("GGEMSMaterials", "PrintInfos", 0) << GGendl;
  GGcout("GGEMSMaterials", "PrintInfos", 0) << "Number of materials: " << static_cast<GGuint>(material_table->number_of_materials_) << GGendl;
  GGcout("GGEMSMaterials", "PrintInfos", 0) << "Total number of chemical elements: " << material_table->total_number_of_chemical_elements_ << GGendl;
  GGcout("GGEMSMaterials", "PrintInfos", 0) << "Activated Materials: " << GGendl;
  GGcout("GGEMSMaterials", "PrintInfos", 0) << "-----------------------------------" << GGendl;
  for (GGuchar i = 0; i < material_table->number_of_materials_; ++i, ++iter_material) {
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "* " << *iter_material << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Number of chemical elements: " << static_cast<GGushort>(material_table->number_of_chemical_elements_[i]) << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Density: " << material_table->density_of_material_[i]/(GGEMSUnits::g/GGEMSUnits::cm3) << " g/cm3" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Radiation length: " << material_table->radiation_length_[i]/(GGEMSUnits::cm) << " cm" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Total atomic density: " << material_table->number_of_atoms_by_volume_[i]/(GGEMSUnits::mol/GGEMSUnits::cm3) << " atom/cm3" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Total Electron density: " << material_table->number_of_electrons_by_volume_[i]/(GGEMSUnits::mol/GGEMSUnits::cm3) << " e-/cm3" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Chemical Elements:" << GGendl;
    for (GGuchar j = 0; j < material_table->number_of_chemical_elements_[i]; ++j) {
      GGushort const kIndexChemicalElement = material_table->index_of_chemical_elements_[i];
      GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + Z = " << static_cast<GGushort>(material_table->atomic_number_Z_[j+kIndexChemicalElement]) << GGendl;
      GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + fraction of chemical element = " << material_table->mass_fraction_[j+kIndexChemicalElement]/GGEMSUnits::PERCENT << " %" << GGendl;
      GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + Atomic number density = " << material_table->atomic_number_density_[j+kIndexChemicalElement]/(GGEMSUnits::mol/GGEMSUnits::cm3) << " atom/cm3" << GGendl;
      GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + Element abundance = " << 100.0*material_table->atomic_number_density_[j+kIndexChemicalElement]/material_table->number_of_atoms_by_volume_[i] << " %" << GGendl;
    }
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Energy loss fluctuation data:" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + Mean electron excitation energy: " << material_table->mean_excitation_energy_[i]/GGEMSUnits::eV << " eV" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + Log mean electron excitation energy: " << material_table->log_mean_excitation_energy_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + f1: " << material_table->f1_fluct_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + f2: " << material_table->f2_fluct_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + energy0: " << material_table->energy0_fluct_[i]/GGEMSUnits::eV << " eV" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + energy1: " << material_table->energy1_fluct_[i]/GGEMSUnits::eV << " eV" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + energy2: " << material_table->energy2_fluct_[i]/GGEMSUnits::eV << " eV" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + log energy 1: " << material_table->log_energy1_fluct_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + log energy 2: " << material_table->log_energy2_fluct_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "    - Density correction data:" << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + x0 = " << material_table->x0_density_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + x1 = " << material_table->x1_density_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + d0 = " << material_table->d0_density_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + -C = " << material_table->c_density_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + a = " << material_table->a_density_[i] << GGendl;
    GGcout("GGEMSMaterials", "PrintInfos", 0) << "        + m = " << material_table->m_density_[i] << GGendl;
  }
  GGcout("GGEMSMaterials", "PrintInfos", 0) << GGendl;

  // Release the pointer, mandatory step!!!
  opencl_manager_.ReleaseDeviceBuffer(material_tables_, material_table);

/*

    ui32 mat_id = 0; while ( mat_id < h_materials->nb_materials )
    {
        ui32 index = h_materials->index[ mat_id ];

        printf("[GGEMS]       Nb atoms per vol: %e\n",  h_materials->nb_atoms_per_vol[ mat_id ]);
        printf("[GGEMS]       Nb electrons per vol: %e\n", h_materials->nb_electrons_per_vol[ mat_id ]);
        printf("[GGEMS]       Electron mean exitation energy: %e\n", h_materials->electron_mean_excitation_energy[ mat_id ]);
        printf("[GGEMS]       Rad length: %e\n", h_materials->rad_length[ mat_id ]);
        printf("[GGEMS]       Density: %e\n", h_materials->density[ mat_id ]);
        printf("[GGEMS]       Photon energy cut: %e\n", h_materials->photon_energy_cut[ mat_id ]);
        printf("[GGEMS]       Electon energy cut: %e\n", h_materials->electron_energy_cut[ mat_id ]);

        printf("[GGEMS]       Density correction:\n");
        printf("[GGEMS]          fX0: %e\n", h_materials->fX0[ mat_id ]);
        printf("[GGEMS]          fX1: %e\n", h_materials->fX1[ mat_id ]);
        printf("[GGEMS]          fD0: %e\n", h_materials->fD0[ mat_id ]);
        printf("[GGEMS]          fC: %e\n", h_materials->fC[ mat_id ]);
        printf("[GGEMS]          fA: %e\n", h_materials->fA[ mat_id ]);
        printf("[GGEMS]          fM: %e\n", h_materials->fM[ mat_id ]);

        printf("[GGEMS]       Energy loss fluctuation:\n");
        printf("[GGEMS]          fF1: %e\n", h_materials->fF1[ mat_id ]);
        printf("[GGEMS]          fF2: %e\n", h_materials->fF2[ mat_id ]);
        printf("[GGEMS]          fEnergy0: %e\n", h_materials->fEnergy0[ mat_id ]);
        printf("[GGEMS]          fEnergy1: %e\n", h_materials->fEnergy1[ mat_id ]);
        printf("[GGEMS]          fEnergy2: %e\n", h_materials->fEnergy2[ mat_id ]);
        printf("[GGEMS]          fLogEnergy1: %e\n", h_materials->fLogEnergy1[ mat_id ]);
        printf("[GGEMS]          fLogEnergy2: %e\n", h_materials->fLogEnergy2[ mat_id ]);
        printf("[GGEMS]          fLogMeanExcitationEnergy: %e\n", h_materials->fLogMeanExcitationEnergy[ mat_id ]);

        printf("[GGEMS]       Mixture:\n");

        ui32 elt_id = 0; while ( elt_id < h_materials->nb_elements[ mat_id ] )
        {
            printf("[GGEMS]          Z: %i   Atom Num Dens: %e\n", h_materials->mixture[ index+elt_id ], h_materials->atom_num_dens[ index+elt_id ]);
            ++elt_id;
        }

        ++mat_id;
    }
  */
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterials::BuildMaterialTables(void)
{
  GGcout("GGEMSMaterials", "BuildMaterialTables", 3) << "Building the material tables..." << GGendl;

  // Allocating memory for material tables in OpenCL device
  material_tables_ = opencl_manager_.Allocate(nullptr, sizeof(GGEMSMaterialTables), CL_MEM_READ_WRITE);

  // Getting the OpenCL pointer on material tables
  GGEMSMaterialTables* material_table = opencl_manager_.GetDeviceBuffer<GGEMSMaterialTables>(material_tables_, sizeof(GGEMSMaterialTables));

  // Get the number of activated materials
  material_table->number_of_materials_ = static_cast<GGuchar>(materials_.size());

  // Loop over the materials
  std::set<std::string>::iterator iter_material = materials_.begin();
  GGushort index_to_chemical_element = 0;
  for (std::size_t i = 0; i < materials_.size(); ++i, ++iter_material) {
    // Getting the material infos from database
    GGEMSSingleMaterial const& kSingleMaterial = material_manager_.GetMaterial(*iter_material);

    // Storing infos about material
    material_table->number_of_chemical_elements_[i] = kSingleMaterial.nb_elements_;
    material_table->density_of_material_[i] = kSingleMaterial.density_;

    // Initialize some counters
    material_table->number_of_atoms_by_volume_[i] = 0.0f;
    material_table->number_of_electrons_by_volume_[i] = 0.0f;

    // Loop over the chemical elements by material
    for (GGuchar j = 0; j < kSingleMaterial.nb_elements_; ++j) {
      // Getting the chemical element
      GGEMSChemicalElement const& kChemicalElement = material_manager_.GetChemicalElement(kSingleMaterial.chemical_element_name_[j]);

      // Atomic number Z
      material_table->atomic_number_Z_[j+index_to_chemical_element] = kChemicalElement.atomic_number_Z_;

      // Mass fraction of element by material
      material_table->mass_fraction_[j+index_to_chemical_element] = kSingleMaterial.mixture_f_[j];

      // Atomic number density
      material_table->atomic_number_density_[j+index_to_chemical_element] = static_cast<GGfloat>(static_cast<GGdouble>(GGEMSPhysicalConstant::AVOGADRO) / kChemicalElement.molar_mass_M_ * kSingleMaterial.density_ * kSingleMaterial.mixture_f_[j]);

      // Increment density of atoms and electrons
      material_table->number_of_atoms_by_volume_[i] += material_table->atomic_number_density_[j+index_to_chemical_element];
      material_table->number_of_electrons_by_volume_[i] += material_table->atomic_number_density_[j+index_to_chemical_element] * kChemicalElement.atomic_number_Z_;
    }

    // Computing ionization params for a material
    GGEMSIonizationParamsMaterial ionization_params(&kSingleMaterial);
    material_table->mean_excitation_energy_[i] = ionization_params.GetMeanExcitationEnergy();
    material_table->log_mean_excitation_energy_[i] = ionization_params.GetLogMeanExcitationEnergy();
    material_table->x0_density_[i] = ionization_params.GetX0Density();
    material_table->x1_density_[i] = ionization_params.GetX1Density();
    material_table->d0_density_[i] = ionization_params.GetD0Density();
    material_table->c_density_[i] = ionization_params.GetCDensity();
    material_table->a_density_[i] = ionization_params.GetADensity();
    material_table->m_density_[i] = ionization_params.GetMDensity();

    // Energy fluctuation parameters
    material_table->f1_fluct_[i] = ionization_params.GetF1Fluct();
    material_table->f2_fluct_[i] = ionization_params.GetF2Fluct();
    material_table->energy0_fluct_[i] = ionization_params.GetEnergy0Fluct();
    material_table->energy1_fluct_[i] = ionization_params.GetEnergy1Fluct();
    material_table->energy2_fluct_[i] = ionization_params.GetEnergy2Fluct();
    material_table->log_energy1_fluct_[i] = ionization_params.GetLogEnergy1Fluct();
    material_table->log_energy2_fluct_[i] = ionization_params.GetLogEnergy2Fluct();

    // others stuffs
    material_table->radiation_length_[i] = GetRadiationLength(*iter_material);

    // Computing the access to chemical element by material
    material_table->index_of_chemical_elements_[i] = index_to_chemical_element;
    index_to_chemical_element += static_cast<GGushort>(material_table->number_of_chemical_elements_[i]);
  }

  // Storing number total of chemical elements
  material_table->total_number_of_chemical_elements_ = index_to_chemical_element;

  // Release the pointer, mandatory step!!!
  opencl_manager_.ReleaseDeviceBuffer(material_tables_, material_table);

/*
    h_materials->electron_mean_excitation_energy = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->rad_length = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->photon_energy_cut = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->electron_energy_cut = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fX0 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fX1 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fD0 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fC = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fA = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fM = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fF1 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fF2 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fEnergy0 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fEnergy1 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fEnergy2 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fLogEnergy1 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fLogEnergy2 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fLogMeanExcitationEnergy = (f32*)malloc(sizeof(f32)*m_nb_materials);

    i32 i, j;
    ui32 access_index = 0;
    ui32 fill_index = 0;
    std::string mat_name, elt_name;
    aMaterial cur_mat;

    i=0; while (i < m_nb_materials) {
        // get mat name
        mat_name = mats_list[i];

        // read mat from databse
        cur_mat = m_material_db.materials[mat_name];
        if (cur_mat.name == "") {
            printf("[ERROR] Material %s is not on your database (%s function)\n", mat_name.c_str(),__FUNCTION__);
            exit_simulation();
        }
        // get nb of elements
        h_materials->nb_elements[i] = cur_mat.nb_elements;

        // compute index
        h_materials->index[i] = access_index;
        access_index += cur_mat.nb_elements;

        ++i;
    }

    // nb of total elements
    m_nb_elements_total = access_index;
    h_materials->nb_elements_total = access_index;
    h_materials->mixture = (ui16*)malloc(sizeof(ui16)*access_index);
    h_materials->atom_num_dens = (f32*)malloc(sizeof(f32)*access_index);
    h_materials->mass_fraction = (f32*)malloc(sizeof(f32)*access_index);


    // store mixture element and compute atomic density
    i=0; while (i < m_nb_materials) {

        // get mat name
        mat_name = mats_list[i];

        // read mat from database
        cur_mat = m_material_db.materials[mat_name];

        // get density
        h_materials->density[i] = m_material_db.get_density( mat_name );     // in g/cm3

        h_materials->nb_atoms_per_vol[i] = 0.0f;
        h_materials->nb_electrons_per_vol[i] = 0.0f;

        j=0; while (j < m_material_db.get_nb_elements( mat_name )) {
            // read element name
            //elt_name = cur_mat.mixture_Z[j];
            elt_name = m_material_db.get_element_name( mat_name, j );

            // store Z
            h_materials->mixture[fill_index] = m_material_db.get_element_Z( elt_name );

            // compute atom num dens (Avo*fraction*dens) / Az
            h_materials->atom_num_dens[fill_index] = m_material_db.get_atom_num_dens( mat_name, j );

            // get mass fraction
            h_materials->mass_fraction[fill_index] = m_material_db.get_mass_fraction( mat_name, j );

            // compute nb atoms per volume
            h_materials->nb_atoms_per_vol[i] += h_materials->atom_num_dens[fill_index];

            // compute nb electrons per volume
            h_materials->nb_electrons_per_vol[i] += h_materials->atom_num_dens[fill_index] *
                                                     m_material_db.get_element_Z( elt_name );

            ++j;
            ++fill_index;
        }

        /// electron Ionisation data
        m_material_db.compute_ioni_parameters( mat_name );

        h_materials->electron_mean_excitation_energy[i] = m_material_db.get_mean_excitation();
        h_materials->fLogMeanExcitationEnergy[i] = logf( m_material_db.get_mean_excitation() );

        // correction
        h_materials->fX0[i] = m_material_db.get_X0_density();
        h_materials->fX1[i] = m_material_db.get_X1_density();
        h_materials->fD0[i] = m_material_db.get_D0_density();
        h_materials->fC[i] = m_material_db.get_C_density();
        h_materials->fA[i] = m_material_db.get_A_density();
        h_materials->fM[i] = m_material_db.get_M_density();

        //eFluctuation parameters
        h_materials->fF1[i] = m_material_db.get_F1_fluct();
        h_materials->fF2[i] = m_material_db.get_F2_fluct();
        h_materials->fEnergy0[i] = m_material_db.get_Energy0_fluct();
        h_materials->fEnergy1[i] = m_material_db.get_Energy1_fluct();
        h_materials->fEnergy2[i] = m_material_db.get_Energy2_fluct();
        h_materials->fLogEnergy1[i] = m_material_db.get_LogEnergy1_fluct();
        h_materials->fLogEnergy2[i] = m_material_db.get_LogEnergy2_fluct();

        /// others stuffs

        h_materials->rad_length[i] = m_material_db.get_rad_len( mat_name );

        /// Compute energy cut
        f32 gEcut = m_rangecut.convert_gamma(h_params->photon_cut, h_materials->mixture, h_materials->nb_elements[i],
                                             h_materials->atom_num_dens, h_materials->index[i]);

        f32 eEcut = m_rangecut.convert_electron(h_params->electron_cut, h_materials->mixture, h_materials->nb_elements[i],
                                                h_materials->atom_num_dens, h_materials->density[i], h_materials->index[i]);

        h_materials->photon_energy_cut[i] = gEcut;
        h_materials->electron_energy_cut[i] = eEcut;

        if ( h_params->display_energy_cuts )
        {
            printf( "[GGEMS]    material: %s\t\tgamma: %s electron: %s\n", mat_name.c_str(),
                   Energy_str( gEcut ).c_str(), Energy_str( eEcut ).c_str() );
        }

        ++i;
    }

    */
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSMaterials::GetRadiationLength(std::string const& material) const
{
  GGfloat inverse_radiation = 0.0f;
  GGfloat tsai_radiation = 0.0f;
  GGfloat zeff = 0.0f;
  GGfloat coulomb = 0.0f;

  static constexpr GGfloat l_rad_light[]  = {5.310f , 4.790f , 4.740f, 4.710f};
  static constexpr GGfloat lp_rad_light[] = {6.144f , 5.621f , 5.805f, 5.924f};
  static constexpr GGfloat k1 = 0.00830f;
  static constexpr GGfloat k2 = 0.20206f;
  static constexpr GGfloat k3 = 0.00200f;
  static constexpr GGfloat k4 = 0.03690f;

  // Getting the material infos from database
  GGEMSSingleMaterial const& kSingleMaterial = material_manager_.GetMaterial(material);

  // Loop over the chemical elements by material
  for (GGuchar i = 0; i < kSingleMaterial.nb_elements_; ++i) {
    // Getting the chemical element
    GGEMSChemicalElement const& kChemicalElement = material_manager_.GetChemicalElement(kSingleMaterial.chemical_element_name_[i]);

    // Z effective
    zeff = static_cast<GGfloat>(kChemicalElement.atomic_number_Z_);

    //  Compute Coulomb correction factor (Phys Rev. D50 3-1 (1994) page 1254)
    GGfloat az2 = (GGEMSPhysicalConstant::FINE_STRUCTURE_CONST*zeff)*(GGEMSPhysicalConstant::FINE_STRUCTURE_CONST*zeff);
    GGfloat az4 = az2 * az2;
    coulomb = ( k1*az4 + k2 + 1.0f/ (1.0f+az2) ) * az2 - ( k3*az4 + k4 ) * az4;

    //  Compute Tsai's Expression for the Radiation Length
    //  (Phys Rev. D50 3-1 (1994) page 1254)
    GGfloat const logZ3 = std::log(zeff) / 3.0f;

    GGfloat l_rad = 0.0f;
    GGfloat lp_rad = 0.0f;
    GGint iz = static_cast<GGint>(( zeff + 0.5f ) - 1);
    if (iz <= 3){
      l_rad = l_rad_light[iz];
      lp_rad = lp_rad_light[iz];
    }
    else {
      l_rad = std::log(184.15f) - logZ3;
      lp_rad = std::log(1194.0f) - 2.0f*logZ3;
    }

    tsai_radiation = 4.0f * GGEMSPhysicalConstant::ALPHA_RCL2 * zeff * ( zeff * ( l_rad - coulomb ) + lp_rad );
    inverse_radiation += static_cast<GGfloat>(static_cast<GGdouble>(GGEMSPhysicalConstant::AVOGADRO) / kChemicalElement.molar_mass_M_ * kSingleMaterial.density_ * kSingleMaterial.mixture_f_[i] * tsai_radiation);
  }

  return (inverse_radiation <= 0.0f ? std::numeric_limits<GGfloat>::max() : 1.0f / inverse_radiation);
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
