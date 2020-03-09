/*!
  \file GGEMSMaterials.cc

  \brief GGEMS class handling material(s) for a specific navigator

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 4, 2020
*/

#include "GGEMS/physics/GGEMSMaterials.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/tools/GGEMSTools.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMaterials::GGEMSMaterials(void)
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
  /*GGuint index_label = 0;
  for (auto&& i : materials_) {
    GGcout("GGEMSMaterials", "PrintLabels", 0) << "Material: " << i << ", label: " << index_label << GGendl;
    ++index_label;
  }*/
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterials::BuildMaterialTables(void)
{
  GGcout("GGEMSMaterials", "BuildMaterialTables", 3) << "Building the material tables..." << GGendl;

  // Allocating memory for material tables in OpenCL device
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  material_tables_ = opencl_manager.Allocate(nullptr, sizeof(GGEMSMaterialTables), CL_MEM_READ_WRITE);

  // Getting the OpenCL pointer on material tables
  GGEMSMaterialTables* material_table = opencl_manager.GetDeviceBuffer<GGEMSMaterialTables>(material_tables_, sizeof(GGEMSMaterialTables));


  // Release the pointer, mandatory step!!!
  opencl_manager.ReleaseDeviceBuffer(material_tables_, material_table);

/*
    // First allocated data to the structure according the number of materials
    m_nb_materials = mats_list.size();

    **** h_materials = (MaterialsData*)malloc( sizeof(MaterialsData) );
    h_materials->nb_elements = (ui16*)malloc(sizeof(ui16)*m_nb_materials);
    h_materials->index = (ui16*)malloc(sizeof(ui16)*m_nb_materials);
    h_materials->nb_atoms_per_vol = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->nb_electrons_per_vol = (f32*)malloc(sizeof(f32)*m_nb_materials);
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
    h_materials->density = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fF1 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fF2 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fEnergy0 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fEnergy1 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fEnergy2 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fLogEnergy1 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fLogEnergy2 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->fLogMeanExcitationEnergy = (f32*)malloc(sizeof(f32)*m_nb_materials);
    h_materials->nb_materials = mats_list.size();



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

    // Display energy cuts
    if ( h_params->display_energy_cuts )
    {
        GGcout << GGendl;
        GGcout << "Energy cuts:" << GGendl;        
    }

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

    // Display energy cuts
    if ( h_params->display_energy_cuts )
    {
        GGcout << GGendl;
    }
    */
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
