// GGEMS Copyright (C) 2015

#ifndef MATERIALS_CU
#define MATERIALS_CU

#include "materials.cuh"

//////////////////////////////////////////////////////////////////
//// MaterialDataBase class //////////////////////////////////////
//////////////////////////////////////////////////////////////////

MaterialDataBase::MaterialDataBase() {} // Open and load the material database


// Load materials from data base
void MaterialDataBase::load_materials(std::string filename) {
    //printf("load material ... \n");
    std::ifstream file(filename.c_str());

    std::string line, elt_name;
    f32 mat_f;
    ui16 i;
    ui16 ind = 0;

    while (file) {
        m_txt_reader.skip_comment(file);
        std::getline(file, line);

        if (file) {
            Material mat;
            mat.name = m_txt_reader.read_material_name(line);
            //printf("mat name ... %s \n",mat.name.c_str());   // Too much verbose - JB
            mat.density = m_txt_reader.read_material_density(line);
            mat.nb_elements = m_txt_reader.read_material_nb_elements(line);

            i=0; while (i<mat.nb_elements) {
                std::getline(file, line);
                elt_name = m_txt_reader.read_material_element(line);
                mat_f = m_txt_reader.read_material_fraction(line);

                mat.mixture_Z.push_back(elt_name);
                mat.mixture_f.push_back(mat_f);

                ++i;
            }

            materials[mat.name] = mat;
            ++ind;

        } // if

    } // while

}


// Load elements from data file
void MaterialDataBase::load_elements(std::string filename) {

    std::ifstream file(filename.c_str());

    std::string line, elt_name;
    i32 elt_Z;
    f32 elt_A;

    while (file) {
        m_txt_reader.skip_comment(file);
        std::getline(file, line);

        if (file) {
            elt_name = m_txt_reader.read_element_name(line);
            elt_Z = m_txt_reader.read_element_Z(line);
            elt_A = m_txt_reader.read_element_A(line);

            elements_Z[elt_name] = elt_Z;
            elements_A[elt_name] = elt_A;
        }

    }

}

//////////////////////////////////////////////////////////////////
//// MaterialManager class //////////////////////////////////////
//////////////////////////////////////////////////////////////////

MaterialManager::MaterialManager() {} // // This class is used to build the material table

///:: Privates

// Search and return the material index for a given material name
ui16 MaterialManager::m_get_material_index(std::string material_name) {

    // Check if this material is already used, if it is return the corresponding index
    ui16 index = 0;
    while (index < m_materials_list.size()) {
        if (m_materials_list[index] == material_name) return index;
        ++index;
    }

    // If it is not, add a new entry into the material table
    index = m_materials_list.size();
    m_materials_list.push_back(material_name);

    return index;
}

// Check mandatory
bool MaterialManager::m_check_mandatory() {

    if (materials.nb_materials == 0 || materials.nb_elements_total == 0) return false;
    else return true;
}

// Copy data to the GPU
void MaterialManager::m_copy_materials_table_cpu2gpu() {

    ui32 n = materials.nb_materials;
    ui32 k = materials.nb_elements_total;

    // First allocate the GPU mem for the scene
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.nb_elements, n*sizeof(ui16)) );
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.index, n*sizeof(ui16)) );

    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.mixture, k*sizeof(ui16)) );
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.atom_num_dens, k*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.nb_atoms_per_vol, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.nb_electrons_per_vol, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.electron_mean_excitation_energy, n*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.rad_length, n*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.photon_energy_cut, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.electron_energy_cut, n*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.fX0, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.fX1, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.fD0, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.fC, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.fA, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.fM, n*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.fF1, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.fF2, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.fEnergy0, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.fEnergy1, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.fEnergy2, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.fLogEnergy1, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.fLogEnergy2, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.fLogMeanExcitationEnergy, n*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &materials.data_d.density, n*sizeof(f32)) );

    // Copy data to the GPU
    materials.data_d.nb_materials = materials.data_h.nb_materials;
    materials.data_d.nb_elements_total = materials.data_h.nb_elements_total;

    HANDLE_ERROR( cudaMemcpy( materials.data_d.nb_elements, materials.data_h.nb_elements,
                             n*sizeof(ui16), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( materials.data_d.index, materials.data_h.index,
                             n*sizeof(ui16), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy( materials.data_d.mixture, materials.data_h.mixture,
                             k*sizeof(ui16), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( materials.data_d.atom_num_dens, materials.data_h.atom_num_dens,
                             k*sizeof(f32), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy( materials.data_d.nb_atoms_per_vol, materials.data_h.nb_atoms_per_vol,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( materials.data_d.nb_electrons_per_vol, materials.data_h.nb_electrons_per_vol,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( materials.data_d.electron_mean_excitation_energy, materials.data_h.electron_mean_excitation_energy,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( materials.data_d.rad_length, materials.data_h.rad_length,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy( materials.data_d.photon_energy_cut, materials.data_h.photon_energy_cut,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( materials.data_d.electron_energy_cut, materials.data_h.electron_energy_cut,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy( materials.data_d.fX0, materials.data_h.fX0,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( materials.data_d.fX1, materials.data_h.fX1,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( materials.data_d.fD0, materials.data_h.fD0,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( materials.data_d.fC, materials.data_h.fC,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( materials.data_d.fA, materials.data_h.fA,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( materials.data_d.fM, materials.data_h.fM,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy( materials.data_d.fF1, materials.data_h.fF1,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( materials.data_d.fF2, materials.data_h.fF2,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( materials.data_d.fEnergy0, materials.data_h.fEnergy0,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( materials.data_d.fEnergy1, materials.data_h.fEnergy1,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( materials.data_d.fEnergy2, materials.data_h.fEnergy2,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( materials.data_d.fLogEnergy1, materials.data_h.fLogEnergy1,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( materials.data_d.fLogEnergy2, materials.data_h.fLogEnergy2,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( materials.data_d.fLogMeanExcitationEnergy, materials.data_h.fLogMeanExcitationEnergy,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy( materials.data_d.density, materials.data_h.density,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );

}

// Build the materials table according the list of materials
void MaterialManager::m_build_materials_table(GlobalSimulationParameters params) {

    // First allocated data to the structure according the number of materials
    materials.nb_materials = m_materials_list.size();
    materials.data_h.nb_materials = m_materials_list.size();
    materials.data_h.nb_elements = (ui16*)malloc(sizeof(ui16)*materials.nb_materials);
    materials.data_h.index = (ui16*)malloc(sizeof(ui16)*materials.nb_materials);
    materials.data_h.nb_atoms_per_vol = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.nb_electrons_per_vol = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.electron_mean_excitation_energy = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.rad_length = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.photon_energy_cut = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.electron_energy_cut = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.fX0 = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.fX1 = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.fD0 = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.fC = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.fA = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.fM = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.density = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.fF1 = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.fF2 = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.fEnergy0 = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.fEnergy1 = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.fEnergy2 = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.fLogEnergy1 = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.fLogEnergy2 = (f32*)malloc(sizeof(f32)*materials.nb_materials);
    materials.data_h.fLogMeanExcitationEnergy = (f32*)malloc(sizeof(f32)*materials.nb_materials);

    i32 i, j;
    ui32 access_index = 0;
    ui32 fill_index = 0;
    std::string mat_name, elt_name;
    Material cur_mat;

    i=0; while (i < materials.nb_materials) {
        // get mat name
        mat_name = m_materials_list[i];

        // read mat from databse
        cur_mat = m_material_db.materials[mat_name];
        if (cur_mat.name == "") {
            printf("[ERROR] Material %s is not on your database (%s function)\n", mat_name.c_str(),__FUNCTION__);
            exit_simulation();
        }
        // get nb of elements
        materials.data_h.nb_elements[i] = cur_mat.nb_elements;

        // compute index
        materials.data_h.index[i] = access_index;
        access_index += cur_mat.nb_elements;

        ++i;
    }

    // nb of total elements
    materials.nb_elements_total = access_index;
    materials.data_h.nb_elements_total = access_index;
    materials.data_h.mixture = (ui16*)malloc(sizeof(ui16)*access_index);
    materials.data_h.atom_num_dens = (f32*)malloc(sizeof(f32)*access_index);

    // store mixture element and compute atomic density
    i=0; while (i < materials.nb_materials) {

        // get mat name
        mat_name = m_materials_list[i];

        // read mat from database
        cur_mat = m_material_db.materials[mat_name];

        // get density
        materials.data_h.density[i] = cur_mat.density / gram;

        // G4 material
        G4Material *g4mat = new G4Material("tmp", cur_mat.density, cur_mat.nb_elements);

        materials.data_h.nb_atoms_per_vol[i] = 0.0f;
        materials.data_h.nb_electrons_per_vol[i] = 0.0f;

        j=0; while (j < cur_mat.nb_elements) {
            // read element name
            elt_name = cur_mat.mixture_Z[j];

            // store Z
            materials.data_h.mixture[fill_index] = m_material_db.elements_Z[elt_name];

            // compute atom num dens (Avo*fraction*dens) / Az
            materials.data_h.atom_num_dens[fill_index] = Avogadro/m_material_db.elements_A[elt_name] *
                                                         cur_mat.mixture_f[j]*cur_mat.density;

            // compute nb atoms per volume
            materials.data_h.nb_atoms_per_vol[i] += materials.data_h.atom_num_dens[fill_index];

            // compute nb electrons per volume
            materials.data_h.nb_electrons_per_vol[i] += materials.data_h.atom_num_dens[fill_index] *
                                                        m_material_db.elements_Z[elt_name];

            // build G4 material
            G4Element *elt = new G4Element("element", "ELT", m_material_db.elements_Z[elt_name],
                                                             m_material_db.elements_A[elt_name]);
            g4mat->AddElement(elt, cur_mat.mixture_f[j]);

            ++j;
            ++fill_index;
        }

        // electron data
        materials.data_h.electron_mean_excitation_energy[i] = g4mat->GetIonisation()->GetMeanExcitationEnergy();
        materials.data_h.rad_length[i] = g4mat->GetRadlen();

        // cut (in energy for now, but should be change for range cut) TODO
        materials.data_h.photon_energy_cut[i] = params.data_h.photon_cut;
        materials.data_h.electron_energy_cut[i] = params.data_h.electron_cut;

        // eIonisation correction
        materials.data_h.fX0[i] = g4mat->GetIonisation()->GetX0density();
        materials.data_h.fX1[i] = g4mat->GetIonisation()->GetX1density();
        materials.data_h.fD0[i] = g4mat->GetIonisation()->GetD0density();
        materials.data_h.fC[i] = g4mat->GetIonisation()->GetCdensity();
        materials.data_h.fA[i] = g4mat->GetIonisation()->GetAdensity();
        materials.data_h.fM[i] = g4mat->GetIonisation()->GetMdensity();

        //eFluctuation parameters
        materials.data_h.fF1[i] = g4mat->GetIonisation()->GetF1fluct();
        materials.data_h.fF2[i] = g4mat->GetIonisation()->GetF2fluct();
        materials.data_h.fEnergy0[i] = g4mat->GetIonisation()->GetEnergy0fluct();
        materials.data_h.fEnergy1[i] = g4mat->GetIonisation()->GetEnergy1fluct();
        materials.data_h.fEnergy2[i] = g4mat->GetIonisation()->GetEnergy2fluct();
        materials.data_h.fLogEnergy1[i] = g4mat->GetIonisation()->GetLogEnergy1fluct();
        materials.data_h.fLogEnergy2[i] = g4mat->GetIonisation()->GetLogEnergy2fluct();
        materials.data_h.fLogMeanExcitationEnergy[i] = g4mat->GetIonisation()->GetLogMeanExcEnergy();
        ++i;
    }

}


///:: Mains

// Load default elements database (wrapper to the class MaterialDataBase)
void MaterialManager::load_elements_database() {
    std::string filename = std::string(getenv("GGEMSHOME"));
    filename += "/data/elts.dat";
    m_material_db.load_elements(filename);
}

// Load default materials database (wrapper to the class MaterialDataBase)
void MaterialManager::load_materials_database() {
    std::string filename = std::string(getenv("GGEMSHOME"));
    filename += "/data/mats.dat";
    m_material_db.load_materials(filename);
}

// Load elements database from a given file (wrapper to the class MaterialDataBase)
void MaterialManager::load_elements_database(std::string filename) {
    m_material_db.load_elements(filename);
}

// Load materials database from a given file (wrapper to the class MaterialDataBase)
void MaterialManager::load_materials_database(std::string filename) {
    m_material_db.load_materials(filename);
}

// Add materials to the main list and update the corresponding indices
void MaterialManager::add_materials_and_update_indices(std::vector<std::string> mats_list, ui16 *data, ui32 ndata) {


    ui16 local_id_mat=0; while (local_id_mat<mats_list.size()) {

        ui16 glb_id_mat = m_get_material_index(mats_list[local_id_mat]);

        // If the material index is different to the mats_list index,
        // the object local material list have to be
        // re-index considering the main (and global) material list
        if (glb_id_mat != local_id_mat) {

            ui32 i=0; while (i<ndata) {
                if (data[i] == local_id_mat) {
                    data[i] = glb_id_mat;
                }
                ++i;
            }

        }

        ++local_id_mat;
    }


}



//// Build the materials table according the object contains in the world
//void MaterialManager::free_materials_table() {


//    free(materials_table.nb_elements);
//    free(materials_table.index);
//    free(materials_table.nb_atoms_per_vol);
//    free(materials_table.nb_electrons_per_vol);
//    free(materials_table.electron_mean_excitation_energy);
//    free(materials_table.rad_length);
//    free(materials_table.fX0);
//    free(materials_table.fX1);
//    free(materials_table.fD0);
//    free(materials_table.fC);
//    free(materials_table.fA);
//    free(materials_table.fM);
//    free(materials_table.density);
//    free(materials_table.fF1);
//    free(materials_table.fF2);
//    free(materials_table.fEnergy0);
//    free(materials_table.fEnergy1);
//    free(materials_table.fEnergy2);
//    free(materials_table.fLogEnergy1);
//    free(materials_table.fLogEnergy2);
//    free(materials_table.fLogMeanExcitationEnergy);
//    free(materials_table.mixture);
//    free(materials_table.atom_num_dens);


//    //delete mat_table_h;

//}



// Init
void MaterialManager::initialize(GlobalSimulationParameters params) {
    
    ///WARNING fonction m_build_materials_table deplacee au dessus de check. Non verifié
    
    // Build materials table
    m_build_materials_table(params);
 
    // Check if everything was set properly
    if ( !m_check_mandatory() ) {
        print_error("Missing materials definition!");
        exit_simulation();
    }


    // Copy data to the GPU
    if (params.data_h.device_target == GPU_DEVICE) m_copy_materials_table_cpu2gpu();
}





#endif
