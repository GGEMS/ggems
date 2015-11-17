// GGEMS Copyright (C) 2015

#ifndef MATERIALS_CU
#define MATERIALS_CU

#include "materials.cuh"

//////////////////////////////////////////////////////////////////
//// MaterialDataBase class //////////////////////////////////////
//////////////////////////////////////////////////////////////////

MaterialDataBase::MaterialDataBase() {} // Open and load the material database

// Read material name
std::string MaterialDataBase::read_material_name(std::string txt) {
    return txt.substr(0, txt.find(":"));
}

// Read material density
f32 MaterialDataBase::read_material_density(std::string txt) {
    f32 res;
    // density
    txt = txt.substr(txt.find("d=")+2);
    std::string txt1 = txt.substr(0, txt.find(" "));
    std::stringstream(txt1) >> res;
    // unit
    txt = txt.substr(txt.find(" ")+1);
    txt = txt.substr(0, txt.find(";"));
    txt = remove_white_space(txt);
    if (txt=="g/cm3")  return res *gram/cm3;
    if (txt=="mg/cm3") return res *mg/cm3;
        printf("read densité %f\n",res);
    return res;

}

// Read material number of elements
ui16 MaterialDataBase::read_material_nb_elements(std::string txt) {
    ui16 res;
    txt = txt.substr(txt.find("n=")+2);
    txt = txt.substr(0, txt.find(";"));
    txt = remove_white_space(txt);
    std::stringstream(txt) >> res;
    return res;
}

// Read material element name
std::string MaterialDataBase::read_material_element(std::string txt) {
    txt = txt.substr(txt.find("name=")+5);
    txt = txt.substr(0, txt.find(";"));
    txt = remove_white_space(txt);
    return txt;
}

// Read material element fraction TODO Add compound definition
f32 MaterialDataBase::read_material_fraction(std::string txt) {
    f32 res;
    txt = txt.substr(txt.find("f=")+2);
    txt = remove_white_space(txt);
    std::stringstream(txt) >> res;
    return res;
}

// Load materials from data base
void MaterialDataBase::load_materials(std::string filename) {
    //printf("load material ... \n");
    std::ifstream file(filename.c_str());

    std::string line, elt_name;
    f32 mat_f;
    ui16 i;
    ui16 ind = 0;

    while (file) {
        skip_comment(file);
        std::getline(file, line);

        if (file) {
            Material mat;
            mat.name = read_material_name(line);
            //printf("mat name ... %s \n",mat.name.c_str());   // Too much verbose - JB
            mat.density = read_material_density(line);
            mat.nb_elements = read_material_nb_elements(line);

            i=0; while (i<mat.nb_elements) {
                std::getline(file, line);
                elt_name = read_material_element(line);
                mat_f = read_material_fraction(line);

                mat.mixture_Z.push_back(elt_name);
                mat.mixture_f.push_back(mat_f);

                ++i;
            }

            materials[mat.name] = mat;
            ++ind;

        } // if

    } // while

}

// Read element name
std::string MaterialDataBase::read_element_name(std::string txt) {
    return txt.substr(0, txt.find(":"));
}

// Read element Z
i32 MaterialDataBase::read_element_Z(std::string txt) {
    i32 res;
    txt = txt.substr(txt.find("Z=")+2);
    txt = txt.substr(0, txt.find("."));
    std::stringstream(txt) >> res;
    return res;
}

// Read element A
f32 MaterialDataBase::read_element_A(std::string txt) {
    f32 res;
    txt = txt.substr(txt.find("A=")+2);
    txt = txt.substr(0, txt.find("g/mole"));
    std::stringstream(txt) >> res;
    return res *gram/mole;
}


// Load elements from data file
void MaterialDataBase::load_elements(std::string filename) {
    std::ifstream file(filename.c_str());

    std::string line, elt_name;
    i32 elt_Z;
    f32 elt_A;

    while (file) {
        skip_comment(file);
        std::getline(file, line);

        if (file) {
            elt_name = read_element_name(line);
            elt_Z = read_element_Z(line);
            elt_A = read_element_A(line);

            elements_Z[elt_name] = elt_Z;
            elements_A[elt_name] = elt_A;
        }

    }
}

// Skip comment starting with "#"
void MaterialDataBase::skip_comment(std::istream & is) {
    i8 c;
    i8 line[1024];
    if (is.eof()) return;
    is >> c;
    while (is && (c=='#')) {
        is.getline(line, 1024);
        is >> c;
        if (is.eof()) return;
    }
    is.unget();
}

// Remove all white space
std::string MaterialDataBase::remove_white_space(std::string txt) {
    txt.erase(remove_if(txt.begin(), txt.end(), isspace), txt.end());
    return txt;
}


//////////////////////////////////////////////////////////////////
//// MaterialBuilder class //////////////////////////////////////
//////////////////////////////////////////////////////////////////

MaterialBuilder::MaterialBuilder() {} // // This class is used to build the material table

// Load elements database (wrapper to the class MaterialDataBase)
void MaterialBuilder::load_elements_database(std::string filename) {
    material_db.load_elements(filename);
}

// Load materials database (wrapper to the class MaterialDataBase)
void MaterialBuilder::load_materials_database(std::string filename) {
    material_db.load_materials(filename);
}

// Build the materials table according the object contains in the world
void MaterialBuilder::free_materials_table() {

    /*
    free(materials_table.nb_elements);
    free(materials_table.index);
    free(materials_table.nb_atoms_per_vol);
    free(materials_table.nb_electrons_per_vol);
    free(materials_table.electron_mean_excitation_energy);
    free(materials_table.rad_length);
    free(materials_table.fX0);
    free(materials_table.fX1);
    free(materials_table.fD0);
    free(materials_table.fC);
    free(materials_table.fA);
    free(materials_table.fM);
    free(materials_table.density);
    free(materials_table.fF1);
    free(materials_table.fF2);
    free(materials_table.fEnergy0);
    free(materials_table.fEnergy1);
    free(materials_table.fEnergy2);
    free(materials_table.fLogEnergy1);
    free(materials_table.fLogEnergy2);
    free(materials_table.fLogMeanExcitationEnergy);
    free(materials_table.mixture);
    free(materials_table.atom_num_dens);
    */

    //delete mat_table_h;

}

/*
// Build the materials table according the object contains in the world
void MaterialBuilder::get_materials_table_from_world(GeometryBuilder World) {

    // First allocated data to the structure according the number of materials
    materials_table.nb_materials = World.materials_list.size();
    materials_table.nb_elements = (ui16*)malloc(sizeof(ui16)*materials_table.nb_materials);
    materials_table.index = (ui16*)malloc(sizeof(ui16)*materials_table.nb_materials);
    materials_table.nb_atoms_per_vol = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);
    materials_table.nb_electrons_per_vol = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);
    materials_table.electron_mean_excitation_energy = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);
    materials_table.rad_length = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);
    materials_table.fX0 = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);
    materials_table.fX1 = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);
    materials_table.fD0 = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);
    materials_table.fC = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);
    materials_table.fA = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);
    materials_table.fM = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);
    materials_table.density = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);
    materials_table.fF1 = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);
    materials_table.fF2 = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);
    materials_table.fEnergy0 = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);
    materials_table.fEnergy1 = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);
    materials_table.fEnergy2 = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);
    materials_table.fLogEnergy1 = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);
    materials_table.fLogEnergy2 = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);
    materials_table.fLogMeanExcitationEnergy = (f32*)malloc(sizeof(f32)*materials_table.nb_materials);

    i32 i, j;
    ui32 access_index = 0;
    ui32 fill_index = 0;
    std::string mat_name, elt_name;
    Material cur_mat;

    i=0; while (i < materials_table.nb_materials) {
        // get mat name
        mat_name = World.materials_list[i];
        
        printf("Material %s \n", mat_name.c_str());

        // read mat from databse
        cur_mat = material_db.materials[mat_name];
        if (cur_mat.name == "") {
            printf("[ERROR] Material %s is not on your database (%s function)\n", mat_name.c_str(),__FUNCTION__);
            exit(EXIT_FAILURE);
        }
        // get nb of elements
        materials_table.nb_elements[i] = cur_mat.nb_elements;

        // compute index
        materials_table.index[i] = access_index;
        access_index += cur_mat.nb_elements;

        ++i;
    }

    // nb of total elements
    materials_table.nb_elements_total = access_index;
    materials_table.mixture = (ui16*)malloc(sizeof(ui16)*access_index);
    materials_table.atom_num_dens = (f32*)malloc(sizeof(f32)*access_index);

    // store mixture element and compute atomic density
    i=0; while (i < materials_table.nb_materials) {

        // get mat name
        mat_name = World.materials_list[i];

        // read mat from database
        cur_mat = material_db.materials[mat_name];

        // get density
        materials_table.density[i] = cur_mat.density / gram;

        // G4 material
        G4Material *g4mat = new G4Material("tmp", cur_mat.density, cur_mat.nb_elements);

        materials_table.nb_atoms_per_vol[i] = 0.0f;
        materials_table.nb_electrons_per_vol[i] = 0.0f;

        j=0; while (j < cur_mat.nb_elements) {
            // read element name
            elt_name = cur_mat.mixture_Z[j];

            // store Z
            materials_table.mixture[fill_index] = material_db.elements_Z[elt_name];

            // compute atom num dens (Avo*fraction*dens) / Az
            materials_table.atom_num_dens[fill_index] = Avogadro/material_db.elements_A[elt_name] *
                                                        cur_mat.mixture_f[j]*cur_mat.density;

            // compute nb atoms per volume
            materials_table.nb_atoms_per_vol[i] += materials_table.atom_num_dens[fill_index];

            // compute nb electrons per volume
            materials_table.nb_electrons_per_vol[i] += materials_table.atom_num_dens[fill_index] *
                                                       material_db.elements_Z[elt_name];

            // build G4 material
            G4Element *elt = new G4Element("element", "ELT", material_db.elements_Z[elt_name],
                                                             material_db.elements_A[elt_name]);
            g4mat->AddElement(elt, cur_mat.mixture_f[j]);

            ++j;
            ++fill_index;
        }

        // electron data
        materials_table.electron_mean_excitation_energy[i] = g4mat->GetIonisation()->GetMeanExcitationEnergy();
        materials_table.rad_length[i] = g4mat->GetRadlen();

        // eIonisation correction
        materials_table.fX0[i] = g4mat->GetIonisation()->GetX0density();
        materials_table.fX1[i] = g4mat->GetIonisation()->GetX1density();
        materials_table.fD0[i] = g4mat->GetIonisation()->GetD0density();
        materials_table.fC[i] = g4mat->GetIonisation()->GetCdensity();
        materials_table.fA[i] = g4mat->GetIonisation()->GetAdensity();
        materials_table.fM[i] = g4mat->GetIonisation()->GetMdensity();

        //eFluctuation parameters
        materials_table.fF1[i] = g4mat->GetIonisation()->GetF1fluct();
        materials_table.fF2[i] = g4mat->GetIonisation()->GetF2fluct();
        materials_table.fEnergy0[i] = g4mat->GetIonisation()->GetEnergy0fluct();
        materials_table.fEnergy1[i] = g4mat->GetIonisation()->GetEnergy1fluct();
        materials_table.fEnergy2[i] = g4mat->GetIonisation()->GetEnergy2fluct();
        materials_table.fLogEnergy1[i] = g4mat->GetIonisation()->GetLogEnergy1fluct();
        materials_table.fLogEnergy2[i] = g4mat->GetIonisation()->GetLogEnergy2fluct();
        materials_table.fLogMeanExcitationEnergy[i] = g4mat->GetIonisation()->GetLogMeanExcEnergy();
        ++i;
    }

}
*/

// Copy data to the GPU
void MaterialBuilder::copy_materials_table_cpu2gpu() {

    ui32 n = mat_table_h.nb_materials;
    ui32 k = mat_table_h.nb_elements_total;

    // First allocate the GPU mem for the scene
    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.nb_elements, n*sizeof(ui16)) );
    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.index, n*sizeof(ui16)) );

    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.mixture, k*sizeof(ui16)) );
    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.atom_num_dens, k*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.nb_atoms_per_vol, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.nb_electrons_per_vol, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.electron_mean_excitation_energy, n*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.rad_length, n*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.fX0, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.fX1, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.fD0, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.fC, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.fA, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.fM, n*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.fF1, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.fF2, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.fEnergy0, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.fEnergy1, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.fEnergy2, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.fLogEnergy1, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.fLogEnergy2, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.fLogMeanExcitationEnergy, n*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &mat_table_d.density, n*sizeof(f32)) );

    // Copy data to the GPU
    mat_table_d.nb_materials = mat_table_h.nb_materials;
    mat_table_d.nb_elements_total = mat_table_h.nb_elements_total;

    HANDLE_ERROR( cudaMemcpy(mat_table_d.nb_elements, mat_table_h.nb_elements,
                             n*sizeof(ui16), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(mat_table_d.index, mat_table_h.index,
                             n*sizeof(ui16), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy(mat_table_d.mixture, mat_table_h.mixture,
                             k*sizeof(ui16), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(mat_table_d.atom_num_dens, mat_table_h.atom_num_dens,
                             k*sizeof(f32), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy(mat_table_d.nb_atoms_per_vol, mat_table_h.nb_atoms_per_vol,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(mat_table_d.nb_electrons_per_vol, mat_table_h.nb_electrons_per_vol,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(mat_table_d.electron_mean_excitation_energy, mat_table_h.electron_mean_excitation_energy,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(mat_table_d.rad_length, mat_table_h.rad_length,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy(mat_table_d.fX0, mat_table_h.fX0,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(mat_table_d.fX1, mat_table_h.fX1,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(mat_table_d.fD0, mat_table_h.fD0,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(mat_table_d.fC, mat_table_h.fC,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(mat_table_d.fA, mat_table_h.fA,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(mat_table_d.fM, mat_table_h.fM,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy(mat_table_d.fF1, mat_table_h.fF1,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(mat_table_d.fF2, mat_table_h.fF2,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(mat_table_d.fEnergy0, mat_table_h.fEnergy0,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(mat_table_d.fEnergy1, mat_table_h.fEnergy1,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(mat_table_d.fEnergy2, mat_table_h.fEnergy2,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(mat_table_d.fLogEnergy1, mat_table_h.fLogEnergy1,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(mat_table_d.fLogEnergy2, mat_table_h.fLogEnergy2,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(mat_table_d.fLogMeanExcitationEnergy, mat_table_h.fLogMeanExcitationEnergy,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy(mat_table_d.density, mat_table_h.density,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );

}







































































#endif
