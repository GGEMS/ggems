// This file is part of GGEMS
//
// FIREwork is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FIREwork is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with FIREwork.  If not, see <http://www.gnu.org/licenses/>.
//
// GGEMS Copyright (C) 2013-2014 Julien Bert

#ifndef MATERIALS_CU
#define MATERIALS_CU

#include "materials.cuh"

//////////////////////////////////////////////////////////////////
//// Material class //////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

Material::Material() {} // To handle one material

//////////////////////////////////////////////////////////////////
//// MaterialDataBase class //////////////////////////////////////
//////////////////////////////////////////////////////////////////

MaterialDataBase::MaterialDataBase() {} // Open and load the material database

// Read material name
std::string MaterialDataBase::read_material_name(std::string txt) {
    return txt.substr(0, txt.find(":"));
}

// Read material density
float MaterialDataBase::read_material_density(std::string txt) {
    float res;
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
unsigned short int MaterialDataBase::read_material_nb_elements(std::string txt) {
    unsigned short int res;
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
float MaterialDataBase::read_material_fraction(std::string txt) {
    float res;
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
    float mat_f;
    unsigned short int i;
    unsigned short int ind = 0;

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
int MaterialDataBase::read_element_Z(std::string txt) {
    int res;
    txt = txt.substr(txt.find("Z=")+2);
    txt = txt.substr(0, txt.find("."));
    std::stringstream(txt) >> res;
    return res;
}

// Read element A
float MaterialDataBase::read_element_A(std::string txt) {
    float res;
    txt = txt.substr(txt.find("A=")+2);
    txt = txt.substr(0, txt.find("g/mole"));
    std::stringstream(txt) >> res;
    return res *gram/mole;
}


// Load elements from data file
void MaterialDataBase::load_elements(std::string filename) {
    std::ifstream file(filename.c_str());

    std::string line, elt_name;
    int elt_Z;
    float elt_A;

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
    char c;
    char line[1024];
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
void MaterialBuilder::load_elements(std::string filename) {
    MaterialDataBase.load_elements(filename);
}

// Load materials database (wrapper to the class MaterialDataBase)
void MaterialBuilder::load_materials(std::string filename) {
    MaterialDataBase.load_materials(filename);
}

// Build the materials table according the object contains in the world
void MaterialBuilder::get_materials_table_from_world(Geometry World) {

    // First allocated data to the structure according the number of materials
    MaterialsTable.nb_materials = World.materials_list.size();
    MaterialsTable.nb_elements = (unsigned short int*)malloc(sizeof(unsigned short int)*MaterialsTable.nb_materials);
    MaterialsTable.index = (unsigned short int*)malloc(sizeof(unsigned short int)*MaterialsTable.nb_materials);
    MaterialsTable.nb_atoms_per_vol = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);
    MaterialsTable.nb_electrons_per_vol = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);
    MaterialsTable.electron_mean_excitation_energy = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);
    MaterialsTable.rad_length = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);
    MaterialsTable.fX0 = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);
    MaterialsTable.fX1 = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);
    MaterialsTable.fD0 = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);
    MaterialsTable.fC = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);
    MaterialsTable.fA = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);
    MaterialsTable.fM = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);
    MaterialsTable.density = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);
    MaterialsTable.fF1 = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);
    MaterialsTable.fF2 = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);
    MaterialsTable.fEnergy0 = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);
    MaterialsTable.fEnergy1 = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);
    MaterialsTable.fEnergy2 = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);
    MaterialsTable.fLogEnergy1 = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);
    MaterialsTable.fLogEnergy2 = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);
    MaterialsTable.fLogMeanExcitationEnergy = (float*)malloc(sizeof(float)*MaterialsTable.nb_materials);

    int i, j;
    unsigned int access_index = 0;
    unsigned int fill_index = 0;
    std::string mat_name, elt_name;
    Material cur_mat;

    i=0; while (i < MaterialsTable.nb_materials) {
        // get mat name
        mat_name = World.materials_list[i];

        // read mat from databse
        cur_mat = material_db.materials[mat_name];
        if (cur_mat.name == "") {
            printf("[ERROR] Material %s is not on your database (%s function)\n", mat_name.c_str(),__FUNCTION__);
            exit(EXIT_FAILURE);
        }
        // get nb of elements
        MaterialsTable.nb_elements[i] = cur_mat.nb_elements;

        // compute index
        MaterialsTable.index[i] = access_index;
        access_index += cur_mat.nb_elements;

        ++i;
    }

    // nb of total elements
    MaterialsTable.nb_elements_total = access_index;
    MaterialsTable.mixture = (unsigned short int*)malloc(sizeof(unsigned short int)*access_index);
    MaterialsTable.atom_num_dens = (float*)malloc(sizeof(float)*access_index);

    // store mixture element and compute atomic density
    i=0; while (i < MaterialsTable.nb_materials) {

        // get mat name
        mat_name = m_list_of_materials[i];

        // read mat from database
        cur_mat = db.materials_database[mat_name];

        // get density
        MaterialsTable.density[i] = cur_mat.density / gramme;

        // G4 material
        G4Material *g4mat = new G4Material("tmp", cur_mat.density, cur_mat.nb_elements);

        MaterialsTable.nb_atoms_per_vol[i] = 0.0f;
        MaterialsTable.nb_electrons_per_vol[i] = 0.0f;

        j=0; while (j < cur_mat.nb_elements) {
            // read element name
            elt_name = cur_mat.mixture_Z[j];

            // store Z
            MaterialsTable.mixture[fill_index] = db.elements_Z[elt_name];

            // compute atom num dens (Avo*fraction*dens) / Az
            MaterialsTable.atom_num_dens[fill_index] = Avogadro/db.elements_A[elt_name] *
                                                       cur_mat.mixture_f[j]*cur_mat.density;

            // compute nb atoms per volume
            MaterialsTable.nb_atoms_per_vol[i] += MaterialsTable.atom_num_dens[fill_index];

            // compute nb electrons per volume
            MaterialsTable.nb_electrons_per_vol[i] += MaterialsTable.atom_num_dens[fill_index] *
                                                      db.elements_Z[elt_name];

            // build G4 material
            G4Element *elt = new G4Element("element", "ELT", db.elements_Z[elt_name],
                                                             db.elements_A[elt_name]);
            g4mat->AddElement(elt, cur_mat.mixture_f[j]);

            ++j;
            ++fill_index;
        }

        // electron data
        MaterialsTable.electron_mean_excitation_energy[i] = g4mat->GetIonisation()->GetMeanExcitationEnergy();
        MaterialsTable.rad_length[i] = g4mat->GetRadlen();

        // eIonisation correction
        MaterialsTable.fX0[i] = g4mat->GetIonisation()->GetX0density();
        MaterialsTable.fX1[i] = g4mat->GetIonisation()->GetX1density();
        MaterialsTable.fD0[i] = g4mat->GetIonisation()->GetD0density();
        MaterialsTable.fC[i] = g4mat->GetIonisation()->GetCdensity();
        MaterialsTable.fA[i] = g4mat->GetIonisation()->GetAdensity();
        MaterialsTable.fM[i] = g4mat->GetIonisation()->GetMdensity();

        //eFluctuation parameters
        MaterialsTable.fF1[i] = g4mat->GetIonisation()->GetF1fluct();
        MaterialsTable.fF2[i] = g4mat->GetIonisation()->GetF2fluct();
        MaterialsTable.fEnergy0[i] = g4mat->GetIonisation()->GetEnergy0fluct();
        MaterialsTable.fEnergy1[i] = g4mat->GetIonisation()->GetEnergy1fluct();
        MaterialsTable.fEnergy2[i] = g4mat->GetIonisation()->GetEnergy2fluct();
        MaterialsTable.fLogEnergy1[i] = g4mat->GetIonisation()->GetLogEnergy1fluct();
        MaterialsTable.fLogEnergy2[i] = g4mat->GetIonisation()->GetLogEnergy2fluct();
        MaterialsTable.fLogMeanExcitationEnergy[i] = g4mat->GetIonisation()->GetLogMeanExcEnergy();
        ++i;
    }

}








































































#endif
