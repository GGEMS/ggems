// This file is part of GGEMS
//
// GGEMS is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// GGEMS is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with GGEMS.  If not, see <http://www.gnu.org/licenses/>.
//
// GGEMS Copyright (C) 2013-2014 Julien Bert

#ifndef VOXELIZED_SOURCE_CU
#define VOXELIZED_SOURCE_CU

#include "voxelized_source.cuh"

//// External function
__host__ __device__ void voxelized_source_primary_generator(ParticleStack particles, ui32 id,
                                                            f32 *cdf_index, f32 *cdf_act, ui32 nb_acts,
                                                            f32 px, f32 py, f32 pz,
                                                            ui32 nb_vox_x, ui32 nb_vox_y, ui32 nb_vox_z,
                                                            f32 sx, f32 sy, f32 sz,
                                                            f32 energy, ui8 type, ui32 geom_id) {

    f32 jump = (f32)(nb_vox_x*nb_vox_y);
    f32 ind, x, y, z;

    // use cdf to find the next emission spot
    f32 rnd = JKISS32(particles, id);
    ui32 pos = binary_search(rnd, cdf_act, nb_acts);
    
    // convert position index to emitted position
    ind = cdf_index[pos];
    z = floor(ind / jump);
    ind -= (z*jump);
    y = floor(ind / (f32)nb_vox_x);
    x = ind - y*nb_vox_x;

    // random positon within the voxel
    x += JKISS32(particles, id);
    y += JKISS32(particles, id);
    z += JKISS32(particles, id);

    // Due to float operation aproximation: 1+(1-Epsilon) = 2
    // we need to check that x, y, z are not equal to the size of the vox source
    // x, y, z must be in [0, size[
    if (x == nb_vox_x) x -= EPSILON3;
    if (y == nb_vox_y) y -= EPSILON3;
    if (z == nb_vox_z) z -= EPSILON3;

    // convert in mm
    x *= sx;
    y *= sy;
    z *= sz;

    // shift according to center of phantom and translation
    x = x - nb_vox_x*sx*0.5 + px;
    y = y - nb_vox_y*sy*0.5 + py;
    z = z - nb_vox_z*sz*0.5 + pz;

    // random orientation
    f32 phi = JKISS32(particles, id);
    f32 theta = JKISS32(particles, id);
    phi *= gpu_twopi;
    theta = acosf(1.0f - 2.0f*theta);

    // compute direction vector
    f32 dx = cos(phi)*sin(theta);
    f32 dy = sin(phi)*sin(theta);
    f32 dz = cos(theta);

    // set particle stack 1
    particles.E[id] = energy;
    particles.dx[id] = dx;
    particles.dy[id] = dy;
    particles.dz[id] = dz;
    particles.px[id] = x;
    particles.py[id] = y;
    particles.pz[id] = z;
    particles.tof[id] = 0.0;
    particles.endsimu[id] = PARTICLE_ALIVE;
    particles.level[id] = PRIMARY;
    particles.pname[id] = type;
    particles.geometry_id[id] = geom_id;

    // set particle stack 2
    ++id;
    particles.E[id] = energy;
    particles.dx[id] = -dx;    // back2back
    particles.dy[id] = -dy;
    particles.dz[id] = -dz;
    particles.px[id] = x;
    particles.py[id] = y;
    particles.pz[id] = z;
    particles.tof[id] = 0.0;
    particles.endsimu[id] = PARTICLE_ALIVE;
    particles.level[id] = PRIMARY;
    particles.pname[id] = type;
    particles.geometry_id[id] = geom_id;
}

__host__ __device__ void voxelized_source_primary_mono_generator(ParticleStack particles, ui32 id,
                                                            f32 *cdf_index, f32 *cdf_act, ui32 nb_acts,
                                                            f32 px, f32 py, f32 pz,
                                                            ui32 nb_vox_x, ui32 nb_vox_y, ui32 nb_vox_z,
                                                            f32 sx, f32 sy, f32 sz,
                                                            f32 energy, ui8 type, ui32 geom_id) {

    f32 jump = (f32)(nb_vox_x*nb_vox_y);
    f32 ind, x, y, z;

    // use cdf to find the next emission spot
    f32 rnd = JKISS32(particles, id);
    ui32 pos = binary_search(rnd, cdf_act, nb_acts);

    // convert position index to emitted position
    ind = cdf_index[pos];
    z = floor(ind / jump);
    ind -= (z*jump);
    y = floor(ind / (f32)nb_vox_x);
    x = ind - y*nb_vox_x;
    
    // random positon within the voxel
    x += JKISS32(particles, id);
    y += JKISS32(particles, id);
    z += JKISS32(particles, id);

    // Due to float operation aproximation: 1+(1-Epsilon) = 2
    // we need to check that x, y, z are not equal to the size of the vox source
    // x, y, z must be in [0, size[
    if (x == nb_vox_x) x -= EPSILON3;
    if (y == nb_vox_y) y -= EPSILON3;
    if (z == nb_vox_z) z -= EPSILON3;

    // convert in mm
    x *= sx;
    y *= sy;
    z *= sz;

    // shift according to center of phantom and translation
    x = x - nb_vox_x*sx*0.5 + px;
    y = y - nb_vox_y*sy*0.5 + py;
    z = z - nb_vox_z*sz*0.5 + pz;

    // random orientation
    f32 phi = JKISS32(particles, id);
    f32 theta = JKISS32(particles, id);
    phi *= gpu_twopi;
    theta = acosf(1.0f - 2.0f*theta);

    // compute direction vector
    f32 dx = cos(phi)*sin(theta);
    f32 dy = sin(phi)*sin(theta);
    f32 dz = cos(theta);

    // set particle stack 1
    particles.E[id] = energy;
    particles.dx[id] = dx;
    particles.dy[id] = dy;
    particles.dz[id] = dz;
    particles.px[id] = x;
    particles.py[id] = y;
    particles.pz[id] = z;
    particles.tof[id] = 0.0;
    particles.endsimu[id] = PARTICLE_ALIVE;
    particles.level[id] = PRIMARY;
    particles.pname[id] = type;
    particles.geometry_id[id] = geom_id;
}

VoxelizedSource::VoxelizedSource() {
    // Default values
    seed=10;
    geometry_id=0;
    source_name="VoxSrc01";
    source_type="back2back";
    px=0.0; py=0.0; pz=0.0;
    energy=511*keV;

    // Init pointer
    activity_volume = NULL;
    activity_cdf = NULL;
    activity_index = NULL;
}

void VoxelizedSource::set_position(f32 vpx, f32 vpy, f32 vpz) {
    px = vpx; py = vpy; pz = vpz;
}

void VoxelizedSource::set_energy(f32 venergy) {
    energy = venergy;
}

void VoxelizedSource::set_histpoint(f32 venergy, f32 vpart) {
      energy_hist.push_back(venergy);
      partpdec.push_back(vpart);
}  

void VoxelizedSource::set_source_type(std::string vtype) {
    source_type = vtype;
}

void VoxelizedSource::set_seed(ui32 vseed) {
    seed = vseed;
}

void VoxelizedSource::set_in_geometry(ui32 vgeometry_id) {
    geometry_id = vgeometry_id;
}

void VoxelizedSource::set_source_name(std::string vsource_name) {
    source_name = vsource_name;
}

//// MHD //////////////////////////////////////////////////////:

// Skip comment starting with "#"
void VoxelizedSource::skip_comment(std::istream & is) {
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
std::string VoxelizedSource::remove_white_space(std::string txt) {
    txt.erase(remove_if(txt.begin(), txt.end(), isspace), txt.end());
    return txt;
}

// Read mhd key
std::string VoxelizedSource::read_mhd_key(std::string txt) {
    txt = txt.substr(0, txt.find("="));
    return remove_white_space(txt);
}

// Read string mhd arg
std::string VoxelizedSource::read_mhd_string_arg(std::string txt) {
    txt = txt.substr(txt.find("=")+1);
    return remove_white_space(txt);
}

// Read i32 mhd arg
i32 VoxelizedSource::read_mhd_int(std::string txt) {
    i32 res;
    txt = txt.substr(txt.find("=")+1);
    txt = remove_white_space(txt);
    std::stringstream(txt) >> res;
    return res;
}

// Read int mhd arg
i32 VoxelizedSource::read_mhd_int_atpos(std::string txt, i32 pos) {
    i32 res;
    txt = txt.substr(txt.find("=")+2);
    if (pos==0) {
        txt = txt.substr(0, txt.find(" "));
    }
    if (pos==1) {
        txt = txt.substr(txt.find(" ")+1);
        txt = txt.substr(0, txt.find(" "));
    }
    if (pos==2) {
        txt = txt.substr(txt.find(" ")+1);
        txt = txt.substr(txt.find(" ")+1);
    }
    std::stringstream(txt) >> res;
    return res;
}

// Read f32 mhd arg
f32 VoxelizedSource::read_mhd_f32_atpos(std::string txt, i32 pos) {
    f32 res;
    txt = txt.substr(txt.find("=")+2);
    if (pos==0) {
        txt = txt.substr(0, txt.find(" "));
    }
    if (pos==1) {
        txt = txt.substr(txt.find(" ")+1);
        txt = txt.substr(0, txt.find(" "));
    }
    if (pos==2) {
        txt = txt.substr(txt.find(" ")+1);
        txt = txt.substr(txt.find(" ")+1);
    }
    std::stringstream(txt) >> res;
    return res;
}

// Load activities from mhd file (only f32 data)
void VoxelizedSource::load_from_mhd(std::string filename) {

    /////////////// First read the MHD file //////////////////////

    std::string line, key;
    nb_vox_x=0, nb_vox_y=0, nb_vox_z=0;
    spacing_x=0, spacing_y=0, spacing_z=0;

    // Watchdog
    std::string ObjectType="", BinaryData="", BinaryDataByteOrderMSB="", CompressedData="",
                ElementType="", ElementDataFile="";
    i32 NDims=0;

    // Read range file
    std::ifstream file(filename.c_str());
    if(!file) { printf("Error, file %s not found \n", filename.c_str()); exit(EXIT_FAILURE);}
    while (file) {
        skip_comment(file);
        std::getline(file, line);

        if (file) {
            key = read_mhd_key(line);
            if (key=="ObjectType")              ObjectType = read_mhd_string_arg(line);
            if (key=="NDims")                   NDims = read_mhd_int(line);
            if (key=="BinaryData")              BinaryData = read_mhd_string_arg(line);
            if (key=="BinaryDataByteOrderMSB")  BinaryDataByteOrderMSB=read_mhd_string_arg(line);
            if (key=="CompressedData")          CompressedData = read_mhd_string_arg(line);
            //if (key=="TransformMatrix") printf("Matrix\n");
            //if (key=="Offset")  printf("Offset\n");
            //if (key=="CenterOfRotation") printf("CoR\n");
            if (key=="ElementSpacing") {
                                                spacing_x=read_mhd_f32_atpos(line, 0);
                                                spacing_y=read_mhd_f32_atpos(line, 1);
                                                spacing_z=read_mhd_f32_atpos(line, 2);
            }
            if (key=="DimSize") {
                                                nb_vox_x=read_mhd_int_atpos(line, 0);
                                                nb_vox_y=read_mhd_int_atpos(line, 1);
                                                nb_vox_z=read_mhd_int_atpos(line, 2);
            }

            //if (key=="AnatomicalOrientation") printf("Anato\n");
            if (key=="ElementType")             ElementType = read_mhd_string_arg(line);
            if (key=="ElementDataFile")         ElementDataFile = read_mhd_string_arg(line);
        }

    } // read file

    // Check header
    if (ObjectType != "Image") {
        printf("Error, mhd header: ObjectType = %s\n", ObjectType.c_str());
        exit(EXIT_FAILURE);
    }
    if (BinaryData != "True") {
        printf("Error, mhd header: BinaryData = %s\n", BinaryData.c_str());
        exit(EXIT_FAILURE);
    }
    if (BinaryDataByteOrderMSB != "False") {
        printf("Error, mhd header: BinaryDataByteOrderMSB = %s\n", BinaryDataByteOrderMSB.c_str());
        exit(EXIT_FAILURE);
    }
    if (CompressedData != "False") {
        printf("Error, mhd header: CompressedData = %s\n", CompressedData.c_str());
        exit(EXIT_FAILURE);
    }
    if (ElementType != "MET_FLOAT") {
        printf("Error, mhd header: ElementType = %s\n", ElementType.c_str());
        exit(EXIT_FAILURE);
    }
    if (ElementDataFile == "") {
        printf("Error, mhd header: ElementDataFile = %s\n", ElementDataFile.c_str());
        exit(EXIT_FAILURE);
    }
    if (NDims != 3) {
        printf("Error, mhd header: NDims = %i\n", NDims);
        exit(EXIT_FAILURE);
    }

    if (nb_vox_x == 0 || nb_vox_y == 0 || nb_vox_z == 0 ||
            spacing_x == 0 || spacing_y == 0 || spacing_z == 0) {
        printf("Error when loading mhd file (unknown dimension and spacing)\n");
        printf("   => dim %i %i %i - spacing %f %f %f\n", nb_vox_x, nb_vox_y, nb_vox_z,
                                                          spacing_x, spacing_y, spacing_z);
        exit(EXIT_FAILURE);
    }
    // Read data
    FILE *pfile = fopen(ElementDataFile.c_str(), "rb");
    if (!pfile) {
        std::string nameWithRelativePath = filename;
        i32 lastindex = nameWithRelativePath.find_last_of(".");
        nameWithRelativePath = nameWithRelativePath.substr(0, lastindex);
        nameWithRelativePath+=".raw";
        pfile = fopen(nameWithRelativePath.c_str(), "rb");
        if (!pfile) {
            printf("Error when loading mhd file: %s\n", ElementDataFile.c_str());
            exit(EXIT_FAILURE);
        }
    }

    number_of_voxels = nb_vox_x*nb_vox_y*nb_vox_z;

    activity_volume = (f32*)malloc(sizeof(f32) * number_of_voxels);
    fread(activity_volume, sizeof(f32), number_of_voxels, pfile);
    fclose(pfile);

    // Compute the associated CDF of the activities
    compute_cdf();

}

// Compute the CDF of the activities
void VoxelizedSource::compute_cdf() {

    // count nb of non zeros activities
    ui32 nb=0;
    ui32 i=0; while (i<number_of_voxels) {
        if (activity_volume[i] != 0.0f) ++nb;
        ++i;
    }
    activity_size = nb;

    // mem allocation
    activity_index = (f32*)malloc(nb*sizeof(f32));
    activity_cdf = (f32*)malloc(nb*sizeof(f32));

    // Buffer
    f64* cdf = new f64[nb];

    // fill array with non zeros values activity
    ui32 index = 0;
    f64 val;
    f64 sum = 0.0; // for the cdf
    i=0; while (i<number_of_voxels) {
        val = activity_volume[i];
        if (val != 0.0f) {
            activity_index[index] = i;
            cdf[index] = val;
            sum += val;
            //printf("cdf i %d val %lf \n", index, cdf[index]);
            ++index;
        }
        ++i;
    }
    tot_activity = sum;
    printf("tot_activity %lf \n", sum);
    
    // compute cummulative density function
    cdf[0] /= sum;
    activity_cdf[0] = cdf[0];
      
    i = 1; while (i<nb) {
       // printf("i %d test div %4.12lf \n", i, (cdf[i]/sum));
        cdf[i] = (cdf[i]/sum) + cdf[i-1];
        activity_cdf[i]= (f32) cdf[i];
       // printf("i %d test div %4.12lf \n", i, cdf[i]);
        ++i;
    }

    delete cdf;

}









#endif
