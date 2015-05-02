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

#ifndef DIGITIZER_CU
#define DIGITIZER_CU

#include "digitizer.cuh"

Digitizer::Digitizer() {
    pulses.size = 0;
    global_time = 0;
    E_low = 0.0;
    E_high = F64_MAX;

    flag_singles = false;
    flag_coincidences = false;
      
    flag_sp_blurring = false;
    flag_energy_blurring = false;
    
    flag_projXY = false;
    flag_projYZ = false;
    flag_projXZ = false;
    
    flag_spect_proj = false;
    
    nb_proj = 1;
    index_run = 1;
    
}

// Allocate and init the singles list file (CPU)
void Digitizer::cpu_init_pulses(ui32 nb) {
    pulses.size = nb;

    // Block Pulse 1
    pulses.pu1_px = (f32*)malloc(pulses.size*sizeof(f32));
    pulses.pu1_py = (f32*)malloc(pulses.size*sizeof(f32));
    pulses.pu1_pz = (f32*)malloc(pulses.size*sizeof(f32));
    pulses.pu1_E = (f32*)malloc(pulses.size*sizeof(f32));
    pulses.pu1_tof = (f32*)malloc(pulses.size*sizeof(f32));
    pulses.pu1_id_part = (ui32*)malloc(pulses.size*sizeof(ui32));
    pulses.pu1_id_geom = (ui32*)malloc(pulses.size*sizeof(ui32));
    pulses.pu1_nb_hits = (ui32*)malloc(pulses.size*sizeof(ui32));
    // Block Pulse 2
    pulses.pu2_px = (f32*)malloc(pulses.size*sizeof(f32));
    pulses.pu2_py = (f32*)malloc(pulses.size*sizeof(f32));
    pulses.pu2_pz = (f32*)malloc(pulses.size*sizeof(f32));
    pulses.pu2_E = (f32*)malloc(pulses.size*sizeof(f32));
    pulses.pu2_tof = (f32*)malloc(pulses.size*sizeof(f32));
    pulses.pu2_id_part = (ui32*)malloc(pulses.size*sizeof(ui32));
    pulses.pu2_id_geom = (ui32*)malloc(pulses.size*sizeof(ui32));
    pulses.pu2_nb_hits = (ui32*)malloc(pulses.size*sizeof(ui32));

    ui32 i=0; while (i<pulses.size) {
        pulses.pu1_nb_hits[i]=0;
        pulses.pu2_nb_hits[i]=0;
        ++i;
    }
}

// Allocate and init the singles list file (CPU)
void Digitizer::clear_cpu_pulses() {
 
    ui32 i=0; while (i<pulses.size) {
        pulses.pu1_nb_hits[i]=0;
        pulses.pu2_nb_hits[i]=0;
        ++i;
    }
}


// Allocate and init the singles list file (CPU)
void Digitizer::free_cpu_pulses() {

    // Block Pulse 1
    free(pulses.pu1_px);
    free(pulses.pu1_py);
    free(pulses.pu1_pz);
    free(pulses.pu1_E);
    free(pulses.pu1_tof);
    free(pulses.pu1_id_part);
    free(pulses.pu1_id_geom);
    free(pulses.pu1_nb_hits);
    // Block Pulse 2
    free(pulses.pu2_px);
    free(pulses.pu2_py);
    free(pulses.pu2_pz);
    free(pulses.pu2_E);
    free(pulses.pu2_tof);
    free(pulses.pu2_id_part);
    free(pulses.pu2_id_geom);
    free(pulses.pu2_nb_hits);
}


// Allocate and init the pulses list file (GPU)
void Digitizer::gpu_init_pulses(ui32 nb) {

    ui32 n = pulses.size;

    // First allocate mem on GPU
    HANDLE_ERROR( cudaMalloc((void**) &dpulses.pu1_px, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dpulses.pu1_py, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dpulses.pu1_pz, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dpulses.pu1_E, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dpulses.pu1_tof, n*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &dpulses.pu1_id_part, n*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dpulses.pu1_id_geom, n*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dpulses.pu1_nb_hits, n*sizeof(ui32)) );

    HANDLE_ERROR( cudaMalloc((void**) &dpulses.pu2_px, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dpulses.pu2_py, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dpulses.pu2_pz, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dpulses.pu2_E, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dpulses.pu2_tof, n*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &dpulses.pu2_id_part, n*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dpulses.pu2_id_geom, n*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dpulses.pu2_nb_hits, n*sizeof(ui32)) );

    // Init values
    ui32 *vec = (ui32*)malloc(n*sizeof(ui32));
    ui32 i=0; while (i<n) {
        vec[i]=0;
        ++i;
    }

    // Copy data to the GPU
    dpulses.size = pulses.size;

//    HANDLE_ERROR( cudaMemcpy(dpulses.pu1_px, pulses.pu1_px,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dpulses.pu1_py, pulses.pu1_py,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dpulses.pu1_pz, pulses.pu1_pz,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dpulses.pu1_E, pulses.pu1_E,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dpulses.pu1_tof, pulses.pu1_tof,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );

//    HANDLE_ERROR( cudaMemcpy(dpulses.pu1_id_part, pulses.pu1_id_part,
//                             n*sizeof(ui32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dpulses.pu1_id_geom, pulses.pu1_id_geom,
//                             n*sizeof(ui32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dpulses.pu1_nb_hits, vec,
                             n*sizeof(ui32), cudaMemcpyHostToDevice) );

//    HANDLE_ERROR( cudaMemcpy(dpulses.pu2_px, pulses.pu2_px,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dpulses.pu2_py, pulses.pu2_py,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dpulses.pu2_pz, pulses.pu2_pz,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dpulses.pu2_E, pulses.pu2_E,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dpulses.pu2_tof, pulses.pu2_tof,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );

//    HANDLE_ERROR( cudaMemcpy(dpulses.pu2_id_part, pulses.pu2_id_part,
//                             n*sizeof(ui32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dpulses.pu2_id_geom, pulses.pu2_id_geom,
//                             n*sizeof(ui32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dpulses.pu2_nb_hits, vec,
                             n*sizeof(ui32), cudaMemcpyHostToDevice) );
    
    free(vec);

}

// Copy the pulses list from the GPU to the CPU
void Digitizer::copy_pulses_gpu2cpu() {
   
    ui32 n = dpulses.size;
    
    HANDLE_ERROR( cudaMemcpy(pulses.pu1_px, dpulses.pu1_px,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );  
    HANDLE_ERROR( cudaMemcpy(pulses.pu1_py, dpulses.pu1_py,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(pulses.pu1_pz, dpulses.pu1_pz,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(pulses.pu1_E, dpulses.pu1_E,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(pulses.pu1_tof, dpulses.pu1_tof,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaMemcpy(pulses.pu1_id_part, dpulses.pu1_id_part,
                             n*sizeof(ui32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(pulses.pu1_id_geom, dpulses.pu1_id_geom,
                             n*sizeof(ui32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(pulses.pu1_nb_hits, dpulses.pu1_nb_hits,
                             n*sizeof(ui32), cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaMemcpy(pulses.pu2_px, dpulses.pu2_px,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(pulses.pu2_py, dpulses.pu2_py,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(pulses.pu2_pz, dpulses.pu2_pz,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(pulses.pu2_E, dpulses.pu2_E,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(pulses.pu2_tof, dpulses.pu2_tof,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaMemcpy(pulses.pu2_id_part, dpulses.pu2_id_part,
                             n*sizeof(ui32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(pulses.pu2_id_geom, dpulses.pu2_id_geom,
                             n*sizeof(ui32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(pulses.pu2_nb_hits, dpulses.pu2_nb_hits,
                             n*sizeof(ui32), cudaMemcpyDeviceToHost) );
    
}

void Digitizer::clear_gpu_pulses() {
  
    ui32 n = dpulses.size;
    ui32 *vec = (ui32*)malloc(n*sizeof(ui32));
    ui32 i=0; while (i<n) {
            vec[i]=0;
            ++i;
    }

    HANDLE_ERROR( cudaMemcpy(dpulses.pu1_nb_hits, vec,
                            n*sizeof(ui32), cudaMemcpyHostToDevice) );
                            
    HANDLE_ERROR( cudaMemcpy(dpulses.pu2_nb_hits, vec,
                            n*sizeof(ui32), cudaMemcpyHostToDevice) );
    
    free(vec);
    
}

// Set the output filename
void Digitizer::set_output_singles(std::string name) {
    singles_filename = name;
    flag_singles = true;
}

void Digitizer::set_output_coincidences(std::string name) {
    coincidences_filename = name;
    flag_coincidences = true;
}

void Digitizer::set_spect_projections(std::string name,
                                      f32 xmin, f32 xmax,
                                      f32 ymin, f32 ymax,
                                      f32 zmin, f32 zmax,
                                      f32 sx, f32 sy, f32 sz) {
     
    projection_filename = name;
    projection_sx = sx;
    projection_sy = sy;
    projection_sz = sz;
    
    projection_nx = (ui32)((xmax-xmin) / sx) + 1;
    if (projection_nx == 1) 
      flag_projYZ = true;
    
    projection_ny = (ui32)((ymax-ymin) / sy) + 1;
    
    if (projection_ny == 1) 
      flag_projXZ = true;
    
    projection_nz = (ui32)((zmax-zmin) / sz) + 1;
    if (projection_nz == 1) 
      flag_projXY = true;
    
    projection_xmin = xmin;
    projection_ymin = ymin;
    projection_zmin = zmin;
    
    flag_spect_proj = true;
    
    printf("proj nx %d, ny %d, nz %d \n", projection_nx, projection_ny, projection_nz);
}

void Digitizer::set_number_of_projections(ui32 nb_head) {
    
    nb_proj = nb_head;
}

void Digitizer::set_run(ui32 id_run) {
    
    index_run = id_run + 1;
}

void Digitizer::set_output_projection(std::string name, ui32 volid,
                                      f32 xmin, f32 xmax,
                                      f32 ymin, f32 ymax,
                                      f32 zmin, f32 zmax,
                                      f32 sx, f32 sy, f32 sz) {
    projection_filename = name;
    projection_idvol = volid;
    projection_sx = sx;
    projection_sy = sy;
    projection_sz = sz;
    
    projection_nx = (ui32)round((xmax-xmin) / sx);
    if (projection_nx == 0) 
      flag_projYZ = true;
    
    projection_ny = (ui32)round((ymax-ymin) / sy);
    
    if (projection_ny == 0) 
      flag_projXZ = true;
    
    projection_nz = (ui32)round((zmax-zmin) / sz);
    if (projection_nz == 0) 
      flag_projXY = true;
    
    projection_xmin = xmin;
    projection_ymin = ymin;
    projection_zmin = zmin;
    
    flag_projection = true;
    
    printf("proj nx %d, ny %d, nz %d \n", projection_nx, projection_ny, projection_nz);
}

void Digitizer::set_spatial_blurring(f32 vSP_res) {
    SP_res = vSP_res;
    flag_sp_blurring = true;
}
        
void Digitizer::set_energy_blurring(std::string law, f32 vE_res, f32 vE_ref, f32 vE_slope) {
    law_name = law;
    E_res = vE_res;
    E_ref = vE_ref;
    E_slope = vE_slope;
    flag_energy_blurring = true;
}

// Set parameters for coincidences
void Digitizer::set_energy_window(f32 vE_low, f32 vE_high) {
    E_low = vE_low;
    E_high = vE_high;
}

void Digitizer::set_time_window(f32 vwin_time) {
    win_time = vwin_time;
}

/// Process Singles /////////////////////////////////////////

// Process singles
void Digitizer::process_singles(ui32 iter, f64 tot_activity) {

    singles.clear();
    ui32 nb_record_singles=0;
  
    // Count the number of recorded singles
    ui32 i=0; while(i < pulses.size) {
        if (pulses.pu1_nb_hits[i]>0) {
            nb_record_singles++;
        }
        ++i;
    }

#ifdef DEBUG_DIGITIZER
        printf("Number of singles: %i\n", nb_record_singles);
#endif

    // Process the list of pulses into singles
    tot_activity = 1.0 / tot_activity;
    aSingle single;
    i=0; while(i<pulses.size) {

        // Update the global time
        f64 rnd = rand()/(f64)(RAND_MAX);
        global_time += -log(rnd)*tot_activity;

        // DEBUG
        //printf("glb time %e\n", global_time);

        if (pulses.pu1_nb_hits[i] > 0) {

            f32 E1=0.0; f32 E2=0.0;
            E1=pulses.pu1_E[i];
            if (pulses.pu2_nb_hits[i] > 0) E2=pulses.pu2_E[i];

            // Keep the first block
            if (E1 > E2) {
                single.px = pulses.pu1_px[i] / E1;
                single.py = pulses.pu1_py[i] / E1;
                single.pz = pulses.pu1_pz[i] / E1;
                single.E = E1;
                single.tof = pulses.pu1_tof[i];
                single.id_part = iter*pulses.size + i; // Absolute ID over the complete simulation
                single.id_geom = pulses.pu1_id_geom[i];
                single.time = global_time + pulses.pu1_tof[i];
            // Keep the second block
            } else {
                single.px = pulses.pu2_px[i] / E2;
                single.py = pulses.pu2_py[i] / E2;
                single.pz = pulses.pu2_pz[i] / E2;
                single.E = E2;
                single.tof = pulses.pu2_tof[i];
                single.id_part = iter*pulses.size + i; // Absolute ID over the complete simulation
                single.id_geom = pulses.pu2_id_geom[i];
                single.time = global_time + pulses.pu2_tof[i];
            }
            // Record the single
            singles.push_back(single);
        }
        ++i;
    } // while

}

// Export singles
void Digitizer::export_singles() {

    // check extension
    std::string ext = singles_filename.substr(singles_filename.size()-3);
    if (ext!="txt") {
        printf("Error, to export a Singles file, the exension must be '.txt'!\n");
        return;
    }
    
    std::string fullname;
    fullname.clear();
    
    std::string prefix = singles_filename.substr(0,singles_filename.size()-4);
    fullname.append(prefix);
    
    std::ostringstream oss;
    oss << index_run;
    fullname.append(oss.str());
    
    fullname.append(".txt");

    // first write te header
    FILE *pfile = fopen(fullname.c_str(), "w");
    ui32 i=0; while (i < singles.size()) {
        fprintf(pfile, "BLOCK ID %i PART ID %i POS %e %e %e E %e TOF %e TIME %e\n",
                singles[i].id_geom, singles[i].id_part,
                singles[i].px, singles[i].py, singles[i].pz,
                singles[i].E, singles[i].tof, singles[i].time);
        ++i;
    }

    fclose(pfile);

}

// Get recorded and processed singles
std::vector<aSingle> Digitizer::get_singles() {
    return singles;
}

/// Process Coincidences /////////////////////////////////////////

// Compare the time between two singles
bool compare_single_time(aSingle s1, aSingle s2) {
   return (s1.time < s2.time);
}

void Digitizer::process_coincidences() {

    printf("DEBUG\n");
    printf("Single 0: %i %e\n", singles[0].id_part, singles[0].time);
    printf("Single 1: %i %e\n", singles[1].id_part, singles[1].time);
    printf("Single 2: %i %e\n", singles[2].id_part, singles[2].time);
    printf("Single 3: %i %e\n", singles[3].id_part, singles[3].time);
    printf("Single 4: %i %e\n", singles[4].id_part, singles[0].time);

    // First sort singles
    std::sort(singles.begin(), singles.end(), compare_single_time);
    
    printf("SORT\n");
    printf("Single 0: %i %e\n", singles[0].id_part, singles[0].time);
    printf("Single 1: %i %e\n", singles[1].id_part, singles[1].time);
    printf("Single 2: %i %e\n", singles[2].id_part, singles[2].time);
    printf("Single 3: %i %e\n", singles[3].id_part, singles[3].time);
    printf("Single 4: %i %e\n", singles[4].id_part, singles[4].time);

    // Loop over singles
    aCoincidence coin;
    ui32 i=0;
    while (i < singles.size()) {

        // Open a coincidence window and read the singles within it
        ui32 ct = 0;
        while (singles[i].time < singles[i+ct].time+win_time && (i+ct) < singles.size()) {
            ++ct;
        }

        // Now depend on the number of singles find within the time window

#ifdef DEBUG_DIGITIZER
        printf("Singles %i find %i singles in coin win\n", i, ct);
#endif

        // If only one, discard it and continue
        if (ct == 1) {
            ++i;
            continue;
        }

        // If only two, store the coincidence
        if (ct == 2) {
            // first single
            coin.s1_px = singles[i].px;
            coin.s1_py = singles[i].py;
            coin.s1_pz = singles[i].pz;
            coin.s1_E = singles[i].E;
            coin.s1_tof = singles[i].tof;
            coin.s1_id_part = singles[i].id_part;
            coin.s1_id_geom = singles[i].id_geom;
            ++i;
            // second single
            coin.s2_px = singles[i].px;
            coin.s2_py = singles[i].py;
            coin.s2_pz = singles[i].pz;
            coin.s2_E = singles[i].E;
            coin.s2_tof = singles[i].tof;
            coin.s2_id_part = singles[i].id_part;
            coin.s2_id_geom = singles[i].id_geom;
            ++i;

            coincidences.push_back(coin);
            continue;
        }

        // If multiple coincidence, apply a simplify version of the rule Gate
        // named takeWinnerOfGoods. We look the first three singles and then keep the
        // coincidence that have the higher single energy
        if (ct >= 3) {
            // Store the first single
            coin.s1_px = singles[i].px;
            coin.s1_py = singles[i].py;
            coin.s1_pz = singles[i].pz;
            coin.s1_E = singles[i].E;
            coin.s1_tof = singles[i].tof;
            coin.s1_id_part = singles[i].id_part;
            coin.s1_id_geom = singles[i].id_geom;

            // and keep the higher single energy between the two next singles
            if (singles[i+1].E > singles[i+2].E) {
                coin.s2_px = singles[i+1].px;
                coin.s2_py = singles[i+1].py;
                coin.s2_pz = singles[i+1].pz;
                coin.s2_E = singles[i+1].E;
                coin.s2_tof = singles[i+1].tof;
                coin.s2_id_part = singles[i+1].id_part;
                coin.s2_id_geom = singles[i+1].id_geom;
            } else {
                coin.s2_px = singles[i+2].px;
                coin.s2_py = singles[i+2].py;
                coin.s2_pz = singles[i+2].pz;
                coin.s2_E = singles[i+2].E;
                coin.s2_tof = singles[i+2].tof;
                coin.s2_id_part = singles[i+2].id_part;
                coin.s2_id_geom = singles[i+2].id_geom;
            }

            coincidences.push_back(coin);
            i += ct;

        } // ct 3

    } // while single

#ifdef DEBUG_DIGITIZER
        printf("Number of coincidences: %i\n", coincidences.size());
#endif


}


/// Process Projection /////////////////////////////////////////

void Digitizer::init_projection() {
  
    ui32 n;

    if (flag_projXY) {
        n = projection_nx*projection_ny;
    }
    else if (flag_projXZ) {
        n = projection_nx*projection_nz;
    }
    else if (flag_projYZ) {
        n = projection_ny*projection_nz;
    }
    else {
        n = projection_nx*projection_ny*projection_nz;
    }
    
   // projection.clear();
   projection.resize(nb_proj, single_proj ( n , 0 ));
    
   /* for (ui32 id = 0; id < nb_proj; id++) {
        for (ui32 i = 0; i < n; i++) {
            projection[id].clear();
            projection[id].reserve(n);
            ui32 j=0; while (j<n) {
                projection[id][j] = 0;
                ++j;
            }
        }
    }*/
}

void Digitizer::process_projection() {

    // Loop over singles
    ui32 i=0;
    
    #ifdef VALID_GGEMS
    FILE *pfile = fopen("Projection.txt", "a");
    FILE *efile = fopen("Energy.txt", "a");
    #endif
    
    //printf("dim proj %d %d \n",projection_ny, projection_nz);

    while (i < singles.size()) {

        // If single hit the right detector
        if (singles[i].id_geom == projection_idvol) {
            
            f32 E_New = singles[i].E;
            f32 PxNew = singles[i].px;
            f32 PyNew = singles[i].py;
            f32 PzNew = singles[i].pz;
          
            // Apply energy blurring
            if (flag_energy_blurring) {
                f32 resolution = E_slope * (singles[i].E - E_ref) + E_res;
                E_New = G4RandGauss::shoot(singles[i].E, resolution * singles[i].E / 2.35482);
                
                #ifdef VALID_GGEMS
                fprintf(efile, "%f\n", E_New);
                #endif
            }
            
            if (E_New >= E_low && E_New <= E_high) {
            
                // Apply spatial blurring
                if (flag_sp_blurring) {
                    PxNew = G4RandGauss::shoot(singles[i].px,SP_res/2.35);
                    PyNew = G4RandGauss::shoot(singles[i].py,SP_res/2.35);
                    PzNew = G4RandGauss::shoot(singles[i].pz,SP_res/2.35);                    
                }
            
                if (flag_projXY) {
                    
                    // Change single frame to voxel space
                    ui32 ppx = (PxNew - projection_xmin) / projection_sx;
                    ui32 ppy = (PyNew - projection_ymin) / projection_sy;
             
                    assert(ppx >= 0);
                    assert(ppy >= 0);

                    assert(ppx < projection_nx);
                    assert(ppy < projection_ny);

                    // Assign value
                    projection[0][ppy*projection_nx + ppx] += 1; 
                    
                    #ifdef VALID_GGEMS
                    fprintf(pfile, "%f %f\n", PxNew, PyNew);
                    #endif
                 }
                else if (flag_projYZ) {
                    
                    // Change single frame to voxel space
                    ui32 ppy = (PyNew - projection_ymin) / projection_sy;
                    ui32 ppz = (PzNew - projection_zmin) / projection_sz;
                    
                    //printf("ppy %d ppz %d \n", ppy, ppz);
                    
                    assert(ppy >= 0);
                    assert(ppz >= 0);

                    assert(ppy < projection_ny);
                    assert(ppz < projection_nz);

                    // Assign value
                    projection[0][ppz*projection_ny + ppy] += 1; 
                    
                    #ifdef VALID_GGEMS
                    fprintf(pfile, "%f %f\n", PyNew, PzNew);
                    #endif
                }
                else if (flag_projXZ) {
               
                    // Change single frame to voxel space
                    ui32 ppx = (PxNew - projection_xmin) / projection_sx;
                    ui32 ppz = (PzNew - projection_zmin) / projection_sz;
             
                    assert(ppx >= 0);
                    assert(ppz >= 0);

                    assert(ppx < projection_nx);
                    assert(ppz < projection_nz);

                    // Assign value
                    projection[0][ppz*projection_nx + ppx] += 1; 
                    
                    #ifdef VALID_GGEMS
                    fprintf(pfile, "%f %f\n", PxNew, PzNew);
                    #endif
                }
                else {
                    // Change single frame to voxel space
                    ui32 ppx = (PxNew - projection_xmin) / projection_sx;
                    ui32 ppy = (PyNew - projection_ymin) / projection_sy;
                    ui32 ppz = (PzNew - projection_zmin) / projection_sz;

                    assert(ppx >= 0);
                    assert(ppy >= 0);
                    assert(ppz >= 0);

                    assert(ppx < projection_nx);
                    assert(ppy < projection_ny);
                    assert(ppz < projection_nz);

                    projection[0][ppz*projection_ny*projection_nx + ppy*projection_nx + ppx] += 1;
                }
            }
        }
        ++i;
    }
    
    #ifdef VALID_GGEMS
    fclose(pfile);
    fclose(efile);
    #endif
}

void Digitizer::process_spect_projections(Scene geometry) {

    // Loop over singles
    ui32 i=0;
    
    #ifdef VALID_GGEMS
    FILE *pfile = fopen("Projection.txt", "a");
    FILE *efile = fopen("Energy.txt", "a");
    #endif
    
    printf("dim proj %d %d single size %d\n",projection_ny, projection_nz, singles.size());

    while (i < singles.size()) {

        // If single hit the right detector
        //if (singles[i].id_geom == projection_idvol) {
            
            f32 E_New = singles[i].E;
            f32xyz PNew;
            PNew.x = singles[i].px;
            PNew.y = singles[i].py;
            PNew.z = singles[i].pz;
          
            //printf("Energy %f singles[i].id_geom %d \n", E_New, singles[i].id_geom);
            
            // Apply energy blurring
            if (flag_energy_blurring) {
                f32 resolution = E_slope * (singles[i].E - E_ref) + E_res;
                E_New = G4RandGauss::shoot(singles[i].E, resolution * singles[i].E / 2.35482);
                
                #ifdef VALID_GGEMS
                fprintf(efile, "%f\n", E_New);
                #endif
            }
            
            if (E_New >= E_low && E_New <= E_high) {
              
                ui32 mother_id = geometry.mother_node[singles[i].id_geom];
                
                ui32 adr_geom = geometry.ptr_objects[singles[i].id_geom];
                
                    
                f64 aabb_xmin = (f64)geometry.data_objects[adr_geom+ADR_AABB_XMIN];
                f64 aabb_xmax = (f64)geometry.data_objects[adr_geom+ADR_AABB_XMAX];
                f64 aabb_ymin = (f64)geometry.data_objects[adr_geom+ADR_AABB_YMIN];
                f64 aabb_ymax = (f64)geometry.data_objects[adr_geom+ADR_AABB_YMAX];
                f64 aabb_zmin = (f64)geometry.data_objects[adr_geom+ADR_AABB_ZMIN];
                f64 aabb_zmax = (f64)geometry.data_objects[adr_geom+ADR_AABB_ZMAX];
              
                ui32 id_head = (singles[i].id_geom + 1) / (geometry.size_of_nodes[mother_id] + 1);
            
                // Apply spatial blurring
                if (flag_sp_blurring) {
                    PNew.x = G4RandGauss::shoot(singles[i].px,SP_res/2.35);              
                    PNew.y = G4RandGauss::shoot(singles[i].py,SP_res/2.35);
                    PNew.z = G4RandGauss::shoot(singles[i].pz,SP_res/2.35);
                }
            
                // If new position still inside the detector 
                if (test_point_AABB(PNew, aabb_xmin, aabb_xmax, aabb_ymin, aabb_ymax, aabb_zmin, aabb_zmax)) {
                    if (flag_projXY) {
                        
                        // Change single frame to voxel space
                        ui32 ppx = (PNew.x - projection_xmin) / projection_sx;
                        ui32 ppy = (PNew.y - projection_ymin) / projection_sy;
                
                        assert(ppx >= 0);
                        assert(ppy >= 0);

                        assert(ppx < projection_nx);
                        assert(ppy < projection_ny);

                        // Assign value
                        projection[id_head-1][ppy*projection_nx + ppx] += 1; 
                        
                        #ifdef VALID_GGEMS
                        fprintf(pfile, "%f %f\n", PNew.x, PNew.y);
                        #endif
                    }
                    else if (flag_projYZ) {
                        
                        // Change single frame to voxel space
                        ui32 ppy = (PNew.y - projection_ymin) / projection_sy;
                        ui32 ppz = (PNew.z - projection_zmin) / projection_sz;
                        
                        //printf("i %d head %d: ppy %d ppz %d \n", i, id_head, ppy, ppz);
                        
                        if(ppz >= projection_nz) {
                            printf("ppz %d proj_nz %d\n", ppz, projection_nz);
                            printf("pos before %f %f %f \n", singles[i].px, singles[i].py, singles[i].pz);
                            printf("pos spblur %f %f %f \n", PNew.x, PNew.y, PNew.z);
                            printf("id geom %d \n", singles[i].id_geom);
                        }
                        
                        if(ppy >= projection_ny) {
                            printf("ppy %f proj_y %d\n", (PNew.y - projection_ymin) / projection_sy, projection_ny);
                            printf("pos before %f %f %f \n", singles[i].px, singles[i].py, singles[i].pz);
                            printf("pos spblur %f %f %f \n", PNew.x, PNew.y, PNew.z);
                            printf("id geom %d \n", singles[i].id_geom);
                        }
                        
                        assert(ppy >= 0);
                        assert(ppz >= 0);

                        assert(ppy < projection_ny);
                        assert(ppz < projection_nz);
                        
                        // Assign value
                        projection[id_head-1][ppz*projection_ny + ppy] += 1; 
                        
                        #ifdef VALID_GGEMS
                        fprintf(pfile, "%f %f\n", PNew.y, PNew.z);
                        #endif
                    }
                    else if (flag_projXZ) {
                  
                        // Change single frame to voxel space
                        ui32 ppx = (PNew.x - projection_xmin) / projection_sx;
                        ui32 ppz = (PNew.z - projection_zmin) / projection_sz;
                
                        assert(ppx >= 0);
                        assert(ppz >= 0);

                        assert(ppx < projection_nx);
                        assert(ppz < projection_nz);

                        // Assign value
                        projection[id_head-1][ppz*projection_nx + ppx] += 1; 
                        
                        #ifdef VALID_GGEMS
                        fprintf(pfile, "%f %f\n", PNew.x, PNew.z);
                        #endif
                    }
                    else {
                        // Change single frame to voxel space
                        ui32 ppx = (PNew.x - projection_xmin) / projection_sx;
                        ui32 ppy = (PNew.y - projection_ymin) / projection_sy;
                        ui32 ppz = (PNew.z - projection_zmin) / projection_sz;

                        assert(ppx >= 0);
                        assert(ppy >= 0);
                        assert(ppz >= 0);

                        assert(ppx < projection_nx);
                        assert(ppy < projection_ny);
                        assert(ppz < projection_nz);

                        projection[id_head-1][ppz*projection_ny*projection_nx + ppy*projection_nx + ppx] += 1;
                    }
                }
            }
        ++i;
    }
    
    #ifdef VALID_GGEMS
    fclose(pfile);
    fclose(efile);
    #endif
}

void Digitizer::export_spect_projections(ui32 nb_proj, ui32 id_run) {

    std::string proj_fullname = "";
  
   for (ui32 i = 0; i < nb_proj; i++) {
  
        ui32 proj = (id_run * nb_proj) + (i+1);
        
        std::ostringstream oss;
        
        oss << proj;
       
        proj_fullname.clear();
        
        proj_fullname = proj_fullname.append(projection_filename);
        proj_fullname = proj_fullname.append(oss.str());
        proj_fullname = proj_fullname.append(".mhd");
        
        printf("projection %s \n", proj_fullname.c_str());
  
        // check extension
        /*std::string ext = projection_filename.substr(projection_filename.size()-3);
        
        if (ext!="mhd") {
            printf("Error, to export an mhd file, the exension must be '.mhd'!\n");
            return;
        }*/

        // first write te header
        FILE *pfile = fopen(proj_fullname.c_str(), "w");
        fprintf(pfile, "ObjectType = Image \n");
        fprintf(pfile, "BinaryData = True \n");
        fprintf(pfile, "BinaryDataByteOrderMSB = False \n");
        fprintf(pfile, "CompressedData = False \n");
        fprintf(pfile, "ElementType = MET_UINT \n");
        if (flag_projXY) {
            fprintf(pfile, "NDims = 2 \n");
            fprintf(pfile, "ElementSpacing = %f %f\n", projection_sx, projection_sy);
            fprintf(pfile, "DimSize = %i %i\n", projection_nx, projection_ny);
        }
        else if (flag_projXZ) {
            fprintf(pfile, "NDims = 2 \n");
            fprintf(pfile, "ElementSpacing = %f %f\n", projection_sx, projection_sz);
            fprintf(pfile, "DimSize = %i %i\n", projection_nx, projection_nz);
        }        
        else if (flag_projYZ) {
            fprintf(pfile, "NDims = 2 \n");
            fprintf(pfile, "ElementSpacing = %f %f\n", projection_sy, projection_sz);
            fprintf(pfile, "DimSize = %i %i\n", projection_ny, projection_nz);
        }         
        else {
            fprintf(pfile, "NDims = 3 \n");
            fprintf(pfile, "ElementSpacing = %f %f %f\n", projection_sx, projection_sy, projection_sz);
            fprintf(pfile, "DimSize = %i %i %i\n", projection_nx, projection_ny, projection_nz);
        }
        
        std::string export_name = proj_fullname.replace(proj_fullname.size()-3, 3, "raw");
        fprintf(pfile, "ElementDataFile = %s \n", export_name.c_str());
        fclose(pfile);

        // then export data
        pfile = fopen(export_name.c_str(), "wb");
        
        if (flag_projXY) {
            fwrite(projection[i].data(), projection_nx*projection_ny, sizeof(f32), pfile);
        }
        else if (flag_projXZ) {
            fwrite(projection[i].data(), projection_nx*projection_nz, sizeof(f32), pfile);
        }
        else if (flag_projYZ) {
            fwrite(projection[i].data(), projection_ny*projection_nz, sizeof(f32), pfile);
        }
        else {
            fwrite(projection[i].data(), projection_nx*projection_ny*projection_nz, sizeof(f32), pfile);
        }
        
        fclose(pfile);
   }
}


void Digitizer::export_projection() {

   
    // check extension
    /*std::string ext = projection_filename.substr(projection_filename.size()-3);
    
    if (ext!="mhd") {
        printf("Error, to export an mhd file, the exension must be '.mhd'!\n");
        return;
    }*/

    // first write te header
    FILE *pfile = fopen(projection_filename.c_str(), "w");
    fprintf(pfile, "ObjectType = Image \n");
    fprintf(pfile, "BinaryData = True \n");
    fprintf(pfile, "BinaryDataByteOrderMSB = False \n");
    fprintf(pfile, "CompressedData = False \n");
    fprintf(pfile, "ElementType = MET_UINT \n");
    if (flag_projXY) {
        fprintf(pfile, "NDims = 2 \n");
        fprintf(pfile, "ElementSpacing = %f %f\n", projection_sx, projection_sy);
        fprintf(pfile, "DimSize = %i %i\n", projection_nx, projection_ny);
    }
    else if (flag_projXZ) {
        fprintf(pfile, "NDims = 2 \n");
        fprintf(pfile, "ElementSpacing = %f %f\n", projection_sx, projection_sz);
        fprintf(pfile, "DimSize = %i %i\n", projection_nx, projection_nz);
    }        
    else if (flag_projYZ) {
        fprintf(pfile, "NDims = 2 \n");
        fprintf(pfile, "ElementSpacing = %f %f\n", projection_sy, projection_sz);
        fprintf(pfile, "DimSize = %i %i\n", projection_ny, projection_nz);
    }         
    else {
        fprintf(pfile, "NDims = 3 \n");
        fprintf(pfile, "ElementSpacing = %f %f %f\n", projection_sx, projection_sy, projection_sz);
        fprintf(pfile, "DimSize = %i %i %i\n", projection_nx, projection_ny, projection_nz);
    }
    
    std::string export_name = projection_filename.replace(projection_filename.size()-3, 3, "raw");
    fprintf(pfile, "ElementDataFile = %s \n", export_name.c_str());
    fclose(pfile);

    // then export data
    pfile = fopen(export_name.c_str(), "wb");
    
    if (flag_projXY) {
        fwrite(projection[0].data(), projection_nx*projection_ny, sizeof(f32), pfile);
    }
    else if (flag_projXZ) {
        fwrite(projection[0].data(), projection_nx*projection_nz, sizeof(f32), pfile);
    }
    else if (flag_projYZ) {
        fwrite(projection[0].data(), projection_ny*projection_nz, sizeof(f32), pfile);
    }
    else {
        fwrite(projection[0].data(), projection_nx*projection_ny*projection_nz, sizeof(f32), pfile);
    }
    
    fclose(pfile);

}


/// Main function /////////////////////////////////////////

void Digitizer::process_chain(ui32 iter, f64 tot_activity, Scene geometry) {

    process_singles(iter, tot_activity);


    if (flag_singles) {
        export_singles();
    }


    if (flag_coincidences) {   
        process_coincidences();  
        // TODO
        // Export
    }


    if (flag_projection) {
        process_projection();
    }

    if (flag_spect_proj) {
        process_spect_projections(geometry);
    }


}











#endif
