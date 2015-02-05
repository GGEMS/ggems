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

// Set the output filename
void Digitizer::set_output_singles(std::string name) {
    singles_filename = name;
}

void Digitizer::set_output_coincidences(std::string name) {
    coincidences_filename = name;
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
                single.time = global_time + tof;
            // Keep the second block
            } else {
                single.px = pulses.pu2_px[i] / E2;
                single.py = pulses.pu2_py[i] / E2;
                single.pz = pulses.pu2_pz[i] / E2;
                single.E = E2;
                single.tof = pulses.pu2_tof[i];
                single.id_part = iter*pulses.size + i; // Absolute ID over the complete simulation
                single.id_geom = pulses.pu2_id_geom[i];
                single.time = global_time + tof;
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
    std::string ext = filename.substr(filename.size()-3);
    if (ext!="txt") {
        printf("Error, to export a Singles file, the exension must be '.txt'!\n");
        return;
    }

    // first write te header
    FILE *pfile = fopen(filename.c_str(), "a");
    ui32 i=0; while (i<record_singles.size) {
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
bool Digitizer::compare_single_time(aSingle s1, aSingle s2) {
   return (s1.time < s2.time);
}

void Digitizer::process_coincidences() {

    // First sort singles
    std::sort(singles.begin(), singles.end(), compare_single_time);

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



















#endif
