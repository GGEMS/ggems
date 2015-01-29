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
    singles.size = 0;
}

// Allocate and init the singles list file (CPU)
void Digitizer::cpu_init_singles(ui32 nb) {
    singles.size = nb;

    // Block Pulse 1
    singles.pu1_px = (f32*)malloc(singles.size*sizeof(f32));
    singles.pu1_py = (f32*)malloc(singles.size*sizeof(f32));
    singles.pu1_pz = (f32*)malloc(singles.size*sizeof(f32));
    singles.pu1_E = (f32*)malloc(singles.size*sizeof(f32));
    singles.pu1_tof = (f32*)malloc(singles.size*sizeof(f32));
    singles.pu1_id_part = (ui32*)malloc(singles.size*sizeof(ui32));
    singles.pu1_id_geom = (ui32*)malloc(singles.size*sizeof(ui32));
    singles.pu1_nb_hits = (ui32*)malloc(singles.size*sizeof(ui32));
    // Block Pulse 2
    singles.pu2_px = (f32*)malloc(singles.size*sizeof(f32));
    singles.pu2_py = (f32*)malloc(singles.size*sizeof(f32));
    singles.pu2_pz = (f32*)malloc(singles.size*sizeof(f32));
    singles.pu2_E = (f32*)malloc(singles.size*sizeof(f32));
    singles.pu2_tof = (f32*)malloc(singles.size*sizeof(f32));
    singles.pu2_id_part = (ui32*)malloc(singles.size*sizeof(ui32));
    singles.pu2_id_geom = (ui32*)malloc(singles.size*sizeof(ui32));
    singles.pu2_nb_hits = (ui32*)malloc(singles.size*sizeof(ui32));

    ui32 i=0; while (i<singles.size) {
        singles.pu1_nb_hits[i]=0;
        singles.pu2_nb_hits[i]=0;
        ++i;
    }
}

// Allocate and init the singles list file (GPU)
void Digitizer::gpu_init_singles(ui32 nb) {

    ui32 n = singles.size;

    // First allocate mem on GPU
    HANDLE_ERROR( cudaMalloc((void**) &dsingles.pu1_px, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dsingles.pu1_py, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dsingles.pu1_pz, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dsingles.pu1_E, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dsingles.pu1_tof, n*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &dsingles.pu1_id_part, n*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dsingles.pu1_id_geom, n*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dsingles.pu1_nb_hits, n*sizeof(ui32)) );

    HANDLE_ERROR( cudaMalloc((void**) &dsingles.pu2_px, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dsingles.pu2_py, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dsingles.pu2_pz, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dsingles.pu2_E, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dsingles.pu2_tof, n*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &dsingles.pu2_id_part, n*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dsingles.pu2_id_geom, n*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dsingles.pu2_nb_hits, n*sizeof(ui32)) );

    // Init values
    ui32 *vec = (ui32*)malloc(n*sizeof(ui32));
    ui32 i=0; while (i<n) {
        vec[i]=0;
        ++i;
    }

    // Copy data to the GPU
    dsingles.size = singles.size;

//    HANDLE_ERROR( cudaMemcpy(dsingles.pu1_px, singles.pu1_px,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dsingles.pu1_py, singles.pu1_py,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dsingles.pu1_pz, singles.pu1_pz,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dsingles.pu1_E, singles.pu1_E,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dsingles.pu1_tof, singles.pu1_tof,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );

//    HANDLE_ERROR( cudaMemcpy(dsingles.pu1_id_part, singles.pu1_id_part,
//                             n*sizeof(ui32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dsingles.pu1_id_geom, singles.pu1_id_geom,
//                             n*sizeof(ui32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dsingles.pu1_nb_hits, vec,
                             n*sizeof(ui32), cudaMemcpyHostToDevice) );

//    HANDLE_ERROR( cudaMemcpy(dsingles.pu2_px, singles.pu2_px,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dsingles.pu2_py, singles.pu2_py,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dsingles.pu2_pz, singles.pu2_pz,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dsingles.pu2_E, singles.pu2_E,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dsingles.pu2_tof, singles.pu2_tof,
//                             n*sizeof(f32), cudaMemcpyHostToDevice) );

//    HANDLE_ERROR( cudaMemcpy(dsingles.pu2_id_part, singles.pu2_id_part,
//                             n*sizeof(ui32), cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemcpy(dsingles.pu2_id_geom, singles.pu2_id_geom,
//                             n*sizeof(ui32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dsingles.pu2_nb_hits, vec,
                             n*sizeof(ui32), cudaMemcpyHostToDevice) );

}

// Copy the singles list from the GPU to the CPU
void Digitizer::copy_singles_gpu2cpu() {

    ui32 n = dsingles.size;

    HANDLE_ERROR( cudaMemcpy(singles.pu1_px, dsingles.pu1_px,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(singles.pu1_py, dsingles.pu1_py,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(singles.pu1_pz, dsingles.pu1_pz,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(singles.pu1_E, dsingles.pu1_E,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(singles.pu1_tof, dsingles.pu1_tof,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaMemcpy(singles.pu1_id_part, dsingles.pu1_id_part,
                             n*sizeof(ui32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(singles.pu1_id_geom, dsingles.pu1_id_geom,
                             n*sizeof(ui32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(singles.pu1_nb_hits, dsingles.pu1_nb_hits,
                             n*sizeof(ui32), cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaMemcpy(singles.pu2_px, dsingles.pu2_px,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(singles.pu2_py, dsingles.pu2_py,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(singles.pu2_pz, dsingles.pu2_pz,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(singles.pu2_E, dsingles.pu2_E,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(singles.pu2_tof, dsingles.pu2_tof,
                             n*sizeof(f32), cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaMemcpy(singles.pu2_id_part, dsingles.pu2_id_part,
                             n*sizeof(ui32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(singles.pu2_id_geom, dsingles.pu2_id_geom,
                             n*sizeof(ui32), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(singles.pu2_nb_hits, dsingles.pu2_nb_hits,
                             n*sizeof(ui32), cudaMemcpyDeviceToHost) );
}

// Set the output filename
void Digitizer::set_output_filename(std::string name) {
    filename = name;
}

// Process singles
void Digitizer::process_singles(ui32 iter) {

    record_singles.size=0;
    ui32 nb_record_singles=0;

    // Count the number of recorded singles
    ui32 i=0; while(i<singles.size) {
        if (singles.pu1_nb_hits[i]>0) {
            nb_record_singles++;
        }
        ++i;
    }

    // Init the list
    record_singles.size = nb_record_singles;
    record_singles.pu1_px = (f32*)malloc(singles.size*sizeof(f32));
    record_singles.pu1_py = (f32*)malloc(singles.size*sizeof(f32));
    record_singles.pu1_pz = (f32*)malloc(singles.size*sizeof(f32));
    record_singles.pu1_E = (f32*)malloc(singles.size*sizeof(f32));
    record_singles.pu1_tof = (f32*)malloc(singles.size*sizeof(f32));
    record_singles.pu1_id_part = (ui32*)malloc(singles.size*sizeof(ui32));
    record_singles.pu1_id_geom = (ui32*)malloc(singles.size*sizeof(ui32));
    record_singles.pu1_nb_hits = (ui32*)malloc(singles.size*sizeof(ui32));

    // Process the list of pulses into singles
    ui32 index=0;
    i=0; while(i<singles.size) {

        if (singles.pu1_nb_hits[i] > 0) {

            f32 E1=0.0; f32 E2=0.0;
            E1=singles.pu1_E[i];
            if (singles.pu2_nb_hits[i] > 0) E2=singles.pu2_E[i];

            // Keep the first block
            if (E1 > E2) {
                record_singles.pu1_px[index] = singles.pu1_px[i]/E1;
                record_singles.pu1_py[index] = singles.pu1_py[i]/E1;
                record_singles.pu1_pz[index] = singles.pu1_pz[i]/E1;
                record_singles.pu1_E[index] = E1;
                record_singles.pu1_tof[index] = singles.pu1_tof[i];
                record_singles.pu1_id_part[index] = iter*singles.size + i; // Absolute ID over the complete simulation
                record_singles.pu1_id_geom[index] = singles.pu1_id_geom[i];
                record_singles.pu1_nb_hits[index] = singles.pu1_nb_hits[i];
            // Keep the second block
            } else {
                record_singles.pu1_px[index] = singles.pu2_px[i]/E2;
                record_singles.pu1_py[index] = singles.pu2_py[i]/E2;
                record_singles.pu1_pz[index] = singles.pu2_pz[i]/E2;
                record_singles.pu1_E[index] = E2;
                record_singles.pu1_tof[index] = singles.pu2_tof[i];
                record_singles.pu1_id_part[index] = iter*singles.size + i; // Absolute ID over the complete simulation
                record_singles.pu1_id_geom[index] = singles.pu2_id_geom[i];
                record_singles.pu1_nb_hits[index] = singles.pu2_nb_hits[i];
            }
            ++index;
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
        fprintf(pfile, "BLOCK ID %i PART ID %i POS %e %e %e E %e TOF %e NB HITS %i\n",
                record_singles.pu1_id_geom[i], record_singles.pu1_id_part[i],
                record_singles.pu1_px[i], record_singles.pu1_py[i], record_singles.pu1_pz[i],
                record_singles.pu1_E[i], record_singles.pu1_tof[i], record_singles.pu1_nb_hits[i]);
        ++i;
    }

    fclose(pfile);

}

// Get recorded and processed singles
Singles Digitizer::get_singles() {
    return record_singles;
}


#endif
