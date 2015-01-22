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

// Alocate and init the singles list file
void Digitizer::init_singles(ui32 nb) {
    singles.size = nb;
    singles.px = (f32*)malloc(singles.size*sizeof(f32));
    singles.py = (f32*)malloc(singles.size*sizeof(f32));
    singles.pz = (f32*)malloc(singles.size*sizeof(f32));
    singles.E = (f32*)malloc(singles.size*sizeof(f32));
    singles.tof = (f32*)malloc(singles.size*sizeof(f32));
    singles.id = (ui32*)malloc(singles.size*sizeof(ui32));
    singles.nb_hits = (ui32*)malloc(singles.size*sizeof(ui32));
    ui32 i=0; while (i<singles.size) {
        singles.nb_hits[i]=0;
        ++i;
    }
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
        if (singles.nb_hits[i]>0) {
            nb_record_singles++;
        }
        ++i;
    }

    // Init the list
    record_singles.size = nb_record_singles;
    record_singles.px = (f32*)malloc(singles.size*sizeof(f32));
    record_singles.py = (f32*)malloc(singles.size*sizeof(f32));
    record_singles.pz = (f32*)malloc(singles.size*sizeof(f32));
    record_singles.E = (f32*)malloc(singles.size*sizeof(f32));
    record_singles.tof = (f32*)malloc(singles.size*sizeof(f32));
    record_singles.id = (ui32*)malloc(singles.size*sizeof(ui32));
    record_singles.nb_hits = (ui32*)malloc(singles.size*sizeof(ui32));

    // Process the list
    ui32 index=0;
    i=0; while(i<singles.size) {
        if (singles.nb_hits[i]>0) {
            record_singles.px[index] = singles.px[i]/singles.E[i];
            record_singles.py[index] = singles.py[i]/singles.E[i];
            record_singles.pz[index] = singles.pz[i]/singles.E[i];
            record_singles.E[index] = singles.E[i];
            record_singles.tof[index] = singles.tof[i];
            record_singles.id[index] = iter*singles.size + i; // Absolute ID over the complete simulation
            record_singles.nb_hits[index] = singles.nb_hits[i];
            ++index;
        }
        ++i;
    }
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
        fprintf(pfile, "ID %i POS %e %e %e E %e TOF %e NB HITS %i\n", record_singles.id[i],
                record_singles.px[i], record_singles.py[i], record_singles.pz[i],
                record_singles.E[i], record_singles.tof[i], record_singles.nb_hits[i]);
        ++i;
    }

    fclose(pfile);

}

// Get recorded and processed singles
Singles Digitizer::get_singles() {
    return record_singles;
}


#endif
