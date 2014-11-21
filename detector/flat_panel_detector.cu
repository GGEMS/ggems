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

#ifndef FLAT_PANEL_DETECTOR_CU
#define FLAT_PANEL_DETECTOR_CU

#include "flat_panel_detector.cuh"

FlatPanelDetector::FlatPanelDetector() {

    panel_detector.data = NULL;
    panel_detector.nx = 0;
    panel_detector.ny = 0;
    panel_detector.nz = 0;
    panel_detector.sx = 0;
    panel_detector.sy = 0;
    panel_detector.sz = 0;
    panel_detector.geometry_id = 0;
    panel_detector.countp = 0;

}

// Setting function

void FlatPanelDetector::attach_to(unsigned int geometry_id) {
    panel_detector.geometry_id = geometry_id;
}

void FlatPanelDetector::set_resolution(float sx, float sy, float sz) {
    panel_detector.sx = sx;
    panel_detector.sy = sy;
    panel_detector.sz = sz;
}

// Init
void FlatPanelDetector::init(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax) {

    panel_detector.nx = (unsigned int)((xmax-xmin) / panel_detector.sx);
    panel_detector.ny = (unsigned int)((ymax-ymin) / panel_detector.sy);
    panel_detector.nz = (unsigned int)((zmax-zmin) / panel_detector.sz);
    panel_detector.nb_voxels = panel_detector.nx*panel_detector.ny*panel_detector.nz;

    panel_detector.xmin = xmin;
    panel_detector.xmax = xmax;
    panel_detector.ymin = ymin;
    panel_detector.ymax = ymax;
    panel_detector.zmin = zmin;
    panel_detector.zmax = zmax;

    panel_detector.data = (float*)malloc(panel_detector.nb_voxels*sizeof(float));

    unsigned int i=0;
    while (i<panel_detector.nb_voxels) {
        panel_detector.data[i] = 0.0f;
        ++i;
    }
}

// Export image into mhd format
void FlatPanelDetector::save_image(std::string outputname) {

    // check extension
    std::string ext = outputname.substr(outputname.size()-3);
    if (ext!="mhd") {
        printf("Error, to export an mhd file, the exension must be '.mhd'!\n");
        return;
    }

    // first write te header
    FILE *pfile = fopen(outputname.c_str(), "w");
    fprintf(pfile, "ObjectType = Image \n");
    fprintf(pfile, "NDims = 3 \n");
    fprintf(pfile, "BinaryData = True \n");
    fprintf(pfile, "BinaryDataByteOrderMSB = False \n");
    fprintf(pfile, "CompressedData = False \n");
    fprintf(pfile, "ElementSpacing = %f %f %f\n", panel_detector.sx, panel_detector.sy, panel_detector.sz);
    fprintf(pfile, "DimSize = %i %i %i\n", panel_detector.nx, panel_detector.ny, panel_detector.nz);
    fprintf(pfile, "ElementType = MET_FLOAT \n");

    std::string export_name = outputname.replace(outputname.size()-3, 3, "raw");
    fprintf(pfile, "ElementDataFile = %s \n", export_name.c_str());
    fclose(pfile);

    // then export data
    pfile = fopen(export_name.c_str(), "wb");
    fwrite(panel_detector.data, panel_detector.nb_voxels, sizeof(float), pfile);
    fclose(pfile);

}








#endif
