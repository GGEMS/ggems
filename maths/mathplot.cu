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

#ifndef MATHPLOT_CU
#define MATHPLOT_CU

#include "mathplot.cuh"


MathPlotBuilder::MathPlotBuilder(){}

//// Private functions ///////////////////////////////////////////////////////

// Build an histogramm based on 1D data
void MathPlotBuilder::get_histogramm(float* xdata, unsigned int nxdata,
                                    float* bins, float* nbelt, unsigned int nbins) {
    unsigned int i;
    float xmin = FLT_MAX;
    float xmax = FLT_MIN;
    float val;
    i=0; while (i<nxdata) {
        val = xdata[i];
        if (val < xmin) {xmin=val;};
        if (val > xmax) {xmax=val;};
        ++i;
    }
    assert(xmin != FLT_MAX);
    assert(xmax != -FLT_MIN);
    // Build hist
    float di = (xmax-xmin) / float(nbins-1);
    float posi;
    i=0; while (i<nxdata) {
        posi = ((xdata[i]-xmin) / di)+0.5f;
        ++nbelt[int(posi)];
        ++i;
    }
    // Build bins
    i=0; while (i<nbins) {
        bins[i] = xmin + i*di;
        ++i;
    }
}

// Build a weigthed histogramm based on 2D data
void MathPlotBuilder::get_weighted_histogramm(float* xdata, float* ydata, unsigned int nxdata,
                                              float* hist, float* nbelt, float* bins,
                                              unsigned int nbins) {
    int i;
    float xmin = FLT_MAX;
    float xmax = FLT_MIN;
    float val;
    i=0; while (i<nxdata) {
        val = xdata[i];
        if (val < xmin) {xmin=val;};
        if (val > xmax) {xmax=val;};
        ++i;
    }
    assert(xmin != FLT_MAX);
    assert(xmax != -FLT_MIN);
    // Build hist
    float di = (xmax-xmin) / float(nbins-1);
    float posi;
    i=0; while (i<nxdata) {
        posi = ((xdata[i]-xmin) / di)+0.5f;
        ++nbelt[int(posi)];
        hist[int(posi)] += ydata[i];
        ++i;
    }
    // Build bins
    i=0; while (i<nbins) {
        bins[i] = xmin + i*di;
        ++i;
    }
}

//// Pulbic functions /////////////////////////////////////////////////////////

// Plot a 1D histogramm
void MathPlotBuilder::plot_distribution(float *xdata, unsigned int nxdata,
                                        unsigned int n_bin, std::string filename,
                                        std::string xlabel, std::string ylabel) {

    // Memory allocation
    float* bins = (float*)malloc(n_bin * sizeof(float));
    float* nbelt = (float*)malloc(n_bin * sizeof(float));

    // Compute the histogramm
    get_histogramm(xdata, nxdata, bins, nbelt, n_bin);

    // Export data
    std::string data_name = filename + ".dat";
    FILE* pfile = fopen(data_name.c_str(), "wb");
    float buffer = (float)n_bin;
    fwrite(&buffer, sizeof(float), 1, pfile);
    fwrite(bins, sizeof(float), n_bin, pfile);
    fwrite(nbelt, sizeof(float), n_bin, pfile);
    fclose(pfile);

    // Export MatPlotLib script
    std::string script_name = filename + ".py";
    pfile = fopen(script_name.c_str(), "w");

    // header
    fprintf(pfile, "#!/usr/bin/env python\n");
    fprintf(pfile, "from numpy import *\n");
    fprintf(pfile, "import matplotlib.pyplot as plt\n");
    fprintf(pfile, "from matplotlib.ticker import ScalarFormatter\n\n");

    // load data
    fprintf(pfile, "data  = fromfile('%s', 'float32')\n", data_name.c_str());
    fprintf(pfile, "ndata = int(data[0])\n", data_name.c_str());
    fprintf(pfile, "bins  = data[1:ndata+1]\n", data_name.c_str());
    fprintf(pfile, "nelt  = data[ndata+1:2*ndata+1]\n\n", data_name.c_str());

    // set figure
    fprintf(pfile, "fig = plt.figure()\n");
    fprintf(pfile, "ax = fig.add_subplot(111)\n\n");

    // plot
    fprintf(pfile, "plt.plot(bins, nelt, c='0.0', drawstyle='steps')\n\n");

    // axes
    fprintf(pfile, "mymft = ScalarFormatter(useOffset=True)\n");
    fprintf(pfile, "mymft.set_scientific(True)\n");
    fprintf(pfile, "mymft.set_powerlimits((0, 2))\n");
    fprintf(pfile, "ax.yaxis.set_major_formatter(mymft)\n\n");

    // label
    //fprintf(pfile, "plt.legend(fancybox=True)\n");
    fprintf(pfile, "plt.xlabel('%s', fontsize=14)\n", xlabel.c_str());
    fprintf(pfile, "plt.ylabel('%s', fontsize=14)\n\n", ylabel.c_str());

    // show & close
    fprintf(pfile, "plt.show()\n");
    fclose(pfile);
}


















#endif
