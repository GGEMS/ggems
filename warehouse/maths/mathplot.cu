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
void MathPlotBuilder::get_histogramm(f32* xdata, ui32 nxdata,
                                    f32* bins, f32* nbelt, ui32 nbins) {
    ui32 i;

    // Find min max values
    f32 xmin = FLT_MAX;
    f32 xmax = FLT_MIN;
    f32 val;
    i=0; while (i<nxdata) {
        val = xdata[i];
        if (val < xmin) {xmin=val;};
        if (val > xmax) {xmax=val;};
        ++i;
    }
    assert(xmin != FLT_MAX);
    assert(xmax != -FLT_MIN);

    // Init nbelt vector
    i=0; while (i<nbins) {
        nbelt[i] = 0;
        ++i;
    }

    // Build hist
    f32 di = (xmax-xmin) / f32(nbins-1);
    f32 posi;
    i=0; while (i<nxdata) {
        posi = ((xdata[i]-xmin) / di)+0.5f;

        assert(i32(posi) != nbins);
        assert(i32(posi) >= 0);

        ++nbelt[i32(posi)];
        ++i;
    }

    // Build bins
    i=0; while (i<nbins) {
        bins[i] = xmin + i*di;
        ++i;
    }

}

// Build a weigthed histogramm based on 2D data
void MathPlotBuilder::get_weighted_histogramm(f32* xdata, f32* ydata, ui32 nxdata,
                                              f32* hist, f32* nbelt, f32* bins,
                                              ui32 nbins) {
    i32 i;
    f32 xmin = FLT_MAX;
    f32 xmax = FLT_MIN;
    f32 val;
    i=0; while (i<nxdata) {
        val = xdata[i];
        if (val < xmin) {xmin=val;};
        if (val > xmax) {xmax=val;};
        ++i;
    }
    assert(xmin != FLT_MAX);
    assert(xmax != -FLT_MIN);

    // Init vectors
    i=0; while (i<nbins) {
        nbelt[i] = 0;
        hist[i] = 0;
        ++i;
    }

    // Build hist
    f32 di = (xmax-xmin) / f32(nbins-1);
    f32 posi;
    i=0; while (i<nxdata) {
        posi = ((xdata[i]-xmin) / di)+0.5f;
        ++nbelt[i32(posi)];
        hist[i32(posi)] += ydata[i];
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
void MathPlotBuilder::plot_distribution(f32 *xdata, ui32 nxdata,
                                        ui32 n_bin, std::string filename,
                                        std::string xlabel, std::string ylabel) {

    // Memory allocation
    f32* bins = (f32*)malloc(n_bin * sizeof(f32));
    f32* nbelt = (f32*)malloc(n_bin * sizeof(f32));

    // Compute the histogramm
    get_histogramm(xdata, nxdata, bins, nbelt, n_bin);

    // Export data
    std::string data_name = filename + ".dat";
    FILE* pfile = fopen(data_name.c_str(), "wb");
    f32 buffer = (f32)n_bin;
    fwrite(&buffer, sizeof(f32), 1, pfile);
    fwrite(bins, sizeof(f32), n_bin, pfile);
    fwrite(nbelt, sizeof(f32), n_bin, pfile);
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


// Plot energy distribution
void MathPlotBuilder::plot_energy_distribution(ParticleBuilder particles,
                                               ui32 n_bin, std::string filename) {

    // Looking only on particle alive
    ui32 nxdata = 0;
    ui32 i= 0;
    while (i < particles.stack.size) {
        if (particles.stack.endsimu[i] == PARTICLE_ALIVE) ++nxdata;
        ++i;
    }
    f32* xdata = (f32*)malloc(nxdata * sizeof(f32));
    i=0; while (i < particles.stack.size) {
        if (particles.stack.endsimu[i] == PARTICLE_ALIVE) {
            xdata[i] = particles.stack.E[i];
            printf("%e\n", xdata[i]);
        }
        ++i;
    }

    // Memory allocation
    f32* bins = (f32*)malloc(n_bin * sizeof(f32));
    f32* nbelt = (f32*)malloc(n_bin * sizeof(f32));

    // Compute the histogramm
    get_histogramm(xdata, nxdata, bins, nbelt, n_bin);

    // Export data
    std::string data_name = filename + ".dat";
    FILE* pfile = fopen(data_name.c_str(), "wb");
    f32 buffer = (f32)n_bin;
    fwrite(&buffer, sizeof(f32), 1, pfile);
    fwrite(bins, sizeof(f32), n_bin, pfile);
    fwrite(nbelt, sizeof(f32), n_bin, pfile);
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
    fprintf(pfile, "plt.xlabel('Energy [MeV]', fontsize=14)\n");
    fprintf(pfile, "plt.ylabel('Number of particles', fontsize=14)\n\n");

    // show & close
    fprintf(pfile, "plt.show()\n");
    fclose(pfile);
}


















#endif
