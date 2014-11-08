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

#ifndef MATHPLOT_CUH
#define MATHPLOT_CUH

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <vector>
#include <cfloat>
#include <string>

#include "particles.cuh"

class MathPlotBuilder {
    public:
        MathPlotBuilder();

        void plot_distribution(float* xdata, unsigned int nxdata,
                               unsigned int n_bins, std::string filename,
                               std::string xlabel, std::string ylabel);

        void plot_energy_distribution(ParticleBuilder particles,
                                      unsigned int n_bins, std::string filename);

    private:
        void get_histogramm(float* xdata, unsigned int nxdata,
                           float* bins, float* nbelt, unsigned int nbins);
        void get_weighted_histogramm(float* xdata, float* ydata, unsigned int nxdata,
                                     float* hist, float* nbelt, float* bins,
                                     unsigned int nbins);



};




#endif
