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

#include "global.cuh"
#include "particles.cuh"

class MathPlotBuilder {
    public:
        MathPlotBuilder();

        void plot_distribution(f32* xdata, ui32 nxdata,
                               ui32 n_bins, std::string filename,
                               std::string xlabel, std::string ylabel);

        void plot_energy_distribution(ParticleBuilder particles,
                                      ui32 n_bins, std::string filename);

    private:
        void get_histogramm(f32* xdata, ui32 nxdata,
                           f32* bins, f32* nbelt, ui32 nbins);
        void get_weighted_histogramm(f32* xdata, f32* ydata, ui32 nxdata,
                                     f32* hist, f32* nbelt, f32* bins,
                                     ui32 nbins);



};




#endif
