// GGEMS Copyright (C) 2015

/*!
 * \file point_source.cuh
 * \brief Header of point source class
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 * Point source class
 *
 */

#ifndef POINT_SOURCE_CUH
#define POINT_SOURCE_CUH

#include "global.cuh"
#include "ggems_vsource.cuh"

// Sphere
class PointSource : public GGEMSVSource {
    public:        
        PointSource();
        ~PointSource();

        void set_position(f32 vpx, f32 vpy, f32 vpz);
        void set_particle_type(std::string pname);
        void set_mono_energy(f32 valE);
        void set_energy_spectrum(f64 *valE, f64 *hist, ui32 nb);

        // Virtual from GGEMSVSource
        void get_primaries_generator(Particles particles);
        void initialize(GlobalSimulationParameters params);       

    private:
        bool m_check_mandatory();

        GlobalSimulationParameters m_params;

        f32 m_px, m_py, m_pz;
        ui32 m_nb_of_energy_bins;
        f64 *m_spectrumE_h;
        f64 *m_spectrumE_d;
        f64 *m_spectrumCDF_h;
        f64 *m_spectrumCDF_d;
        ui8 m_particle_type;
};

#endif
