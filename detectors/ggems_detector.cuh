// GGEMS Copyright (C) 2015

/*!
 * \file ggems_detector.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 2 december 2015
 *
 *
 *
 */

#ifndef GGEMSDETECTOR_CUH
#define GGEMSDETECTOR_CUH

#include "global.cuh"
#include "particles.cuh"

class GGEMSDetector {
    public:
        GGEMSDetector();
        virtual ~GGEMSDetector() {}

        //void set_detector(VoxPhanImgNav &aPhantom);
        virtual void initialize(GlobalSimulationParameters params) = 0;

        virtual void track_to_in(Particles particles) = 0;
        virtual void track_to_out(Particles particles) = 0;

        void digitizer();
        //virtual void save_data(std::string filename) = 0;

        std::string get_detector_name();

    protected:
      void set_name(std::string name);

    protected:
        //VoxPhanImgNav m_vox_phan_img;
        std::string m_detector_name;
};

#endif
