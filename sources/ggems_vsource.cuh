// GGEMS Copyright (C) 2015

/*!
 * \file ggems_vsource.cuh
 * \brief Header of virtual source class
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 * Virtual class that handle every sources used in GGEMS
 *
 */

#ifndef GGEMS_VSOURCE_CUH
#define GGEMS_VSOURCE_CUH

class GGEMSVSource {
    public:
        GGEMSVSource();
        ~GGEMSVSource();
        virtual void get_primaries_generator();

    private:
};

#endif
