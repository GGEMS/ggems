#ifndef DOSIMETRY_ANALOG_CUH
#define DOSIMETRY_ANALOG_CUH

#include "dosimetry_actor.cuh"



class DoseAnalogCalculator : public DoseCalculator {

    public:
        DoseAnalogCalculator() {};
        ~DoseAnalogCalculator() {}
    
        
        void store_dose();
        
        
        
};









#endif