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

#ifndef FUN_CU
#define FUN_CU
#include "prng.cu"

// rotateUz, function from CLHEP
inline __host__ __device__ float3 rotateUz(float3 vector, float3 newUz) {
    float u1 = newUz.x;
    float u2 = newUz.y;
    float u3 = newUz.z;
    float up = u1*u1 + u2*u2;

    if (up>0) {
        up = sqrtf(up);
        float px = vector.x,  py = vector.y, pz = vector.z;
        vector.x = (u1*u3*px - u2*py)/up + u1*pz;
        vector.y = (u2*u3*px + u1*py)/up + u2*pz;
        vector.z =    -up*px +             u3*pz;
    }
    else if (u3 < 0.) { vector.x = -vector.x; vector.z = -vector.z; } // phi=0  theta=gpu_pi

    return make_float3(vector.x, vector.y, vector.z);
}

// Loglog interpolation
inline __host__ __device__ float loglog_interpolation(float x, float x0, float y0, float x1, float y1) {
    if (x < x0) return y0;
    if (x > x1) return y1;
    x0 = 1.0f / x0;
    return powf(10.0f, log10f(y0) + log10f(y1 / y0) * (log10f(x * x0) / log10f(x1 * x0)));
}

/**
 * \brief Research in array by dicotomy
 * \param[in] key Value to get in array tab
 * \param[in] tab Array to look in
 * \param[in] size Max index in array 
 * \param[in] min Min index in array
 * \param[out] index return array index corresponding to key value
 */
// Binary search
inline __host__ __device__ int binary_search(float key, float* tab, int size, int min=0) {
    int max=size, mid;
    while ((min < max)) {
        mid = (min + max) >> 1;
        if (key > tab[mid]) {
            min = mid + 1;
        } else {
            max = mid;
        }
    }
    return min;
}


/**
 * \brief Linear interpolation
 * \param[in] xa Xa value
 * \param[in] ya Ya value
 * \param[in] xb Xb value
 * \param[in] yb Yb value
 * \param[in] x value to interpolate
 * \param[out] return Y interpolated
 */
inline __host__ __device__ float linear_interpolation(float xa,float ya, float xb,  float yb, float x) { // Taylor young 1st order
    if (xa > x) return ya;
    if (xb < x) return yb;
    return ya + (x-xa) * (yb-ya) / (xb-xa);
}

// Poisson distribution from Geant4 using JKISS32 Generator
inline __device__ int G4Poisson(float mean,ParticleStack &particles, int id) {
    float    number=0.;

    float  position,poissonValue,poissonSum;
    float  value,y,t;
    if(mean<=16.) { // border == 16
        do{
        position=JKISS32(particles, id);
        }while((1.-position)<2.e-7); // to avoid 1 due to float approximation
        poissonValue=expf(-mean);
        poissonSum=poissonValue;
        while((poissonSum<=position)&&(number<40000.)) {
            number++;
            poissonValue*=mean/number;
            if((poissonSum+poissonValue)==poissonSum) break;
            poissonSum+=poissonValue;
        }

        return  (int)number;
    }
    float toto = JKISS32(particles, id);

    t=sqrtf(-2.*logf(toto));
    
    y=2.*pi*JKISS32(particles, id);
    t*=cosf(y);
    value=mean+t*sqrtf(mean)+.5;

    if(value<=0.)
        return  0;
    else if(value>=2.e9) // float limit = 2.e9
        return  (int)2.e9;
    return  (int)value;
}

// Gaussian distribution using JKISS32 Generator
inline __device__ float Gaussian(float mean,float rms,ParticleStack &particles, int id) {
    float  data;
    float  U1,U2,Disp,Fx;
    
    do {
        U1=2.*JKISS32(particles, id)-1.;
        U2=2.*JKISS32(particles, id)-1.;
        Fx=U1*U1+U2*U2;

    } while((Fx>=1.));
    
    float temps = Fx;
    Fx=sqrtf((-2.*logf(Fx))/Fx);

    Disp=U1*Fx;
    data=mean+Disp*rms;

    return  data;
}


#endif
