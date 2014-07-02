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

// rotateUz, function from CLHEP
__device__ float3 rotateUz(float3 vector, float3 newUz) {
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
__device__ float loglog_interpolation(float x, float x0, float y0, float x1, float y1) {
    if (x < x0) return y0;
    if (x > x1) return y1;
    x0 = 1.0f / x0;
    return powf(10.0f, log10f(y0) + log10f(y1 / y0) * (log10f(x * x0) / log10f(x1 * x0)));
}


#endif
