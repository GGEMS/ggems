// GGEMS Copyright (C) 2017

/*!
 * \file fun.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 *
 *
 */

#ifndef FUN_CU
#define FUN_CU
#include "fun.cuh"
#include "prng.cuh"



// rotateUz, function from CLHEP
__host__ __device__ f32xyz rotateUz ( f32xyz vector, f32xyz newUz )
{
    f32 u1 = newUz.x;
    f32 u2 = newUz.y;
    f32 u3 = newUz.z;
    f32 up = u1*u1 + u2*u2;

    if ( up>0 )
    {
        up = sqrtf ( up );
        f32 px = vector.x,  py = vector.y, pz = vector.z;
        vector.x = ( u1*u3*px - u2*py ) /up + u1*pz;
        vector.y = ( u2*u3*px + u1*py ) /up + u2*pz;
        vector.z =    -up*px +             u3*pz;
    }
    else if ( u3 < 0. )
    {
        vector.x = -vector.x;    // phi=0  theta=gpu_pi
        vector.z = -vector.z;
    }

    return make_f32xyz ( vector.x, vector.y, vector.z );
}

// Loglog interpolation
__host__ __device__ f32 loglog_interpolation ( f32 x, f32 x0, f32 y0, f32 x1, f32 y1 )
{
    if ( x < x0 ) return y0;
    if ( x > x1 ) return y1;
    x0 = 1.0f / x0;
    return powf ( 10.0f, log10f ( y0 ) + log10f ( y1 / y0 ) * ( log10f ( x * x0 ) / log10f ( x1 * x0 ) ) );
}

// // Binary search f32
// __host__ __device__ ui32 binary_search(f32 key, f32* tab, ui32 size, ui32 min=0) {
//     ui32 max=size, mid;
//     while ((min < max)) {
//         mid = (min + max) >> 1;
//         if (key > tab[mid]) {
//             min = mid + 1;
//         } else {
//             max = mid;
//         }
//     }
//     return min;
// }
//
// // Binary search f64
// __host__ __device__ ui32 binary_search(f64 key, f64* tab, ui32 size, ui32 min=0) {
//     ui32 max=size, mid;
//     while ((min < max)) {
//         mid = (min + max) >> 1;
//         if (key > tab[mid]) {
//             min = mid + 1;
//         } else {
//             max = mid;
//         }
//     }
//     return min;
// }

// Linear interpolation
__host__ __device__ f32 linear_interpolation ( f32 xa, f32 ya, f32 xb, f32 yb, f32 x )
{
    // Taylor young 1st order
//     if ( xa > x ) return ya;
//     if ( xb < x ) return yb;
    
    if (xa > xb) return yb;
    if (xa >= x) return ya;
    if (xb <= x) return yb;
    
    return ya + ( x-xa ) * ( yb-ya ) / ( xb-xa );
}




__host__ __device__ f32 Gaussian (f32 mean, f32 rms, ParticlesData *particles, ui32 id )
{
    f32  data;
    f32  U1,U2,Disp,Fx;

    do
    {
        U1 = 2.*prng_uniform( particles, id )-1.;
        U2 = 2.*prng_uniform( particles, id )-1.;
        Fx = U1*U1 + U2*U2;

    }
    while ( ( Fx >= 1. ) );

    Fx = sqrtf ( ( -2.*logf ( Fx ) ) /Fx );
    Disp = U1*Fx;
    data = mean + Disp*rms;

    return  data;
}


/// Filtering

#define SWAP(a, b) {float tmp=(a); (a)=(b); (b)=tmp;}
// Quick sort O(n(log n))
void inkernel_quicksort(f32* vec, i32 l, i32 r) {
    ui32 j;

    if (l < r)
    {
        ui32 i;
        f32 pivot;

        pivot = vec[l];
        i = l;
        j = r+1;

        //printf("l %i r %i - i %i j %i  pivot %f\n", l, r, i, j, pivot);

        while (1) {
            do ++i; while ( vec[i] <= pivot && i <= r );
            do --j; while ( vec[j] > pivot );
            if ( i >= j ) break;
            //printf("   swap  i %i %f  j %i %f\n", i, j, vec[i], vec[j]);
            SWAP( vec[i], vec[j] );
        }
        //printf("   swap  l %i %f  j %i %f\n", l, j, vec[l], vec[j]);
        SWAP( vec[l], vec[j] );
        inkernel_quicksort( vec, l, j-1 );
        inkernel_quicksort( vec, j+1, r );
    }
}
#undef SWAP

// Get stats from a spatial window
f32xyz get_win_min_max_mead( f32* input, ui32 w_size, ui32 nx, ui32 ny, ui32 x, ui32 y, ui32 z )
{
    i32 nwin = w_size * w_size * w_size;
    f32* win = new f32[nwin];

    i32 edgex = w_size / 2;
    i32 edgey = w_size / 2;
    i32 edgez = w_size / 2;

    i32 mpos = nwin / 2;

    i32 step = ny * nx;
    i32 wx, wy, wz, indy, indz;
    i32 nwa;

    // read windows
    nwa = 0;
    for ( wz = 0; wz < w_size; ++wz )
    {
        indz = step * (z + wz - edgez);

        for (wy=0; wy<w_size; ++wy)
        {
            indy = indz + nx * (y + wy - edgey);

            for ( wx = 0; wx < w_size; ++wx )
            {
                win[ nwa++ ] = input[ indy + x + wx - edgex ];
            } // wx

        } // wy

    } // wz

    // sort win
    inkernel_quicksort(win, 0, nwin-1);

    // get values
    f32xyz res;
    res.x = win[ 0 ];     // min
    res.y = win[ mpos ];  // mead
    res.z = win[ nwa-1 ]; // max

    delete[] win;

    return res;
}

// 3D Median Filter
f32* Filter::median( f32* input,  ui32 nx, ui32 ny, ui32 nz, ui32 w_size ) {

    // init output
    f32 *output = new f32[ nx*ny*nz ];
    ui32 i=0; while ( i < nx*ny*nz )
    {
        output[ i ] = input[ i ];
        i++;
    }

    i32 nwin = w_size * w_size * w_size;
    f32* win = new f32[nwin];
    i32 edgex = w_size / 2;
    i32 edgey = w_size / 2;
    i32 edgez = w_size / 2;
    i32 mpos = nwin / 2;
    i32 step = ny * nx;
    i32 x, y, z, wx, wy, wz, ind, indy, indz, indw;
    i32 nwa;

    for ( z = edgez; z < (nz-edgez); ++z )
    {
        indz = z * step;
        for ( y = edgey; y < (ny-edgey); ++y)
        {
            ind = indz + y*nx;
            for ( x = edgex; x < (nx-edgex); ++x)
            {

                nwa = 0;

                for ( wz = 0; wz < w_size; ++wz )
                {
                    indw = step * (z + wz - edgez);
                    for ( wy=0; wy < w_size; ++wy )
                    {
                        indy = indw + nx*(y + wy - edgey);

                        for ( wx = 0; wx < w_size; ++wx )
                        {
                            win[ nwa ] = input[ indy + x + wx - edgex ];
                            ++nwa;
                        }
                    }
                }

                // sort win
                inkernel_quicksort(win, 0, nwin-1);

                // select mpos
                output[ ind + x ] = win[ mpos ];

            } // x
        } // y
    } // z

    return output;
}

// 3D Mean Filter
f32* Filter::mean( f32* input,  ui32 nx, ui32 ny, ui32 nz, ui32 w_size ) {

    // init output
    f32 *output = new f32[ nx*ny*nz ];
    ui32 i=0; while ( i < nx*ny*nz )
    {
        output[ i ] = input[ i ];
        i++;
    }

    i32 nwin = w_size * w_size * w_size;
    i32 edgex = w_size / 2;
    i32 edgey = w_size / 2;
    i32 edgez = w_size / 2;
    i32 step = ny * nx;
    i32 x, y, z, wx, wy, wz, ind, indy, indz, indw;
    f32 sum;

    for ( z = edgez; z < (nz-edgez); ++z )
    {
        indz = z * step;
        for ( y = edgey; y < (ny-edgey); ++y)
        {
            ind = indz + y*nx;
            for ( x = edgex; x < (nx-edgex); ++x)
            {

                sum = 0.0;

                for ( wz = 0; wz < w_size; ++wz )
                {
                    indw = step * (z + wz - edgez);
                    for ( wy=0; wy < w_size; ++wy )
                    {
                        indy = indw + nx*(y + wy - edgey);

                        for ( wx = 0; wx < w_size; ++wx )
                        {
                            sum += input[ indy + x + wx - edgex ];
                        }
                    }
                }

                // select mpos
                output[ ind + x ] = sum / f32( nwin );

            } // x
        } // y
    } // z

    return output;
}

// 3D Adaptive Median Filter
f32* Filter::adaptive_median(f32* input,  ui32 nx, ui32 ny, ui32 nz,
                             ui32 w_size, ui32 w_size_max )
{
    // init output
    f32 *output = new f32[ nx*ny*nz ];
    ui32 i=0; while ( i < nx*ny*nz )
    {
        output[ i ] = input[ i ];
        i++;
    }

    ui32 step = ny * nx;
    f32 smin, smead, smax;
    ui32 edgex, edgey, edgez;
    ui32 wa;
    ui32 x, y, z, ind, indimz;

    edgex = w_size_max / 2;
    edgey = w_size_max / 2;
    edgez = w_size_max / 2;

    f32xyz stat;

    // Loop over position
    for ( z = edgez; z < ( nz-edgez ); ++z )
    {
        GGcout << " Adaptive median filter slice " << z << GGendl;
        indimz = step * z;

        for ( y = edgey; y < ( ny - edgey ); ++y )
        {
            ind = indimz + y * nx;

            for ( x = edgex; x < ( nx - edgex ); ++x)
            {
                // Loop over win size
                for (wa = w_size; wa <= w_size_max; wa+=2)
                {
                    // Get stat values from the current win size
                    stat = get_win_min_max_mead(input, wa, nx, ny, x, y, z);
                    smin = stat.x; smead = stat.y; smax = stat.z;

                    printf("wa %i   min %f mead %f max %f    cur val %f\n", wa, smin, smead, smax, input[ ind + x ]);

                    // if smin < smead < smaw
                    if ( ( smin < smead ) && ( smead < smax ) )
                    {
                        // if smin < val < smax
                        if ( ( smin < input[ ind + x ] ) && ( input[ ind + x ] < smax ) )
                        {
                            output[ ind + x ] = input[ ind + x ];
                            printf("   Assign cur val\n");
                        }
                        else
                        {
                            output[ ind + x ] = smead;
                            printf("   Assign smead\n");
                        }

                        printf("   Next position\n");
                        // move to the next position
                        break;
                    }
                    else
                    {
                        // Else let's increase the win size and restart
                        printf("   Next win size\n");

                        // In anycase if win size max is reached, assigned the value
                        if ( wa == w_size_max )
                        {
                            output[ ind + x ] = smead;
                        }
                    }

                } // Win size

                break; //DEBUG
            } // x
            break; //DEBUG

        } // y
        break; //DEBUG

    } // z

    return output;
}



// 3D Resampling by Lanczos3 (uses backwarp mapping)
#define pi 3.141592653589793238462643383279
#define SINC(x) ((x)==(0)?1:sin(pi*(x))/(pi*(x)))
f32* Filter::resampling_lanczos3(f32* input, ui32 nx, ui32 ny, ui32 nz, ui32 new_nx, ui32 new_ny, ui32 new_nz) {

    // init output
    f32 *output = new f32[ new_nx*new_ny*new_nz ];
    ui32 i=0; while ( i < new_nx*new_ny*new_nz )
    {
        output[ i++ ] = 0.0f;
    }

    // scale factor
    f32 scalez = nz / ( f32 )new_nz;
    f32 scaley = ny / ( f32 )new_ny;
    f32 scalex = nx / ( f32 )new_nx;
    i32 stepo = nx*ny;
    i32 stept = new_nx*new_ny;

    // backward mapping, thus scan from the target
    i32 x, y, z;
    i32 xi, yi, zi;
    f32 xt, yt, zt;
    i32 u, v, w;
    i32 wz, wy, wx;
    f32 p, q, r;
    f32 dx, dy, dz;

    for ( z = 0; z < new_nz; ++z )
    {
        GGcout << "Resmapling: slice " << z+1 << "/" << new_nz << GGendl;
        zt = ( z + 0.5f ) * scalez - 0.5f;
        zi = ( i32 )zt;

        for ( y = 0; y < new_ny; ++y )
        {
            yt = ( y + 0.5f) * scaley - 0.5f;
            yi = ( i32 )yt;

            for ( x = 0; x < new_nx; ++x )
            {
                xt = ( x + 0.5f ) * scalex - 0.5f;
                xi = ( i32 )xt;

                // window loop
                r = 0;
                for (wz = -2; wz < 4; ++wz)
                {
                    w = zi + wz;
                    if ( w >= nz ) continue;
                    if ( w < 0 ) continue;
                    dz = zt - w;
                    if ( abs( dz ) > 3.0f ) dz = 3.0f;
                    q = 0;

                    for ( wy = -2; wy < 4; ++wy )
                    {
                        v = yi + wy;
                        if ( v >= ny ) continue;
                        if ( v < 0 ) continue;
                        dy = yt - v;
                        if ( abs( dy ) > 3.0f) dy = 3.0f;
                        p = 0;

                        for ( wx = -2; wx < 4; ++wx )
                        {
                            u = xi + wx;
                            if ( u >= nx ) continue;
                            if ( u < 0 ) continue;
                            dx = xt - u;
                            if ( abs( dx ) > 3.0f ) dx = 3.0f;

                            p = p + input[ w*stepo + v*nx + u ] * SINC( dx ) * SINC( dx * 0.333333f );
                        } // wx

                        q = q + p * SINC( dy ) * SINC( dy * 0.333333f );

                    } // wy

                    r = r + q * SINC( dz ) * SINC( dz * 0.333333f );

                } // wz

                // assign the new value
                output[ z*stept + y*new_nx + x ] = r;

            } // x
        } // y
    } // z

    return output;

}
#undef pi
#undef SINC

f32* Filter::cropping_vox_around_center( f32* input, ui32 nx, ui32 ny, ui32 nz,
                                         i32 xmin, i32 xmax, i32 ymin, i32 ymax, i32 zmin, i32 zmax )
{
    // get center
    ui32 cx = nx / 2;
    ui32 cy = ny / 2;
    ui32 cz = nz / 2;

    // get params
    ui32 ojump = nx*ny;
    ui32 onx = nx;

    // get new dimension
    nx = xmax-xmin;
    ny = ymax-ymin;
    nz = zmax-zmin;

    // init output
    f32 *output = new f32[ nx*ny*nz ];
    ui32 i=0; while ( i < nx*ny*nz )
    {
        output[ i++ ] = 0.0f;
    }

    // get abs crop value
    xmin = cx+xmin; xmax = cx+xmax;
    ymin = cy+ymin; ymax = cy+ymax;
    zmin = cz+zmin; zmax = cz+zmax;

    // Cropping
    ui32 ix, iy, iz, index;

    index = 0;
    iz = zmin; while ( iz < zmax )
    {
        iy = ymin; while ( iy < ymax )
        {
            ix = xmin; while ( ix < xmax )
            {
                output[ index++ ] = input[ iz*ojump + iy*onx + ix ];
                ++ix;
            } // ix

            ++iy;
        } // iy

        ++iz;
    } // iz

    return output;
}

void Filter::capping_values( f32* input, ui32 nx, ui32 ny, ui32 nz, f32 val_min, f32 val_max )
{
    ui32 index = nx*ny*nz;
    ui32 i=0; while ( i < index )
    {
        if ( input[ i ] > val_max ) input[ i ] = val_max;
        if ( input[ i ] < val_min ) input[ i ] = val_min;
        ++i;
    }
}

/// Atomic functions

/*   OLD VERSION
        __device__ double ggems_atomic_add_f64(f64* array, ui32 pos, f64 val)
        {
            f64 *address = &array[pos];
            unsigned long long int* address_as_ull = (unsigned long long int*)address;
            unsigned long long int old = *address_as_ull, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,
                                __double_as_longlong(val +
                                __longlong_as_double(assumed)));
            } while (assumed != old);
            return __longlong_as_double(old);
        }
*/

__host__ __device__ void ggems_atomic_add_f64(f64* array, ui32 pos, f64 val)
{

    #ifdef __CUDA_ARCH__

        // Single precision f64 is in fact a f32
        #ifdef SINGLE_PRECISION

            ggems_atomic_add(array, pos, val);

        // Double precision
        #else

            f64 *address = &array[pos];
            unsigned long long oldval, newval, readback;

            oldval = __double_as_longlong(*address);
            newval = __double_as_longlong(__longlong_as_double(oldval) + val);
            while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
            {
                oldval = readback;
                newval = __double_as_longlong(__longlong_as_double(oldval) + val);
            }

        #endif

    #else
        array[pos] += val;
    #endif

}

#endif
