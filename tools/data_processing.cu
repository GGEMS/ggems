// GGEMS Copyright (C) 2017

/*!
 * \file data_procesing.cu
 * \brief Functions allowing processing data (filtering, resampling, etc.)
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date May 4 2017
 *
 *
 *
 */

#ifndef PROCESSING_CU
#define PROCESSING_CU

#include "data_processing.cuh"

/// Filtering helper function ///////////////////////////////////////////

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

/// Misc. ////////////////////////////////////////////////////////////////////////////////

// New volume init to zeros from an existing vol
VoxVolumeData<f32>* DataProcessing::new_volume_zeros( const VoxVolumeData<f32> *input )
{
    VoxVolumeData<f32> *output = new VoxVolumeData<f32>;
    output->nb_vox_x = input->nb_vox_x;
    output->nb_vox_y = input->nb_vox_y;
    output->nb_vox_z = input->nb_vox_z;

    output->number_of_voxels = input->number_of_voxels;

    output->off_x = input->off_x;
    output->off_y = input->off_y;
    output->off_z = input->off_z;

    output->spacing_x = input->spacing_x;
    output->spacing_y = input->spacing_y;
    output->spacing_z = input->spacing_z;

    output->xmin = input->xmin;
    output->xmax = input->xmax;
    output->ymin = input->ymin;
    output->ymax = input->ymax;
    output->zmin = input->zmin;
    output->zmax = input->zmax;

    output->values = new f32[ input->number_of_voxels ];
    ui32 i = 0; while( i < input->number_of_voxels )
    {
        output->values[ i++ ] = 0.0f;
    }

    return output;
}

// Return stat info on a voxelized volume (min, max, mean, std)
f32xyzw DataProcessing::mask_info( const VoxVolumeData<f32> *input, std::string mask_name, ui32 id_mask )
{
    f32 min =  F32_MAX;
    f32 max = -F32_MAX;
    f64 mean = 0.0;
    f64 std = 0.0;
    f64 sum = 0.0;
    f64 sum2 = 0.0;
    f64 val;

    ui32 ct_vox = 0;
    ui32 i =0;

    // Open and read mask
    ImageIO *image_io = new ImageIO;
    image_io->open( mask_name );
    f32 *mask = image_io->get_image_in_f32();
    ui32xyz nbvox = image_io->get_size();
    ui32 totvox = nbvox.x * nbvox.y * nbvox.z;

    if (totvox != input->number_of_voxels)
    {
        GGwarn << "Mask file and volume have different number of voxels!" << GGendl;
        exit_simulation();
    }
    delete image_io;

    // Get stats
    while (i < input->number_of_voxels)
    {
        if ( mask[ i ] == id_mask )
        {
            val = input->values[ i ];

            if (val > max) max = val;
            if (val < min) min = val;

            sum += val;
            sum2  += (val * val);
            ct_vox++;
        }

        ++i;
    }

    mean = sum / f64( ct_vox );
    std  = ( sum2 / f64( ct_vox ) ) - ( sum*sum / f64( ct_vox*ct_vox ) );
    std  = sqrt( std );

    return make_f32xyzw( min, max, mean, std );
}

/// Filtering ////////////////////////////////////////////////////////////////////////////

// 3D Median Filter
VoxVolumeData<f32> *DataProcessing::filter_median( const VoxVolumeData<f32> *input, ui32 w_size )
{
    // init output
    VoxVolumeData<f32> *output = DataProcessing::new_volume_zeros( input );

    // copy data from input to output
    ui32 i=0; while ( i < input->number_of_voxels )
    {
        output->values[ i ] = input->values[ i ];
        i++;
    }

    i32 nwin = w_size * w_size * w_size;
    f32* win = new f32[nwin];
    i32 edgex = w_size / 2;
    i32 edgey = w_size / 2;
    i32 edgez = w_size / 2;
    i32 mpos = nwin / 2;
    i32 step = input->nb_vox_y * input->nb_vox_x;
    i32 x, y, z, wx, wy, wz, ind, indy, indz, indw;
    i32 nwa;

    for ( z = edgez; z < (input->nb_vox_z-edgez); ++z )
    {
        indz = z * step;
        for ( y = edgey; y < (input->nb_vox_y-edgey); ++y)
        {
            ind = indz + y*input->nb_vox_x;
            for ( x = edgex; x < (input->nb_vox_x-edgex); ++x)
            {

                nwa = 0;

                for ( wz = 0; wz < w_size; ++wz )
                {
                    indw = step * (z + wz - edgez);
                    for ( wy = 0; wy < w_size; ++wy )
                    {
                        indy = indw + input->nb_vox_x*(y + wy - edgey);

                        for ( wx = 0; wx < w_size; ++wx )
                        {
                            win[ nwa ] = input->values[ indy + x + wx - edgex ];
                            ++nwa;
                        }
                    }
                }

                // sort win
                inkernel_quicksort(win, 0, nwin-1);

                // select mpos
                output->values[ ind + x ] = win[ mpos ];

            } // x
        } // y
    } // z

    return output;
}

// 3D Mean Filter
VoxVolumeData<f32> *DataProcessing::filter_mean( const VoxVolumeData<f32> *input, ui32 w_size )
{

    // init output
    VoxVolumeData<f32> *output = new_volume_zeros( input );

    // copy data from input to output
    ui32 i=0; while ( i < input->number_of_voxels )
    {
        output->values[ i ] = input->values[ i ];
        i++;
    }

    i32 nwin = w_size * w_size * w_size;
    i32 edgex = w_size / 2;
    i32 edgey = w_size / 2;
    i32 edgez = w_size / 2;
    i32 step = input->nb_vox_y * input->nb_vox_x;
    i32 x, y, z, wx, wy, wz, ind, indy, indz, indw;
    f32 sum;

    for ( z = edgez; z < (input->nb_vox_z-edgez); ++z )
    {
        indz = z * step;
        for ( y = edgey; y < (input->nb_vox_y-edgey); ++y)
        {
            ind = indz + y*input->nb_vox_x;
            for ( x = edgex; x < (input->nb_vox_x-edgex); ++x)
            {

                sum = 0.0;

                for ( wz = 0; wz < w_size; ++wz )
                {
                    indw = step * (z + wz - edgez);
                    for ( wy = 0; wy < w_size; ++wy )
                    {
                        indy = indw + input->nb_vox_x*(y + wy - edgey);

                        for ( wx = 0; wx < w_size; ++wx )
                        {
                            sum += input->values[ indy + x + wx - edgex ];
                        }
                    }
                }

                // select mpos
                output->values[ ind + x ] = sum / f32( nwin );

            } // x
        } // y
    } // z

    return output;
}


// 3D Adaptive Median Filter
VoxVolumeData<f32> *DataProcessing::filter_adaptive_median( const VoxVolumeData<f32> *input, ui32 w_size, ui32 w_size_max )
{
    // init output
    VoxVolumeData<f32> *output = new_volume_zeros( input );

    // copy data from input to output
    ui32 i=0; while ( i < input->number_of_voxels )
    {
        output->values[ i ] = input->values[ i ];
        i++;
    }

    ui32 step = input->nb_vox_y * input->nb_vox_x;
    f32 smin, smead, smax;
    ui32 edgex, edgey, edgez;
    ui32 wa;
    ui32 x, y, z, ind, indimz;

    edgex = w_size_max / 2;
    edgey = w_size_max / 2;
    edgez = w_size_max / 2;

    f32xyz stat;

    // Loop over position
    for ( z = edgez; z < ( input->nb_vox_z-edgez ); ++z )
    {
        GGcout << " Adaptive median filter slice " << z << GGendl;
        indimz = step * z;

        for ( y = edgey; y < ( input->nb_vox_y - edgey ); ++y )
        {
            ind = indimz + y * input->nb_vox_x;

            for ( x = edgex; x < ( input->nb_vox_x - edgex ); ++x)
            {
                // Loop over win size
                for (wa = w_size; wa <= w_size_max; wa+=2)
                {
                    // Get stat values from the current win size
                    stat = get_win_min_max_mead(input->values, wa, input->nb_vox_x, input->nb_vox_y, x, y, z);
                    smin = stat.x; smead = stat.y; smax = stat.z;

                    printf("wa %i   min %f mead %f max %f    cur val %f\n", wa, smin, smead, smax, input[ ind + x ]);

                    // if smin < smead < smaw
                    if ( ( smin < smead ) && ( smead < smax ) )
                    {
                        // if smin < val < smax
                        if ( ( smin < input->values[ ind + x ] ) && ( input->values[ ind + x ] < smax ) )
                        {
                            output->values[ ind + x ] = input->values[ ind + x ];
                            printf("   Assign cur val\n");
                        }
                        else
                        {
                            output->values[ ind + x ] = smead;
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
                            output->values[ ind + x ] = smead;
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

/// Resampling ///////////////////////////////////////////////////////////////

// 3D Resampling by Lanczos3 (uses backwarp mapping)
#define pi 3.141592653589793238462643383279
#define SINC(x) ((x)==(0)?1:sin(pi*(x))/(pi*(x)))
VoxVolumeData<f32> *DataProcessing::resampling_lanczos3( const VoxVolumeData<f32> *input, ui32 new_nx, ui32 new_ny, ui32 new_nz )
{
    // init output
    VoxVolumeData<f32> *output = new_volume_zeros( input );

    // copy data from input to output
    ui32 i=0; while ( i < input->number_of_voxels )
    {
        output->values[ i ] = input->values[ i ];
        i++;
    }

    // scale factor
    f32 scalez = input->nb_vox_z / ( f32 )new_nz;
    f32 scaley = input->nb_vox_y / ( f32 )new_ny;
    f32 scalex = input->nb_vox_x / ( f32 )new_nx;
    i32 stepo = input->nb_vox_x*input->nb_vox_y;
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
                    if ( w >= input->nb_vox_z ) continue;
                    if ( w < 0 ) continue;
                    dz = zt - w;
                    if ( abs( dz ) > 3.0f ) dz = 3.0f;
                    q = 0;

                    for ( wy = -2; wy < 4; ++wy )
                    {
                        v = yi + wy;
                        if ( v >= input->nb_vox_y ) continue;
                        if ( v < 0 ) continue;
                        dy = yt - v;
                        if ( abs( dy ) > 3.0f) dy = 3.0f;
                        p = 0;

                        for ( wx = -2; wx < 4; ++wx )
                        {
                            u = xi + wx;
                            if ( u >= input->nb_vox_x ) continue;
                            if ( u < 0 ) continue;
                            dx = xt - u;
                            if ( abs( dx ) > 3.0f ) dx = 3.0f;

                            p = p + input->values[ w*stepo + v*input->nb_vox_x + u ] * SINC( dx ) * SINC( dx * 0.333333f );
                        } // wx

                        q = q + p * SINC( dy ) * SINC( dy * 0.333333f );

                    } // wy

                    r = r + q * SINC( dz ) * SINC( dz * 0.333333f );

                } // wz

                // assign the new value
                output->values[ z*stept + y*new_nx + x ] = r;

            } // x
        } // y
    } // z

    return output;

}
#undef pi
#undef SINC


/*

f32* DataProcessing::cropping_vox_around_center( f32* input, ui32 nx, ui32 ny, ui32 nz,
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

void DataProcessing::capping_values( f32* input, ui32 nx, ui32 ny, ui32 nz, f32 val_min, f32 val_max )
{
    ui32 index = nx*ny*nz;
    ui32 i=0; while ( i < index )
    {
        if ( input[ i ] > val_max ) input[ i ] = val_max;
        if ( input[ i ] < val_min ) input[ i ] = val_min;
        ++i;
    }
}

void DataProcessing::scale_values( f32 *input, ui32 nx, ui32 ny, ui32 nz, f32 val_scale )
{
    ui32 index = nx*ny*nz;
    ui32 i=0; while ( i < index )
    {
        input[ i ] = input[ i ] * val_scale;
        ++i;
    }
}
*/

#endif
