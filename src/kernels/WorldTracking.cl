// ************************************************************************
// * This file is part of GGEMS.                                          *
// *                                                                      *
// * GGEMS is free software: you can redistribute it and/or modify        *
// * it under the terms of the GNU General Public License as published by *
// * the Free Software Foundation, either version 3 of the License, or    *
// * (at your option) any later version.                                  *
// *                                                                      *
// * GGEMS is distributed in the hope that it will be useful,             *
// * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
// * GNU General Public License for more details.                         *
// *                                                                      *
// * You should have received a copy of the GNU General Public License    *
// * along with GGEMS.  If not, see <https://www.gnu.org/licenses/>.      *
// *                                                                      *
// ************************************************************************


/*!
  \file WorldTracking.cl

  \brief OpenCL kernel tracking particles through world

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday March 3, 2021
*/

#include "GGEMS/physics/GGEMSPrimaryParticles.hh"
#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/physics/GGEMSParticleConstants.hh"

/*!
  \fn kernel void world_tracking(GGsize const particle_id_limit, global GGEMSPrimaryParticles* primary_particle)
  \param particle_id_limit - particle id limit
  \param primary_particle - pointer to primary particles on OpenCL memory
  \brief tracking particles through world volume
*/
kernel void world_tracking(
  GGsize const particle_id_limit,
  global GGEMSPrimaryParticles* primary_particle,
  GGsize width,
  GGsize height,
  GGsize depth,
  GGfloat size_x,
  GGfloat size_y,
  GGfloat size_z
)
{
  // Getting index of thread
  GGsize global_id = get_global_id(0);

  // Return if index > to particle limit
  if (global_id >= particle_id_limit) return;

  // In world, the particles in tracked using a DDA algorithm
  // Get direction od particle
  GGfloat3 direction = {primary_particle->dx_[global_id], primary_particle->dy_[global_id], primary_particle->dz_[global_id]};
  // Get point x1, y1 and z1
  GGfloat3 p1 = {primary_particle->px_[global_id], primary_particle->py_[global_id], primary_particle->pz_[global_id]};

  // Computing point x2, y2, z2
  GGfloat distance = primary_particle->particle_solid_distance_[global_id] == OUT_OF_WORLD ? 10000.0f : primary_particle->particle_solid_distance_[global_id];
  GGfloat3 p2 = p1 + distance * direction;

  // Start index
  int i_start_index = (p1.x - size_x*(GGfloat)width*-0.5f)/size_x;
  int j_start_index = (p1.y - size_y*(GGfloat)height*-0.5f)/size_y;
  int k_start_index = (p1.z - size_z*(GGfloat)depth*-0.5f)/size_z;

  // Stop index
  int i_stop_index = (p2.x - size_x*(GGfloat)width*-0.5f)/size_x;
  int j_stop_index = (p2.y - size_y*(GGfloat)height*-0.5f)/size_y;
  int k_stop_index = (p2.z - size_z*(GGfloat)depth*-0.5f)/size_z;

  // Computing difference between p1 and p2 and length for each axis
  GGfloat3 diff_p1_p2 = p2 - p1;
  GGfloat3 len_p1_p2 = fabs(diff_p1_p2);

  // Getting main direction
  GGint length = (GGint)fmax(fmax(len_p1_p2.x, len_p1_p2.y), len_p1_p2.z);
  GGfloat inv_length = 1.0f / (GGfloat)length;

  // Floating points with a maximum value of 2^(32-1-18)=8192
  GGint3 f = convert_int3(p1 * 262144.0f); // 262144 = 2^18
  GGint3 incr_f = convert_int3(diff_p1_p2 * inv_length * 262144.0f);

  printf("p1: %e %e %e mm\n", p1.x/mm, p1.y/mm, p1.z/mm);
  printf("p2: %e %e %e mm\n", p2.x/mm, p2.y/mm, p2.z/mm);
  printf("%d %d %d\n", i_start_index, j_start_index, k_start_index);
  printf("%d %d %d\n", i_stop_index, j_stop_index, k_stop_index);

  // printf("f: %d %d %d\n", f.x, f.y, f.z);
  printf("incr f: %d %d %d\n", incr_f.x, incr_f.y, incr_f.z);
  printf("incr f: %d %d %d\n", incr_f.x >> 18, incr_f.y  >> 18, incr_f.z  >> 18);
  // printf("diff_p1_p2: %e %e %e mm\n", diff_p1_p2.x/mm, diff_p1_p2.y/mm, diff_p1_p2.z/mm);
  // printf("len_p1_p2: %e %e %e mm\n", len_p1_p2.x/mm, len_p1_p2.y/mm, len_p1_p2.z/mm);
  // printf("length: %d\n", length);
  // printf("inv. length: %e\n", inv_length);

  printf("%d %d %d\n", f.x >> 18, f.y >> 18, f.z >> 18);

  f += incr_f;
  f += incr_f;
  f += incr_f;

  printf("%d %d %d\n", f.x >> 18, f.y >> 18, f.z >> 18);
  //  printf("%d\n", f.x >> 18);
  //  f += incr_f;
  //  printf("%d\n", f.x >> 18);
  //  f += incr_f;

  #ifdef GGEMS_TRACKING
  if (global_id == primary_particle->particle_tracking_id) {
    printf("[GGEMS OpenCL kernel world_tracking] ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf("[GGEMS OpenCL kernel world_tracking] World tracking particle\n");
    printf("[GGEMS OpenCL kernel world_tracking] Particle status: %d, DEAD: %d, ALIVE: %d\n", primary_particle->status_[global_id], DEAD, ALIVE);
    printf("[GGEMS OpenCL kernel world_tracking] Particle position: %e %e %e mm\n", primary_particle->px_[global_id]/mm, primary_particle->py_[global_id]/mm, primary_particle->pz_[global_id]/mm);
    printf("[GGEMS OpenCL kernel world_tracking] Particle direction: %e %e %e\n", primary_particle->dx_[global_id], primary_particle->dy_[global_id], primary_particle->dz_[global_id]);
    printf("[GGEMS OpenCL kernel world_tracking] Particle energy: %e keV\n", primary_particle->E_[global_id]/keV);
    printf("[GGEMS OpenCL kernel world_tracking] Distance to next solid: %e mm\n", distance/mm);
  }
  #endif
}

/*
#define pi  3.141592653589
// Floating points with a maximum value of 2^(32-1-18)=8192
#define CONST (int) 262144 //2^18
#define float2fixed(X) ((int) (X * CONST))
#define intfixed(X) (X >> 18)

__device__ float ct_dda_raytracing(float z1, float y1, float x1, float z2, float y2, float x2, float* volume, int depth, int height, int width)
{
    float value = 0.0f;

    float diffx, diffy, diffz;
    unsigned short int lx, ly, lz, length;
    float invlength;
    int fxinc, fyinc, fzinc;
    int fx, fy, fz;
    unsigned int ind;;

    fx = float2fixed(x1);
    fy = float2fixed(y1);
    fz = float2fixed(z1);

    diffx = x2 - x1;
    diffy = y2 - y1;
    diffz = z2 - z1;
    lx = abs(diffx);
    ly = abs(diffy);
    lz = abs(diffz);
    length = (unsigned short int) (max(max(lx, ly), lz));
    invlength = 1.0f / (float) length;
    fxinc = float2fixed(diffx*invlength);
    fyinc = float2fixed(diffy*invlength);
    fzinc = float2fixed(diffz*invlength);

    unsigned short int i = 0;
    while (true)
    {
        if (intfixed(fz) >= 0 && intfixed(fy) >= 0 && intfixed(fx) >= 0
                && intfixed(fz) < depth && intfixed(fy) < height
                && intfixed(fx) < width)
            break;
        else if (i >= length)
            return 0.0f;
        ++i;
        fx += fxinc;
        fy += fyinc;
        fz += fzinc;
    }

    while (true)
    {
        if (intfixed(fz) >= 0 && intfixed(fy) >= 0 && intfixed(fx) >= 0
                && intfixed(fz) < depth && intfixed(fy) < height
                && intfixed(fx) < width && i < length)
        {
            ind = intfixed(fx) + intfixed(fy)*width + intfixed(fz)*width*height;
            value += volume[ind];
        }
        else
            break;

        ++i;
        fx += fxinc;
        fy += fyinc;
        fz += fzinc;
    }

    return value;
}
*/