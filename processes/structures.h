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

#ifndef STRUCTURES_H
#define STRUCTURES_H



#ifndef PARTICLESTACK
#define PARTICLESTACK
// Stack of particles, format data is defined as SoA
struct ParticleStack{
    // property
    float* E;
    float* dx;
    float* dy;
    float* dz;
    float* px;
    float* py;
    float* pz;
    float* tof;
    // PRNG
    unsigned int* prng_state_1;
    unsigned int* prng_state_2;
    unsigned int* prng_state_3;
    unsigned int* prng_state_4;
    unsigned int* prng_state_5;
    // simulation
    unsigned char* endsimu;
    unsigned char* level;
    unsigned char* pname; // particle name (photon, electron, etc)
    // stack size
    unsigned int size;
}; //
#endif



#ifndef DOSIMETRY
#define DOSIMETRY
/**
 * \struct Dosimetry
 * \brief Dosimetry structure
 *
 * Structure where dosimetry parameters are store during the simulation (edep and edep squared). The size of the dosimetry volume is the same as the voxelised volume
 * \param edep Energy deposited inside the volume
 * \param dose Dose deposited inside the volume
 * \param edep_Squared_value Energy squared deposited inside the volume
 * \param uncertainty Uncertainty associated with the Energy deposited inside the volume
 * \param nb_of_voxels Number of voxels inside the volume, and also the size of the dosimetrics array
 * \param morton_key Morton key of the voxels. Each byte of the integer contains the index of the seed inside the radius of the voxel.
 */
struct Dosimetry
    {

#ifdef FLOAT_DOSI
    float * edep;
    float * dose;
    float * edep_Squared_value;
    float * uncertainty;   
#else
    double * edep;
    double * dose;
    double * edep_Squared_value;
    double * uncertainty;
#endif  
    
    // For double precision using integers
    unsigned int *edep_Trigger;
    unsigned int *edep_Squared_Trigger;
    unsigned int *edep_Int;
    unsigned int *edep_Squared_Int;
    
    unsigned int nb_of_voxels;
    
    // Number of voxels per dimension
    unsigned short int nx;
    unsigned short int ny;
    unsigned short int nz;
    
    // voxels size of the dosemap
    float spacing_x;
    float spacing_y;
    float spacing_z;
    
    // position of the origin of the dosemap
    float x0;
    float y0;
    float z0;

    };
#endif

    
// Some error "checkers"
// comes from "cuda by example" book
static void HandleError( cudaError_t err,
                         const char *file,
                         int line );

// comes from "cuda programming" book
__host__ void cuda_error_check (const char * prefix, const char * postfix);

// Stack device allocation
void _stack_device_malloc(ParticleStack &stackpart, int stack_size);

// Init particle seeds with the main seed
void wrap_init_particle_seeds(ParticleStack &d_p, int seed);

#endif
