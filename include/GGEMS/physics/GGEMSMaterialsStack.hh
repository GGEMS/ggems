#ifndef GUARD_GGEMS_PHYSICS_GGEMSMATERIALSSTACK_HH
#define GUARD_GGEMS_PHYSICS_GGEMSMATERIALSSTACK_HH

/*!
  \file GGEMSMaterialsStack.hh

  \brief Structure storing the material tables on OpenCL device

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday March 9, 2020
*/

#include "GGEMS/global/GGEMSConfiguration.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \struct GGEMSMaterialTables_t
  \brief Structure storing the material tables on OpenCL device
*/
#ifdef OPENCL_COMPILER
typedef struct __attribute__((aligned (1))) GGEMSMaterialTables_t
#else
#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
typedef struct PACKED GGEMSMaterialTables_t
#endif
{
  GGuchar number_of_materials_; /*!< Number of the materials */

/*    ui16 *nb_elements;        // n
    ui16 *index;              // n
    ui16 *mixture;            // k
    f32 *atom_num_dens;       // k
    f32 *mass_fraction;       // k
    f32 *nb_atoms_per_vol;                // n
    f32 *nb_electrons_per_vol;            // n
    f32 *electron_mean_excitation_energy; // n
    f32 *rad_length;                      // n
    f32 *photon_energy_cut;               // n
    f32 *electron_energy_cut;             // n
    f32 *fX0;                             // n
    f32 *fX1;
    f32 *fD0;
    f32 *fC;
    f32 *fA;
    f32 *fM;
    f32 *fF1;
    f32 *fF2;
    f32 *fEnergy0;
    f32 *fEnergy1;
    f32 *fEnergy2;
    f32 *fLogEnergy1;
    f32 *fLogEnergy2;
    f32 *fLogMeanExcitationEnergy;
    f32 *density;
    ui32 nb_materials;              // n
    ui32 nb_elements_total;         // k
*/
  //GGfloat E_[MAXIMUM_PARTICLES]; /*!< Energies of particles */
  //GGfloat dx_[MAXIMUM_PARTICLES]; /*!< Position of the particle in x */
  //GGfloat dy_[MAXIMUM_PARTICLES]; /*!< Position of the particle in y */
  //GGfloat dz_[MAXIMUM_PARTICLES]; /*!< Position of the particle in z */
  //GGfloat px_[MAXIMUM_PARTICLES]; /*!< Momentum of the particle in x */
  //GGfloat py_[MAXIMUM_PARTICLES]; /*!< Momentum of the particle in y */
  //GGfloat pz_[MAXIMUM_PARTICLES]; /*!< Momentum of the particle in z */
  //GGfloat tof_[MAXIMUM_PARTICLES]; /*!< Time of flight */

  //GGuint geometry_id_[MAXIMUM_PARTICLES]; /*!< current geometry crossed by the particle */
  //GGushort E_index_[MAXIMUM_PARTICLES]; /*!< Energy index within CS and Mat tables */
  //GGuchar scatter_order_[MAXIMUM_PARTICLES]; /*!< Scatter order, usefull for the imagery */

  //GGfloat next_interaction_distance_[MAXIMUM_PARTICLES]; /*!< Distance to the next interaction */
  //GGuchar next_discrete_process_[MAXIMUM_PARTICLES]; /*!< Next process */

  //GGuchar status_[MAXIMUM_PARTICLES]; /*!< */
  //GGuchar level_[MAXIMUM_PARTICLES]; /*!< */
  //GGuchar pname_[MAXIMUM_PARTICLES]; /*!< particle name (photon, electron, etc) */
} GGEMSMaterialTables;
#ifndef OPENCL_COMPILER
#ifdef _MSC_VER
#pragma pack(pop)
#endif
#endif

#endif // GUARD_GGEMS_PHYSICS_GGEMSMATERIALSSTACK_HH
