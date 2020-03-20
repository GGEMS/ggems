#ifndef GUARD_GGEMS_GLOBAL_GGEMSCONSTANTS_HH
#define GUARD_GGEMS_GLOBAL_GGEMSCONSTANTS_HH

/*!
  \file GGEMSConstants.hh

  \brief Different namespaces storing constants useful for GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Wednesday October 2, 2019
*/

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

/*!
  \namespace GGEMSProcessName
  \brief Namespace storing constants about processes
*/
#ifndef OPENCL_COMPILER
namespace GGEMSProcessName
{
#endif
  __constant GGuchar NUMBER_PROCESSES = 7; /*!< Maximum number of processes */
  __constant GGuchar NUMBER_PHOTON_PROCESSES = 3; /*!< Maximum number of photon processes */
  __constant GGuchar NUMBER_ELECTRON_PROCESSES = 3; /*!< Maximum number of electron processes */
  __constant GGuchar NUMBER_PARTICLES = 3; /*!< Maximum number of different particles */

  __constant GGuchar PHOTON_COMPTON = 0; /*!< Compton process */
  __constant GGuchar PHOTON_PHOTOELECTRIC = 1; /*!< Photoelectric process */
  __constant GGuchar PHOTON_RAYLEIGH = 2; /*!< Rayleigh process */
  __constant GGuchar PHOTON_BONDARY_VOXEL = 3; /*!< Photon on the boundaries */

  __constant GGuchar ELECTRON_IONISATION = 4; /*!< Electron ionisation process */
  __constant GGuchar ELECTRON_MSC = 5; /*!< Electron multiple scattering process */
  __constant GGuchar ELECTRON_BREMSSTRAHLUNG = 6; /*!< Bremsstralung electron process */

  __constant GGuchar NO_PROCESS = 99; /*!< No process */
#ifndef OPENCL_COMPILER
}
#endif

/*!
  \namespace GGEMSParticleName
  \brief Namespace storing particles handling by GGEMS
*/
#ifndef OPENCL_COMPILER
namespace GGEMSParticleName
{
#endif
  __constant GGuchar PHOTON = 0; /*!< Photon particle */
  __constant GGuchar ELECTRON = 1; /*!< Electron particle */
#ifndef OPENCL_COMPILER
}
#endif

/*!
  \namespace GGEMSParticleState
  \brief Namespace storing the state of the particle
*/
#ifndef OPENCL_COMPILER
namespace GGEMSParticleState
{
#endif
  __constant GGuchar PRIMARY = 0; /*!< Primary particle */
  __constant GGuchar GEOMETRY_BOUNDARY = 99; /*!< Particle on the boundary */
  __constant GGuchar ALIVE = 0; /*!< Particle alive */
  __constant GGuchar DEAD = 1; /*!< Particle dead */
  __constant GGuchar FREEZE = 2; /*!< Particle freeze */
#ifndef OPENCL_COMPILER
}
#endif

/*!
  \namespace GGEMSTolerance
  \brief Namespace storing the tolerance for the float computations
*/
#ifndef OPENCL_COMPILER
namespace GGEMSTolerance
{
#endif
  __constant GGfloat EPSILON2 = 1.0e-02f; /*!< Epsilon of 0.01 */
  __constant GGfloat EPSILON3 = 1.0e-03f; /*!< Epsilon of 0.001 */
  __constant GGfloat EPSILON6 = 1.0e-06f; /*!< Epsilon of 0.000001 */
  __constant GGfloat GEOMETRY = 100.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::nm; /*!< Tolerance for the geometry navigation */
  #else
  1.e-6f;
  #endif
#ifndef OPENCL_COMPILER
}
#endif

/*!
  \namespace GGEMSState
  \brief Namespace storing the state of the particle
*/
#ifndef OPENCL_COMPILER
namespace GGEMSState
{
#endif
  __constant GGuchar SOLID = 0; /*!< Solid state */
  __constant GGuchar GAS = 1; /*!< Gas state */
#ifndef OPENCL_COMPILER
}
#endif

/*!
  \namespace GGEMSDefaultParams
  \brief Namespace storing the default parameters
*/
#ifndef OPENCL_COMPILER
namespace GGEMSDefaultParams
{
#endif
  __constant GGfloat KINETIC_ENERGY_MIN = 1.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::eV; /*!< Min kinetic energy */
  #else
  1.e-6f;
  #endif

  __constant GGushort CROSS_SECTION_TABLE_NUMBER_BINS = 220; /*!< Number of bins in the cross section table */
  __constant GGfloat CROSS_SECTION_TABLE_ENERGY_MIN = 990.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::eV; /*!< Min energy in the cross section table */
  #else
  1.e-6f;
  #endif

  __constant GGfloat CROSS_SECTION_TABLE_ENERGY_MAX = 250.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::MeV; /*!< Max energy in the cross section table */
  #else
  1.f;
  #endif

  __constant GGfloat PHOTON_CUT = 1.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::um; /*!< Photon cut */
  #else
  1.e-3f;
  #endif

  __constant GGfloat ELECTRON_CUT = 1.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::um; /*!< Electron cut */
  #else
  1.e-3f;
  #endif
#ifndef OPENCL_COMPILER
}
#endif

/*!
  \namespace GGEMSMathematicalConstant
  \brief namespace storing mathematical constants
*/
#ifndef OPENCL_COMPILER
namespace GGEMSMathematicalConstant
{
#endif

  // PI variables
  __constant GGfloat PI         = 3.141592653589793f; /*!< Pi */
  __constant GGfloat TWO_PI     = 6.283185307179586f; /*!< Pi * 2 */
  __constant GGfloat HALF_PI    = 1.570796326794896f; /*!< Pi / 2 */
  __constant GGfloat PI_SQUARED = 9.869604401089358f; /*!< Pi * Pi */

#ifndef OPENCL_COMPILER
}
#endif

/*!
  \namespace GGEMSPhysicalConstant
  \brief namespace storing all physical constants
*/
#ifndef OPENCL_COMPILER
namespace GGEMSPhysicalConstant
{
#endif
  __constant GGfloat AVOGADRO = 6.02214179e+23f/
  #ifndef OPENCL_COMPILER
  GGEMSUnits::mol; /*!< Number of Avogadro */
  #else
  1.0f;
  #endif

  __constant GGfloat GASTHRESHOLD = 10.f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::mg/GGEMSUnits::cm3; /*!< Limit between gas and other state in density */
  #else
  6.241510246e+15f;
  #endif

  __constant GGfloat C_LIGHT = 2.99792458e+8f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::m/GGEMSUnits::s; /*!< Speed of light */
  #else
  1.e-6f;
  #endif
  __constant GGfloat C_LIGHT_SQUARED = 89875.5178736817f; /*!< squared speed of ligth in mm.ns-1 */

  __constant GGfloat ELECTRON_CHARGE = 1.0f*
  #ifndef OPENCL_COMPILER
  -GGEMSUnits::eplus; /*!< Charge of the electron */
  #else
  -1.0f;
  #endif
  __constant GGfloat ELECTRON_CHARGE_SQUARED = 1.0f; /*!< Squared charge of electron */

  __constant GGfloat ELECTRON_MASS_C2 = 0.510998910f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::MeV; /*!< Mass of the electron in MeV */
  #else
  1.0f;
  #endif

  __constant GGfloat PROTON_MASS_C2 = 938.272013f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::MeV; /*!< Mass of the proton in MeV */
  #else
  1.0f;
  #endif

  __constant GGfloat NEUTRON_MASS_C2 = 939.56536f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::MeV; /*!< Mass of the neutron in MeV */
  #else
  1.0f;
  #endif

  __constant GGfloat ATOMIC_MASS_UNIT_C2 = 931.494028f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::MeV; /*!< Unified atomic mass unit in Mev */
  #else
  1.0f;
  #endif
  __constant GGfloat ATOMIC_MASS_UNIT = 0.01036426891f; /*!<Atomic mass unit c2 / c2 in MeV/(mm.ns-1)*/

  __constant GGfloat MU0      = 4.0f*3.141592653589793f*1.e-7f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::H / GGEMSUnits::m; /*!< permeability of free space in MeV.(ns.eplus)^2.mm-1 */
  #else
  1.602176383e-10f;
  #endif
  __constant GGfloat EPSILON0 = 5.526349824e+10f; /*!< permittivity of free space in eplus^2/(Mev.mm) */

  // hbar  = 6.58212e-13 MeV*ns
  // hbarc = 197.32705e-12 MeV*mm
  __constant GGfloat H_PLANCK = 6.62606896e-34f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::J * GGEMSUnits::s; /*!< Planck constant in Mev.ns */
  #else
  6.241509704e+12f;
  #endif

  __constant GGfloat HBAR_PLANCK   = 6.582118206e-13f; /*!< h_planck / (2*PI) */
  __constant GGfloat HBARC         = 1.973269187e-10f; /*!< speed of light * h_planck / (2*PI) */
  __constant GGfloat HBARC_SQUARED = 3.89351824e-20F; /*!< hbar * hbar */

  // electromagnetic coupling = 1.43996e-12 MeV*mm/(eplus^2)
  __constant GGfloat ELM_COUPLING            = 1.439964467e-12f; /*!< Electromagnetic coupling in MeV.mm/(eplus^2), ELECTRON_CHARGE^2/(4*PI*EPSILON0) */
  __constant GGfloat FINE_STRUCTURE_CONST    = 0.007297354285f; /*!< Structure fine constant, ELM_COUPLING/HBARC*/
  __constant GGfloat CLASSIC_ELECTRON_RADIUS = 2.817940326e-12f; /*!< Classical radius of electron in mm, ELM_COUPLING/ELECTRON_MASS_C2 */
  __constant GGfloat ELECTRON_COMPTON_LENGTH = 3.86159188e-10f; /*!< Length of electron Compton in mm, HBARC/ELECTRON_MASS_C2 */
  __constant GGfloat BOHR_RADIUS             = 5.291769867e-08f; /*!< Radius of Bohr in mm, ELECTRON_COMPTON_LENGTH/FINE_STRUCTURE_CONST */
  __constant GGfloat ALPHA_RCL2              = 5.794673922e-26f; /*!< Constant FINE_STRUCTURE_CONST * CLASSIC_ELECTRON_RADIUS^2 */

#ifndef OPENCL_COMPILER
}
#endif

#endif // End of GUARD_GGEMS_GLOBAL_GGEMSCONSTANTS_HH
