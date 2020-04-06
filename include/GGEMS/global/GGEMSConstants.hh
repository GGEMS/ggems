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
  1.0f; /*!< Number of Avogadro */
  #endif

  __constant GGfloat GASTHRESHOLD = 10.f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::mg/GGEMSUnits::cm3; /*!< Limit between gas and other state in density */
  #else
  6.241510246e+15f; /*!< Limit between gas and other state in density */
  #endif

  __constant GGfloat C_LIGHT = 2.99792458e+8f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::m/GGEMSUnits::s; /*!< Speed of light */
  #else
  1.e-6f; /*!< Speed of light */
  #endif
  __constant GGfloat C_LIGHT_SQUARED = 89875.5178736817f; /*!< squared speed of ligth in mm.ns-1 */

  __constant GGfloat ELECTRON_CHARGE = 1.0f*
  #ifndef OPENCL_COMPILER
  -GGEMSUnits::eplus; /*!< Charge of the electron */
  #else
  -1.0f; /*!< Charge of the electron */
  #endif
  __constant GGfloat ELECTRON_CHARGE_SQUARED = 1.0f; /*!< Squared charge of electron */

  __constant GGfloat ELECTRON_MASS_C2 = 0.510998910f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::MeV; /*!< Mass of the electron in MeV */
  #else
  1.0f; /*!< Mass of the electron in MeV */
  #endif

  __constant GGfloat POSITRON_MASS_C2 = 0.510998910f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::MeV; /*!< Mass of the electron in MeV */
  #else
  1.0f; /*!< Mass of the electron in MeV */
  #endif

  __constant GGfloat PROTON_MASS_C2 = 938.272013f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::MeV; /*!< Mass of the proton in MeV */
  #else
  1.0f; /*!< Mass of the proton in MeV */
  #endif

  __constant GGfloat NEUTRON_MASS_C2 = 939.56536f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::MeV; /*!< Mass of the neutron in MeV */
  #else
  1.0f; /*!< Mass of the neutron in MeV */
  #endif

  __constant GGfloat ATOMIC_MASS_UNIT_C2 = 931.494028f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::MeV; /*!< Unified atomic mass unit in Mev */
  #else
  1.0f; /*!< Unified atomic mass unit in Mev */
  #endif
  __constant GGfloat ATOMIC_MASS_UNIT = 0.01036426891f; /*!<Atomic mass unit c2 / c2 in MeV/(mm.ns-1)*/

  __constant GGfloat MU0      = 4.0f*3.141592653589793f*1.e-7f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::H / GGEMSUnits::m; /*!< permeability of free space in MeV.(ns.eplus)^2.mm-1 */
  #else
  1.602176383e-10f; /*!< permeability of free space in MeV.(ns.eplus)^2.mm-1 */
  #endif
  __constant GGfloat EPSILON0 = 5.526349824e+10f; /*!< permittivity of free space in eplus^2/(Mev.mm) */

  // hbar  = 6.58212e-13 MeV*ns
  // hbarc = 197.32705e-12 MeV*mm
  __constant GGfloat H_PLANCK = 6.62606896e-34f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::J * GGEMSUnits::s; /*!< Planck constant in Mev.ns */
  #else
  6.241509704e+12f; /*!< Planck constant in Mev.ns */
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
  __constant GGfloat TWO_PI_MC2_RCL2         = 2.549549536e-23f; /*!< 2pi*electron_mc2*CLASSIC_ELECTRON_RADIUS^2 */
#ifndef OPENCL_COMPILER
}
#endif

#endif // End of GUARD_GGEMS_GLOBAL_GGEMSCONSTANTS_HH
