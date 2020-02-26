#ifndef GUARD_GGEMS_TOOLS_GGEMSSYSTEMOFUNITS_HH
#define GUARD_GGEMS_TOOLS_GGEMSSYSTEMOFUNITS_HH

/*!
  \file GGEMSSystemOfUnits.hh

  \brief Namespace storing all the usefull physical units

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Tuesday October 1, 2019
*/

#include "GGEMS/tools/GGEMSTypes.hh"

#ifdef PASCAL
#undef PASCAL
#endif

#ifndef OPENCL_COMPILER
#define __constant inline static constexpr
#include <algorithm>
#include <sstream>
#include "GGEMS/tools/GGEMSTools.hh"
#endif

/*!
  \namespace GGEMSUnits
  \brief namespace storing all the usefull physical units
*/
#ifndef OPENCL_COMPILER
namespace GGEMSUnits
{
#endif
  // Pi definitions
  __constant GGdouble PI = 3.141592653589793238463;
  __constant GGdouble PI_TWICE = 2.0 * 3.141592653589793238463;
  __constant GGdouble PI_HALF = 3.141592653589793238463 / 2.0;
  __constant GGdouble PI_PI = 3.141592653589793238463 * 3.141592653589793238463;

  // Lengths
  __constant GGdouble MILLIMETER = 1.0;
  __constant GGdouble MILLIMETER2 = 1.0*1.0;
  __constant GGdouble MILLIMETER3 = 1.0*1.0*1.0;

  __constant GGdouble CENTIMETER = 10.*1.0;
  __constant GGdouble CENTIMETER2 = 10.*1.0*10.*1.0;
  __constant GGdouble CENTIMETER3 = 10.*1.0*10.*1.0*10.*1.0;

  __constant GGdouble METER = 1000.*1.0;
  __constant GGdouble METER2 = 1000.*1.0*1000.*1.0;
  __constant GGdouble METER3 = 1000.*1.0*1000.*1.0*1000.*1.0;

  __constant GGdouble KILOMETER = 1000.*1000.*1.0;
  __constant GGdouble KILOMETER2 = 1000.*1000.*1.0*1000.*1000.*1.0;
  __constant GGdouble KILOMETER3 = 1000.*1000.*1.0*1000.*1000.*1.0*1000.*1000.*1.0;

  __constant GGdouble PARSEC = 96939420213600000*1000.*1.0/3.141592653589793238463;

  __constant GGdouble MICROMETER = 1.e-6 *1000.*1.0;
  __constant GGdouble NANOMETER = 1.e-9 *1000.*1.0;
  __constant GGdouble ANGSTROM = 1.e-10*1000.*1.0;
  __constant GGdouble FERMI = 1.e-15*1000.*1.0;

  __constant GGdouble BARN = 1.e-28*1000.*1.0;
  __constant GGdouble MILLIBARN = 1.e-3 *1.e-28*1000.*1.0;
  __constant GGdouble MICROBARN = 1.e-6 *1.e-28*1000.*1.0;
  __constant GGdouble NANOBARN = 1.e-9 *1.e-28*1000.*1.0;
  __constant GGdouble PICOBARN = 1.e-12*1.e-28*1000.*1.0;

  // Symbol definitions
  __constant GGdouble nm = 1.e-9 *1000.*1.0;
  __constant GGdouble um = 1.e-6 *1000.*1.0;

  __constant GGdouble mm = 1.0;
  __constant GGdouble mm2 = 1.0*1.0;
  __constant GGdouble mm3 = 1.0*1.0*1.0;

  __constant GGdouble cm  = 10.*1.0;
  __constant GGdouble cm2 = 10.*1.0*10.*1.0;
  __constant GGdouble cm3 = 10.*1.0*10.*1.0*10.*1.0;

  __constant GGdouble m = 1000.*1.0;
  __constant GGdouble m2 = 1000.*1.0*1000.*1.0;
  __constant GGdouble m3 = 1000.*1.0*1000.*1.0*1000.*1.0;

  __constant GGdouble km = 1000.*1000.*1.0;
  __constant GGdouble km2 = 1000.*1000.*1.0*1000.*1000.*1.0;
  __constant GGdouble km3 = 1000.*1000.*1.0*1000.*1000.*1.0*1000.*1000.*1.0;

  __constant GGdouble pc = 96939420213600000*1000.*1.0/3.141592653589793238463;

  // Angles
  __constant GGdouble RADIAN = 1.0;
  __constant GGdouble MILLIRADIAN = 1.e-3*1.0;
  __constant GGdouble DEGREE = (3.141592653589793238463/180.0)*1.0;

  __constant GGdouble STERADIAN = 1.0;

  // Symbols definitions
  __constant GGdouble rad  = 1.0;
  __constant GGdouble mrad = 1.e-3*1.0;
  __constant GGdouble sr = 1.0;
  __constant GGdouble deg  = (3.141592653589793238463/180.0)*1.0;

  // Time
  __constant GGdouble NANOSECOND = 1.;
  __constant GGdouble SECOND = 1.e+9 *1.;
  __constant GGdouble MILLISECOND = 1.e-3 *1.e+9 *1.;
  __constant GGdouble MICROSECOND = 1.e-6 *1.e+9 *1.;
  __constant GGdouble PICOSECOND = 1.e-12*1.e+9 *1.;

  __constant GGdouble HERTZ = 1./(1.e+9 *1.);
  __constant GGdouble KILOHERTZ = 1.e+3*(1./(1.e+9 *1.));
  __constant GGdouble MEGAHERTZ = 1.e+6*(1./(1.e+9 *1.));

  // Symbols definitions
  __constant GGdouble ns = 1.;
  __constant GGdouble s = 1.e+9 *1.;
  __constant GGdouble ms = 1.e-3 *1.e+9 *1.;

  // Electric charge [Q]
  __constant GGdouble EPLUS = 1. ;// positron charge
  __constant GGdouble ESI = 1.602176487e-19;// positron charge in coulomb
  __constant GGdouble COULOMB = 1./1.602176487e-19;// coulomb = 6.24150 e+18 * eplus

  // Energy [E]
  __constant GGdouble MEGAELECTRONVOLT = 1. ;
  __constant GGdouble ELECTRONVOLT = 1.e-6*1.;
  __constant GGdouble KILOELECTRONVOLT = 1.e-3*1.;
  __constant GGdouble GIGAELECTRONVOLT = 1.e+3*1.;
  __constant GGdouble TERAELECTRONVOLT = 1.e+6*1.;
  __constant GGdouble PETAELECTRONVOLT = 1.e+9*1.;

  __constant GGdouble JOULE = 1.e-6*1./1.602176487e-19;// joule = 6.24150 e+12 * MeV

  // Symbols definitions
  __constant GGdouble MeV = 1.;
  __constant GGdouble eV = 1.e-6*1.;
  __constant GGdouble keV = 1.e-3*1.;
  __constant GGdouble GeV = 1.e+3*1.;
  __constant GGdouble TeV = 1.e+6*1.;
  __constant GGdouble PeV = 1.e+9*1.;

  // Mass [E][T^2][L^-2]
  __constant GGdouble KILOGRAM = 1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0);
  __constant GGdouble GRAM = 1.e-3*(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9*1./(1000.*1.0*1000.*1.0));
  __constant GGdouble MILLIGRAM = 1.e-3*1.e-3*(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0));

  // Symbols definitions
  __constant GGdouble kg = 1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0);
  __constant GGdouble g = 1.e-3*(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0));
  __constant GGdouble mg = 1.e-3*1.e-3*(1.e-6*1./1.602176487e-19*1.e+9*1.*1.e+9 *1./(1000.*1.0*1000.*1.0));

  // Power [E][T^-1]
  __constant GGdouble WATT = 1.e-6*1./1.602176487e-19/(1.e+9 *1.);// watt = 6.24150 e+3 * MeV/ns

  // Force [E][L^-1]
  __constant GGdouble NEWTON = 1.e-6*1./1.602176487e-19/(1000.*1.0);// newton = 6.24150 e+9 * MeV/mm

  // Pressure [E][L^-3]
  __constant GGdouble PASCAL = (1.e-6*1./1.602176487e-19/(1000.*1.0))/(1000.*1.0*1000.*1.0);   // pascal = 6.24150 e+3 * MeV/mm3
  __constant GGdouble BAR = 100000*((1.e-6*1./1.602176487e-19/(1000.*1.0))/(1000.*1.0*1000.*1.0)); // bar    = 6.24150 e+8 * MeV/mm3
  __constant GGdouble ATMOSPHERE = 101325*((1.e-6*1./1.602176487e-19/(1000.*1.0))/(1000.*1.0*1000.*1.0)); // atm    = 6.32420 e+8 * MeV/mm3

  // Electric current [Q][T^-1]
  __constant GGdouble AMPERE = (1./1.602176487e-19)/(1.e+9 *1.); // ampere = 6.24150 e+9 * eplus/ns
  __constant GGdouble MILLIAMPERE = 1.e-3*((1./1.602176487e-19)/(1.e+9 *1.));
  __constant GGdouble MICROAMPERE = 1.e-6*((1./1.602176487e-19)/(1.e+9 *1.));
  __constant GGdouble NANOAMPERE = 1.e-9*((1./1.602176487e-19)/(1.e+9 *1.));

  // Electric potential [E][Q^-1]
  __constant GGdouble MEGAVOLT = (1.)/(1.);
  __constant GGdouble KILOVOLT = 1.e-3*((1.)/(1.));
  __constant GGdouble VOLT = 1.e-6*((1.)/(1.));

  // Electric resistance [E][T][Q^-2]
  __constant GGdouble OHM = (1.e-6*((1.)/(1.)))/((1./1.602176487e-19)/(1.e+9 *1.));// ohm = 1.60217e-16*(MeV/eplus)/(eplus/ns)

  // Electric capacitance [Q^2][E^-1]
  __constant GGdouble FARAD = (1./1.602176487e-19)/(1.e-6*((1.)/(1.)));// farad = 6.24150e+24 * eplus/Megavolt
  __constant GGdouble MILLIFARAD = 1.e-3*((1./1.602176487e-19)/(1.e-6*((1.)/(1.))));
  __constant GGdouble MICROFARAD = 1.e-6*((1./1.602176487e-19)/(1.e-6*((1.)/(1.))));
  __constant GGdouble NANOFARAD = 1.e-9*((1./1.602176487e-19)/(1.e-6*((1.)/(1.))));
  __constant GGdouble PICOFARAD = 1.e-12*((1./1.602176487e-19)/(1.e-6*((1.)/(1.))));

  // Magnetic Flux [T][E][Q^-1]
  __constant GGdouble WEBER = (1.e-6*((1.)/(1.)))*(1.e+9 *1.);// weber = 1000*megavolt*ns

  // Magnetic Field [T][E][Q^-1][L^-2]
  __constant GGdouble TESLA = (1.e-6*((1.)/(1.)))*(1.e+9 *1.)/(1000.*1.0*1000.*1.0);// tesla =0.001*megavolt*ns/mm2

  __constant GGdouble GAUSS = 1.e-4*((1.e-6*((1.)/(1.)))*(1.e+9 *1.)/(1000.*1.0*1000.*1.0));
  __constant GGdouble KILOGAUSS = 1.e-1*((1.e-6*((1.)/(1.)))*(1.e+9 *1.)/(1000.*1.0*1000.*1.0));

  // Inductance [T^2][E][Q^-2]
  __constant GGdouble HENRY = ((1.e-6*((1.)/(1.)))*(1.e+9 *1.))/((1./1.602176487e-19)/(1.e+9 *1.));// henry = 1.60217e-7*MeV*(ns/eplus)**2

  // Temperature
  __constant GGdouble KELVIN = 1.;

  // Amount of substance
  __constant GGdouble MOLE = 1.;
  __constant GGdouble mol = 1.;

  // Activity [T^-1]
  __constant GGdouble BECQUEREL = 1./(1.e+9 *1.) ;
  __constant GGdouble CURIE = 3.7e+10 * (1./(1.e+9 *1.));
  __constant GGdouble KILOBECQUEREL = 1.e+3*(1./(1.e+9 *1.));
  __constant GGdouble MEGABECQUEREL = 1.e+6*(1./(1.e+9 *1.));
  __constant GGdouble GIGABECQUEREL = 1.e+9*(1./(1.e+9 *1.));
  __constant GGdouble MILLICURIE = 1.e-3*(3.7e+10 * (1./(1.e+9 *1.)));
  __constant GGdouble MICROCURIE = 1.e-6*(3.7e+10 * (1./(1.e+9 *1.)));

  // Symbols definitions
  __constant GGdouble Bq = (1./(1.e+9 *1.));
  __constant GGdouble kBq = (1.e+3*(1./(1.e+9 *1.)));
  __constant GGdouble MBq = (1.e+6*(1./(1.e+9 *1.)));
  __constant GGdouble GBq = (1.e+9*(1./(1.e+9 *1.)));
  __constant GGdouble Ci = (3.7e+10 * (1./(1.e+9 *1.)));
  __constant GGdouble mCi = (1.e-3*(3.7e+10 * (1./(1.e+9 *1.))));
  __constant GGdouble uCi = (1.e-6*(3.7e+10 * (1./(1.e+9 *1.))));

  // Absorbed dose [L^2][T^-2]
  __constant GGdouble GRAY = (1.e-6*1./1.602176487e-19)/(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0));
  __constant GGdouble KILOGRAY = 1.e+3*((1.e-6*1./1.602176487e-19)/(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0)));
  __constant GGdouble MILLIGRAY = 1.e-3*((1.e-6*1./1.602176487e-19)/(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0)));
  __constant GGdouble MICROGRAY = 1.e-6*((1.e-6*1./1.602176487e-19)/(1.e-6*1./1.602176487e-19*1.e+9*1.*1.e+9 *1./(1000.*1.0*1000.*1.0)));

  // Luminous intensity [I]
  __constant GGdouble CANDELA = 1.;

  // Luminous flux [I]
  __constant GGdouble LUMEN = (1.)*(1.0);

  // Illuminance [I][L^-2]
  __constant GGdouble LUX = ((1.)*(1.0))/(1000.*1.0*1000.*1.0);

  // Miscellaneous
  __constant GGdouble PERCENT = 0.01 ;
  __constant GGdouble PERTHOUSAND = 0.001;
  __constant GGdouble PERMILLION  = 0.000001;

  #ifndef OPENCL_COMPILER
  /*!
    \fn T BestDistanceUnit(T const& value, char const* unit)
    \tparam T - type of the value to convert unit
    \param value - value to check
    \param unit - distance unit
    \brief Choose best distance unit
    \return value in the good unit
  */
  template <typename T>
  T BestDistanceUnit(T const& value, char const* unit)
  {
    // Convert char* to string
    std::string unit_str = unit;

    T new_value = static_cast<T>(0);
    if (unit_str == "nm") {
      new_value = static_cast<T>(value * GGEMSUnits::nm);
    }
    else if (unit_str == "um") {
      new_value = static_cast<T>(value * GGEMSUnits::um);
    }
    else if (unit_str == "mm") {
      new_value = static_cast<T>(value * GGEMSUnits::mm);
    }
    else if (unit_str == "cm") {
      new_value = static_cast<T>(value * GGEMSUnits::cm);
    }
    else if (unit_str == "m") {
      new_value = static_cast<T>(value * GGEMSUnits::m);
    }
    else if (unit_str == "km") {
      new_value = static_cast<T>(value * GGEMSUnits::km);
    }
    else if (unit_str == "pc") {
      new_value = static_cast<T>(value * GGEMSUnits::pc);
    }
    else {
      std::ostringstream oss(std::ostringstream::out);
      oss << "Unknown unit!!! You have choice between:" << std::endl;
      oss << "    - \"nm\": nanometer" << std::endl;
      oss << "    - \"um\": micrometer" << std::endl;
      oss << "    - \"mm\": millimeter" << std::endl;
      oss << "    - \"cm\": centimeter" << std::endl;
      oss << "    - \"m\": meter" << std::endl;
      oss << "    - \"km\": kilometer" << std::endl;
      oss << "    - \"pc\": parsec";
      GGEMSMisc::ThrowException("GGEMSUnits", "BestDistanceUnit", oss.str());
    }
    return new_value;
  }

  /*!
    \fn T BestEnergyUnit(T const& value, char const* unit)
    \tparam T - type of the value to convert unit
    \param value - value to check
    \param unit - energy unit
    \brief Choose best energy unit
    \return value in the good unit
  */
  template <typename T>
  T BestEnergyUnit(T const& value, char const* unit)
  {
    // Convert char* to string
    std::string unit_str = unit;

    T new_value = static_cast<T>(0);
    if (unit_str == "ev") {
      new_value = static_cast<T>(value * GGEMSUnits::eV);
    }
    else if (unit_str == "keV") {
      new_value = static_cast<T>(value * GGEMSUnits::keV);
    }
    else if (unit_str == "MeV") {
      new_value = static_cast<T>(value * GGEMSUnits::MeV);
    }
    else if (unit_str == "GeV") {
      new_value = static_cast<T>(value * GGEMSUnits::GeV);
    }
    else if (unit_str == "TeV") {
      new_value = static_cast<T>(value * GGEMSUnits::TeV);
    }
    else if (unit_str == "PeV") {
      new_value = static_cast<T>(value * GGEMSUnits::PeV);
    }
    else {
      std::ostringstream oss(std::ostringstream::out);
      oss << "Unknown unit!!! You have choice between:" << std::endl;
      oss << "    - \"eV\": electronvolt" << std::endl;
      oss << "    - \"keV\": kiloelectronvolt" << std::endl;
      oss << "    - \"MeV\": megaelectronvolt" << std::endl;
      oss << "    - \"GeV\": gigaelectronvolt" << std::endl;
      oss << "    - \"TeV\": teraelectronvolt" << std::endl;
      oss << "    - \"PeV\": petaelectronvolt" << std::endl;
      GGEMSMisc::ThrowException("GGEMSUnits", "BestEnergyUnit", oss.str());
    }
    return new_value;
  }

  /*!
    \fn T BestAngleUnit(T const& value, char const* unit)
    \tparam T - type of the value to convert unit
    \param value - value to check
    \param unit - angle unit
    \brief Choose best angle unit
    \return value in the good unit
  */
  template <typename T>
  T BestAngleUnit(T const& value, char const* unit)
  {
    // Convert char* to string
    std::string unit_str = unit;

    T new_value = static_cast<T>(0);
    if (unit_str == "rad") {
      new_value = static_cast<T>(value * GGEMSUnits::rad);
    }
    else if (unit_str == "mrad") {
      new_value = static_cast<T>(value * GGEMSUnits::mrad);
    }
    else if (unit_str == "deg") {
      new_value = static_cast<T>(value * GGEMSUnits::deg);
    }
    else if (unit_str == "sr") {
      new_value = static_cast<T>(value * GGEMSUnits::sr);
    }
    else {
      std::ostringstream oss(std::ostringstream::out);
      oss << "Unknown unit!!! You have choice between:" << std::endl;
      oss << "    - \"rad\": radian" << std::endl;
      oss << "    - \"mrad\": milliradian" << std::endl;
      oss << "    - \"deg\": degree" << std::endl;
      oss << "    - \"sr\": steradian" << std::endl;
      GGEMSMisc::ThrowException("GGEMSUnits", "BestAngleUnit", oss.str());
    }
    return new_value;
  }
  #endif
#ifndef OPENCL_COMPILER
}
#endif

#endif // End of GUARD_GGEMS_TOOLS_GGEMSSYSTEMOFUNITS_HH
