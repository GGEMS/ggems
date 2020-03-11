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

  // Lengths
  __constant GGfloat MILLIMETER = 1.0f;
  __constant GGfloat MILLIMETER2 = 1.0f*1.0f;
  __constant GGfloat MILLIMETER3 = 1.0f*1.0f*1.0f;

  __constant GGfloat CENTIMETER = 10.f*1.0f;
  __constant GGfloat CENTIMETER2 = 10.f*1.0f*10.f*1.0f;
  __constant GGfloat CENTIMETER3 = 10.f*1.0f*10.f*1.0*10.*1.0f;

  __constant GGfloat METER = 1000.f*1.0f;
  __constant GGfloat METER2 = 1000.f*1.0f*1000.f*1.0f;
  __constant GGfloat METER3 = 1000.f*1.0f*1000.f*1.0f*1000.*1.0f;

  __constant GGfloat KILOMETER = 1000.f*1000.f*1.0f;
  __constant GGfloat KILOMETER2 = 1000.f*1000.f*1.0f*1000.f*1000.f*1.0f;
  __constant GGfloat KILOMETER3 = 1000.f*1000.f*1.0f*1000.f*1000.f*1.0f*1000.f*1000.f*1.0f;

  __constant GGfloat PARSEC = 96939420213600000.0f*1000.f*1.0f/3.141592653589793238463f;

  __constant GGfloat MICROMETER = 1.e-6f *1000.f*1.0f;
  __constant GGfloat NANOMETER = 1.e-9f *1000.f*1.0f;
  __constant GGfloat ANGSTROM = 1.e-10f*1000.f*1.0f;
  __constant GGfloat FERMI = 1.e-15f*1000.f*1.0f;

  __constant GGfloat BARN = 1.e-28f*1000.f*1.0f;
  __constant GGfloat MILLIBARN = 1.e-3f *1.e-28f*1000.f*1.0f;
  __constant GGfloat MICROBARN = 1.e-6f *1.e-28f*1000.f*1.0f;
  __constant GGfloat NANOBARN = 1.e-9f *1.e-28f*1000.f*1.0f;
  __constant GGfloat PICOBARN = 1.e-12f*1.e-28f*1000.f*1.0f;

  // Symbol definitions
  __constant GGfloat nm = 1.e-9f *1000.f*1.0f;
  __constant GGfloat um = 1.e-6f *1000.f*1.0f;

  __constant GGfloat mm = 1.0f;
  __constant GGfloat mm2 = 1.0f*1.0f;
  __constant GGfloat mm3 = 1.0f*1.0f*1.0f;

  __constant GGfloat cm  = 10.f*1.0f;
  __constant GGfloat cm2 = 10.f*1.0f*10.f*1.0f;
  __constant GGfloat cm3 = 10.f*1.0f*10.f*1.0f*10.f*1.0f;

  __constant GGfloat m = 1000.*1.0f;
  __constant GGfloat m2 = 1000.*1.0f*1000.f*1.0f;
  __constant GGfloat m3 = 1000.f*1.0f*1000.f*1.0f*1000.f*1.0f;

  __constant GGfloat km = 1000.f*1000.f*1.0f;
  __constant GGfloat km2 = 1000.f*1000.f*1.0f*1000.f*1000.f*1.0f;
  __constant GGfloat km3 = 1000.f*1000.f*1.0f*1000.f*1000.f*1.0f*1000.f*1000.f*1.0f;

  __constant GGfloat pc = 96939420213600000.0f*1000.f*1.0f/3.141592653589793238463f;

  // Angles
  __constant GGfloat RADIAN = 1.0f;
  __constant GGfloat MILLIRADIAN = 1.e-3f*1.0f;
  __constant GGfloat DEGREE = (3.141592653589793238463f/180.0f)*1.0f;

  __constant GGfloat STERADIAN = 1.0f;

  // Symbols definitions
  __constant GGfloat rad  = 1.0f;
  __constant GGfloat mrad = 1.e-3f*1.0f;
  __constant GGfloat sr = 1.0f;
  __constant GGfloat deg  = (3.141592653589793238463f/180.0f)*1.0f;

  // Time
  __constant GGfloat NANOSECOND = 1.f;
  __constant GGfloat SECOND = 1.e+9f *1.f;
  __constant GGfloat MILLISECOND = 1.e-3f *1.e+9f *1.f;
  __constant GGfloat MICROSECOND = 1.e-6f *1.e+9f *1.f;
  __constant GGfloat PICOSECOND = 1.e-12f*1.e+9f *1.f;

  __constant GGfloat HERTZ = 1.f/(1.e+9f *1.f);
  __constant GGfloat KILOHERTZ = 1.e+3f*(1.f/(1.e+9f *1.f));
  __constant GGfloat MEGAHERTZ = 1.e+6f*(1.f/(1.e+9f *1.f));

  // Symbols definitions
  __constant GGfloat ns = 1.f;
  __constant GGfloat s = 1.e+9f *1.f;
  __constant GGfloat ms = 1.e-3f *1.e+9f *1.f;

  // Electric charge [Q]
  __constant GGfloat eplus = 1.f ;// positron charge
  __constant GGfloat ESI = 1.602176487e-19f;// positron charge in coulomb
  __constant GGfloat COULOMB = 1.f/1.602176487e-19f;// coulomb = 6.24150 e+18 * eplus

  // Energy [E]
  __constant GGfloat MEGAELECTRONVOLT = 1.f;
  __constant GGfloat ELECTRONVOLT = 1.e-6f*1.f;
  __constant GGfloat KILOELECTRONVOLT = 1.e-3f*1.f;
  __constant GGfloat GIGAELECTRONVOLT = 1.e+3f*1.f;
  __constant GGfloat TERAELECTRONVOLT = 1.e+6f*1.f;
  __constant GGfloat PETAELECTRONVOLT = 1.e+9f*1.f;

  __constant GGfloat joule = 1.e-6f*1.f/1.602176487e-19f;// joule = 6.24150 e+12 * MeV

  // Symbols definitions
  __constant GGfloat MeV = 1.f;
  __constant GGfloat eV = 1.e-6f*1.f;
  __constant GGfloat keV = 1.e-3f*1.f;
  __constant GGfloat GeV = 1.e+3f*1.f;
  __constant GGfloat TeV = 1.e+6f*1.f;
  __constant GGfloat PeV = 1.e+9f*1.f;

  // Mass [E][T^2][L^-2]
  __constant GGfloat KILOGRAM = 1.e-6f*1.f/1.602176487e-19f*1.e+9f*1.f*1.e+9f*1.f/(1000.f*1.0f*1000.f*1.0f);
  __constant GGfloat GRAM = 1.e-3f*(1.e-6f*1.f/1.602176487e-19f*1.e+9f *1.f*1.e+9f*1.f/(1000.f*1.0f*1000.f*1.0f));
  __constant GGfloat MILLIGRAM = 1.e-3f*1.e-3f*(1.e-6f*1.f/1.602176487e-19f*1.e+9f *1.f*1.e+9f *1.f/(1000.f*1.0f*1000.f*1.0f));

  // Symbols definitions
  __constant GGfloat kg = 1.e-6f*1.f/1.602176487e-19f*1.e+9f *1.f*1.e+9f *1.f/(1000.f*1.0f*1000.f*1.0f);
  __constant GGfloat g = 1.e-3f*(1.e-6f*1.f/1.602176487e-19f*1.e+9f *1.f*1.e+9f *1.f/(1000.f*1.0f*1000.f*1.0f));
  __constant GGfloat mg = 1.e-3f*1.e-3f*(1.e-6f*1.f/1.602176487e-19f*1.e+9f*1.f*1.e+9f *1.f/(1000.f*1.0f*1000.f*1.0f));

  // Power [E][T^-1]
  __constant GGfloat WATT = 1.e-6f*1.f/1.602176487e-19f/(1.e+9f*1.f);// watt = 6.24150 e+3 * MeV/ns

  // Force [E][L^-1]
  __constant GGfloat NEWTON = 1.e-6f*1.f/1.602176487e-19f/(1000.f*1.0f);// newton = 6.24150 e+9 * MeV/mm

  // Pressure [E][L^-3]
  __constant GGfloat PASCAL = (1.e-6f*1.f/1.602176487e-19f/(1000.f*1.0f))/(1000.f*1.0f*1000.f*1.0f);   // pascal = 6.24150 e+3 * MeV/mm3
  __constant GGfloat BAR = 100000.0f*((1.e-6f*1.f/1.602176487e-19f/(1000.f*1.0f))/(1000.f*1.0f*1000.f*1.0f)); // bar    = 6.24150 e+8 * MeV/mm3
  __constant GGfloat ATMOSPHERE = 101325.0f*((1.e-6f*1.f/1.602176487e-19f/(1000.f*1.0f))/(1000.f*1.0f*1000.f*1.0f)); // atm    = 6.32420 e+8 * MeV/mm3

  // Electric current [Q][T^-1]
  __constant GGfloat AMPERE = (1.f/1.602176487e-19f)/(1.e+9f *1.f); // ampere = 6.24150 e+9 * eplus/ns
  __constant GGfloat MILLIAMPERE = 1.e-3f*((1.f/1.602176487e-19f)/(1.e+9f *1.f));
  __constant GGfloat MICROAMPERE = 1.e-6f*((1.f/1.602176487e-19f)/(1.e+9f *1.f));
  __constant GGfloat NANOAMPERE = 1.e-9f*((1.f/1.602176487e-19f)/(1.e+9f *1.f));

  // Electric potential [E][Q^-1]
  __constant GGfloat MEGAVOLT = (1.f)/(1.f);
  __constant GGfloat KILOVOLT = 1.e-3f*((1.f)/(1.f));
  __constant GGfloat VOLT = 1.e-6f*((1.f)/(1.f));

  // Electric resistance [E][T][Q^-2]
  __constant GGfloat OHM = (1.e-6f*((1.f)/(1.f)))/((1.f/1.602176487e-19f)/(1.e+9f *1.f));// ohm = 1.60217e-16*(MeV/eplus)/(eplus/ns)

  // Electric capacitance [Q^2][E^-1]
  __constant GGfloat FARAD = (1.f/1.602176487e-19f)/(1.e-6f*((1.f)/(1.f)));// farad = 6.24150e+24 * eplus/Megavolt
  __constant GGfloat MILLIFARAD = 1.e-3f*((1.f/1.602176487e-19f)/(1.e-6f*((1.f)/(1.f))));
  __constant GGfloat MICROFARAD = 1.e-6f*((1.f/1.602176487e-19f)/(1.e-6f*((1.f)/(1.f))));
  __constant GGfloat NANOFARAD = 1.e-9f*((1.f/1.602176487e-19f)/(1.e-6f*((1.f)/(1.f))));
  __constant GGfloat PICOFARAD = 1.e-12f*((1.f/1.602176487e-19f)/(1.e-6f*((1.f)/(1.f))));

  // Magnetic Flux [T][E][Q^-1]
  __constant GGfloat WEBER = (1.e-6f*((1.f)/(1.f)))*(1.e+9f *1.f);// weber = 1000*megavolt*ns

  // Magnetic Field [T][E][Q^-1][L^-2]
  __constant GGfloat TESLA = (1.e-6f*((1.f)/(1.f)))*(1.e+9f *1.f)/(1000.f*1.0f*1000.f*1.0f);// tesla =0.001*megavolt*ns/mm2

  __constant GGfloat GAUSS = 1.e-4f*((1.e-6f*((1.f)/(1.f)))*(1.e+9f *1.f)/(1000.f*1.0f*1000.f*1.0f));
  __constant GGfloat KILOGAUSS = 1.e-1f*((1.e-6f*((1.f)/(1.f)))*(1.e+9f *1.f)/(1000.f*1.0f*1000.f*1.0f));

  // Inductance [T^2][E][Q^-2]
  __constant GGfloat henry = ((1.e-6f*((1.f)/(1.f)))*(1.e+9f *1.f))/((1.f/1.602176487e-19f)/(1.e+9f *1.f));// henry = 1.60217e-7*MeV*(ns/eplus)**2

  // Temperature
  __constant GGfloat KELVIN = 1.f;

  // Amount of substance
  __constant GGfloat MOLE = 1.f;
  __constant GGfloat mol = 1.f;

  // Activity [T^-1]
  __constant GGfloat BECQUEREL = 1.f/(1.e+9f *1.f) ;
  __constant GGfloat CURIE = 3.7e+10f * (1.f/(1.e+9f *1.f));
  __constant GGfloat KILOBECQUEREL = 1.e+3f*(1.f/(1.e+9f *1.f));
  __constant GGfloat MEGABECQUEREL = 1.e+6f*(1.f/(1.e+9f *1.f));
  __constant GGfloat GIGABECQUEREL = 1.e+9f*(1.f/(1.e+9f *1.f));
  __constant GGfloat MILLICURIE = 1.e-3f*(3.7e+10f * (1.f/(1.e+9f *1.f)));
  __constant GGfloat MICROCURIE = 1.e-6f*(3.7e+10f * (1.f/(1.e+9f *1.f)));

  // Symbols definitions
  __constant GGfloat Bq = (1.f/(1.e+9f *1.f));
  __constant GGfloat kBq = (1.e+3f*(1.f/(1.e+9f *1.f)));
  __constant GGfloat MBq = (1.e+6f*(1.f/(1.e+9f *1.f)));
  __constant GGfloat GBq = (1.e+9f*(1.f/(1.e+9f *1.f)));
  __constant GGfloat Ci = (3.7e+10f * (1.f/(1.e+9f *1.f)));
  __constant GGfloat mCi = (1.e-3f*(3.7e+10f * (1.f/(1.e+9f *1.f))));
  __constant GGfloat uCi = (1.e-6f*(3.7e+10f * (1.f/(1.e+9f *1.f))));

  // Absorbed dose [L^2][T^-2]
  __constant GGfloat GRAY = (1.e-6f*1.f/1.602176487e-19f)/(1.e-6f*1.f/1.602176487e-19f*1.e+9f *1.f*1.e+9f *1.f/(1000.f*1.0f*1000.f*1.0f));
  __constant GGfloat KILOGRAY = 1.e+3f*((1.e-6f*1.f/1.602176487e-19f)/(1.e-6f*1.f/1.602176487e-19f*1.e+9f *1.f*1.e+9f *1.f/(1000.f*1.0f*1000.f*1.0f)));
  __constant GGfloat MILLIGRAY = 1.e-3f*((1.e-6f*1.f/1.602176487e-19f)/(1.e-6f*1.f/1.602176487e-19f*1.e+9f *1.f*1.e+9f *1.f/(1000.f*1.0f*1000.f*1.0f)));
  __constant GGfloat MICROGRAY = 1.e-6f*((1.e-6f*1.f/1.602176487e-19f)/(1.e-6f*1.f/1.602176487e-19f*1.e+9f*1.f*1.e+9f *1.f/(1000.f*1.0f*1000.f*1.0f)));

  // Luminous intensity [I]
  __constant GGfloat CANDELA = 1.f;

  // Luminous flux [I]
  __constant GGfloat LUMEN = (1.f)*(1.0f);

  // Illuminance [I][L^-2]
  __constant GGfloat LUX = ((1.f)*(1.0f))/(1000.f*1.0f*1000.f*1.0f);

  // Miscellaneous
  __constant GGfloat PERCENT = 0.01f ;
  __constant GGfloat PERTHOUSAND = 0.001f;
  __constant GGfloat PERMILLION  = 0.000001f;

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

/*!
  \namespace GGEMSPhysicalConstants
  \brief namespace storing all physical constants
*/
#ifndef OPENCL_COMPILER
namespace GGEMSPhysicalConstants
{
#endif

  __constant GGfloat PI = 3.141592653589793238463f;
  __constant GGfloat PI_TWICE = 2.0 * 3.141592653589793238463f;
  __constant GGfloat PI_HALF = 3.141592653589793238463f / 2.0f;
  __constant GGfloat PI_PI = 3.141592653589793238463f * 3.141592653589793238463f;

  __constant GGfloat AVOGADRO = 6.02214179e+23f/
  #ifndef OPENCL_COMPILER
  GGEMSUnits::mol;
  #else
  1.0f;
  #endif

  __constant GGfloat GASTHRESHOLD = 10.f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::mg/GGEMSUnits::cm3;
  #else
  1.e-3f*1.e-3f*(1.e-6f*1.f/1.602176487e-19f*1.e+9f*1.f*1.e+9f *1.f/(1000.f*1.0f*1000.f*1.0f))*10.f*1.0f*10.f*1.0f*10.f*1.0f;
  #endif

  // c   = 299.792458 mm/ns
  // c^2 = 898.7404 (mm/ns)^2
  __constant GGfloat C_LIGHT = 2.99792458e+8f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::m/GGEMSUnits::s;
  #else
  1000.*1.0f/1.e+9f *1.f;
  #endif

  __constant GGfloat C_SQUARED = C_LIGHT * C_LIGHT;

  __constant GGfloat ELECTRON_CHARGE =
  #ifndef OPENCL_COMPILER
  -GGEMSUnits::eplus;
  #else
  -1.0f;
  #endif

  __constant GGfloat E_SQUARED =
  #ifndef OPENCL_COMPILER
  GGEMSUnits::eplus * GGEMSUnits::eplus;
  #else
  1.0f*1.0f;
  #endif

  // amu_c2 - atomic equivalent mass unit
  //        - AKA, unified atomic mass unit (u)
  // amu    - atomic mass unit
  __constant GGfloat ELECTRON_MASS_C2 = 0.510998910f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::MeV;
  #else
  1.0f;
  #endif

  __constant GGfloat PROTON_MASS_C2 = 938.272013f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::MeV;
  #else
  1.0f;
  #endif

  __constant GGfloat NEUTRON_MASS_C2 = 939.56536f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::MeV;
  #else
  1.0f;
  #endif

  __constant GGfloat AMU_C2 = 931.494028f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::MeV;
  #else
  1.0f;
  #endif

  __constant GGfloat AMU = AMU_C2/C_SQUARED;

  // permeability of free space mu0    = 2.01334e-16 Mev*(ns*eplus)^2/mm
  // permittivity of free space epsil0 = 5.52636e+10 eplus^2/(MeV*mm)
  __constant GGfloat MU0      = 4.0f*PI*1.e-7f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::henry / GGEMSUnits::m;
  #else
  ((1.e-6f*((1.f)/(1.f)))*(1.e+9f *1.f))/((1.f/1.602176487e-19f)/(1.e+9f *1.f))/1000.0f;
  #endif

  __constant GGfloat EPSILON0 = 1.0f/(C_SQUARED*MU0);

  // h     = 4.13566e-12 MeV*ns
  // hbar  = 6.58212e-13 MeV*ns
  // hbarc = 197.32705e-12 MeV*mm
  __constant GGfloat H_PLANCK = 6.62606896e-34f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::joule * GGEMSUnits::s;
  #else
  1.e-6f*1.f/1.602176487e-19f*1.e+9f *1.f;
  #endif

  __constant GGfloat HBAR_PLANCK   = H_PLANCK/PI_TWICE;
  __constant GGfloat HBARC         = HBAR_PLANCK * C_LIGHT;
  __constant GGfloat HBARC_SQUARED = HBARC * HBARC;

  // electromagnetic coupling = 1.43996e-12 MeV*mm/(eplus^2)
  __constant GGfloat ELM_COUPLING           = E_SQUARED/(4.0f*PI*EPSILON0);
  __constant GGfloat FINE_STRUCTURE_CONST   = ELM_COUPLING/HBARC;
  __constant GGfloat CLASSIC_ELECTRON_RADIUS  = ELM_COUPLING/ELECTRON_MASS_C2;
  __constant GGfloat ELECTRON_COMPTON_LENGTH = HBARC/ELECTRON_MASS_C2;
  __constant GGfloat BOHR_RADIUS = ELECTRON_COMPTON_LENGTH/FINE_STRUCTURE_CONST;

  __constant GGfloat ALPHA_RCL2 = FINE_STRUCTURE_CONST * CLASSIC_ELECTRON_RADIUS * CLASSIC_ELECTRON_RADIUS;

#ifndef OPENCL_COMPILER
}
#endif

#endif // End of GUARD_GGEMS_TOOLS_GGEMSSYSTEMOFUNITS_HH
