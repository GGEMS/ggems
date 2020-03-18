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

  // Lengths [L] (mm) (mm2) (mm3)
  __constant GGfloat nm  = 1.e-6f;
  __constant GGfloat um  = 1.e-3f;
  __constant GGfloat mm  = 1.0f; // Reference
  __constant GGfloat mm2 = 1.0f; // Reference
  __constant GGfloat mm3 = 1.0f; // Reference
  __constant GGfloat cm  = 10.f;
  __constant GGfloat cm2 = 1.e2f;
  __constant GGfloat cm3 = 1.e3f;
  __constant GGfloat m   = 1.e3f;
  __constant GGfloat m2  = 1.e6f;
  __constant GGfloat m3  = 1.e9f;
  __constant GGfloat km  = 1.e6f;
  __constant GGfloat km2 = 1.e12f;
  __constant GGfloat km3 = 1.e18f;
  __constant GGfloat pc  = 3.0856775807e+19f;

  // Angles (rad)
  __constant GGfloat rad  = 1.0f; // Reference
  __constant GGfloat mrad = 1.e-3f;
  __constant GGfloat sr   = 1.0f;
  __constant GGfloat deg  = 3.141592653589793238463f/180.0f;

  // Time [T] (ns)
  __constant GGfloat ns = 1.f; // Reference
  __constant GGfloat s  = 1.e+9f;
  __constant GGfloat ms = 1.e+6f;
  __constant GGfloat us = 1.e+3f;
  __constant GGfloat ps = 1.e-3f;

  // Frequency [T^-1] (ns-1)
  __constant GGfloat Hz  = 1.f/(1.e+9f);
  __constant GGfloat kHz = 1.e-6f;
  __constant GGfloat MHz = 1.e-3f;

  // Electric charge [Q]
  __constant GGfloat eplus = 1.f ;// positron charge
  __constant GGfloat qe    = 1.602176487e-19f;// elementary charge in coulomb
  __constant GGfloat C     = 1.f/1.602176487e-19f;// coulomb = 6.24150 e+18 * eplus

  // Energy [E] (MeV)
  __constant GGfloat eV    = 1.e-6f;
  __constant GGfloat keV   = 1.e-3f;
  __constant GGfloat MeV   = 1.f; // Reference
  __constant GGfloat GeV   = 1.e+3f;
  __constant GGfloat TeV   = 1.e+6f;
  __constant GGfloat PeV   = 1.e+9f;
  __constant GGfloat J     = 1.e-6f/1.602176487e-19f;// joule = 6.24150 e+12 * MeV

  // Mass [E][T^2][L^-2] (MeV.ns2.mm-2)
  __constant GGfloat kg = 6.241509704e+24f;
  __constant GGfloat g  = 6.241509704e+21f;
  __constant GGfloat mg = 6.241509704e+18f;

  // Power [E][T^-1] (MeV.ns-1)
  __constant GGfloat W = 6.241509766e+3f;

  // Force [E][L^-1] (MeV.mm-1)
  __constant GGfloat N = 6.241509766e+9f;

  // Pressure [E][L^-3] (MeV.mm-3)
  __constant GGfloat Pa = 6.241509766e+3f;
  __constant GGfloat bar = 100000.0f*6.241509766e+3f;
  __constant GGfloat atm = 101325.0f*6.241509766e+3f;

  // Electric current [Q][T^-1] (C.ns-1)
  __constant GGfloat A  = 6.241509696e+9f;
  __constant GGfloat mA = 6.241509696e+6f;
  __constant GGfloat uA = 6.241509696e+3f;
  __constant GGfloat nA = 6.241509696f;

  // Electric potential [E][Q^-1] 
  __constant GGfloat MV = 1.0f; // Reference
  __constant GGfloat kV = 1.e-3f;
  __constant GGfloat V  = 1.e-6f;

  // Electric resistance [E][T][Q^-2] (MeV.ns.C-2)
  __constant GGfloat OHM = 1.602176452e-16f;// ohm = 1.60217e-16*(MeV/eplus)/(eplus/ns)

  // Electric capacitance [Q^2][E^-1] (C.MV-1)
  __constant GGfloat F  = 6.241509468e+24f;
  __constant GGfloat mF = 6.241509468e+21f;
  __constant GGfloat uF = 6.241509468e+18f;
  __constant GGfloat nF = 6.241509468e+15f;
  __constant GGfloat pF = 6.241509468e+12f;

  // Magnetic Flux [T][E][Q^-1] (ns.MV)
  __constant GGfloat Wb = 1000.0f;// weber = 1000*megavolt*ns

  // Magnetic Field [T][E][Q^-1][L^-2] (MV.ns.mm2)
  __constant GGfloat T = 0.001f;// tesla =0.001*megavolt*ns/mm2
  __constant GGfloat G = 1.e-7f; // 0.0001 T
  __constant GGfloat kG = 1.e-4f;

  // Inductance [T^2][E][Q^-2] (MeV.ns2.C-2)
  __constant GGfloat H = 1.602176383e-07f;// henry = 1.60217e-7*MeV*(ns/eplus)**2

  // Temperature (K)
  __constant GGfloat K = 1.0f; //Reference

  // Amount of substance (mol)
  __constant GGfloat mol = 1.0f; //Reference

  // Activity [T^-1] (ns-1)
  __constant GGfloat Bq  = 1.e-9f;
  __constant GGfloat kBq = 1.e-6f;
  __constant GGfloat MBq = 1.e-3f;
  __constant GGfloat GBq = 1.0f;
  __constant GGfloat Ci  = 3.7e+10f/1.e+9f; // Bq.ns-1
  __constant GGfloat mCi = 3.7e-2f;
  __constant GGfloat uCi = 3.7e-5f;

  // Absorbed dose [L^2][T^-2] (mm2.ns-2)
  __constant GGfloat Gy  = 1.0e-12f;
  __constant GGfloat kGy = 1.0e-9f;
  __constant GGfloat mGy = 1.0e-15f;
  __constant GGfloat uGy = 1.0e-18f;

  // Luminous intensity [I] (cd)
  __constant GGfloat cd = 1.0f;

  // Luminous flux [I] (cd.sr)
  __constant GGfloat lm = 1.0f;

  // Illuminance [I][L^-2] (cd.sr.mm-2)
  __constant GGfloat lx = 1.e-6f;

  // Miscellaneous
  __constant GGfloat PERCENT = 0.01f ;
  __constant GGfloat PERTHOUSAND = 0.001f;
  __constant GGfloat PERMILLION  = 0.000001f;

  #ifndef OPENCL_COMPILER
  /*!
    \fn T DistanceUnit(T const& value, char const* unit)
    \tparam T - type of the value to convert unit
    \param value - value to check
    \param unit - distance unit
    \brief Choose best distance unit
    \return value in the good unit
  */
  template <typename T>
  T DistanceUnit(T const& value, char const* unit)
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
    \fn T EnergyUnit(T const& value, char const* unit)
    \tparam T - type of the value to convert unit
    \param value - value to check
    \param unit - energy unit
    \brief Choose best energy unit
    \return value in the good unit
  */
  template <typename T>
  T EnergyUnit(T const& value, char const* unit)
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
    \fn T AngleUnit(T const& value, char const* unit)
    \tparam T - type of the value to convert unit
    \param value - value to check
    \param unit - angle unit
    \brief Choose best angle unit
    \return value in the good unit
  */
  template <typename T>
  T AngleUnit(T const& value, char const* unit)
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
