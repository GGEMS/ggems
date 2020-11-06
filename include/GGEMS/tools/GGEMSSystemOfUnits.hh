#ifndef GUARD_GGEMS_TOOLS_GGEMSSYSTEMOFUNITS_HH
#define GUARD_GGEMS_TOOLS_GGEMSSYSTEMOFUNITS_HH

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

#ifndef __OPENCL_C_VERSION__
/*!
  \def __constant
  \brief __constant is known for OpenCL, but for C++ we define __constant as a constexpr
*/
#define constant inline static constexpr
#include <algorithm>
#include <sstream>
#include "GGEMS/tools/GGEMSTools.hh"
#endif

  // Lengths [L] (mm)
constant GGfloat nm  = 1.e-6f; /*!< Nanometer */
constant GGfloat um  = 1.e-3f; /*!< Micrometer */
constant GGfloat mm  = 1.0f; /*!< Millimeter (REFERENCE) */
constant GGfloat mm2 = 1.0f; /*!< Squared millimeter (REFERENCE) */
constant GGfloat mm3 = 1.0f; /*!< Cubic millimeter (REFERENCE) */
constant GGfloat cm  = 10.f; /*!< Centimeter */
constant GGfloat cm2 = 1.e2f; /*!< Squared centimeter */
constant GGfloat cm3 = 1.e3f; /*!< Cubic centimeter */
constant GGfloat m   = 1.e3f; /*!< Meter */
constant GGfloat m2  = 1.e6f; /*!< Squared meter */
constant GGfloat m3  = 1.e9f; /*!< Cubic meter */
constant GGfloat km  = 1.e6f; /*!< Kilometer */
constant GGfloat km2 = 1.e12f; /*!< Squared kilometer */
constant GGfloat km3 = 1.e18f; /*!< Cubic kilometer */
constant GGfloat pc  = 3.0856775807e+19f; /*!< Parsec */

// Cross section unit (mm2)
constant GGfloat b = 1.e-22f; /*!< Barn */
constant GGfloat mb = 1.e-25f; /*!< millibarn */
constant GGfloat ub = 1.e-28f; /*!< microbarn */
constant GGfloat nb = 1.e-31f; /*!< nanobarn */
constant GGfloat pb = 1.e-34f; /*!< picobarn */

// Angles
constant GGfloat rad  = 1.0f; /*!< Radian (REFERENCE) */
constant GGfloat mrad = 1.e-3f; /*!< milliradian */
constant GGfloat deg  = 3.141592653589793238463f/180.0f; /*!< Degree */

// Solid angle
constant GGfloat sr   = 1.0f; /*!< Steradian (REFERENCE) */

// Time [T] (ns)
constant GGfloat ns = 1.f; /*!< Nanosecond (REFERENCE) */
constant GGfloat s  = 1.e+9f; /*!< Second */
constant GGfloat ms = 1.e+6f; /*!< Millisecond */
constant GGfloat us = 1.e+3f; /*!< Microsecond */
constant GGfloat ps = 1.e-3f; /*!< Picosecond */

// Frequency [T^-1] (ns-1)
constant GGfloat Hz  = 1.f/(1.e+9f); /*!< Hertz */
constant GGfloat kHz = 1.e-6f; /*!< Kilohertz */
constant GGfloat MHz = 1.e-3f; /*!< Megahertz */

// Electric charge [Q] (eplus)
constant GGfloat eplus = 1.f; /*!< Positron charge */
constant GGfloat qe    = 1.602176487e-19f; /*!< elementary charge in coulomb (C) */
constant GGfloat C     = 1.f/1.602176487e-19f; /*!< Coulomb = 6.24150e+18 * eplus */

// Energy [E] (MeV)
constant GGfloat eV    = 1.e-6f; /*!< Electronvolt */
constant GGfloat keV   = 1.e-3f; /*!< kiloelectronvolt */
constant GGfloat MeV   = 1.f; /*!< Megaelectronvolt (REFERENCE) */
constant GGfloat GeV   = 1.e+3f; /*!< Gigaelectronvolt */
constant GGfloat TeV   = 1.e+6f; /*!< Teraelectronvolt */
constant GGfloat PeV   = 1.e+9f; /*!< Petaelectronvolt */
constant GGfloat J     = 1.e-6f/1.602176487e-19f; /*!< Joule 6.24150 e+12 * MeV */

// Mass [E][T^2][L^-2] (MeV.ns2.mm-2)
constant GGfloat kg = 6.241509704e+24f; /*!< Kilogram */
constant GGfloat g  = 6.241509704e+21f; /*!< gram */
constant GGfloat mg = 6.241509704e+18f; /*!< milligram */

// Power [E][T^-1] (MeV.ns-1)
constant GGfloat W = 6.241509766e+3f; /*!< Watt */

// Force [E][L^-1] (MeV.mm-1)
constant GGfloat N = 6.241509766e+9f; /*!< Newton */

// Pressure [E][L^-3] (MeV.mm-3)
constant GGfloat Pa = 6.241509766e+3f; /*!< Pascal */
constant GGfloat bar = 100000.0f*6.241509766e+3f; /*!< Bar */
constant GGfloat atm = 101325.0f*6.241509766e+3f; /*!< Atmosphere */

// Electric current [Q][T^-1] (C.ns-1)
constant GGfloat A  = 6.241509696e+9f; /*!< Ampere */
constant GGfloat mA = 6.241509696e+6f; /*!< Milliampere */
constant GGfloat uA = 6.241509696e+3f; /*!< Microampere */
constant GGfloat nA = 6.241509696f; /*!< Nanoampere */

// Electric potential [E][Q^-1] 
constant GGfloat MV = 1.0f; /*!< Megavolt (REFERENCE) */
constant GGfloat kV = 1.e-3f; /*!< Kilovolt */
constant GGfloat V  = 1.e-6f; /*!< Volt */

// Electric resistance [E][T][Q^-2] (MeV.ns.C-2)
constant GGfloat OHM = 1.602176452e-16f; /*!< OHM 1.60217e-16*(MeV/eplus)/(eplus/ns) */

// Electric capacitance [Q^2][E^-1] (C.MV-1)
constant GGfloat F  = 6.241509468e+24f; /*!< Farad */
constant GGfloat mF = 6.241509468e+21f; /*!< millifarad */
constant GGfloat uF = 6.241509468e+18f; /*!< microfarad */
constant GGfloat nF = 6.241509468e+15f; /*!< nanofarad */
constant GGfloat pF = 6.241509468e+12f; /*!< picofarad */

// Magnetic Flux [T][E][Q^-1] (ns.MV)
constant GGfloat Wb = 1000.0f; /*!< Weber 1000*megavolt*ns */

// Magnetic Field [T][E][Q^-1][L^-2] (MV.ns.mm2)
constant GGfloat T = 0.001f; /*!< Tesla 0.001*megavolt*ns/mm2 */
constant GGfloat G = 1.e-7f; /*!< Gauss */
constant GGfloat kG = 1.e-4f; /*!< Kilogauss */

// Inductance [T^2][E][Q^-2] (MeV.ns2.C-2)
constant GGfloat H = 1.602176383e-07f; /*!< Henry 1.60217e-7*MeV*(ns/eplus)^2 */

// Temperature (K)
constant GGfloat K = 1.0f; /*!< Kelvin (REFERENCE) */

// Amount of substance (mol)
constant GGfloat mol = 1.0f; /*!< Mole (REFERENCE) */

// Activity [T^-1] (ns-1)
constant GGfloat Bq  = 1.e-9f; /*!< Becquerel */
constant GGfloat kBq = 1.e-6f; /*!< Kilobecquerel */
constant GGfloat MBq = 1.e-3f; /*!< Megabecquerel */
constant GGfloat GBq = 1.0f; /*!< Gigabecquerel (REFERENCE) */
constant GGfloat Ci  = 3.7e+10f/1.e+9f; /*!< Curie (Bq.ns-1) */
constant GGfloat mCi = 3.7e-2f; /*!< Millicurie */
constant GGfloat uCi = 3.7e-5f; /*!< Microcurie */

// Absorbed dose [L^2][T^-2] (mm2.ns-2)
constant GGfloat Gy  = 1.0e-12f; /*!< Gray */
constant GGfloat kGy = 1.0e-9f; /*!< Kilogray */
constant GGfloat mGy = 1.0e-15f; /*!< Milligray */
constant GGfloat uGy = 1.0e-18f; /*!< Microgray */

// Luminous intensity [I] (cd)
constant GGfloat cd = 1.0f; /*!< Candela (REFERENCE) */

// Luminous flux [I] (cd.sr)
constant GGfloat lm = 1.0f; /*!< Lumen (REFERENCE) */

// Illuminance [I][L^-2] (cd.sr.mm-2)
constant GGfloat lx = 1.e-6f; /*!< Lux */

// Miscellaneous
constant GGfloat percent = 0.01f; /*!< Percent value */
constant GGfloat perthousant = 0.001f; /*!< Perthousand value */
constant GGfloat permillion  = 0.000001f; /*!< Permillion value */

#ifndef __OPENCL_C_VERSION__
/*!
  \fn T DistanceUnit(T const& value, std::string const& unit)
  \tparam T - type of the value to convert unit
  \param value - value to check
  \param unit - distance unit
  \brief Choose best distance unit
  \return value in the good unit
*/
template <typename T>
T DistanceUnit(T const& value, std::string const& unit)
{
  T new_value = static_cast<T>(0);
  if (unit == "nm") {
    new_value = static_cast<T>(value * nm);
  }
  else if (unit == "um") {
    new_value = static_cast<T>(value * um);
  }
  else if (unit == "mm") {
    new_value = static_cast<T>(value * mm);
  }
  else if (unit == "cm") {
    new_value = static_cast<T>(value * cm);
  }
  else if (unit == "m") {
    new_value = static_cast<T>(value * m);
  }
  else if (unit == "km") {
    new_value = static_cast<T>(value * km);
  }
  else if (unit == "pc") {
    new_value = static_cast<T>(value * pc);
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
    GGEMSMisc::ThrowException("", "DistanceUnit", oss.str());
  }
  return new_value;
}

/*!
  \fn T EnergyUnit(T const& value, std::string const& unit)
  \tparam T - type of the value to convert unit
  \param value - value to check
  \param unit - energy unit
  \brief Choose best energy unit
  \return value in the good unit
*/
template <typename T>
T EnergyUnit(T const& value, std::string const& unit)
{
  T new_value = static_cast<T>(0);
  if (unit == "eV") {
    new_value = static_cast<T>(value * eV);
  }
  else if (unit == "keV") {
    new_value = static_cast<T>(value * keV);
  }
  else if (unit == "MeV") {
    new_value = static_cast<T>(value * MeV);
  }
  else if (unit == "GeV") {
    new_value = static_cast<T>(value * GeV);
  }
  else if (unit == "TeV") {
    new_value = static_cast<T>(value * TeV);
  }
  else if (unit == "PeV") {
    new_value = static_cast<T>(value * PeV);
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
    GGEMSMisc::ThrowException("", "EnergyUnit", oss.str());
  }
  return new_value;
}

/*!
  \fn T AngleUnit(T const& value, std::string const& unit)
  \tparam T - type of the value to convert unit
  \param value - value to check
  \param unit - angle unit
  \brief Choose best angle unit
  \return value in the good unit
*/
template <typename T>
T AngleUnit(T const& value, std::string const& unit)
{
  T new_value = static_cast<T>(0);
  if (unit == "rad") {
    new_value = static_cast<T>(value * rad);
  }
  else if (unit == "mrad") {
    new_value = static_cast<T>(value * mrad);
  }
  else if (unit == "deg") {
    new_value = static_cast<T>(value * deg);
  }
  else if (unit == "sr") {
    new_value = static_cast<T>(value * sr);
  }
  else {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Unknown unit!!! You have choice between:" << std::endl;
    oss << "    - \"rad\": radian" << std::endl;
    oss << "    - \"mrad\": milliradian" << std::endl;
    oss << "    - \"deg\": degree" << std::endl;
    oss << "    - \"sr\": steradian" << std::endl;
    GGEMSMisc::ThrowException("", "AngleUnit", oss.str());
  }
  return new_value;
}
#endif

#endif // End of GUARD_GGEMS_TOOLS_GGEMSSYSTEMOFUNITS_HH
