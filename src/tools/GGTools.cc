/*!
  \file GGTools.cc

  \brief Namespaces for different useful fonctions

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Wednesday October 10, 2018
*/

#include <sstream>
#include <cerrno>
#include <cstring>
#include <cmath>
#include <iostream>

#include "GGEMS/tools/GGTools.hh"
#include "GGEMS/tools/GGPrint.hh"

#include "GGEMS/global/GGEMSManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGStream::CheckInputStream(std::ifstream const& input_stream,
  std::string const& filename)
{
  if (!input_stream) {
    std::ostringstream oss(std::ostringstream::out);
    #ifdef _WIN32
    char buffer_error[ 256 ];
    oss << "Problem reading filename '" << filename << "': "
        << strerror_s(buffer_error, 256, errno);
    #else
    oss << "Problem reading filename '" << filename << "': "
        << strerror(errno);
    #endif
    throw std::runtime_error(oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// @cond
template GGbool GGMisc::IsEqual<GGdouble>(GGdouble const&, GGdouble const&);
template GGbool GGMisc::IsEqual<float>(GGfloat const&, GGfloat const&);
/// @endcond

template <typename T>
GGbool GGMisc::IsEqual(T const& a, T const& b)
{
  return std::nextafter(a, std::numeric_limits<T>::lowest()) <= b
    && std::nextafter(a, std::numeric_limits<T>::max()) >= b;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGMisc::ThrowException(std::string const& class_name,
  std::string const& method_name, std::string const& message)
{
  std::ostringstream oss(std::ostringstream::out);
  oss << message;
  GGcerr(class_name, method_name, 0) << oss.str() << GGendl;
  throw std::runtime_error("");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSTools::PrintBanner()
{
  // Call the GGEMS singleton
  GGEMSManager& ggems_manager = GGEMSManager::GetInstance();
  // Get the GGEMS version
  std::string const kGGEMSVersion = ggems_manager.GetVersion();

  // Print the banner
  GGcout("GGEMSTools", "PrintBanner", 0) << "      ____                  "
    << GGendl;
  GGcout("GGEMSTools", "PrintBanner", 0) << ".--. /\\__/\\ .--.            "
    << GGendl;
  GGcout("GGEMSTools", "PrintBanner", 0) << "`O  / /  \\ \\  .`     GGEMS "
    << kGGEMSVersion << GGendl;
  GGcout("GGEMSTools", "PrintBanner", 0) << "  `-| |  | |O`              "
    << GGendl;
  GGcout("GGEMSTools", "PrintBanner", 0) << "   -|`|..|`|-        "
    << GGendl;
  GGcout("GGEMSTools", "PrintBanner", 0) << " .` \\.\\__/./ `.    "
    << GGendl;
  GGcout("GGEMSTools", "PrintBanner", 0) << "'.-` \\/__\\/ `-.'   "
    << GGendl;
  GGcout("GGEMSTools", "PrintBanner", 0) << GGendl;
}
