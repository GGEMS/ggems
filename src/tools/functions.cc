/*!
  \file functions.cc

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

#include "GGEMS/tools/functions.hh"
#include "GGEMS/global/ggems_manager.hh"
#include "GGEMS/tools/print.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void Stream::CheckInputStream(std::ifstream const& input_stream,
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
template bool Misc::IsEqual<double>(double const&, double const&);
template bool Misc::IsEqual<float>(float const&, float const&);
/// @endcond

template <typename T>
bool Misc::IsEqual(T const& a, T const& b)
{
  return std::nextafter(a, std::numeric_limits<T>::lowest()) <= b
    && std::nextafter(a, std::numeric_limits<T>::max()) >= b;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void Misc::ThrowException(std::string const& class_name,
  std::string const& method_name, std::string const& message)
{
  std::ostringstream oss(std::ostringstream::out);
  oss << "[GGEMS " << class_name << "::" << method_name << "] EXCEPTION!!! "
      << message;
  #if defined _MSC_VER
  GGEMScout(class_name, method_name, 0) << oss.str() << GGEMSendl;
  #endif
  throw std::runtime_error(oss.str());
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
  GGEMScout("GGEMSTools", "PrintBanner", 0) << "      ____                  "
    << GGEMSendl;
  GGEMScout("GGEMSTools", "PrintBanner", 0) << ".--. /\\__/\\ .--.            "
    << GGEMSendl;
  GGEMScout("GGEMSTools", "PrintBanner", 0) << "`O  / /  \\ \\  .`     GGEMS "
    << kGGEMSVersion << GGEMSendl;
  GGEMScout("GGEMSTools", "PrintBanner", 0) << "  `-| |  | |O`              "
    << GGEMSendl;
  GGEMScout("GGEMSTools", "PrintBanner", 0) << "   -|`|..|`|-        "
    << GGEMSendl;
  GGEMScout("GGEMSTools", "PrintBanner", 0) << " .` \\.\\__/./ `.    "
    << GGEMSendl;
  GGEMScout("GGEMSTools", "PrintBanner", 0) << "'.-` \\/__\\/ `-.'   "
    << GGEMSendl;
  GGEMScout("GGEMSTools", "PrintBanner", 0) << GGEMSendl;
}
