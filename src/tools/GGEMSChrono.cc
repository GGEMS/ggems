/*!
  \file GGEMSChrono.cc

  \brief Structure storing static method computing/displaying the time

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Friday October 4, 2019
*/

#include <iomanip>

#include "GGEMS/tools/GGEMSChrono.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGChrono::DisplayTime(DurationNano const& duration,
  std::string const& displayed_text)
{
  #if __MINGW64__ || __clang__ || (_MSC_VER > 1800) || __GNUC__
  // Display the iteration time
  GGEMScout("Chrono", "DisplayTime", 0) << "Elapsed time (" << displayed_text
    << "): " << std::setfill( '0' ) << std::setw(2)
    << std::chrono::duration_cast<Hs>((duration)).count()
    << " hours " << std::setw(2)
    << std::chrono::duration_cast<Mins>((duration) % Hs(1)).count()
    << " mins " << std::setw(2)
    << std::chrono::duration_cast<Secs>(
        (duration) % Mins(1) ).count()
    << " secs " << std::setw(3)
    << std::chrono::duration_cast<Ms>((duration) % Secs(1)).count()
    << " ms" << GGEMSendl;
  #else
  GGEMScout("Chrono", "DisplayTime", 0) << "Elapsed time (" << displayedText
    << "): " << duration.count() / 1000000000.0f
    << "sec";
  #endif
}
