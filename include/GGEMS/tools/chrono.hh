#ifndef GUARD_GGEMS_TOOLS_CHRONO_HH
#define GUARD_GGEMS_TOOLS_CHRONO_HH

/*!
  \file chrono.hh

  \brief Namespace computing/displaying the time

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Friday October 4, 2019
*/

#include <chrono>
#include <string>

#include "GGEMS/global/ggems_export.hh"

typedef std::chrono::time_point<std::chrono::system_clock> ChronoTime;
typedef std::chrono::duration<int64_t,std::nano> DurationNano;

#if __MINGW64__ || __clang__ || (_MSC_VER > 1800) || __GNUC__
typedef std::chrono::milliseconds Ms;
typedef std::chrono::seconds Secs;
typedef std::chrono::minutes Mins;
typedef std::chrono::hours Hs;
#endif

/*!
  \namespace Chrono
  \brief namespace computing/displaying the time
*/
namespace Chrono
{
  /*!
    \fn void DisplayTime( DurationNano const& duration, std::string const& displayedText )
    \param duration - Duration of code/method in nanoseconds
    \param displayedText - Text into the screen/file
    \brief Print the execution time
    \return no value is returned
  */
  void GGEMS_EXPORT DisplayTime(DurationNano const& duration,
    std::string const& displayed_text);

  /*!
    \fn inline ChronoTime Now(void)
    \return the current time in nanoseconds
  */
  inline ChronoTime Now(void) {return std::chrono::system_clock::now();}

  /*!
    \fn inline DurationNano Zero(void)
    \brief Initialization at zero nanosecond
  */
  inline DurationNano Zero(void)
  {
    return std::chrono::duration<int64_t,std::nano>::zero();
  }
}

#endif // End of GUARD_GGEMS_TOOLS_CHRONO_HH
