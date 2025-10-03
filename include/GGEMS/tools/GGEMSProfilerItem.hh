#ifndef GUARD_GGEMS_TOOLS_GGEMSPROFILERITEM_HH
#define GUARD_GGEMS_TOOLS_GGEMSPROFILERITEM_HH

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
  \file GGEMSProfilerItem.hh

  \brief GGEMS class handling a specific item profiler

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 16, 2021
*/

#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/tools/GGEMSChrono.hh"

/*!
  \enum GGEMSEventInfo
  \brief infos from OpenCL event
*/
enum GGEMSEventInfo : GGint
{
  /*!
    \brief START event flag
  */
  START = 0,

  /*!
    \brief END event flag
  */
  END,

  /*!
    \brief ELAPSED event flag
  */
  ELAPSED
};

/*!
  \class GGEMSProfilerItem
  \brief GGEMS handling a specific item profiler_item
*/
class GGEMS_EXPORT GGEMSProfilerItem
{
  public:
    /*!
      \param event - OpenCL event
      \brief GGEMSProfilerItem constructor
    */
    explicit GGEMSProfilerItem(cl_event event);

    /*!
      \brief GGEMSProfilerItem destructor
    */
    ~GGEMSProfilerItem(void) {}

    /*!
      \fn GGEMSProfilerItem(GGEMSProfilerItem const& profiler_item) = delete
      \param profiler_item - reference on the GGEMS profiler item
      \brief Avoid copy by reference
    */
    GGEMSProfilerItem(GGEMSProfilerItem const& profiler_item) = delete;

    /*!
      \fn GGEMSProfilerItem& operator=(GGEMSProfilerItem const& profiler_item) = delete
      \param profiler_item - reference on the GGEMS profiler item
      \brief Avoid assignement by reference
    */
    GGEMSProfilerItem& operator=(GGEMSProfilerItem const& profiler_item) = delete;

    /*!
      \fn GGEMSProfilerItem(GGEMSProfilerItem const&& profiler_item) = delete
      \param profiler_item - rvalue reference on the GGEMS profiler item
      \brief Avoid copy by rvalue reference
    */
    GGEMSProfilerItem(GGEMSProfilerItem const&& profiler_item) = delete;

    /*!
      \fn GGEMSProfilerItem& operator=(GGEMSProfilerItem const&& profiler_item) = delete
      \param profiler_item - rvalue reference on the GGEMS profiler item
      \brief Avoid copy by rvalue reference
    */
    GGEMSProfilerItem& operator=(GGEMSProfilerItem const&& profiler_item) = delete;

    /*!
      \fn inline DurationNano GetElapsedTime(void) const
      \return time in ns
      \brief Get elapsed time
    */
    inline DurationNano GetElapsedTime(void) const {return static_cast<DurationNano>(times_[GGEMSEventInfo::ELAPSED]);}

    /*!
      \fn void UpdateEvent(cl_event event)
      \param event - OpenCL event
      \brief Update elapsed time in OpenCL command
    */
    void UpdateEvent(cl_event event);

  private:
    GGulong times_[3]; /*!< Variables storing start and end of computation time in OpenCL command */
};

#endif // End of GUARD_GGEMS_TOOLS_GGEMSPROFILERITEM_HH
