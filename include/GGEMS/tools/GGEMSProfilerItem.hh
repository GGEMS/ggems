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

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/tools/GGEMSChrono.hh"

/*!
  \enum GGEMSEventInfo
  \brief infos from OpenCL event
*/
enum GGEMSEventInfo : GGint
{
  QUEUE = 0,
  SUBMIT,
  START,
  END
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
    ~GGEMSProfilerItem(void);

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
      \fn void PrintInfos(void) const
      \brief print infos about timing for event
    */
    void PrintInfos(void) const;

    /*!
      \fn inline DurationNano GetQueueTime(void) const
      \return time in ns
      \brief Get moment in time of enqueueing command
    */
    inline DurationNano GetQueueTime(void) const {return static_cast<DurationNano>(times_[GGEMSEventInfo::QUEUE]);}

    /*!
      \fn inline DurationNano GetSubmitFromQueueTime(void) const
      \return time in ns
      \brief Get submission time from queue
    */
    inline DurationNano GetSubmitFromQueueTime(void) const {return static_cast<DurationNano>(times_[GGEMSEventInfo::SUBMIT] - times_[GGEMSEventInfo::QUEUE]);}

    /*!
      \fn inline DurationNano GetStartFromSubmitTime(void) const
      \return time in ns
      \brief Get start time from submit
    */
    inline DurationNano GetStartFromSubmitTime(void) const {return static_cast<DurationNano>(times_[GGEMSEventInfo::START] - times_[GGEMSEventInfo::SUBMIT]);}

    /*!
      \fn inline DurationNano GetEndFromStartTime(void) const
      \return time in ns
      \brief Get end time from start
    */
    inline DurationNano GetEndFromStartTime(void) const {return static_cast<DurationNano>(times_[GGEMSEventInfo::END] - times_[GGEMSEventInfo::START]);}

  private:
    GGulong times_[4]; /*!< Buffer storing times from event */
};

#endif // End of GUARD_GGEMS_TOOLS_GGEMSPROFILERITEM_HH
