#ifndef GUARD_GGEMS_TOOLS_GGEMSPROFILER_HH
#define GUARD_GGEMS_TOOLS_GGEMSPROFILER_HH

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
  \file GGEMSProfiler.hh

  \brief GGEMS class handling a specific profiler type

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 16, 2021
*/

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/tools/GGEMSChrono.hh"
#include "GGEMS/tools/GGEMSProfilerItem.hh"

/*!
  \class GGEMSProfiler
  \brief GGEMS class handling a specific profiler type
*/
class GGEMS_EXPORT GGEMSProfiler
{
  public:
    /*!
      \brief GGEMSProfiler constructor
    */
    GGEMSProfiler(void);

    /*!
      \brief GGEMSProfiler destructor
    */
    ~GGEMSProfiler(void);

    /*!
      \fn void HandleEvent(cl::Event event)
      \param event - OpenCL event
      \brief handle an OpenCL event in profile_name type
    */
    void HandleEvent(cl::Event event);

    /*!
      \fn inline DurationNano GetSummaryTime(void) const
      \brief get elapsed time in ns in OpenCL operation
    */
    inline DurationNano GetSummaryTime(void) const
    {
      DurationNano time = GGEMSChrono::Zero();
      for (std::size_t i = 0; i < number_of_data_; ++i) {
        time += profiler_items_[i]->GetEndFromStartTime();
      }

      return time;
    }

  private:
    /*!
      \fn static void CallBackFunction(cl_event event, GGint event_command_exec_status, void* user_data)
      \param event - OpenCL event
      \param event_command_exec_status - status of OpenCL event
      \param user_data - adress of GGEMSProfiler object
      \brief call back function analyzing event
    */
    static void CallBackFunction(cl_event event, GGint event_command_exec_status, void* user_data);

    /*!
      \fn void AddProfilerItem(cl_event event)
      \param event - event to profile
      \brief add new profiler item in profile
    */
    void AddProfilerItem(cl_event event);

  private:
    GGEMSProfilerItem** profiler_items_; /*!< Buffer storing profiling data of same type */
    std::size_t number_of_data_; /*!< Number of profiler data */
};

#endif // End of GUARD_GGEMS_TOOLS_GGEMSPROFILER_HH
