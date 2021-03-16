#ifndef GUARD_GGEMS_TOOLS_GGEMSPROFILERMANAGER_HH
#define GUARD_GGEMS_TOOLS_GGEMSPROFILERMANAGER_HH

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
  \file GGEMSProfilerManager.hh

  \brief GGEMS class managing profiler data

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 16, 2021
*/

#include <unordered_map>

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSProfiler.hh"

typedef std::unordered_map<std::string, GGEMSProfiler> ProfilerUMap; /*!< Unordered map with key : name of profile, profile object */

/*!
  \class GGEMSProfilerManager
  \brief GGEMS class managing profiler data
*/
class GGEMS_EXPORT GGEMSProfilerManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGEMSProfilerManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGEMSProfilerManager(void);

  public:
    /*!
      \fn static GGEMSProfilerManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSProfilerManager
    */
    static GGEMSProfilerManager& GetInstance(void)
    {
      static GGEMSProfilerManager instance;
      return instance;
    }

    /*!
      \fn GGEMSProfilerManager(GGEMSProfilerManager const& profiler_manager) = delete
      \param profiler_manager - reference on the profiler manager
      \brief Avoid copy of the class by reference
    */
    GGEMSProfilerManager(GGEMSProfilerManager const& profiler_manager) = delete;

    /*!
      \fn GGEMSProfilerManager& operator=(GGEMSProfilerManager const& profiler_manager) = delete
      \param profiler_manager - reference on the profiler manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSProfilerManager& operator=(GGEMSProfilerManager const& profiler_manager) = delete;

    /*!
      \fn GGEMSProfilerManager(GGEMSProfilerManager const&& profiler_manager) = delete
      \param profiler_manager - rvalue reference on the profiler manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSProfilerManager(GGEMSProfilerManager const&& profiler_manager) = delete;

    /*!
      \fn GGEMSProfilerManager& operator=(GGEMSProfilerManager const&& profiler_manager) = delete
      \param profiler_manager - rvalue reference on the profiler manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSProfilerManager& operator=(GGEMSProfilerManager const&& profiler_manager) = delete;

    /*!
      \fn void HandleEvent(cl::Event const& event, std::string const& profile_name)
      \param event - OpenCL event
      \param profile_name - type of profile
      \brief handle an OpenCL event in profile_name type
    */
    void HandleEvent(cl::Event const& event, std::string const& profile_name);

  private:
    ProfilerUMap profilers_; /*!< Map storing all types of profiles */
};

#endif // End of GUARD_GGEMS_TOOLS_GGEMSPROFILERMANAGER_HH
