#ifndef GUARD_GGEMS_PHYSICS_GGEMSCROSSSECTIONS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSCROSSSECTIONS_HH

/*!
  \file GGEMSCrossSections.hh

  \brief GGEMS class handling the cross sections tables

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 31, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include <vector>
#include <string>

#include "GGEMS/global/GGEMSOpenCLManager.hh"

class GGEMSEMProcess;
class GGEMSMaterials;
class GGEMSProcessesManager;

typedef std::vector<std::shared_ptr<GGEMSEMProcess>> GGEMSEMProcessesList; /*!< vector of pointer storing physical processes */

/*!
  \class GGEMSCrossSections
  \brief GGEMS class handling the cross sections tables
*/
class GGEMS_EXPORT GGEMSCrossSections
{
  public:
    /*!
      \brief GGEMSCrossSections constructor
    */
    GGEMSCrossSections(void);

    /*!
      \brief GGEMSCrossSections destructor
    */
    ~GGEMSCrossSections(void);

    /*!
      \fn GGEMSCrossSections(GGEMSCrossSections const& cross_sections) = delete
      \param cross_sections - reference on the GGEMS cross sections
      \brief Avoid copy by reference
    */
    GGEMSCrossSections(GGEMSCrossSections const& cross_sections) = delete;

    /*!
      \fn GGEMSCrossSections& operator=(GGEMSCrossSections const& cross_sections) = delete
      \param cross_sections - reference on the GGEMS cross sections
      \brief Avoid assignement by reference
    */
    GGEMSCrossSections& operator=(GGEMSCrossSections const& cross_sections) = delete;

    /*!
      \fn GGEMSCrossSections(GGEMSCrossSections const&& cross_sections) = delete
      \param cross_sections - rvalue reference on the GGEMS cross sections
      \brief Avoid copy by rvalue reference
    */
    GGEMSCrossSections(GGEMSCrossSections const&& cross_sections) = delete;

    /*!
      \fn GGEMSCrossSections& operator=(GGEMSCrossSections const&& cross_sections) = delete
      \param cross_sections - rvalue reference on the GGEMS cross sections
      \brief Avoid copy by rvalue reference
    */
    GGEMSCrossSections& operator=(GGEMSCrossSections const&& cross_sections) = delete;

    /*!
      \fn void AddProcess(std::string const& process_name, std::string const& particle_name)
      \param process_name - name of the process
      \param particle_name - name of the particle
      \brief add a process to the GGEMS simulation
    */
    void AddProcess(std::string const& process_name, std::string const& particle_name);

    /*!
      \fn void Initialize(std::shared_ptr<GGEMSMaterials> const materials)
      \param materials - activated materials for a specific phantom
      \brief Initialize all the activated processes computing tables on OpenCL device
    */
    void Initialize(std::shared_ptr<GGEMSMaterials> const materials);

    /*!
      \fn inline GGEMSEMProcessesList GetProcessesList(void) const
      \return pointer to process list
      \brief get the pointer on activated process
    */
    inline GGEMSEMProcessesList GetProcessesList(void) const {return em_processes_list_;}

  private:
    GGEMSEMProcessesList em_processes_list_; /*!< vector of electromagnetic processes */
    std::vector<bool> is_process_activated_; /*!< Boolean checking if the process is already activated */
    std::shared_ptr<cl::Buffer> particle_cross_sections_; /*!< Pointer storing cross sections for each particles */
    GGEMSOpenCLManager& opencl_manager_; /*!< Reference to OpenCL manager singleton */
    GGEMSProcessesManager& process_manager_; /*!< Reference to process manager */
};

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSCROSSSECTIONS_HH
