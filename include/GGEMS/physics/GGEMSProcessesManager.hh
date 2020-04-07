#ifndef GUARD_GGEMS_PHYSICS_GGEMSPROCESSESMANAGER_HH
#define GUARD_GGEMS_PHYSICS_GGEMSPROCESSESMANAGER_HH

/*!
  \file GGEMSProcessesManager.hh

  \brief GGEMS class managing the processes in GGEMS simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday March 9, 2020
*/

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

/*!
  \namespace GGEMSProcessParams
  \brief Namespace storing the default parameters
*/
#ifndef OPENCL_COMPILER
namespace GGEMSProcessParams
{
#endif
  __constant GGfloat KINETIC_ENERGY_MIN = 1.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::eV; /*!< Min kinetic energy */
  #else
  1.e-6f; /*!< Min kinetic energy */
  #endif

  __constant GGushort CROSS_SECTION_TABLE_NUMBER_BINS = 220; /*!< Number of bins in the cross section table */
  __constant GGfloat CROSS_SECTION_TABLE_ENERGY_MIN = 990.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::eV; /*!< Min energy in the cross section table */
  #else
  1.e-6f; /*!< Min energy in the cross section table */
  #endif

  __constant GGfloat CROSS_SECTION_TABLE_ENERGY_MAX = 250.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::MeV; /*!< Max energy in the cross section table */
  #else
  1.f; /*!< Max energy in the cross section table */
  #endif

  __constant GGfloat PHOTON_DISTANCE_CUT = 1.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::um; /*!< Photon cut */
  #else
  1.e-3f; /*!< Photon cut */
  #endif

  __constant GGfloat ELECTRON_DISTANCE_CUT = 1.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::um; /*!< Electron cut */
  #else
  1.e-3f; /*!< Electron cut */
  #endif

  __constant GGfloat POSITRON_DISTANCE_CUT = 1.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::um; /*!< Positron cut */
  #else
  1.e-3f; /*!< Positron cut */
  #endif
#ifndef OPENCL_COMPILER
}
#endif

/*!
  \class GGEMSProcessesManager
  \brief GGEMS class managing the processes in GGEMS simulation
*/
class GGEMS_EXPORT GGEMSProcessesManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGEMSProcessesManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGEMSProcessesManager(void);

  public:
    /*!
      \fn static GGEMSProcessesManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSProcessesManager
    */
    static GGEMSProcessesManager& GetInstance(void)
    {
      static GGEMSProcessesManager instance;
      return instance;
    }

    /*!
      \fn GGEMSProcessesManager(GGEMSProcessesManager const& processes_manager) = delete
      \param processes_manager - reference on the processes manager
      \brief Avoid copy of the class by reference
    */
    GGEMSProcessesManager(GGEMSProcessesManager const& processes_manager) = delete;

    /*!
      \fn GGEMSProcessesManager& operator=(GGEMSProcessesManager const& processes_manager) = delete
      \param processes_manager - reference on the processes manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSProcessesManager& operator=(GGEMSProcessesManager const& processes_manager) = delete;

    /*!
      \fn GGEMSProcessesManager(GGEMSProcessesManager const&& processes_manager) = delete
      \param processes_manager - rvalue reference on the processes manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSProcessesManager(GGEMSProcessesManager const&& processes_manager) = delete;

    /*!
      \fn GGEMSProcessesManager& operator=(GGEMSProcessesManager const&& processes_manager) = delete
      \param processes_manager - rvalue reference on the processes manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSProcessesManager& operator=(GGEMSProcessesManager const&& processes_manager) = delete;

    /*!
      \fn void SetCrossSectionTableNumberOfBins(GGushort const& number_of_bins)
      \param number_of_bins - number of bins in cross section table
      \brief set the number of bins in the cross section table
    */
    void SetCrossSectionTableNumberOfBins(GGushort const& number_of_bins);

    /*!
      \fn void SetCrossSectionTableMinimumEnergy(GGfloat const& energy, char const* unit = "keV")
      \param energy - minimum energy in the cross section table
      \param unit - unit of energy
      \brief set the minimum energy in the cross section table
    */
    void SetCrossSectionTableMinimumEnergy(GGfloat const& energy, char const* unit = "keV");

    /*!
      \fn void SetCrossSectionTableMaximumEnergy(GGfloat const& energy, char const* unit = "keV")
      \param energy - maximum energy in the cross section table
      \param unit - unit of energy
      \brief set the maximum energy in the cross section table
    */
    void SetCrossSectionTableMaximumEnergy(GGfloat const& energy, char const* unit = "keV");

    /*!
      \fn inline GGfloat GetCrossSectionTableMinEnergy(void) const
      \return the minimum energy in the cross section table
      \brief get the minimum energy in the cross section table
    */
    inline GGfloat GetCrossSectionTableMinEnergy(void) const {return cross_section_table_min_energy_;}

    /*!
      \fn inline GGfloat GetCrossSectionTableMaxEnergy(void) const
      \return the maximum energy in the cross section table
      \brief get the maximum energy in the cross section table
    */
    inline GGfloat GetCrossSectionTableMaxEnergy(void) const {return cross_section_table_max_energy_;}

    /*!
      \fn inline GGushort GetCrossSectionTableNumberOfBins(void) const
      \return the number of bins in the cross section table
      \brief get the number of bins in the cross section table
    */
    inline GGushort GetCrossSectionTableNumberOfBins(void) const {return cross_section_table_number_of_bins_;}

    /*!
      \fn void AddProcess(std::string const& process_name, std::string const& particle_name, std::string const& phantom_name)
      \param process_name - Name of the process
      \param particle_name - Name of the particle
      \param phantom_name - Name of the phantom
      \brief add a process for a specific phantom or all the phantom
    */
    void AddProcess(std::string const& process_name, std::string const& particle_name, std::string const& phantom_name);

    /*!
      \fn void AddProcessRAM(GGulong const& size)
      \param size - allocated RAM for processes in GGEMS
      \brief add RAM memory size for processes
    */
    void AddProcessRAM(GGulong const& size);

    /*!
      \fn void PrintInfos(void) const
      \brief Print all infos about processes
    */
    void PrintInfos(void) const;

    /*!
      \fn void PrintAllocatedRAM(void) const
      \brief Print allocated RAM for processes
    */
    void PrintAllocatedRAM(void) const;

    /*!
      \fn void PrintAvailableProcesses(void) const
      \brief Print all infos about available processes
    */
    void PrintAvailableProcesses(void) const;

  private:
    GGushort cross_section_table_number_of_bins_; /*!< Number of bins in the cross section table */
    GGfloat cross_section_table_min_energy_; /*!< Minimum energy in the cross section table */
    GGfloat cross_section_table_max_energy_; /*!< Maximum energy in the cross section table */
    GGulong allocated_RAM_for_processes_; /*!< Allocated RAM in bytes for processes in GGEMS */
};

/*!
  \fn GGEMSProcessesManager* get_instance_processes_manager(void)
  \return the pointer on the singleton
  \brief Get the GGEMSProcessesManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSProcessesManager* get_instance_processes_manager(void);

/*!
  \fn void add_process_processes_manager(GGEMSProcessesManager* processes_manager, char const* process_name, char const* particle_name, char const* phantom_name)
  \param processes_manager - pointer on the processes manager
  \param process_name - Name of the process
  \param particle_name - Name of the particle
  \param phantom_name - Name of the phantom
  \brief add a process for a specific phantom or all the phantom
*/
extern "C" GGEMS_EXPORT void add_process_processes_manager(GGEMSProcessesManager* processes_manager, char const* process_name, char const* particle_name, char const* phantom_name);

/*!
  \fn void set_cross_section_table_number_of_bins_processes_manager(GGEMSProcessesManager* processes_manager, GGushort const number_of_bins)
  \param processes_manager - pointer on the processes manager
  \param number_of_bins - number of the bins for the cross section table
  \brief set the number of the bins in the cross section table
*/
extern "C" GGEMS_EXPORT void set_cross_section_table_number_of_bins_processes_manager(GGEMSProcessesManager* processes_manager, GGushort const number_of_bins);

/*!
  \fn void set_cross_section_table_minimum_energy_processes_manager(GGEMSProcessesManager* processes_manager, GGfloat const energy, char const* unit)
  \param processes_manager - pointer on the processes manager
  \param energy - minimum energy
  \param unit - unit of energy
  \brief set the minimum energy in the cross section table
*/
extern "C" GGEMS_EXPORT void set_cross_section_table_minimum_energy_processes_manager(GGEMSProcessesManager* processes_manager, GGfloat const energy, char const* unit);

/*!
  \fn void set_cross_section_table_maximum_energy_processes_manager(GGEMSProcessesManager* processes_manager, GGfloat const energy, char const* unit)
  \param processes_manager - pointer on the processes manager
  \param energy - maximum energy
  \param unit - unit of energy
  \brief set the maximum energy in the cross section table
*/
extern "C" GGEMS_EXPORT void set_cross_section_table_maximum_energy_processes_manager(GGEMSProcessesManager* processes_manager, GGfloat const energy, char const* unit);

/*!
  \fn void print_infos_processes_manager(GGEMSProcessesManager* processes_manager)
  \param processes_manager - pointer on the processes manager
  \brief print infos about processes
*/
extern "C" GGEMS_EXPORT void print_infos_processes_manager(GGEMSProcessesManager* processes_manager);

/*!
  \fn void print_available_processes_manager(GGEMSProcessesManager* processes_manager)
  \param processes_manager - pointer on the processes manager
  \brief print infos about available processes
*/
extern "C" GGEMS_EXPORT void print_available_processes_manager(GGEMSProcessesManager* processes_manager);

#endif // GUARD_GGEMS_PHYSICS_GGEMSRANGECUTSMANAGER_HH
