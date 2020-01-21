#ifndef GUARD_GGEMS_GLOBAL_GGEMSMANAGER_HH
#define GUARD_GGEMS_GLOBAL_GGEMSMANAGER_HH

/*!
  \file GGEMSManager.hh

  \brief GGEMS class managing the GGEMS simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday September 30, 2019
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include <cstdint>
#include <string>
#include <vector>

#include "GGEMS/global/GGEMSExport.hh"

class GGEMSSourceManager;
class GGEMSOpenCLManager;

/*!
  \class GGEMSManager
  \brief GGEMS class managing the complete simulation
*/
class GGEMS_EXPORT GGEMSManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGEMSManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGEMSManager(void);

  public:
    /*!
      \fn static GGEMSManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSManager
    */
    static GGEMSManager& GetInstance(void)
    {
      static GGEMSManager instance;
      return instance;
    }

  private:
    /*!
      \fn GGEMSManager(GGEMSManager const& ggems_manager) = delete
      \param ggems_manager - reference on the ggems manager
      \brief Avoid copy of the class by reference
    */
    GGEMSManager(GGEMSManager const& ggems_manager) = delete;

    /*!
      \fn GGEMSManager& operator=(GGEMSManager const& ggems_manager) = delete
      \param ggems_manager - reference on the ggems manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSManager& operator=(GGEMSManager const& ggems_manager) = delete;

    /*!
      \fn GGEMSManager(GGEMSManager const&& ggems_manager) = delete
      \param ggems_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSManager(GGEMSManager const&& ggems_manager) = delete;

    /*!
      \fn GGEMSManager& operator=(GGEMSManager const&& ggems_manager) = delete
      \param ggems_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSManager& operator=(GGEMSManager const&& ggems_manager) = delete;

  public:
    /*!
      \fn void Initialize(void)
      \brief Initialization of the GGEMS simulation and check parameters
    */
    void Initialize(void);

    /*!
      \fn void Run(void)
      \brief run the GGEMS simulation
    */
    void Run(void);

  private:
    /*!
      \fn void CheckParameters()
      \brief check the mandatory parameters for the GGEMS simulation
    */
    void CheckParameters();

  public: // Seed
    /*!
      \fn void SetSeed(GGuint const& seed)
      \param seed - seed for the random generator
      \brief Set the seed of random for the simulation
    */
    void SetSeed(GGuint const& seed);

    /*!
      \fn inline GGuint GetSeed() const
      \return the seed given by the user or generated by GGEMS
      \brief Get the general seed for the simulation
    */
    inline GGuint GetSeed() const {return seed_;};

  private:
    /*!
      \fn GGuint GenerateSeed() const
      \return the seed computed by GGEMS
      \brief generate a seed by GGEMS and return it
    */
    GGuint GenerateSeed() const;

  public:
    /*!
      \fn inline std::string GetVersion() const
      \brief Get the version of GGEMS
    */
    inline std::string GetVersion() const {return version_;};

  public:
    /*!
      \fn inline GGulong GetNumberOfParticles() const
      \brief Get the number of simulated particles
      \return the number of simulated particles
    */
    //inline GGulong GetNumberOfParticles() const {return number_of_particles_;};

    /*!
      \fn inline GGuint GetNumberOfBatchs() const
      \brief Get the number of particles in batch
      \return the number of simulated particles in batch
    */
    //inline GGuint GetNumberOfBatchs() const
    //{
      //return static_cast<GGuint>(v_number_of_particles_in_batch_.size());
    //}

  public:
    /*!
      \fn void SetProcess(char const* process_name)
      \param process_name - name of the process to activate
      \brief activate a specific process
    */
    void SetProcess(char const* process_name);

    /*!
      \fn void SetParticleCut(char const* particle_name, GGdouble const& distance)
      \param particle_name - Name of the particle
      \param distance - Cut in distance
      \brief Set the cut in distance for a specific particle
    */
    void SetParticleCut(char const* particle_name, GGdouble const& distance);

    /*!
      \fn void SetParticleSecondary(char const* particle_name, GGuint const& level)
      \param particle_name - Name of the particle
      \param level - Level of the secondary particle
      \brief set the particle to activate to follow the secondaries with a specific level
    */
    void SetParticleSecondaryAndLevel(char const* particle_name,
      GGuint const& level);

    /*!
      \fn void SetGeometryTolerance(GGdouble const& distance)
      \param distance - geometry distance
      \brief Set the geometry tolerance in distance
    */
    void SetGeometryTolerance(GGdouble const& distance);

  private:
    /*!
      \fn void OrganizeParticlesInBatch
      \brief Organize the particles in batch
    */
    //void OrganizeParticlesInBatch(void);

    /*!
      \fn void CheckMemoryForParticles(void) const
      \brief Check the memory for particles and propose an optimized MAXIMUM_NUMBER of particles if necessary
    */
    //void CheckMemoryForParticles(void) const;

  public: // Cross section part
    /*!
      \fn void SetCrossSectionTableNumberOfBins(GGuint const& number_of_bins)
      \param number_of_bins - Number of bins in the cross section table
      \brief set the number of bins in the cross section table
    */
    void SetCrossSectionTableNumberOfBins(GGuint const& number_of_bins);

    /*!
      \fn void SetCrossSectionTableEnergyMin(GGdouble const& min_energy)
      \param min_energy - Min. energy in the cross section table
      \brief set min. energy in the cross section table
    */
    void SetCrossSectionTableEnergyMin(GGdouble const& min_energy);

    /*!
      \fn void SetCrossSectionTableEnergyMax(GGdouble const& max_energy)
      \param max_energy - Max. energy in the cross section table
      \brief set max. energy in the cross section table
    */
    void SetCrossSectionTableEnergyMax(GGdouble const& max_energy);

  private:
    /*!
      \fn void PrintInfos() const
      \brief print infos about the GGEMS simulation
    */
    void PrintInfos(void) const;

    /*!
      \fn void PrintBanner(void) const
      \brief Print GGEMS banner
    */
    void PrintBanner(void) const;

  private: // Global simulation parameters
    GGuint seed_; /*!< Seed for the random generator */
    std::string version_; /*!< Version of GGEMS */
    //std::vector<GGulong> v_number_of_particles_in_batch_; /*!< Number of particles in batch */
    std::vector<GGbool> v_physics_list_; /*!< Vector storing the activated physics list */
    std::vector<GGbool> v_secondaries_list_; /*!< Vector storing the secondaries list */
    GGdouble photon_distance_cut_; /*!< Photon distance cut */
    GGdouble electron_distance_cut_; /*!< Electron distance cut */
    GGdouble geometry_tolerance_; /*!< Geometry tolerance */
    GGuint photon_level_secondaries_; /*!< Level of the secondaries */
    GGuint electron_level_secondaries_; /*!< Level of the secondaries */
    GGuint cross_section_table_number_of_bins_; /*!< Number of bins in the cross section table */
    GGdouble cross_section_table_energy_min_; /*!< Min. energy for the cross section table */
    GGdouble cross_section_table_energy_max_; /*!< Max. energy for the cross section table */

  private: // Source management
    GGEMSSourceManager& source_manager_; /*!< Reference to source manager singleton */
    GGEMSOpenCLManager& opencl_manager_; /*!< Reference to opencl manager singleton */
};

/*!
  \fn GGEMSManager* get_instance_ggems_manager(void)
  \brief Get the GGEMSManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSManager* get_instance_ggems_manager(void);

/*!
  \fn void set_seed_ggems_manager(GGEMSManager* ggems_manager, uint32_t const seed)
  \param ggems_manager - pointer on the singleton
  \param seed - seed given by the user
  \brief Set the seed for the simulation
*/
extern "C" GGEMS_EXPORT void set_seed_ggems_manager(GGEMSManager* ggems_manager,
  GGuint const seed);

/*!
  \fn void initialize_ggems_manager(GGEMSManager* p_ggems_manager)
  \param p_ggems_manager - pointer on the singleton
  \brief Initialize GGEMS simulation
*/
extern "C" GGEMS_EXPORT void initialize_ggems_manager(
  GGEMSManager* p_ggems_manager);

/*!
  \fn void set_process(GGEMSManager* p_ggems_manager, std::string const process_name)
  \param p_ggems_manager - pointer on the singleton
  \param process_name - name of the process to activate
  \brief activate a specific process
*/
extern "C" GGEMS_EXPORT void set_process_ggems_manager(
  GGEMSManager* p_ggems_manager, char const* process_name);

/*!
  \fn void set_particle_cut_ggems_manager(GGEMSManager* p_ggems_manager, char const* particle_name, GGdouble const distance)
  \param p_ggems_manager - pointer on the singleton
  \param particle_name - name of the particle
  \param distance - cut in distance for the particle
  \brief set a cut in distance for a specific particle
*/
extern "C" GGEMS_EXPORT void set_particle_cut_ggems_manager(
  GGEMSManager* p_ggems_manager, char const* particle_name,
  GGdouble const distance);

/*!
  \fn void set_geometry_tolerance_ggems_manager(GGEMSManager* p_ggems_manager, GGdouble const distance)
  \param p_ggems_manager - pointer on the singleton
  \param distance - geometry distance tolerance
  \brief set the geometry distance tolerance
*/
extern "C" GGEMS_EXPORT void set_geometry_tolerance_ggems_manager(
  GGEMSManager* p_ggems_manager, GGdouble const distance);

/*!
  \fn void set_secondary_particle_and_level_ggems_manager(GGEMSManager* p_ggems_manager, char const* particle_name, GGuint const level)
  \param p_ggems_manager - pointer on the singleton
  \param particle_name - name of the particle
  \param level - level of the secondary
  \brief set the particle to follow secondary and set the level
*/
extern "C" GGEMS_EXPORT void set_secondary_particle_and_level_ggems_manager(
  GGEMSManager* p_ggems_manager, char const* particle_name, GGuint const level);

/*!
  \fn void set_cross_section_table_number_of_bins_ggems_manager(GGEMSManager* p_ggems_manager, GGuint const number_of_bins)
  \param p_ggems_manager - pointer on the singleton
  \param number_of_bins - number of the bins in the cross section table
  \brief set the number of bins in the cross section table
*/
extern "C" GGEMS_EXPORT
void set_cross_section_table_number_of_bins_ggems_manager(
  GGEMSManager* p_ggems_manager, GGuint const number_of_bins);

/*!
  \fn void set_cross_section_table_energy_min_ggems_manager(GGEMSManager* p_ggems_manager, GGdouble const min_energy)
  \param p_ggems_manager - pointer on the singleton
  \param min_energy - min. energy in the cross section table
  \brief set the min. energy in the cross section table
*/
extern "C" GGEMS_EXPORT void set_cross_section_table_energy_min_ggems_manager(
  GGEMSManager* p_ggems_manager, GGdouble const min_energy);

/*!
  \fn void set_cross_section_table_energy_max_ggems_manager(GGEMSManager* p_ggems_manager, GGdouble const min_energy)
  \param p_ggems_manager - pointer on the singleton
  \param max_energy - max. energy in the cross section table
  \brief set the max. energy in the cross section table
*/
extern "C" GGEMS_EXPORT void set_cross_section_table_energy_max_ggems_manager(
  GGEMSManager* p_ggems_manager, GGdouble const max_energy);

/*!
  \fn void run_ggems_manager(GGEMSManager* p_ggems_manager)
  \param ggems_manager - pointer on the singleton
  \brief Run the GGEMS simulation
*/
extern "C" GGEMS_EXPORT void run_ggems_manager(GGEMSManager* p_ggems_manager);

#endif // End of GUARD_GGEMS_GLOBAL_GGEMSMANAGER_HH
