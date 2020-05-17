#ifndef GUARD_GGEMS_TOOLS_GGEMSRAMMANAGER_HH
#define GUARD_GGEMS_TOOLS_GGEMSRAMMANAGER_HH

/*!
  \file GGEMSRAMManager.hh

  \brief GGEMS class handling RAM memory

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday May 5, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include <vector>

#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/global/GGEMSExport.hh"

/*!
  \enum GGEMSRAMType
  \brief define a type of RAM allocation
*/
enum GGEMSRAMType : GGuchar
{
  total = 0,
  material,
  geometry,
  process,
  particle,
  random,
  source
};

/*!
  \class GGEMSRAMManager
  \brief GGEMS class handling RAM memory
*/
class GGEMS_EXPORT GGEMSRAMManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGEMSRAMManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGEMSRAMManager(void);

  public:
    /*!
      \fn static GGEMSRAMManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSRAMManager
    */
    static GGEMSRAMManager& GetInstance(void)
    {
      static GGEMSRAMManager instance;
      return instance;
    }

    /*!
      \fn GGEMSRAMManager(GGEMSRAMManager const& ram_manager) = delete
      \param ram_manager - reference on the ram manager
      \brief Avoid copy of the class by reference
    */
    GGEMSRAMManager(GGEMSRAMManager const& ram_manager) = delete;

    /*!
      \fn GGEMSRAMManager& operator=(GGEMSRAMManager const& ram_manager) = delete
      \param ram_manager - reference on the ram manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSRAMManager& operator=(GGEMSRAMManager const& ram_manager) = delete;

    /*!
      \fn GGEMSRAMManager(GGEMSRAMManager const&& ram_manager) = delete
      \param ram_manager - rvalue reference on the ram manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSRAMManager(GGEMSRAMManager const&& ram_manager) = delete;

    /*!
      \fn GGEMSRAMManager& operator=(GGEMSRAMManager const&& ram_manager) = delete
      \param ram_manager - rvalue reference on the ram manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSRAMManager& operator=(GGEMSRAMManager const&& ram_manager) = delete;

    /*!
      \fn void PrintRAMStatus(void) const
      \brief print the RAM memory status for activated context
    */
    void PrintRAMStatus(void) const;

    /*!
      \fn void CheckRAMMemory(std::size_t const& size)
      \param size - size in bytes to allocate
      \brief Checking RAM memory allocation
    */
    void CheckRAMMemory(std::size_t const& size);

    /*!
      \fn void AddMaterialRAMMemory(GGulong const& size)
      \param size - size of the allocated buffer in byte
      \brief store the size of the material allocated buffer
    */
    void AddMaterialRAMMemory(GGulong const& size);

    /*!
      \fn void AddGeometryRAMMemory(GGulong const& size)
      \param size - size of the allocated buffer in byte
      \brief store the size of the geometry allocated buffer
    */
    void AddGeometryRAMMemory(GGulong const& size);

    /*!
      \fn void AddProcessRAMMemory(GGulong const& size)
      \param size - size of the allocated buffer in byte
      \brief store the size of the process allocated buffer
    */
    void AddProcessRAMMemory(GGulong const& size);

    /*!
      \fn void AddParticleRAMMemory(GGulong const& size)
      \param size - size of the allocated buffer in byte
      \brief store the size of the particle allocated buffer
    */
    void AddParticleRAMMemory(GGulong const& size);

    /*!
      \fn void AddRandomRAMMemory(GGulong const& size)
      \param size - size of the allocated buffer in byte
      \brief store the size of the random allocated buffer
    */
    void AddRandomRAMMemory(GGulong const& size);

    /*!
      \fn void AddSourceRAMMemory(GGulong const& size)
      \param size - size of the allocated buffer in byte
      \brief store the size of the source allocated buffer
    */
    void AddSourceRAMMemory(GGulong const& size);

  private:
    /*!
      \fn void AddTotalRAMMemory(GGulong const& size)
      \param size - size of the allocated buffer in byte
      \brief store the size of the global allocated buffer
    */
    void AddTotalRAMMemory(GGulong const& size);

  private:
    std::vector<GGulong> allocated_ram_; /*!< Allocated RAM on OpenCL device */
    std::vector<std::string> name_of_allocated_memory_; /*!< Name of allocated memory */
};

#endif // End of GUARD_GGEMS_TOOLS_GGEMSRAMMANAGER_HH
