#ifndef GUARD_GGEMS_PHYSICS_GGEMSRANGECUTSMANAGER_HH
#define GUARD_GGEMS_PHYSICS_GGEMSRANGECUTSMANAGER_HH

/*!
  \file GGEMSRangeCutsManager.hh

  \brief GGEMS class managing the range cuts in GGEMS simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Friday March 6, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include <unordered_map>

#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/global/GGEMSExport.hh"

typedef std::unordered_map<std::string, GGfloat> RangeUMap;

/*!
  \class GGEMSRangeCutsManager
  \brief GGEMS class managing the range cuts in GGEMS simulation
*/
class GGEMS_EXPORT GGEMSRangeCutsManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGEMSRangeCutsManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGEMSRangeCutsManager(void);

  public:
    /*!
      \fn static GGEMSRangeCutsManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSRangeCutsManager
    */
    static GGEMSRangeCutsManager& GetInstance(void)
    {
      static GGEMSRangeCutsManager instance;
      return instance;
    }

    /*!
      \fn GGEMSRangeCutsManager(GGEMSRangeCutsManager const& range_cuts_manager) = delete
      \param range_cuts_manager - reference on the range cuts manager
      \brief Avoid copy of the class by reference
    */
    GGEMSRangeCutsManager(GGEMSRangeCutsManager const& range_cuts_manager) = delete;

    /*!
      \fn GGEMSRangeCutsManager& operator=(GGEMSRangeCutsManager const& range_cuts_manager) = delete
      \param range_cuts_manager - reference on the range cuts manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSRangeCutsManager& operator=(GGEMSRangeCutsManager const& range_cuts_manager) = delete;

    /*!
      \fn GGEMSRangeCutsManager(GGEMSRangeCutsManager const&& range_cuts_manager) = delete
      \param range_cuts_manager - rvalue reference on the range cuts manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSRangeCutsManager(GGEMSRangeCutsManager const&& range_cuts_manager) = delete;

    /*!
      \fn GGEMSRangeCutsManager& operator=(GGEMSRangeCutsManager const&& range_cuts_manager) = delete
      \param range_cuts_manager - rvalue reference on the range cuts manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSRangeCutsManager& operator=(GGEMSRangeCutsManager const&& range_cuts_manager) = delete;

    /*!
      \fn void SetRangeCut(char const* phantom_name, char const* particle_name, GGfloat const& value, char const* unit)
      \param phantom_name - name of the phantom
      \param particle_name - name of the particle
      \param value - value of the cut
      \param unit - unit of the cut in length
      \brief set the range cut for a phantom and a particle
    */
    void SetRangeCut(char const* phantom_name, char const* particle_name, GGfloat const& value, char const* unit = "mm");

    /*!
      \fn void CheckRangeCuts(void)
      \brief check the cuts for all phantoms and particle
    */
    void CheckRangeCuts(void);

    /*!
      \fn void PrintInfos(void) const
      \brief print infos about range cut manager
    */
    void PrintInfos(void) const;

  private:
    RangeUMap photon_cuts_; /*!< Photon cut in length for each phantom */
    RangeUMap electron_cuts_; /*!< Electron cuts in length for each phantom */
    RangeUMap positron_cuts_; /*!< Positron cuts in length for each phantom */
};

/*!
  \fn GGEMSRangeCutsManager* get_instance_range_cuts_manager(void)
  \brief Get the GGEMSRangeCutsManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSRangeCutsManager* get_instance_range_cuts_manager(void);

/*!
  \fn void set_cut_range_cuts_manager(GGEMSRangeCutsManager* range_cut_manager, char const* phantom_name, char const* particle_name, GGfloat const value, char const* unit)
  \param range_cut_manager - pointer on the range cut manager
  \param phantom_name - name of the phantom
  \param particle_name - name of the particle
  \param value - value of the cut
  \brief set the range cut for a phantom and a particle
*/
extern "C" GGEMS_EXPORT void set_cut_range_cuts_manager(GGEMSRangeCutsManager* range_cut_manager, char const* phantom_name, char const* particle_name, GGfloat const value, char const* unit);

#endif // GUARD_GGEMS_PHYSICS_GGEMSRANGECUTSMANAGER_HH
