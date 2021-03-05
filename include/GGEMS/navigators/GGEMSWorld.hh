#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSWORLD_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSWORLD_HH

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
  \file GGEMSWorld.hh

  \brief GGEMS class handling global world (space between navigators) in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 11, 2021
*/

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \struct GGEMSWorldRecording_t
  \brief Structure storing data for world data recording
*/
typedef struct GGEMSWorldRecording_t
{
  std::shared_ptr<cl::Buffer> edep_; /*!< Buffer storing energy deposit on OpenCL device */
  std::shared_ptr<cl::Buffer> photon_tracking_; /*!< Buffer storing photon tracking on OpenCL device */
  //std::shared_ptr<cl::Buffer> momentum_; /*!< Buffer storing dose in gray (Gy) */
} GGEMSWorldRecording; /*!< Using C convention name of struct to C++ (_t deletion) */

/*!
  \class GGEMSWorld
  \brief GGEMS class handling global world (space between navigators) in GGEMS
*/
class GGEMS_EXPORT GGEMSWorld
{
  public:
    /*!
      \brief GGEMSWorld constructor
    */
    GGEMSWorld(void);

    /*!
      \brief GGEMSWorld destructor
    */
    ~GGEMSWorld(void);

    /*!
      \fn GGEMSWorld(GGEMSWorld const& world) = delete
      \param world - reference on the GGEMS world
      \brief Avoid copy by reference
    */
    GGEMSWorld(GGEMSWorld const& world) = delete;

    /*!
      \fn GGEMSWorld& operator=(GGEMSWorld const& world) = delete
      \param world - reference on the GGEMS world
      \brief Avoid assignement by reference
    */
    GGEMSWorld& operator=(GGEMSWorld const& world) = delete;

    /*!
      \fn GGEMSWorld(GGEMSWorld const&& world) = delete
      \param world - rvalue reference on the GGEMS world
      \brief Avoid copy by rvalue reference
    */
    GGEMSWorld(GGEMSWorld const&& world) = delete;

    /*!
      \fn GGEMSWorld& operator=(GGEMSWorld const&& world) = delete
      \param world - rvalue reference on the GGEMS world
      \brief Avoid copy by rvalue reference
    */
    GGEMSWorld& operator=(GGEMSWorld const&& world) = delete;

    /*!
      \fn void SetOutputWorldBasename(std::string const& output_basename)
      \param output_basename - name of output world basename
      \brief set output basename storing world tracking
    */
    void SetOutputWorldBasename(std::string const& output_basename);

    /*!
      \fn void SetDimension(GGsize const& dimension_x, GGsize const& dimension_y, GGsize const& dimension_z)
      \param dimension_x - dimension in X
      \param dimension_y - dimension in Y
      \param dimension_z - dimension in Z
      \brief set the dimension of the world in X, Y and Z
    */
    void SetDimension(GGsize const& dimension_x, GGsize const& dimension_y, GGsize const& dimension_z);

    /*!
      \fn void SetElementSize(GGfloat const& size_x, GGfloat const& size_y, GGfloat const& size_z, std::string const& unit = "mm")
      \param size_x - size in X
      \param size_y - size in Y
      \param size_z - size in Z
      \param unit - unit of the distance
      \brief set the size of elements of the world in X, Y and Z
    */
    void SetElementSize(GGfloat const& size_x, GGfloat const& size_y, GGfloat const& size_z, std::string const& unit = "mm");

    /*!
      \fn void SetPhotonTracking(bool const& is_activated)
      \param is_activated - boolean activating photon tracking
      \brief activating photon tracking in world
    */
    void SetPhotonTracking(bool const& is_activated);

    /*!
      \fn void SetEdep(bool const& is_activated)
      \param is_activated - boolean activating energy deposit registration
      \brief activating energy deposit in world
    */
    void SetEdep(bool const& is_activated);

    /*!
      \fn void Initialize(void)
      \brief initialize and check parameters for world
    */
    void Initialize(void);

    /*!
      \fn void Tracking(void)
      \brief track particles through world
    */
    void Tracking(void);

    /*!
      \fn void SaveResults(void) const
      \brief save all results from world
    */
    void SaveResults(void) const;

  private:
    /*!
      \fn void CheckParameters(void) const
      \brief check parameters for world volume
    */
    void CheckParameters(void) const;

    /*!
      \fn void InitializeKernel(void)
      \brief Initialize kernel for world tracking
    */
    void InitializeKernel(void);

    /*!
      \fn void SavePhotonTracking(void) const
      \brief save photon tracking from world
    */
    void SavePhotonTracking(void) const;

    /*!
      \fn void SaveEdep(void) const
      \brief save energy deposit
    */
    void SaveEdep(void) const;

  private:
    std::string world_output_basename_; /*!< Output basename for world results */
    GGsize3 dimensions_; /*!< Dimensions of world */
    GGfloat3 sizes_; /*!< Sizes of elements in world */
    bool is_photon_tracking_; /*!< Boolean for photon tracking */
    bool is_edep_; /*!< Boolean for energy deposit */
    GGEMSWorldRecording world_recording_; /*!< Structure storing OpenCL pointer */
    std::weak_ptr<cl::Kernel> kernel_world_tracking_; /*!< OpenCL kernel computing world tracking */
};

/*!
  \fn GGEMSWorld* create_ggems_world(void)
  \return the pointer on the world
  \brief Get the GGEMSWorld pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSWorld* create_ggems_world(void);

/*!
  \fn void set_dimension_ggems_world(GGEMSWorld* world, GGsize const dimension_x, GGsize const dimension_y, GGsize const dimension_z)
  \param world - pointer on world volume
  \param dimension_x - dimension in X
  \param dimension_y - dimension in Y
  \param dimension_z - dimension in Z
  \brief set the dimenstions of the world in X, Y and Z
*/
extern "C" GGEMS_EXPORT void set_dimension_ggems_world(GGEMSWorld* world, GGsize const dimension_x, GGsize const dimension_y, GGsize const dimension_z);

/*!
  \fn void set_size_ggems_world(GGEMSWorld* world, GGfloat const size_x, GGfloat const size_y, GGfloat const size_z, char const* unit)
  \param world - pointer on world volume
  \param size_x - size of X elements of world
  \param size_y - size of Y elements of world
  \param size_z - size of Z elements of world
  \param unit - unit of the distance
  \brief set the element sizes of the world
*/
extern "C" GGEMS_EXPORT void set_size_ggems_world(GGEMSWorld* world, GGfloat const size_x, GGfloat const size_y, GGfloat const size_z, char const* unit);

/*!
  \fn void photon_tracking_ggems_world(GGEMSWorld* world, bool const is_activated)
  \param world - pointer on world volume
  \param is_activated - boolean activating the photon tracking output
  \brief storing results about photon tracking
*/
extern "C" GGEMS_EXPORT void photon_tracking_ggems_world(GGEMSWorld* world, bool const is_activated);

/*!
  \fn void edep_ggems_world(GGEMSWorld* world, bool const is_activated)
  \param world - pointer on world volume
  \param is_activated - boolean activating deposited energy tracking
  \brief storing results about deposited energy
*/
extern "C" GGEMS_EXPORT void edep_ggems_world(GGEMSWorld* world, bool const is_activated);

/*!
  \fn void set_output_ggems_world(GGEMSWorld* world, char const* world_output_basename)
  \param world - pointer on world
  \param world_output_basename - name of basename storing all results
  \brief set output basename storing world tracking results
*/
extern "C" GGEMS_EXPORT void set_output_ggems_world(GGEMSWorld* world, char const* world_output_basename);

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSWORLD_HH
