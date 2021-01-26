#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSDOSIMETRYCALCULATOR_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSDOSIMETRYCALCULATOR_HH

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
  \file GGEMSDosimetryCalculator.hh

  \brief Class providing tools storing and computing dose in phantom

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Wednesday January 13, 2021
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/navigators/GGEMSDoseRecording.hh"

class GGEMSNavigator;

/*!
  \class GGEMSDosimetryCalculator
  \brief Class providing tools storing and computing dose in phantom
*/
class GGEMS_EXPORT GGEMSDosimetryCalculator
{
  public:
    /*!
      \param navigator_name - name of the navigator associated to dosimetry
      \brief GGEMSDosimetryCalculator constructor
    */
    explicit GGEMSDosimetryCalculator(std::string const& navigator_name);

    /*!
      \brief GGEMSDosimetryCalculator destructor
    */
    ~GGEMSDosimetryCalculator(void);

    /*!
      \fn GGEMSDosimetryCalculator(GGEMSDosimetryCalculator const& dose_calculator) = delete
      \param dose_calculator - reference on the GGEMS dose calculator
      \brief Avoid copy by reference
    */
    GGEMSDosimetryCalculator(GGEMSDosimetryCalculator const& dose_calculator) = delete;

    /*!
      \fn GGEMSDosimetryCalculator& operator=(GGEMSDosimetryCalculator const& dose_calculator) = delete
      \param dose_calculator - reference on the GGEMS dose calculator
      \brief Avoid assignement by reference
    */
    GGEMSDosimetryCalculator& operator=(GGEMSDosimetryCalculator const& dose_calculator) = delete;

    /*!
      \fn GGEMSDosimetryCalculator(GGEMSDosimetryCalculator const&& dose_calculator) = delete
      \param dose_calculator - rvalue reference on the GGEMS dose calculator
      \brief Avoid copy by rvalue reference
    */
    GGEMSDosimetryCalculator(GGEMSDosimetryCalculator const&& dose_calculator) = delete;

    /*!
      \fn GGEMSDosimetryCalculator& operator=(GGEMSDosimetryCalculator const&& dose_calculator) = delete
      \param dose_calculator - rvalue reference on the GGEMS dose calculator
      \brief Avoid copy by rvalue reference
    */
    GGEMSDosimetryCalculator& operator=(GGEMSDosimetryCalculator const&& dose_calculator) = delete;

    /*!
      \fn void Initialize(void)
      \brief Initialize dosimetry calculator class
    */
    void Initialize(void);

    /*!
      \fn void SetDoselSizes(float const& dosel_x, float const& dosel_y, float const& dosel_z, std::string const& unit = "mm")
      \param dosel_x - size of dosel in X global axis
      \param dosel_y - size of dosel in Y global axis
      \param dosel_z - size of dosel in Z global axis
      \param unit - unit of the distance
      \brief set size of dosels
    */
    void SetDoselSizes(float const& dosel_x, float const& dosel_y, float const& dosel_z, std::string const& unit = "mm");

    /*!
      \fn void SetOutputDosimetryFilename(std::string const& output_filename)
      \param output_filename - name of output dosimetry file storing dosimetry
      \brief set output filename storing dosimetry
    */
    void SetOutputDosimetryFilename(std::string const& output_filename);

    /*!
      \fn void SetPhotonTracking(bool const& is_activated)
      \param is_activated - boolean activating photon tracking
      \brief activating photon tracking during dosimetry mode
    */
    void SetPhotonTracking(bool const& is_activated);

    /*!
      \fn void SetEdep(bool const& is_activated)
      \param is_activated - boolean activating energy deposit registration
      \brief activating energy deposit registration during dosimetry mode
    */
    void SetEdep(bool const& is_activated);

    /*!
      \fn void SetHitTracking(bool const& is_activated)
      \param is_activated - boolean activating hit tracking
      \brief activating hit tracking during dosimetry mode
    */
    void SetHitTracking(bool const& is_activated);

    /*!
      \fn void SetEdepSquared(bool const& is_activated)
      \param is_activated - boolean activating energy squared deposit registration
      \brief activating energy squared deposit registration during dosimetry mode
    */
    void SetEdepSquared(bool const& is_activated);

    /*!
      \fn void SetUncertainty(bool const& is_activated)
      \param is_activated - boolean activating uncertainty registration
      \brief activating uncertainty registration during dosimetry mode
    */
    void SetUncertainty(bool const& is_activated);

    /*!
      \fn inline std::shared_ptr<cl::Buffer> GetPhotonTrackingBuffer(void) const
      \return OpenCL buffer for photon tracking in dosimetry mode
      \brief get the buffer for photon tracking in dosimetry mode
    */
    inline std::shared_ptr<cl::Buffer> GetPhotonTrackingBuffer(void) const {return dose_recording_.photon_tracking_;}

    /*!
      \fn inline std::shared_ptr<cl::Buffer> GetHitTrackingBuffer(void) const
      \return OpenCL buffer for hit tracking in dosimetry mode
      \brief get the buffer for hit tracking in dosimetry mode
    */
    inline std::shared_ptr<cl::Buffer> GetHitTrackingBuffer(void) const {return dose_recording_.hit_;}

    /*!
      \fn inline std::shared_ptr<cl::Buffer> GetEdepBuffer(void) const
      \return OpenCL buffer for edep in dosimetry mode
      \brief get the buffer for edep in dosimetry mode
    */
    inline std::shared_ptr<cl::Buffer> GetEdepBuffer(void) const {return dose_recording_.edep_;}

    /*!
      \fn inline std::shared_ptr<cl::Buffer> GetEdepSquaredBuffer(void) const
      \return OpenCL buffer for edep squared in dosimetry mode
      \brief get the buffer for edep squared in dosimetry mode
    */
    inline std::shared_ptr<cl::Buffer> GetEdepSquaredBuffer(void) const {return dose_recording_.edep_squared_;}

    /*!
      \fn inline std::shared_ptr<cl::Buffer> GetDoseParams(void) const
      \return OpenCL buffer storing dosimetry params
      \brief get the buffer storing dosimetry params
    */
    inline std::shared_ptr<cl::Buffer> GetDoseParams(void) const {return dose_params_;}

    /*!
      \fn void ComputeDose(void)
      \brief compute dose in voxelized solid
    */
    void ComputeDoseAndSaveResults(void);

  private:
      /*!
        \fn void CheckParameters(void) const
        \return no returned value
      */
    void CheckParameters(void) const;

    /*!
      \fn void InitializeKernel(void)
      \brief Initialize kernel for dose computation
    */
    void InitializeKernel(void);

    /*!
      \fn void SavePhotonTracking(void) const
      \brief save photon tracking in dose map
    */
    void SavePhotonTracking(void) const;

    /*!
      \fn void SaveHit(void) const
      \brief save hits in dose map
    */
    void SaveHit(void) const;

    /*!
      \fn void SaveEdep(void) const
      \brief save energy deposit in dose map
    */
    void SaveEdep(void) const;

    /*!
      \fn void SaveDose(void) const
      \brief save dose in dose map
    */
    void SaveDose(void) const;

    /*!
      \fn void SaveEdepSquared(void) const
      \brief save energy deposit squared in dose map
    */
    void SaveEdepSquared(void) const;

  private:
    GGfloat3 dosel_sizes_; /*!< Sizes of dosel */
    std::string dosimetry_output_filename_; /*!< Output filename for dosimetry results */
    std::shared_ptr<GGEMSNavigator> navigator_; /*!< Navigator pointer associated to dosimetry object */

    // Buffer storing dose data on OpenCL device and host
    std::shared_ptr<cl::Buffer> dose_params_; /*!< Buffer storing dose parameters in OpenCL device */
    GGEMSDoseRecording dose_recording_; /*!< Structure storing dose data on OpenCL device */
    bool is_photon_tracking_; /*!< Boolean for photon tracking */
    bool is_edep_; /*!< Boolean for energy deposit */
    bool is_hit_tracking_; /*!< Boolean for hit tracking */
    bool is_edep_squared_; /*!< Boolean for energy squared deposit */
    bool is_uncertainty_; /*!< Boolean for uncertainty computation */

    std::weak_ptr<cl::Kernel> kernel_compute_dose_; /*!< OpenCL kernel computing dose in voxelized solid */
};

/*!
  \fn GGEMSDosimetryCalculator* create_ggems_dosimetry_calculator(char const* voxelized_phantom_name)
  \param voxelized_phantom_name - name of voxelized phantom
  \return the pointer on the dosimetry calculator
  \brief Get the GGEMSDosimetryCalculator pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSDosimetryCalculator* create_ggems_dosimetry_calculator(char const* voxelized_phantom_name);

/*!
  \fn void set_dosel_size_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, GGfloat const dose_x, GGfloat const dose_y, GGfloat const dose_z, char const* unit)
  \param dose_calculator - pointer on dose calculator
  \param dose_x - size of dosel in X global axis
  \param dose_y - size of dosel in Z global axis
  \param dose_z - size of dosel in Y global axis
  \param unit - unit of the distance
  \brief set size of dosels
*/
extern "C" GGEMS_EXPORT void set_dosel_size_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, GGfloat const dose_x, GGfloat const dose_y, GGfloat const dose_z, char const* unit);

/*!
  \fn void set_dose_output_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, char const* dose_output_filename)
  \param voxelized_phantom - pointer on dose calculator
  \param dose_output_filename - name of output dosimetry file storing dosimetry
  \brief set output filename storing dosimetry
*/
extern "C" GGEMS_EXPORT void set_dose_output_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, char const* dose_output_filename);

/*!
  \fn void dose_photon_tracking_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated)
  \param dose_calculator - pointer on dose calculator
  \param is_dosimetry_mode - boolean activating the photon tracking output
  \brief storing results about photon tracking
*/
extern "C" GGEMS_EXPORT void dose_photon_tracking_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated);

/*!
  \fn void dose_edep_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated)
  \param dose_calculator - pointer on dose calculator
  \param is_dosimetry_mode - boolean activating energy deposit output
  \brief storing results about energy deposit
*/
extern "C" GGEMS_EXPORT void dose_edep_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated);

/*!
  \fn void dose_hit_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated)
  \param dose_calculator - pointer on dose calculator
  \param is_dosimetry_mode - boolean activating the hit tracking output
  \brief storing results about hit tracking
*/
extern "C" GGEMS_EXPORT void dose_hit_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated);

/*!
  \fn void dose_edep_squared_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated)
  \param dose_calculator - pointer on dose calculator
  \param is_dosimetry_mode - boolean activating energy squared deposit output
  \brief storing results about energy squared deposit
*/
extern "C" GGEMS_EXPORT void dose_edep_squared_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated);

/*!
  \fn void dose_uncertainty_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated)
  \param dose_calculator - pointer on dose calculator
  \param is_dosimetry_mode - boolean activating uncertainty output
  \brief storing results about uncertainty
*/
extern "C" GGEMS_EXPORT void dose_uncertainty_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated);

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSDOSIMETRYCALCULATOR_HH
