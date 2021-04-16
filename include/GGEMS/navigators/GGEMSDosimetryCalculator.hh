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
      \brief GGEMSDosimetryCalculator constructor
    */
    GGEMSDosimetryCalculator(void);

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
      \fn void AttachToNavigator(std::string const& navigator_name)
      \param navigator_name - name of the navigator to attach
      \brief attach a navigator to dosimetry module
    */
    void AttachToNavigator(std::string const& navigator_name);

    /*!
      \fn void Initialize(void)
      \brief Initialize dosimetry calculator class
    */
    void Initialize(void);

    /*!
      \fn void SetDoselSizes(GGfloat const& dosel_x, GGfloat const& dosel_y, GGfloat const& dosel_z, std::string const& unit = "mm")
      \param dosel_x - size of dosel in X global axis
      \param dosel_y - size of dosel in Y global axis
      \param dosel_z - size of dosel in Z global axis
      \param unit - unit of the distance
      \brief set size of dosels
    */
    void SetDoselSizes(GGfloat const& dosel_x, GGfloat const& dosel_y, GGfloat const& dosel_z, std::string const& unit = "mm");

    /*!
      \fn void SetOutputDosimetryBasename(std::string const& output_filename)
      \param output_filename - name of output dosimetry basename storing dosimetry results
      \brief set output basename storing dosimetry
    */
    void SetOutputDosimetryBasename(std::string const& output_filename);

    /*!
      \fn void SetScaleFactor(GGfloat const& scale_factor)
      \param scale_factor - scale factor applied to dose value
      \brief set the scale factor applied to dose value
    */
    void SetScaleFactor(GGfloat const& scale_factor);

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
      \fn void SetWaterReference(bool const& is_activated)
      \param is_activated - boolean activating water reference
      \brief activating water reference during dose computation
    */
    void SetWaterReference(bool const& is_activated);

    /*!
      \fn void SetMinimumDensity(GGfloat const& minimum_density, std::string const& unit = "g/cm3")
      \param minimum_density - minimum of density
      \param unit - unit of the density
      \brief set minimum of density for dose computation
    */
    void SetMinimumDensity(GGfloat const& minimum_density, std::string const& unit = "g/cm3");

    /*!
      \fn inline cl::Buffer* GetPhotonTrackingBuffer(GGsize const& thread_index) const
      \param thread_index - index of activated device (thread index)
      \return OpenCL buffer for photon tracking in dosimetry mode
      \brief get the buffer for photon tracking in dosimetry mode
    */
    inline cl::Buffer* GetPhotonTrackingBuffer(GGsize const& thread_index) const {return dose_recording_.photon_tracking_[thread_index];}

    /*!
      \fn inline cl::Buffer* GetHitTrackingBuffer(GGsize const& thread_index) const
      \param thread_index - index of activated device (thread index)
      \return OpenCL buffer for hit tracking in dosimetry mode
      \brief get the buffer for hit tracking in dosimetry mode
    */
    inline cl::Buffer* GetHitTrackingBuffer(GGsize const& thread_index) const {return dose_recording_.hit_[thread_index];}

    /*!
      \fn inline cl::Buffer* GetEdepBuffer(void) const
      \param thread_index - index of activated device (thread index)
      \return OpenCL buffer for edep in dosimetry mode
      \brief get the buffer for edep in dosimetry mode
    */
    inline cl::Buffer* GetEdepBuffer(GGsize const& thread_index) const {return dose_recording_.edep_[thread_index];}

    /*!
      \fn inline cl::Buffer* GetEdepSquaredBuffer(GGsize const& thread_index) const
      \param thread_index - index of activated device (thread index)
      \return OpenCL buffer for edep squared in dosimetry mode
      \brief get the buffer for edep squared in dosimetry mode
    */
    inline cl::Buffer* GetEdepSquaredBuffer(GGsize const& thread_index) const {return dose_recording_.edep_squared_[thread_index];}

    /*!
      \fn inline cl::Buffer* GetDoseParams(GGsize const& thread_index) const
      \param thread_index - index of activated device (thread index)
      \return OpenCL buffer storing dosimetry params
      \brief get the buffer storing dosimetry params
    */
    inline cl::Buffer* GetDoseParams(GGsize const& thread_index) const {return dose_params_[thread_index];}

    /*!
      \fn void ComputeDose(GGsize const& thread_index)
      \param thread_index - index of activated device (thread index)
      \brief computing dose
    */
    void ComputeDose(GGsize const& thread_index);

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
      \brief save photon tracking
    */
    void SavePhotonTracking(void) const;

    /*!
      \fn void SaveHit(void) const
      \brief save hits in dose map
    */
    void SaveHit(void) const;

    /*!
      \fn void SaveEdep(void) const
      \brief save energy deposit
    */
    void SaveEdep(void) const;

    /*!
      \fn void SaveDose(void) const
      \brief save dose
    */
    void SaveDose(void) const;

    /*!
      \fn void SaveEdepSquared(void) const
      \brief save energy squared deposit
    */
    void SaveEdepSquared(void) const;

    /*!
      \fn void SaveUncertainty(void) const
      \brief save uncertainty values
    */
    void SaveUncertainty(void) const;

  private:
    GGfloat3 dosel_sizes_; /*!< Sizes of dosel */
    GGsize total_number_of_dosels_; /*!< Total number of dosels in image */
    std::string dosimetry_output_filename_; /*!< Output filename for dosimetry results */
    GGEMSNavigator* navigator_; /*!< Navigator pointer associated to dosimetry object */

    // Buffer storing dose data on OpenCL device and host
    cl::Buffer** dose_params_; /*!< Buffer storing dose parameters in OpenCL device */
    GGEMSDoseRecording dose_recording_; /*!< Structure storing dose data on OpenCL device */
    bool is_photon_tracking_; /*!< Boolean for photon tracking */
    bool is_edep_; /*!< Boolean for energy deposit */
    bool is_hit_tracking_; /*!< Boolean for hit tracking */
    bool is_edep_squared_; /*!< Boolean for energy squared deposit */
    bool is_uncertainty_; /*!< Boolean for uncertainty computation */
    GGfloat scale_factor_; /*!< Scale factor */
    GGchar is_water_reference_; /*!< Water reference for dose computation */
    GGfloat minimum_density_; /*!< Minimum density value for dose computation */

    cl::Kernel** kernel_compute_dose_; /*!< OpenCL kernel computing dose in voxelized solid */
    GGsize number_activated_devices_; /*!< Number of activated device */
};

/*!
  \fn GGEMSDosimetryCalculator* create_ggems_dosimetry_calculator(void)
  \return the pointer on the dosimetry calculator
  \brief Get the GGEMSDosimetryCalculator pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSDosimetryCalculator* create_ggems_dosimetry_calculator(void);

/*!
  \fn GGEMSTube* delete_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator)
  \param dose_calculator - pointer on dose calculator
  \brief Delete instance of GGEMSDosimetryCalculator
*/
extern "C" GGEMS_EXPORT void delete_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator);

/*!
  \fn void scale_factor_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, GGfloat const scale_factor)
  \param dose_calculator - pointer on dose calculator
  \param scale_factor - scale factor applied to dose value
  \brief set the scale factor applied to dose value
*/
extern "C" GGEMS_EXPORT void scale_factor_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, GGfloat const scale_factor);

/*!
  \fn void water_reference_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated)
  \param dose_calculator - pointer on dose calculator
  \param is_activated - boolean activating water reference mode for dose computation
  \brief set water reference mode
*/
extern "C" GGEMS_EXPORT void water_reference_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated);

/*!
  \fn void minimum_density_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, GGfloat const minimum_density, char const* unit)
  \param dose_calculator - pointer on dose calculator
  \param minimum_density - minimum of density
  \param unit - unit of the density
  \brief set minimum of density for dose computation
*/
extern "C" GGEMS_EXPORT void minimum_density_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, GGfloat const minimum_density, char const* unit);

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
  \param dose_calculator - pointer on dose calculator
  \param dose_output_filename - name of output dosimetry file storing dosimetry
  \brief set output filename storing dosimetry
*/
extern "C" GGEMS_EXPORT void set_dose_output_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, char const* dose_output_filename);

/*!
  \fn void dose_photon_tracking_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated)
  \param dose_calculator - pointer on dose calculator
  \param is_activated - boolean activating the photon tracking output
  \brief storing results about photon tracking
*/
extern "C" GGEMS_EXPORT void dose_photon_tracking_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated);

/*!
  \fn void dose_edep_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated)
  \param dose_calculator - pointer on dose calculator
  \param is_activated - boolean activating energy deposit output
  \brief storing results about energy deposit
*/
extern "C" GGEMS_EXPORT void dose_edep_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated);

/*!
  \fn void dose_hit_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated)
  \param dose_calculator - pointer on dose calculator
  \param is_activated - boolean activating the hit tracking output
  \brief storing results about hit tracking
*/
extern "C" GGEMS_EXPORT void dose_hit_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated);

/*!
  \fn void dose_edep_squared_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated)
  \param dose_calculator - pointer on dose calculator
  \param is_activated - boolean activating energy squared deposit output
  \brief storing results about energy squared deposit
*/
extern "C" GGEMS_EXPORT void dose_edep_squared_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated);

/*!
  \fn void dose_uncertainty_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated)
  \param dose_calculator - pointer on dose calculator
  \param is_activated - boolean activating uncertainty output
  \brief storing results about uncertainty
*/
extern "C" GGEMS_EXPORT void dose_uncertainty_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, bool const is_activated);

/*!
  \fn void attach_to_navigator_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, char const* navigator)
  \param dose_calculator - pointer on dose calculator
  \param navigator - name of the navigator to attach
  \brief attach dosimetry module to a navigator
*/
extern "C" GGEMS_EXPORT void attach_to_navigator_dosimetry_calculator(GGEMSDosimetryCalculator* dose_calculator, char const* navigator);

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSDOSIMETRYCALCULATOR_HH
