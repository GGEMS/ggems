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
    GGEMSDosimetryCalculator();

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
      \fn void SetDoselSizes(GGfloat3 const& dosel_sizes)
      \param dosel_sizes - size of dosels in X, Y and Z in global axis
      \brief set size of dosels
    */
    void SetDoselSizes(GGfloat3 const& dosel_sizes);

    /*!
      \fn void SetOutputDosimetryFilename(std::string const& output_filename)
      \param output_filename - name of output dosimetry file storing dosimetry
      \brief set output filename storing dosimetry
    */
    void SetOutputDosimetryFilename(std::string const& output_filename);

    /*!
      \fn void SetNavigator(std::string const& navigator_name)
      \param navigator_name - name of navigator associated to dosimetry object
      \brief set navigator associated to dosimetry object
    */
    void SetNavigator(std::string const& navigator_name);

    /*!
      \fn inline std::shared_ptr<cl::Buffer> GetPhotonTrackingBuffer(void) const
      \return OpenCL buffer for photon tracking in dosimetry mode
      \brief get the buffer for photon tracking in dosimetry mode
    */
    inline std::shared_ptr<cl::Buffer> GetPhotonTrackingBuffer(void) const
    {
      return dose_recording_.photon_tracking_;
    }

    /*!
      \fn inline std::shared_ptr<cl::Buffer> GetDoseParams(void) const
      \return OpenCL buffer storing dosimetry params
      \brief get the buffer storing dosimetry params
    */
    inline std::shared_ptr<cl::Buffer> GetDoseParams(void) const
    {
      return dose_params_;
    }

  private:
      /*!
        \fn void CheckParameters(void) const
        \return no returned value
      */
    void CheckParameters(void) const;

  private:
    GGfloat3 dosel_sizes_; /*!< Sizes of dosel */
    std::string dosimetry_output_filename; /*!< Output filename for dosimetry results */
    std::shared_ptr<GGEMSNavigator> navigator_; /*!< Navigator pointer associated to dosimetry object */

    // Buffer storing dose data on OpenCL device and host
    std::shared_ptr<cl::Buffer> dose_params_; /*!< Buffer storing dose parameters in OpenCL device */
    GGEMSDoseRecording dose_recording_; /*!< Structure storing dose data on OpenCL device */
    std::vector<GGfloat> dose_values_; /*!< Buffer storing dose value */
    std::vector<GGfloat> uncertainty_values_; /*!< Buffer storing uncertainty values */
};

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSDOSIMETRYCALCULATOR_HH
