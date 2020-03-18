#ifndef GUARD_GGEMS_PHYSICS_GGEMSRANGECUTS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSRANGECUTS_HH

/*!
  \file GGEMSRangeCuts.hh

  \brief GGEMS class storing and converting the cut in energy cut

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday March 18, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include <unordered_map>
#include <string>

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

typedef std::unordered_map<std::string, GGfloat> EnergyCutUMap;

/*!
  \class GGEMSRangeCuts
  \brief GGEMS class storing and converting the cut in energy cut
*/
class GGEMS_EXPORT GGEMSRangeCuts
{
  public:
    /*!
      \brief GGEMSRangeCuts constructor
    */
    GGEMSRangeCuts(void);

    /*!
      \brief GGEMSRangeCuts destructor
    */
    ~GGEMSRangeCuts(void);

    /*!
      \fn GGEMSRangeCuts(GGEMSRangeCuts const& range_cuts) = delete
      \param range_cuts - reference on the GGEMS range cuts
      \brief Avoid copy by reference
    */
    GGEMSRangeCuts(GGEMSRangeCuts const& range_cuts) = delete;

    /*!
      \fn GGEMSRangeCuts& operator=(GGEMSRangeCuts const& range_cuts) = delete
      \param range_cuts - reference on the GGEMS range cuts
      \brief Avoid assignement by reference
    */
    GGEMSRangeCuts& operator=(GGEMSRangeCuts const& range_cuts) = delete;

    /*!
      \fn GGEMSRangeCuts(GGEMSRangeCuts const&& range_cuts) = delete
      \param range_cuts - rvalue reference on the GGEMS range cuts
      \brief Avoid copy by rvalue reference
    */
    GGEMSRangeCuts(GGEMSRangeCuts const&& range_cuts) = delete;

    /*!
      \fn GGEMSRangeCuts& operator=(GGEMSRangeCuts const&& range_cuts) = delete
      \param range_cuts - rvalue reference on the GGEMS range cuts
      \brief Avoid copy by rvalue reference
    */
    GGEMSRangeCuts& operator=(GGEMSRangeCuts const&& range_cuts) = delete;

    /*!
      \fn void SetPhotonLengthCut(GGfloat const& cut)
      \param cut - cut in length (mm)
      \brief set the photon length cut by the range cut manager (allowed by friend class)
    */
    void SetPhotonLengthCut(GGfloat const& cut);

    /*!
      \fn void SetElectronLengthCut(GGfloat const& cut)
      \param cut - cut in length (mm)
      \brief set the electron length cut by the range cut manager (allowed by friend class)
    */
    void SetElectronLengthCut(GGfloat const& cut);

    /*!
      \fn inline GGfloat GetPhotonLengthCut(void) const
      \return the photon length cut in mm
      \brief get the photon length cut
    */
    inline GGfloat GetPhotonLengthCut(void) const {return length_cut_photon_;}

    /*!
      \fn inline GGfloat GetElectronLengthCut(void) const
      \return the electron length cut in mm
      \brief get the electron length cut
    */
    inline GGfloat GetElectronLengthCut(void) const {return length_cut_electron_;}

  private:
    GGfloat length_cut_photon_; /*!< Photon cut in length */
    GGfloat length_cut_electron_; /*!< Electron cut in length */
    EnergyCutUMap energy_cuts_photon_; /*!< List of energy cuts for photon a material */
    EnergyCutUMap energy_cuts_electron_; /*!< List of energy cuts for electron a material */
};

#endif // GUARD_GGEMS_PHYSICS_GGEMSRANGECUTS_HH
