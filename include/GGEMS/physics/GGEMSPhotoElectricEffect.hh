#ifndef GUARD_GGEMS_PHYSICS_GGEMSPHOTOELECTRICEFFECT_HH
#define GUARD_GGEMS_PHYSICS_GGEMSPHOTOELECTRICEFFECT_HH

/*!
  \file GGEMSPhotoElectricEffect.hh

  \brief Photoelectric Effect process using Sandia table

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday April 13, 2020
*/

#include "GGEMS/physics/GGEMSEMProcess.hh"

/*!
  \class GGEMSPhotoElectricEffect
  \brief Photoelectric Effect process using Sandia table
*/
class GGEMS_EXPORT GGEMSPhotoElectricEffect : public GGEMSEMProcess
{
  public:
    /*!
      \param primary_particle - type of primary particle (gamma)
      \param is_secondary - flag activating secondary (e- for Compton)
      \brief GGEMSPhotoElectricEffect constructor
    */
    GGEMSPhotoElectricEffect(std::string const& primary_particle = "gamma", bool const& is_secondary = false);

    /*!
      \brief GGEMSPhotoElectricEffect destructor
    */
    ~GGEMSPhotoElectricEffect(void);

    /*!
      \fn GGEMSPhotoElectricEffect(GGEMSPhotoElectricEffect const& photoelectric_effect) = delete
      \param photoelectric_effect - reference on the GGEMS photoelectric effect
      \brief Avoid copy by reference
    */
    GGEMSPhotoElectricEffect(GGEMSPhotoElectricEffect const& photoelectric_effect) = delete;

    /*!
      \fn GGEMSPhotoElectricEffect& operator=(GGEMSPhotoElectricEffect const& photoelectric_effect) = delete
      \param photoelectric_effect - reference on the GGEMS photoelectric effect
      \brief Avoid assignement by reference
    */
    GGEMSPhotoElectricEffect& operator=(GGEMSPhotoElectricEffect const& photoelectric_effect) = delete;

    /*!
      \fn GGEMSPhotoElectricEffect(GGEMSPhotoElectricEffect const&& photoelectric_effect) = delete
      \param photoelectric_effect - rvalue reference on the GGEMS photoelectric effect
      \brief Avoid copy by rvalue reference
    */
    GGEMSPhotoElectricEffect(GGEMSPhotoElectricEffect const&& photoelectric_effect) = delete;

    /*!
      \fn GGEMSPhotoElectricEffect& operator=(GGEMSPhotoElectricEffect const&& photoelectric_effect) = delete
      \param photoelectric_effect - rvalue reference on the GGEMS photoelectric effect
      \brief Avoid copy by rvalue reference
    */
    GGEMSPhotoElectricEffect& operator=(GGEMSPhotoElectricEffect const&& photoelectric_effect) = delete;

    private:
    /*!
      \fn GGfloat ComputeCrossSectionPerAtom(GGfloat const& energy, GGuchar const& atomic_number) const
      \param energy - energy of the bin
      \param atomic_number - Z number of the chemical element
      \return Compton cross section by atom
      \brief compute Compton cross section for an atom with Klein-Nishina
    */
    GGfloat ComputeCrossSectionPerAtom(GGfloat const& energy, GGuchar const& atomic_number) const override;
};

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSPHOTOELECTRICEFFECT_HH
