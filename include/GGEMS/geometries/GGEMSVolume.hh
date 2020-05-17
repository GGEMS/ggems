#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSVOLUME_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSVOLUME_HH

/*!
  \file GGEMSVolume.hh

  \brief Mother class handle solid volume

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday January 13, 2020
*/

#include "GGEMS/geometries/GGEMSVolumeCreatorManager.hh"

/*!
  \class GGEMSVolume
  \brief Mother class handle volume
*/
class GGEMS_EXPORT GGEMSVolume
{
  public:
    /*!
      \brief GGEMSVolume constructor
    */
    GGEMSVolume(void);

    /*!
      \brief GGEMSVolume destructor
    */
    virtual ~GGEMSVolume(void);

    /*!
      \fn GGEMSVolume(GGEMSVolume const& volume) = delete
      \param volume - reference on the volume
      \brief Avoid copy of the class by reference
    */
    GGEMSVolume(GGEMSVolume const& volume) = delete;

    /*!
      \fn GGEMSVolume& operator=(GGEMSVolume const& volume) = delete
      \param volume - reference on the volume
      \brief Avoid assignement of the class by reference
    */
    GGEMSVolume& operator=(GGEMSVolume const& volume) = delete;

    /*!
      \fn GGEMSVolume(GGEMSVolume const&& volume) = delete
      \param volume - rvalue reference on the volume
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSVolume(GGEMSVolume const&& volume) = delete;

    /*!
      \fn GGEMSVolume& operator=(GGEMSVolume const&& volume) = delete
      \param volume - rvalue reference on the volume
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSVolume& operator=(GGEMSVolume const&& volume) = delete;

    /*!
      \fn void SetLabelValue(GGfloat const& label_value)
      \param label_value - label value in solid phantom
      \brief Set the label value
    */
    void SetLabelValue(GGfloat const& label_value);

    /*!
      \fn void SetPosition(GGfloat const& pos_x, GGfloat const& pos_y, GGfloat const& pos_z, char const* unit = "mm")
      \param pos_x - position of analytical phantom in X
      \param pos_y - position of analytical phantom in Y
      \param pos_z - position of analytical phantom in Z
      \param unit - unit of the distance
      \brief Set the solid phantom position
    */
    void SetPosition(GGfloat const& pos_x, GGfloat const& pos_y, GGfloat const& pos_z, char const* unit = "mm");

    /*!
      \fn void SetMaterial(char const* material)
      \param material - name of the material
      \brief set the material, Air by default
    */
    void SetMaterial(char const* material);

    /*!
      \fn void Initialize(void)
      \return no returned value
      \brief Initialize the solid and store it in Phantom creator manager
    */
    virtual void Initialize(void) = 0;

    /*!
      \fn void Draw(void)
      \return no returned value
      \brief Draw analytical volume in voxelized phantom
    */
    virtual void Draw(void) = 0;

  protected:
    /*!
      \fn void CheckParameters(void) const
      \return no returned value
      \brief check parameters for each type of volume
    */
    virtual void CheckParameters(void) const = 0;

  protected:
    GGfloat label_value_; /*!< Value of label in volume */
    GGfloat3 positions_; /*!< Position of volume */
    std::shared_ptr<cl::Kernel> kernel_draw_volume_; /*!< Kernel drawing solid using OpenCL */
};

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSVOLUME_HH
