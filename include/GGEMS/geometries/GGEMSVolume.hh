#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSVOLUME_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSVOLUME_HH

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
      \fn void SetPosition(GGfloat const& pos_x, GGfloat const& pos_y, GGfloat const& pos_z, std::string const& unit = "mm")
      \param pos_x - position of analytical phantom in X
      \param pos_y - position of analytical phantom in Y
      \param pos_z - position of analytical phantom in Z
      \param unit - unit of the distance
      \brief Set the solid phantom position
    */
    void SetPosition(GGfloat const& pos_x, GGfloat const& pos_y, GGfloat const& pos_z, std::string const& unit = "mm");

    /*!
      \fn void SetMaterial(std::string const& material)
      \param material - name of the material
      \brief set the material, Air by default
    */
    void SetMaterial(std::string const& material);

    /*!
      \fn void Initialize(void)
      \brief Initialize the solid and store it in Phantom creator manager
    */
    virtual void Initialize(void) = 0;

    /*!
      \fn void Draw(void)
      \brief Draw analytical volume in voxelized phantom
    */
    virtual void Draw(void) = 0;

  protected:
    /*!
      \fn void CheckParameters(void) const
      \brief check parameters for each type of volume
    */
    virtual void CheckParameters(void) const = 0;

  protected:
    GGfloat label_value_; /*!< Value of label in volume */
    GGfloat3 positions_; /*!< Position of volume */
    std::weak_ptr<cl::Kernel> kernel_draw_volume_cl_; /*!< Kernel drawing solid using OpenCL */
};

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSVOLUME_HH
