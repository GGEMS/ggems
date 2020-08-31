#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSBOX_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSBOX_HH

/*!
  \file GGEMSBox.hh

  \brief Class GGEMSBox inheriting from GGEMSVolume handling Box solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday August 31, 2020
*/

#include "GGEMS/geometries/GGEMSVolume.hh"

/*!
  \class GGEMSBox
  \brief Class GGEMSBox inheriting from GGEMSVolume handling Box solid
*/
class GGEMS_EXPORT GGEMSBox : public GGEMSVolume
{
  public:
    /*!
      \brief GGEMSBox constructor
    */
    GGEMSBox(void);

    /*!
      \brief GGEMSBox destructor
    */
    ~GGEMSBox(void);

    /*!
      \fn GGEMSBox(GGEMSBox const& box) = delete
      \param box - reference on the box solid volume
      \brief Avoid copy of the class by reference
    */
    GGEMSBox(GGEMSBox const& box) = delete;

    /*!
      \fn GGEMSBox& operator=(GGEMSBox const& box) = delete
      \param box - reference on the box solid volume
      \brief Avoid assignement of the class by reference
    */
    GGEMSBox& operator=(GGEMSBox const& box) = delete;

    /*!
      \fn GGEMSBox(GGEMSBox const&& box) = delete
      \param box - rvalue reference on the box solid volume
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSBox(GGEMSBox const&& box) = delete;

    /*!
      \fn GGEMSBox& operator=(GGEMSBox const&& box) = delete
      \param box - rvalue reference on the box solid volume
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSBox& operator=(GGEMSBox const&& box) = delete;

    /*!
      \fn void SetHeight(GGfloat const& height, char const* unit = "mm")
      \param height - height of the box
      \param unit - unit of the distance
      \brief set the height of the box
    */
    void SetHeight(GGfloat const& height, char const* unit = "mm");

    /*!
      \fn void SetWidth(GGfloat const& width, char const* unit = "mm")
      \param width - width of the box
      \param unit - unit of the distance
      \brief set the width of the box
    */
    void SetWidth(GGfloat const& width, char const* unit = "mm");

    /*!
      \fn void SetDepth(GGfloat const& depth, char const* unit = "mm")
      \param depth - depth of the box
      \param unit - unit of the distance
      \brief set the depth of the box
    */
    void SetDepth(GGfloat const& depth, char const* unit = "mm");

    /*!
      \fn void Initialize(void) override
      \brief Initialize the solid and store it in Phantom creator manager
    */
    void Initialize(void) override;

    /*!
      \fn void Draw(void)
      \brief Draw analytical volume in voxelized phantom
    */
    void Draw(void) override;

  protected:
    /*!
      \fn void CheckParameters(void) const
      \brief check parameters for each type of volume
    */
    void CheckParameters(void) const override;

  private:
    GGfloat height_; /*!< Height of the box */
    GGfloat width_; /*!< Width of the box */
    GGfloat depth_; /*!< Depth of the box */
};

/*!
  \fn GGEMSBox* create_box(void)
  \return the pointer on the singleton
  \brief Create instance of GGEMSBox
*/
extern "C" GGEMS_EXPORT GGEMSBox* create_box(void);

/*!
  \fn GGEMSBox* delete_tube(GGEMSBox* box)
  \param box - pointer on the solid box
  \brief Delete instance of GGEMSBox
*/
extern "C" GGEMS_EXPORT void delete_box(GGEMSBox* box);

/*!
  \fn void set_height_box(GGEMSBox* box, GGfloat const height, char const* unit)
  \param box - pointer on the solid box
  \param height - height of the box
  \param unit - unit of the distance
  \brief Set the height of the box
*/
extern "C" GGEMS_EXPORT void set_height_box(GGEMSBox* box, GGfloat const height, char const* unit);

/*!
  \fn void set_width_box(GGEMSBox* box, GGfloat const width, char const* unit)
  \param box - pointer on the solid box
  \param width - width of the box
  \param unit - unit of the distance
  \brief Set the width of the box
*/
extern "C" GGEMS_EXPORT void set_width_box(GGEMSBox* box, GGfloat const width, char const* unit);

/*!
  \fn void set_depth_box(GGEMSBox* box, GGfloat const depth, char const* unit)
  \param box - pointer on the solid box
  \param depth - depth of the box
  \param unit - unit of the distance
  \brief Set the depth of the box
*/
extern "C" GGEMS_EXPORT void set_depth_box(GGEMSBox* box, GGfloat const depth, char const* unit);

/*!
  \fn void set_height_box(GGEMSBox* box, GGfloat const height, char const* unit)
  \param box - pointer on the solid box
  \param height - height of the box
  \param unit - unit of the distance
  \brief Set the height of the box
*/
extern "C" GGEMS_EXPORT void set_height_box(GGEMSBox* box, GGfloat const height, char const* unit);

/*!
  \fn void set_position_box(GGEMSBox* box, GGfloat const pos_x, GGfloat const pos_y, GGfloat const pos_z, char const* unit)
  \param box - pointer on the solid box
  \param pos_x - radius of the box
  \param pos_y - radius of the box
  \param pos_z - radius of the box
  \param unit - unit of the distance
  \brief Set the position of the box
*/
extern "C" GGEMS_EXPORT void set_position_box(GGEMSBox* box, GGfloat const pos_x, GGfloat const pos_y, GGfloat const pos_z, char const* unit);

/*!
  \fn void set_material_box(GGEMSBox* box, char const* material)
  \param box - pointer on the solid box
  \param material - material of the box
  \brief Set the material of the box
*/
extern "C" GGEMS_EXPORT void set_material_box(GGEMSBox* box, char const* material);

/*!
  \fn void set_label_value_box(GGEMSBox* box, GGfloat const label_value)
  \param box - pointer on the solid box
  \param label_value - label value in box
  \brief Set the label value in box
*/
extern "C" GGEMS_EXPORT void set_label_value_box(GGEMSBox* box, GGfloat const label_value);

/*!
  \fn void initialize_box(GGEMSBox* box)
  \param box - pointer on the solid box
  \brief Initialize the solid and store it in Phantom creator manager
*/
extern "C" GGEMS_EXPORT void initialize_box(GGEMSBox* box);

/*!
  \fn void draw_box(GGEMSBox* box)
  \param box - pointer on the solid box
  \brief Draw analytical volume in voxelized phantom
*/
extern "C" GGEMS_EXPORT void draw_box(GGEMSBox* box);

#endif // End of GUARD_GGEMS_GEOMETRY_GGEMSTUBE_HH
