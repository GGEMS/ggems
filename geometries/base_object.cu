// GGEMS Copyright (C) 2015

/*!
 * \file base_object.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 18 novembre 2015
 *
 *
 *
 */

#ifndef BASE_OBJECT_CU
#define BASE_OBJECT_CU

#include "base_object.cuh"

BaseObject::BaseObject () {

    xmin = 0.0f;
    xmax = 0.0f;
    ymin = 0.0f;
    ymax = 0.0f;
    zmin = 0.0f;
    zmax = 0.0f;
    material_name = "";
    object_name = "";    

}

void BaseObject::set_material(std::string mat_name) {
    material_name = mat_name;
}

void BaseObject::set_name(std::string obj_name) {
    object_name = obj_name;
}

#endif
