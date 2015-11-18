// GGEMS Copyright (C) 2015

/*!
 * \file base_object.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 18 novembre 2015
 *
 *
 *
 */

#ifndef BASE_OBJECT_CUH
#define BASE_OBJECT_CUH

#include "global.cuh"

// Class that define the base of every object in GGEMS
class BaseObject {
    public:
        BaseObject();

        void set_material(std::string mat_name);
        void set_name(std::string obj_name);
        //void set_color(f32 r, f32 g, f32 b);
        //void set_transparency(f32 val);
        //void set_wireframe(bool val);
        //void set_sensitive(bool val);

        // Bounding box
        f32 xmin, xmax, ymin, ymax, zmin, zmax;

        // Viewing
        //Color color;
        //f32 transparency;
        //bool wireframe;

        // Property
        std::string material_name;
        std::string object_name;

    private:
};

#endif
