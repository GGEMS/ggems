// This file is part of GGEMS
//
// GGEMS is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// GGEMS is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with GGEMS.  If not, see <http://www.gnu.org/licenses/>.
//
// GGEMS Copyright (C) 2013-2014 Julien Bert

#ifndef BUILDER_H
#define BUILDER_H

#include <vector>
#include "aabb.h"
#include "sphere.h"


class BVH {
    public:
        BVH();
        void add_node(unsigned int mother_id);
        void print();

    private:
        void update_address();

        std::vector<unsigned int> ptr_nodes;
        std::vector<unsigned int> size_of_nodes;
        std::vector<unsigned int> child_nodes;
        std::vector<unsigned int> mother_node;
        unsigned int cur_node_id;
};


class GeometryBuilder {
    public:
        GeometryBuilder();
        unsigned int add_aabb(AABB obj, unsigned int mother_id);
        unsigned int add_sphere(Sphere obj, unsigned int mother_id);

    private:
        BVH world_tree;
        unsigned int cur_id;


};

#endif
