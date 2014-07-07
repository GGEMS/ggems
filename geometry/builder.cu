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

#ifndef BUILDER_CU
#define BUILDER_CU

#include "builder.h"

///////// BVH class ////////////////////////////////////////////////////

BVH::BVH() {
    // Add the root
    ptr_nodes.push_back(0);
    size_of_nodes.push_back(0);
    mother_node.push_back(0);
    cur_node_id = 0;
}

// Update the three address
void BVH::update_address() {
    ptr_nodes[0] = 0;
    unsigned int i=1;
    while (i < ptr_nodes.size()) {
        ptr_nodes[i] = ptr_nodes[i-1] + size_of_nodes[i-1];
        ++i;
    }
}

// Add a node
void BVH::add_node(unsigned int mother_id) {
    // New node ID
    cur_node_id++;

    // Insert this object into the three
    child_nodes.insert(child_nodes.begin() + ptr_nodes[mother_id] + size_of_nodes[mother_id],
                       cur_node_id);

    // Update the three
    size_of_nodes[mother_id]++;
    size_of_nodes.push_back(0);
    ptr_nodes.push_back(cur_node_id);
    mother_node.push_back(mother_id);

    // Update three address
    update_address();
}

// Print the BVH
void BVH::print() {
    // print each node
    unsigned int i = 0;
    unsigned int j = 0;
    while (i < size_of_nodes.size()) {
        printf("(%i)--[%i]--(", mother_node[i], i);
        j=0; while (j < size_of_nodes[i]) {
            printf("%i,", child_nodes[ptr_nodes[i]+j]);
            ++j;
        }
        printf(")\n");
        ++i;
    }
}

///////// BVH class ////////////////////////////////////////////////////

GeometryBuilder::GeometryBuilder() {};

unsigned int GeometryBuilder::add_aabb(AABB obj, unsigned int mother_ID) {

    return 1;

}


#endif
