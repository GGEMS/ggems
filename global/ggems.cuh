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

#ifndef GGEMS_CUH
#define GGEMS_CUH

#include "../geometry/materials.cuh"

#include "../detector/dosimetry.cuh"

#include "../geometry/aabb.cuh"
#include "../geometry/builder.cuh"

#include "../geometry/meshed.cuh"
#include "../geometry/sphere.cuh"
#include "../geometry/voxelized.cuh"

#include "global.cuh"

#include "../maths/fun.cuh"
#include "../maths/prng.cuh"
#include "../maths/raytracing.cuh"
#include "../maths/vector.cuh"

#include "../navigation/electron_navigator.cuh"
#include "../navigation/main_navigator.cuh"
#include "../navigation/photon_navigator.cuh"
#include "../navigation/proton_navigator.cuh"

#include "../processes/electron.cuh"
#include "../processes/photon.cuh"
#include "../processes/proton.cuh"
#include "../processes/structures.cuh"


#endif