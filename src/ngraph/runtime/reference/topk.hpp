/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <cmath>
#include <algorithm>

#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T, typename U>
            void topk(
                const T* arg, U* out_indices, T* out_values, const Shape& in_shape, const Shape& out_indices_shape, const Shape & outvalues_shape, size_t axis, size_t k, bool compute_max)
            {
                vector<size_t> in_strides = ngraph::row_major_strides(in_shape);
                size_t reduction_axes_stride = in_strides[axis];
                in_strides.erase(in_strides.begin() + axis);
                size_t in_index_axes_stride = in_strides[0];

                vector<size_t> out_strides = ngraph::row_major_strides(out_shape);
                size_t topk_axes_stride = out_strides[axis];
                out_strides.erase(out_strides.begin() + axis);
                size_t out_index_axes_stride = out_strides[0];

                        out[output_transform.index(output_coord)] =
                            static_cast<U>(input_coord[axis]);
            }
        }
    }
}
