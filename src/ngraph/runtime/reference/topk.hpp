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
#include <numeric>

#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T, typename U>
            void topk(
                const T* arg, U* out_indices, T* out_values, const Shape& in_shape, const Shape& out_indices_shape, const Shape & out_values_shape, size_t axis, size_t k, bool compute_max)
            {
                // reorder source axis visit order and make "axis" inner most
                size_t ndim = static_cast<size_t>(in_shape.size());
                AxisVector axis_order(ndim);
                std::iota(axis_order.begin(), axis_order.end(), 0);
                axis_order.erase(axis_order.begin() + axis);
                axis_order.push_back(axis);
                std::vector<size_t> in_strides = ngraph::row_major_strides(in_shape);
                std::vector<size_t> out_indices_strides = ngraph::row_major_strides(out_indices_shape);
                std::vector<size_t> out_values_strides = ngraph::row_major_strides(out_values_shape);

                Coordinate start_corner(ndim, 0);
                std::vector<size_t> tmp_shape(ndim);
                for(size_t i = 0; i < ndim ; i++)
                {
                    tmp_shape[i] = in_shape[i];
                }
                tmp_shape[axis] = 0;
                Coordinate end_corner(tmp_shape);
                // Create a CoordinateTransform that visits only the first element along "axis"
                CoordinateTransform input_transform(in_shape,
                        start_corner,
                        end_corner,
                        in_strides,
                        axis_order);
                CoordinateTransform output_indices_transform(out_indices_shape,
                        start_corner,
                        end_corner,
                        out_indices_strides,
                        axis_order);
                CoordinateTransform output_values_transform(out_values_shape,
                        start_corner,
                        end_corner,
                        out_values_strides,
                        axis_order);

                auto out_indices_iter = output_indices_transform.begin();
                auto out_values_iter = output_values_transform.begin();
                std::vector<std::tuple<T, U>> workspace(in_shape[axis]);
                for(const Coordinate& in_coord: input_transform)
                {
                    auto index = input_transform.index(in_coord);
                    U i = 0;
                    for(std::tuple<T, U> &entry : workspace)
                    {
                        std::get<0>(entry) = arg[index];
                        std::get<1>(entry) = i;
                        index += in_strides[axis];
                        i++;
                    }
                    std::sort(workspace.begin(),
                            workspace.end(),
                            compute_max ?
                                [] (const std::tuple<T, U>&  a, const std::tuple<T, U>& b) -> bool { return a > b;} :
                                [] (const std::tuple<T, U>&  a, const std::tuple<T, U>& b) -> bool { return a < b;});
                    auto out_indices_index = output_indices_transform.index(*out_indices_iter);
                    auto out_values_index = output_values_transform.index(*out_values_iter);
                    for(size_t i = 0; i < k ; i++)
                    {
                        out_indices[out_indices_index]=std::get<1>(workspace[i]);
                        out_values[out_values_index]=std::get<0>(workspace[i]);
                        out_indices_index += out_indices_strides[axis];
                        out_values_index += out_values_strides[axis];
                    }
                    out_indices_iter++;
                    out_values_iter++;
                }
            }
        }
    }
}
