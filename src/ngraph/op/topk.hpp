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

#include "ngraph/axis_set.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/util/index_reduction.hpp"
#include "ngraph/op/util/requires_tensor_view_args.hpp"

namespace ngraph
{
    namespace op
    {
        //brief Computes indices of top k maximum/minimum index along a specified axis for a given tensor
        class TopK : public op::util::IndexReduction
        {
        public:
            /// \brief Constructs a TopK operation.
            ///
            /// \param arg The input tensor
            /// \param axis The axis along which to compute top k indices
            /// \param index_element_type produce indices. Currently, only int64 or int32 are supported
            /// \param k Number of top indices to compute. Compute all indices if k = 0
            /// \param compute_max Compute top k max or top k min?
            TopK(const std::shared_ptr<Node>& arg,
                 size_t axis,
                 const element::Type& index_element_type
                 size_t k = 0,
                 bool compute_max = true)
                : IndexReduction("TopK", arg, axis, index_element_type),
                  m_k(k),
                  m_compute_max(compute_max)
            {
            }

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            size_t get_k() { return m_k; }

            bool get_compute_max() { return m_compute_max; }

        protected:
            size_t m_k;
            bool m_compute_max;
        };
    }
}
