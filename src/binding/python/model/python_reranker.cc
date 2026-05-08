// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "python_reranker.h"
#include <pybind11/stl.h>
#include <zvec/db/collection.h>
#include <zvec/db/type.h>

namespace zvec {

void ZVecPyReranker::Initialize(py::module_ &m) {
  // Bind Reranker base class (abstract, cannot be instantiated directly)
  py::class_<Reranker, Reranker::Ptr>(m, "_Reranker")
      .def_property_readonly("topn", &Reranker::topn);

  // Bind RrfReRanker
  py::class_<RrfReRanker, Reranker, std::shared_ptr<RrfReRanker>>(
      m, "_RrfReRanker")
      .def(py::init<int, int>(), py::arg("topn") = 10,
           py::arg("rank_constant") = 60)
      .def_property_readonly("topn", &RrfReRanker::topn)
      .def_property_readonly("rank_constant", &RrfReRanker::rank_constant);

  // Bind WeightedReRanker
  py::class_<WeightedReRanker, Reranker, std::shared_ptr<WeightedReRanker>>(
      m, "_WeightedReRanker")
      .def(py::init<int, MetricType, std::map<std::string, double>>(),
           py::arg("topn") = 10, py::arg("metric") = MetricType::L2,
           py::arg("weights") = std::map<std::string, double>{})
      .def_property_readonly("topn", &WeightedReRanker::topn)
      .def_property_readonly("metric", &WeightedReRanker::metric)
      .def_property_readonly("weights", &WeightedReRanker::weights);

  // Bind MultiVectorQuery struct
  py::class_<MultiVectorQuery>(m, "_MultiVectorQuery")
      .def(py::init<>())
      .def_readwrite("queries", &MultiVectorQuery::queries)
      .def_readwrite("topk", &MultiVectorQuery::topk)
      .def_readwrite("filter", &MultiVectorQuery::filter)
      .def_readwrite("include_vector", &MultiVectorQuery::include_vector)
      .def_readwrite("output_fields", &MultiVectorQuery::output_fields)
      .def_readwrite("reranker", &MultiVectorQuery::reranker);
}

}  // namespace zvec
