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
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <zvec/db/doc.h>
#include <zvec/db/query_params.h>

namespace zvec {

struct VectorQuery {
  int topk_;
  std::string field_name_;
  std::string query_vector_;  // fp16, void *
  std::string query_sparse_indices_;
  std::string query_sparse_values_;
  std::string filter_;
  bool include_vector_{false};
  bool include_doc_id_{false};
  // select * by default, select no field if output_fields_ is empty, select
  // specific fields if output_fields_ is not empty
  std::optional<std::vector<std::string>> output_fields_;
  QueryParams::Ptr query_params_;

  Status validate_and_sanitize(const FieldSchema *schema);
};

struct GroupByVectorQuery {
  std::string field_name_;
  std::string query_vector_;
  std::string query_sparse_indices_;
  std::string query_sparse_values_;
  std::string filter_;
  bool include_vector_;
  // select * by default, select no field if output_fields_ is empty, select
  // specific fields if output_fields_ is not empty
  std::optional<std::vector<std::string>> output_fields_;
  std::string group_by_field_name_;
  uint32_t group_count_ = 2;
  uint32_t group_topk_ = 3;
  QueryParams::Ptr query_params_;
};

//! Multi-vector query structure for querying multiple vector fields
//! with optional re-ranking of combined results.
class Reranker;  // forward declaration

struct SubVectorQuery {
  int num_candidates_;
  std::string field_name_;
  std::string query_vector_;  // fp16, void *
  std::string query_sparse_indices_;
  std::string query_sparse_values_;
  QueryParams::Ptr query_params_;
};

struct MultiVectorQuery {
  std::vector<SubVectorQuery> queries;
  int topk{10};
  std::string filter;
  bool include_vector{false};
  bool include_doc_id_{false};
  std::optional<std::vector<std::string>> output_fields;
  std::shared_ptr<Reranker> reranker{nullptr};
};

struct GroupResult {
  std::string group_by_value_;
  std::vector<Doc> docs_;
};

using GroupResults = std::vector<GroupResult>;

}  // namespace zvec
