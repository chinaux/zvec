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

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <zvec/db/doc.h>
#include <zvec/db/type.h>

namespace zvec {

//! Reranker abstract base class for re-ranking search results
class Reranker {
 public:
  using Ptr = std::shared_ptr<Reranker>;

  explicit Reranker(int topn = 10) : topn_(topn) {}
  virtual ~Reranker() = default;

  int topn() const { return topn_; }

  //! Re-rank documents from one or more vector queries.
  //! \param query_results Mapping from vector field name to list of retrieved
  //!   documents (sorted by relevance).
  //! \return Re-ranked list of documents (length <= topn), with updated scores.
  virtual DocPtrList rerank(
      const std::map<std::string, DocPtrList>& query_results) const = 0;

 protected:
  int topn_;
};

//! Re-ranker using Reciprocal Rank Fusion (RRF) for multi-vector search.
//!
//! RRF combines results from multiple vector queries without requiring
//! relevance scores. The RRF score for a document at rank r is:
//!   score = 1 / (k + r + 1)
//! where k is the rank constant.
class RrfReRanker : public Reranker {
 public:
  RrfReRanker(int topn = 10, int rank_constant = 60)
      : Reranker(topn), rank_constant_(rank_constant) {}

  int rank_constant() const { return rank_constant_; }

  DocPtrList rerank(
      const std::map<std::string, DocPtrList>& query_results) const override;

 private:
  int rank_constant_;
};

//! Re-ranker that combines scores from multiple vector fields using weights.
//!
//! Each vector field's relevance score is normalized based on its metric type,
//! then scaled by a user-provided weight. Final scores are summed across
//! fields. Supported metrics: L2, IP, COSINE.
class WeightedReRanker : public Reranker {
 public:
  WeightedReRanker(int topn = 10, MetricType metric = MetricType::L2,
                   const std::map<std::string, double>& weights = {});

  MetricType metric() const { return metric_; }
  const std::map<std::string, double>& weights() const { return weights_; }

  DocPtrList rerank(
      const std::map<std::string, DocPtrList>& query_results) const override;

  //! Normalize a raw distance/similarity score to [0, 1] range
  static double normalize_score(double score, MetricType metric);

 private:
  MetricType metric_;
  std::map<std::string, double> weights_;
};

//! Callback-based re-ranker for cross-language bridging.
//!
//! Wraps a user-provided callback (e.g., a Python callable) as a Reranker.
//! When the callback is a Python function, GIL must be managed by the caller.
class CallbackReRanker : public Reranker {
 public:
  using Callback = std::function<DocPtrList(
      const std::map<std::string, DocPtrList>&)>;

  CallbackReRanker(Callback fn, int topn = 10)
      : Reranker(topn), callback_(std::move(fn)) {}

  DocPtrList rerank(
      const std::map<std::string, DocPtrList>& query_results) const override {
    return callback_(query_results);
  }

 private:
  Callback callback_;
};

}  // namespace zvec
