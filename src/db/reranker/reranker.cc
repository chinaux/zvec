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

#include <zvec/db/reranker.h>

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <utility>

namespace zvec {

// ==================== RrfReRanker ====================

DocPtrList RrfReRanker::rerank(
    const std::map<std::string, DocPtrList>& query_results) const {
  // doc_id -> cumulative RRF score
  std::unordered_map<std::string, double> rrf_scores;
  // doc_id -> first-seen Doc pointer
  std::unordered_map<std::string, Doc::Ptr> id_to_doc;

  for (const auto& [field_name, docs] : query_results) {
    for (size_t rank = 0; rank < docs.size(); ++rank) {
      const auto& doc = docs[rank];
      const std::string& doc_id = doc->pk();
      double score =
          1.0 / (static_cast<double>(rank_constant_) + static_cast<double>(rank) + 1.0);
      rrf_scores[doc_id] += score;
      if (id_to_doc.find(doc_id) == id_to_doc.end()) {
        id_to_doc[doc_id] = doc;
      }
    }
  }

  // Sort by RRF score descending and take topn using a min-heap
  using ScorePair = std::pair<std::string, double>;
  auto cmp = [](const ScorePair& a, const ScorePair& b) {
    return a.second > b.second;  // min-heap: top element is smallest
  };
  std::priority_queue<ScorePair, std::vector<ScorePair>, decltype(cmp)> pq(
      cmp);

  for (const auto& [doc_id, score] : rrf_scores) {
    if (static_cast<int>(pq.size()) < topn_) {
      pq.emplace(doc_id, score);
    } else if (score > pq.top().second) {
      pq.pop();
      pq.emplace(doc_id, score);
    }
  }

  DocPtrList results;
  results.reserve(pq.size());
  while (!pq.empty()) {
    const auto& [doc_id, score] = pq.top();
    auto doc = std::make_shared<Doc>(*id_to_doc[doc_id]);
    doc->set_score(static_cast<float>(score));
    results.push_back(std::move(doc));
    pq.pop();
  }
  // Reverse to get descending order
  std::reverse(results.begin(), results.end());
  return results;
}

// ==================== WeightedReRanker ====================

WeightedReRanker::WeightedReRanker(int topn, MetricType metric,
                                   const std::map<std::string, double>& weights)
    : Reranker(topn), metric_(metric), weights_(weights) {}

double WeightedReRanker::normalize_score(double score, MetricType metric) {
  switch (metric) {
    case MetricType::L2:
      return 1.0 - 2.0 * std::atan(score) / M_PI;
    case MetricType::IP:
      return 0.5 + std::atan(score) / M_PI;
    case MetricType::COSINE:
      return 1.0 - score / 2.0;
    default:
      throw std::invalid_argument("Unsupported metric type for normalization");
  }
}

DocPtrList WeightedReRanker::rerank(
    const std::map<std::string, DocPtrList>& query_results) const {
  // doc_id -> cumulative weighted score
  std::unordered_map<std::string, double> weighted_scores;
  // doc_id -> first-seen Doc pointer
  std::unordered_map<std::string, Doc::Ptr> id_to_doc;

  for (const auto& [vector_name, docs] : query_results) {
    double weight = 1.0;
    auto it = weights_.find(vector_name);
    if (it != weights_.end()) {
      weight = it->second;
    }
    for (const auto& doc : docs) {
      const std::string& doc_id = doc->pk();
      double normalized =
          normalize_score(static_cast<double>(doc->score()), metric_);
      weighted_scores[doc_id] += normalized * weight;
      if (id_to_doc.find(doc_id) == id_to_doc.end()) {
        id_to_doc[doc_id] = doc;
      }
    }
  }

  // Sort by weighted score descending and take topn using a min-heap
  using ScorePair = std::pair<std::string, double>;
  auto cmp = [](const ScorePair& a, const ScorePair& b) {
    return a.second > b.second;  // min-heap
  };
  std::priority_queue<ScorePair, std::vector<ScorePair>, decltype(cmp)> pq(
      cmp);

  for (const auto& [doc_id, score] : weighted_scores) {
    if (static_cast<int>(pq.size()) < topn_) {
      pq.emplace(doc_id, score);
    } else if (score > pq.top().second) {
      pq.pop();
      pq.emplace(doc_id, score);
    }
  }

  DocPtrList results;
  results.reserve(pq.size());
  while (!pq.empty()) {
    const auto& [doc_id, score] = pq.top();
    auto doc = std::make_shared<Doc>(*id_to_doc[doc_id]);
    doc->set_score(static_cast<float>(score));
    results.push_back(std::move(doc));
    pq.pop();
  }
  // Reverse to get descending order
  std::reverse(results.begin(), results.end());
  return results;
}

}  // namespace zvec
