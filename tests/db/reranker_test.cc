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

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/db/doc.h>
#include <zvec/db/reranker.h>
#include <zvec/db/type.h>

using namespace zvec;

namespace {

//! Helper to create a Doc::Ptr with given id and score
Doc::Ptr MakeDoc(const std::string& id, float score) {
  auto doc = std::make_shared<Doc>();
  doc->set_pk(id);
  doc->set_score(score);
  return doc;
}

}  // namespace

// ==================== RrfReRanker Tests ====================

TEST(RrfReRankerTest, BasicRRF) {
  RrfReRanker reranker(/*topn=*/10, /*rank_constant=*/60);

  // Two vector fields, each returning 3 documents with some overlap
  std::map<std::string, DocPtrList> query_results;
  query_results["vec1"] = {MakeDoc("a", 0.9f), MakeDoc("b", 0.8f),
                           MakeDoc("c", 0.7f)};
  query_results["vec2"] = {MakeDoc("b", 0.95f), MakeDoc("a", 0.85f),
                           MakeDoc("d", 0.75f)};

  auto results = reranker.rerank(query_results);

  // "a" appears at rank 0 in vec1 and rank 1 in vec2:
  //   rrf_score = 1/(60+0+1) + 1/(60+1+1) = 1/61 + 1/62
  // "b" appears at rank 1 in vec1 and rank 0 in vec2:
  //   rrf_score = 1/(60+1+1) + 1/(60+0+1) = 1/62 + 1/61
  // So a and b should have equal scores and be at the top
  ASSERT_GE(results.size(), 3u);

  // "a" and "b" should have the highest RRF scores
  EXPECT_EQ(results[0]->pk(), "a");
  EXPECT_EQ(results[1]->pk(), "b");
  // Verify scores are close (a and b have same RRF score)
  EXPECT_NEAR(results[0]->score(), results[1]->score(), 1e-10);
}

TEST(RrfReRankerTest, Topn) {
  RrfReRanker reranker(/*topn=*/2, /*rank_constant=*/60);

  std::map<std::string, DocPtrList> query_results;
  query_results["vec1"] = {MakeDoc("a", 0.9f), MakeDoc("b", 0.8f),
                           MakeDoc("c", 0.7f)};

  auto results = reranker.rerank(query_results);
  ASSERT_EQ(results.size(), 2u);
}

TEST(RrfReRankerTest, SingleField) {
  RrfReRanker reranker(/*topn=*/10, /*rank_constant=*/60);

  std::map<std::string, DocPtrList> query_results;
  query_results["vec1"] = {MakeDoc("a", 0.9f), MakeDoc("b", 0.8f)};

  auto results = reranker.rerank(query_results);
  ASSERT_EQ(results.size(), 2u);
  // With single field, RRF score for rank 0 > rank 1
  EXPECT_GT(results[0]->score(), results[1]->score());
}

TEST(RrfReRankerTest, EmptyResults) {
  RrfReRanker reranker(/*topn=*/10, /*rank_constant=*/60);

  std::map<std::string, DocPtrList> query_results;
  auto results = reranker.rerank(query_results);
  EXPECT_TRUE(results.empty());
}

// ==================== WeightedReRanker Tests ====================

TEST(WeightedReRankerTest, BasicWeighted) {
  WeightedReRanker reranker(/*topn=*/10, MetricType::L2,
                            {{"vec1", 0.7}, {"vec2", 0.3}});

  std::map<std::string, DocPtrList> query_results;
  query_results["vec1"] = {MakeDoc("a", 0.5f), MakeDoc("b", 0.3f)};
  query_results["vec2"] = {MakeDoc("a", 0.8f), MakeDoc("c", 0.6f)};

  auto results = reranker.rerank(query_results);
  ASSERT_GE(results.size(), 2u);
  // "a" appears in both fields, should have highest combined score
  EXPECT_EQ(results[0]->pk(), "a");
}

TEST(WeightedReRankerTest, NormalizeL2) {
  // L2: normalize_score = 1 - 2*atan(score)/pi
  // For score=0: 1 - 0 = 1.0
  // For score->inf: 1 - 2*(pi/2)/pi = 0.0
  EXPECT_NEAR(WeightedReRanker::normalize_score(0.0, MetricType::L2), 1.0,
              1e-10);
  EXPECT_GT(WeightedReRanker::normalize_score(1.0, MetricType::L2), 0.0);
  EXPECT_LT(WeightedReRanker::normalize_score(1.0, MetricType::L2), 1.0);
}

TEST(WeightedReRankerTest, NormalizeIP) {
  // IP: normalize_score = 0.5 + atan(score)/pi
  // For score=0: 0.5 + 0 = 0.5
  EXPECT_NEAR(WeightedReRanker::normalize_score(0.0, MetricType::IP), 0.5,
              1e-10);
  EXPECT_GT(WeightedReRanker::normalize_score(1.0, MetricType::IP), 0.5);
}

TEST(WeightedReRankerTest, NormalizeCosine) {
  // COSINE: normalize_score = 1 - score/2
  // For score=0: 1 - 0 = 1.0
  // For score=1: 1 - 0.5 = 0.5
  // For score=2: 1 - 1.0 = 0.0
  EXPECT_NEAR(WeightedReRanker::normalize_score(0.0, MetricType::COSINE), 1.0,
              1e-10);
  EXPECT_NEAR(WeightedReRanker::normalize_score(1.0, MetricType::COSINE), 0.5,
              1e-10);
  EXPECT_NEAR(WeightedReRanker::normalize_score(2.0, MetricType::COSINE), 0.0,
              1e-10);
}

TEST(WeightedReRankerTest, Topn) {
  WeightedReRanker reranker(/*topn=*/2, MetricType::L2, {});

  std::map<std::string, DocPtrList> query_results;
  query_results["vec1"] = {MakeDoc("a", 0.1f), MakeDoc("b", 0.2f),
                           MakeDoc("c", 0.3f)};

  auto results = reranker.rerank(query_results);
  ASSERT_EQ(results.size(), 2u);
}

TEST(WeightedReRankerTest, UnsupportedMetric) {
  EXPECT_THROW(WeightedReRanker::normalize_score(1.0, MetricType::UNDEFINED),
               std::invalid_argument);
}

// ==================== CallbackReRanker Tests ====================

TEST(CallbackReRankerTest, BasicCallback) {
  // Simple callback that returns docs sorted by score descending
  CallbackReRanker::Callback cb =
      [](const std::map<std::string, DocPtrList>& query_results) -> DocPtrList {
    DocPtrList all_docs;
    for (const auto& [_, docs] : query_results) {
      for (const auto& doc : docs) {
        all_docs.push_back(doc);
      }
    }
    std::sort(all_docs.begin(), all_docs.end(),
              [](const Doc::Ptr& a, const Doc::Ptr& b) {
                return a->score() > b->score();
              });
    return all_docs;
  };

  CallbackReRanker reranker(cb, /*topn=*/10);

  std::map<std::string, DocPtrList> query_results;
  query_results["vec1"] = {MakeDoc("a", 0.5f), MakeDoc("b", 0.9f)};
  query_results["vec2"] = {MakeDoc("c", 0.7f)};

  auto results = reranker.rerank(query_results);
  ASSERT_EQ(results.size(), 3u);
  // Should be sorted by score descending
  EXPECT_EQ(results[0]->pk(), "b");
  EXPECT_EQ(results[1]->pk(), "c");
  EXPECT_EQ(results[2]->pk(), "a");
}
