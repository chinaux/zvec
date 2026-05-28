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
//
// =============================================================================
// Callback Reranker Example
// -----------------------------------------------------------------------------
// This example demonstrates how to register a user-provided C callback as a
// Reranker for a MultiQuery in zvec.
//
// Memory ownership rules (CRITICAL — see c_api.h zvec_reranker_callback_fn):
//   - The outer pointer array (zvec_doc_t**) MUST be allocated by the callback
//     using malloc/calloc; zvec frees it via free().
//   - Each inner pointer (zvec_doc_t*) MUST be picked from the input
//     field_results[i].docs arrays. DO NOT call zvec_doc_create or otherwise
//     allocate new doc objects inside the callback — those would leak.
//
// The callback below implements a simple Reciprocal Rank Fusion (RRF):
//   final_score(doc) = sum over fields( 1 / (k + rank_within_field) )
// and returns the topN docs by this score.
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zvec/c_api.h"

// ---- RRF callback ----------------------------------------------------------

typedef struct {
  int rank_constant;  // RRF k, e.g. 60
} rrf_user_data_t;

// A small dedup table mapping pk_string -> (doc, score). For the example we
// keep it dead-simple O(N^2) instead of a real hash map.
typedef struct {
  zvec_doc_t *doc;
  double score;
} rrf_entry_t;

static int find_entry(const rrf_entry_t *entries, size_t n, const char *pk) {
  if (!pk) return -1;
  for (size_t i = 0; i < n; ++i) {
    const char *epk = zvec_doc_get_pk_pointer(entries[i].doc);
    if (epk && strcmp(epk, pk) == 0) return (int)i;
  }
  return -1;
}

static int cmp_entry_desc(const void *a, const void *b) {
  const rrf_entry_t *ea = (const rrf_entry_t *)a;
  const rrf_entry_t *eb = (const rrf_entry_t *)b;
  if (ea->score < eb->score) return 1;
  if (ea->score > eb->score) return -1;
  return 0;
}

// Reciprocal Rank Fusion callback.
// Selects pointers from the input docs only — no new docs allocated.
static zvec_doc_t **rrf_callback(
    const zvec_reranker_field_results_t *field_results, size_t field_count,
    int topn, size_t *result_count, void *user_data) {
  *result_count = 0;
  if (field_count == 0 || topn <= 0) return NULL;

  rrf_user_data_t *cfg = (rrf_user_data_t *)user_data;
  int k = cfg ? cfg->rank_constant : 60;

  // Upper bound on unique docs.
  size_t total_docs = 0;
  for (size_t f = 0; f < field_count; ++f) {
    total_docs += field_results[f].doc_count;
  }
  if (total_docs == 0) return NULL;

  rrf_entry_t *entries = (rrf_entry_t *)calloc(total_docs, sizeof(rrf_entry_t));
  if (!entries) return NULL;
  size_t n_entries = 0;

  // Accumulate RRF score per doc id.
  for (size_t f = 0; f < field_count; ++f) {
    const zvec_reranker_field_results_t *fr = &field_results[f];
    for (size_t r = 0; r < fr->doc_count; ++r) {
      zvec_doc_t *doc = fr->docs[r];
      if (!doc) continue;

      const char *pk = zvec_doc_get_pk_pointer(doc);
      double contrib = 1.0 / ((double)k + (double)r + 1.0);

      int idx = find_entry(entries, n_entries, pk);
      if (idx >= 0) {
        entries[idx].score += contrib;
      } else {
        entries[n_entries].doc = doc;  // pointer from input — no copy
        entries[n_entries].score = contrib;
        n_entries++;
      }
    }
  }

  // Sort by score descending and pick topn.
  qsort(entries, n_entries, sizeof(rrf_entry_t), cmp_entry_desc);

  size_t out_count = (size_t)topn < n_entries ? (size_t)topn : n_entries;

  // Allocate ONLY the outer array; zvec frees it via free().
  zvec_doc_t **out = (zvec_doc_t **)malloc(sizeof(zvec_doc_t *) * out_count);
  if (!out) {
    free(entries);
    return NULL;
  }
  for (size_t i = 0; i < out_count; ++i) {
    out[i] = entries[i].doc;  // still pointing into input — safe
  }
  free(entries);
  *result_count = out_count;
  return out;
}

// ---- Schema / data helpers -------------------------------------------------

static zvec_error_code_t build_schema(zvec_collection_schema_t **out_schema) {
  zvec_collection_schema_t *schema =
      zvec_collection_schema_create("rrf_example");
  if (!schema) return ZVEC_ERROR_INTERNAL_ERROR;

  // Inverted index for the string pk.
  zvec_index_params_t *invert =
      zvec_index_params_create(ZVEC_INDEX_TYPE_INVERT);
  zvec_index_params_set_invert_params(invert, true, false);

  zvec_field_schema_t *id_field =
      zvec_field_schema_create("id", ZVEC_DATA_TYPE_STRING, false, 0);
  zvec_field_schema_set_index_params(id_field, invert);
  zvec_collection_schema_add_field(schema, id_field);
  zvec_index_params_destroy(invert);

  // Two vector fields (HNSW + L2), dim = 4.
  zvec_index_params_t *hnsw = zvec_index_params_create(ZVEC_INDEX_TYPE_HNSW);
  zvec_index_params_set_metric_type(hnsw, ZVEC_METRIC_TYPE_L2);
  zvec_index_params_set_hnsw_params(hnsw, 16, 100);

  zvec_field_schema_t *v1 = zvec_field_schema_create(
      "embedding1", ZVEC_DATA_TYPE_VECTOR_FP32, false, 4);
  zvec_field_schema_set_index_params(v1, hnsw);
  zvec_collection_schema_add_field(schema, v1);

  zvec_field_schema_t *v2 = zvec_field_schema_create(
      "embedding2", ZVEC_DATA_TYPE_VECTOR_FP32, false, 4);
  zvec_field_schema_set_index_params(v2, hnsw);
  zvec_collection_schema_add_field(schema, v2);

  zvec_index_params_destroy(hnsw);

  *out_schema = schema;
  return ZVEC_OK;
}

static zvec_error_code_t insert_demo_docs(zvec_collection_t *collection) {
  static const float E1[4][4] = {
      {1.0f, 0.0f, 0.0f, 0.0f},
      {0.9f, 0.1f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 1.0f, 0.0f},
  };
  static const float E2[4][4] = {
      {0.0f, 1.0f, 0.0f, 0.0f},
      {0.0f, 0.9f, 0.1f, 0.0f},
      {1.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 1.0f},
  };
  static const char *PKS[4] = {"doc1", "doc2", "doc3", "doc4"};

  zvec_doc_t *docs[4] = {0};
  for (int i = 0; i < 4; ++i) {
    docs[i] = zvec_doc_create();
    if (!docs[i]) return ZVEC_ERROR_INTERNAL_ERROR;
    zvec_doc_set_pk(docs[i], PKS[i]);
    zvec_doc_add_field_by_value(docs[i], "id", ZVEC_DATA_TYPE_STRING, PKS[i],
                                strlen(PKS[i]));
    zvec_doc_add_field_by_value(docs[i], "embedding1",
                                ZVEC_DATA_TYPE_VECTOR_FP32, E1[i],
                                4 * sizeof(float));
    zvec_doc_add_field_by_value(docs[i], "embedding2",
                                ZVEC_DATA_TYPE_VECTOR_FP32, E2[i],
                                4 * sizeof(float));
  }

  size_t success_count = 0;
  size_t error_count = 0;
  zvec_error_code_t err = zvec_collection_insert(
      collection, (const zvec_doc_t **)docs, 4, &success_count, &error_count);
  for (int i = 0; i < 4; ++i) zvec_doc_destroy(docs[i]);
  if (err != ZVEC_OK) return err;
  printf("Inserted %zu docs (errors=%zu)\n", success_count, error_count);

  return zvec_collection_flush(collection);
}

// ---- Main ------------------------------------------------------------------

int main(void) {
  const char *path = "./zvec_callback_reranker_example_db";

  zvec_collection_schema_t *schema = NULL;
  if (build_schema(&schema) != ZVEC_OK) {
    fprintf(stderr, "Failed to build schema\n");
    return 1;
  }

  zvec_collection_t *collection = NULL;
  zvec_error_code_t err =
      zvec_collection_create_and_open(path, schema, NULL, &collection);
  zvec_collection_schema_destroy(schema);
  if (err != ZVEC_OK || !collection) {
    char *msg = NULL;
    zvec_get_last_error(&msg);
    fprintf(stderr, "create_and_open failed: %s\n", msg ? msg : "?");
    zvec_free(msg);
    return 1;
  }

  if (insert_demo_docs(collection) != ZVEC_OK) {
    fprintf(stderr, "insert_demo_docs failed\n");
    zvec_collection_destroy(collection);
    return 1;
  }

  // ---- Build a MultiQuery with the RRF callback reranker -------------------
  rrf_user_data_t cfg = {.rank_constant = 60};
  zvec_reranker_t *reranker = zvec_reranker_create_callback(rrf_callback, &cfg);
  if (!reranker) {
    fprintf(stderr, "create_callback failed\n");
    zvec_collection_destroy(collection);
    return 1;
  }

  zvec_multi_query_t *mq = zvec_multi_query_create();
  zvec_multi_query_set_topk(mq, 3);
  zvec_multi_query_set_include_vector(mq, false);

  const float q1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  const float q2[4] = {0.0f, 1.0f, 0.0f, 0.0f};

  zvec_sub_query_t *sq1 = zvec_sub_query_create();
  zvec_sub_query_set_field_name(sq1, "embedding1");
  zvec_sub_query_set_query_vector(sq1, q1, sizeof(q1));
  zvec_sub_query_set_num_candidates(sq1, 4);
  zvec_multi_query_add_sub_query(mq, sq1);

  zvec_sub_query_t *sq2 = zvec_sub_query_create();
  zvec_sub_query_set_field_name(sq2, "embedding2");
  zvec_sub_query_set_query_vector(sq2, q2, sizeof(q2));
  zvec_sub_query_set_num_candidates(sq2, 4);
  zvec_multi_query_add_sub_query(mq, sq2);

  zvec_multi_query_set_reranker(mq, reranker);

  // ---- Execute and dump results -------------------------------------------
  zvec_doc_t **results = NULL;
  size_t result_count = 0;
  err = zvec_collection_multi_query(collection, mq, &results, &result_count);
  if (err == ZVEC_OK) {
    printf("RRF reranker returned %zu docs:\n", result_count);
    for (size_t i = 0; i < result_count; ++i) {
      const char *pk = zvec_doc_get_pk_pointer(results[i]);
      float score = zvec_doc_get_score(results[i]);
      printf("  rank %zu: pk=%s score=%.6f\n", i + 1, pk ? pk : "?", score);
    }
  } else {
    char *msg = NULL;
    zvec_get_last_error(&msg);
    fprintf(stderr, "multi_query failed: %s\n", msg ? msg : "?");
    zvec_free(msg);
  }

  // ---- Cleanup -------------------------------------------------------------
  if (results) zvec_docs_free(results, result_count);
  zvec_sub_query_destroy(sq1);
  zvec_sub_query_destroy(sq2);
  zvec_multi_query_destroy(mq);
  zvec_reranker_destroy(reranker);
  zvec_collection_destroy(collection);

  return err == ZVEC_OK ? 0 : 1;
}
