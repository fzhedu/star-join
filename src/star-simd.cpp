#ifndef __SIMDTEST__
#define __SIMDTEST__
#include "star-simd.h"
#define _mm256_set_m128i(v0, v1) \
  _mm256_insertf128_si256(_mm256_castsi128_si256(v1), (v0), 1)
struct SIMDMState {
  __m512i v_ht_pos, v_tuple_cell, v_join_id, v_bucket_offset, v_addr_offset;
  __mmask16 m_bucket_pass, m_have_tuple, m_new_cells;
  char stage;
  uint32_t temp_payloads[16][6];
};
// use short mask to avoid long mask
uint64_t PrefetchLinear512Probe(Table* pb, HashTable** ht, int ht_num,
                                char* payloads) {
  uint16_t vector_scale = 16, new_add, done = 0, k = 0;
  __mmask16 m_done = 0, m_match = 0, m_abort = 0, m_new;
  __m512i v_offset = _mm512_set1_epi32(0),
          v_join_num = _mm512_set1_epi32(ht_num),
          v_base_offset_upper =
              _mm512_set1_epi32(pb->tuple_num * pb->tuple_size),
          v_right_index, v_base_offset, v_left_size = _mm512_set1_epi32(8),
          ht_cell, v_factor = _mm512_set1_epi32(table_factor),
          v_ht_cell_offset = _mm512_setzero_epi32(), v_shift, v_buckets_minus_1,
          v_cell_hash, v_neg_one512 = _mm512_set1_epi32(-1),
          v_zero512 = _mm512_set1_epi32(0), v_one = _mm512_set1_epi32(1),
          v_payloads, v_payloads_off, v_ht_global_addr_offset,
          v_word_size = _mm512_set1_epi32(WORDSIZE);
  __attribute__((aligned(64)))
  uint32_t cur_offset = 0,
           cur_payloads = 0, global_addr_offset[16] = {0}, tmp_cell[16],
           *addr_offset, *ht_pos, *join_id, base_off[32],
           tuple_cell_offset[16] = {0}, left_size[16] = {0},
           ht_cell_offset[16] = {0}, payloads_off[32],
           result_size = (ht_num + 1) * PAYLOADSIZE, buckets_minus_1[16] = {0},
           shift[16] = {0};
  for (int i = 0; i <= vector_scale; ++i) {
    base_off[i] = i * pb->tuple_size;
    payloads_off[i] = i * 6;  // it is related to temp_payloads
  }
  for (int i = 0; i < ht_num; ++i) {
    tuple_cell_offset[i] = ht[i]->probe_offset;
    left_size[i] = ht[i]->tuple_size;
    ht_cell_offset[i] = ht[i]->key_offset;
    shift[i] = ht[i]->shift;
    buckets_minus_1[i] = ht[i]->slot_num - 1;
    global_addr_offset[i] = ht[i]->global_addr_offset;
  }

  uint64_t equal_num = 0;
  v_base_offset = _mm512_load_epi32(base_off);
  v_payloads_off = _mm512_load_epi32(payloads_off);
#if OUTPUT
  FILE* fp;
  stringstream sstr;
  sstr << pthread_self();
  fp = fopen(string(("simd512.csv") + sstr.str()).c_str(), "wr");
  if (fp == 0) {
    assert(false && "can not open file!");
  }
#endif

  SIMDMState state[SIMDStateSize];
  for (int i = 0; i < SIMDStateSize; ++i) {
    state[i].m_bucket_pass = 0;
    state[i].m_have_tuple = 0;
    state[i].m_new_cells = -1;
    state[i].stage = 1;
    state[i].v_addr_offset = _mm512_set1_epi32(0);
    state[i].v_bucket_offset = _mm512_set1_epi32(0);
    state[i].v_ht_pos = _mm512_set1_epi32(0);
    state[i].v_join_id = _mm512_set1_epi32(0);
    state[i].v_tuple_cell = _mm512_set1_epi32(0);
    memset(state[i].temp_payloads, 0, sizeof(state[i].temp_payloads));
  }
  k = 0;
  for (uint64_t cur = 0; (cur < pb->tuple_num) || (done < SIMDStateSize);) {
    k = (k >= SIMDStateSize) ? 0 : k;
    if (cur >= pb->tuple_num) {
      if (state[k].m_have_tuple == 0 && state[k].stage != 3) {
        ++done;
        state[k].stage = 3;
        ++k;
        continue;
      }
    }
    switch (state[k].stage) {
      case 1: {
        state[k].stage = 0;
///////// step 1: load new tuples' address offsets
// the offset should be within MAX_32INT_
// the tail depends on the number of joins and tuples in each bucket
#if !SEQPREFETCH
        _mm_prefetch((char*)(pb->start + cur_offset + PDIS), _MM_HINT_T0);
        _mm_prefetch((char*)(pb->start + cur_offset + PDIS + 64), _MM_HINT_T0);
        _mm_prefetch((char*)(pb->start + cur_offset + PDIS + 128), _MM_HINT_T0);
        _mm_prefetch((char*)(pb->start + cur_offset + PDIS + 192), _MM_HINT_T1);
        _mm_prefetch((char*)(pb->start + cur_offset + PDIS + 256), _MM_HINT_T1);
#endif
        v_offset =
            _mm512_add_epi32(_mm512_set1_epi32(cur_offset), v_base_offset);
        state[k].v_addr_offset = _mm512_mask_expand_epi32(
            state[k].v_addr_offset, _mm512_knot(state[k].m_have_tuple),
            v_offset);
        // count the number of empty tuples
        new_add = _mm_popcnt_u32(_mm512_knot(state[k].m_have_tuple));
        cur_offset = cur_offset + base_off[new_add];
        cur = cur + new_add;
        state[k].m_have_tuple = _mm512_cmpgt_epi32_mask(v_base_offset_upper,
                                                        state[k].v_addr_offset);
        /////// step 2: load new cells from tuples
        /*
         * the cases that need to load new cells
         * (1) new tuples -> v_have_tuple
         * (2) last cells are matched for all cells in corresponding buckets ->
         * v_next_cells
         */

        v_right_index =
            _mm512_i32gather_epi32(state[k].v_join_id, tuple_cell_offset, 4);
        state[k].m_new_cells =
            _mm512_kand(state[k].m_new_cells, state[k].m_have_tuple);
        state[k].v_tuple_cell = _mm512_mask_i32gather_epi32(
            state[k].v_tuple_cell, state[k].m_new_cells,
            _mm512_add_epi32(state[k].v_addr_offset, v_right_index), pb->start,
            1);

        ////// step 3: load new values in hash tables
        v_left_size = _mm512_i32gather_epi32(state[k].v_join_id, left_size, 4);
        // v_ht_cell_offset = _mm512_i32gather_epi32(v_join_id, ht_cell_offset,
        // 4);

        v_shift = _mm512_i32gather_epi32(state[k].v_join_id, shift, 4);
        v_buckets_minus_1 =
            _mm512_i32gather_epi32(state[k].v_join_id, buckets_minus_1, 4);
        // hash the cell values
        v_cell_hash = _mm512_mullo_epi32(state[k].v_tuple_cell, v_factor);
        v_cell_hash = _mm512_srlv_epi32(v_cell_hash, v_shift);
        // set 0 for new cells, but add 1 for old cells
        state[k].v_bucket_offset = _mm512_maskz_add_epi32(
            _mm512_knot(state[k].m_new_cells), state[k].v_bucket_offset, v_one);

        v_cell_hash = _mm512_add_epi32(v_cell_hash, state[k].v_bucket_offset);
        // avoid overflow
        v_cell_hash = _mm512_and_si512(v_cell_hash, v_buckets_minus_1);

        state[k].v_ht_pos = _mm512_mullo_epi32(v_cell_hash, v_left_size);

        // global address
        v_ht_global_addr_offset =
            _mm512_i32gather_epi32(state[k].v_join_id, global_addr_offset, 4);
        state[k].v_ht_pos =
            _mm512_add_epi32(state[k].v_ht_pos, v_ht_global_addr_offset);
#if !PREFETCH
        ht_pos = (uint32_t*)&state[k].v_ht_pos;
#if 0  // delay the execution
        m_new = state[k].m_new_cells;
        //_mm512_mask_prefetch_i32gather_ps(v_ht_pos, m_have_tuple,
        //                               ht[0]->global_addr, 1, _MM_HINT_T0);
        for (int j = 0; m_new && j < vector_scale; ++j, m_new = (m_new >> 1)) {
          if (m_new & 1) {
            _mm_prefetch(ht[0]->global_addr + ht_pos[j], _MM_HINT_T0);
          }
        }
#else
        for (int j = 0; j < vector_scale; ++j) {
          _mm_prefetch(ht[0]->global_addr + ht_pos[j], _MM_HINT_T0);
        }
#endif
#endif
      } break;
      case 0: {
        state[k].stage = 1;
        ht_cell = _mm512_mask_i32gather_epi32(
            v_neg_one512, state[k].m_have_tuple, state[k].v_ht_pos,
            ht[0]->global_addr, 1);

        //// step 4: compare
        // load raw cell data, then judge whether they are equal ? the AND get
        // rid of invalid keys
        m_match = _mm512_cmpeq_epi32_mask(state[k].v_tuple_cell, ht_cell);
        m_match = _mm512_kand(m_match, state[k].m_have_tuple);

        // store the global address offset of payloads
        state[k].v_ht_pos = _mm512_add_epi32(state[k].v_ht_pos, v_word_size);
        v_offset = _mm512_add_epi32(state[k].v_join_id, v_payloads_off);
        _mm512_mask_i32scatter_epi32(state[k].temp_payloads, m_match, v_offset,
                                     state[k].v_ht_pos, 4);

        // the bucket is over if ht cells =-1 or early break due to match
        // so need to process new cells
        // then process next buckets (increase join_id and load new cells)
        // or process next tuples
        // or abort
        state[k].m_new_cells = _mm512_cmpeq_epi32_mask(ht_cell, v_neg_one512);
#if EARLYBREAK
        state[k].m_new_cells = _mm512_kor(state[k].m_new_cells, m_match);
#endif
        // the bucket is over, so it is necessary to increase join_id by one
        state[k].v_join_id =
            _mm512_mask_add_epi32(state[k].v_join_id, state[k].m_new_cells,
                                  state[k].v_join_id, v_one);

        state[k].m_bucket_pass = _mm512_kor(state[k].m_bucket_pass, m_match);
        m_done = _mm512_kand(
            state[k].m_bucket_pass,
            _mm512_cmpeq_epi32_mask(state[k].v_join_id, v_join_num));
        // the bucket is over but it isn't passed
        m_abort = _mm512_kandn(state[k].m_bucket_pass, state[k].m_new_cells);
        state[k].m_have_tuple =
            _mm512_kandn(_mm512_kor(m_done, m_abort), state[k].m_have_tuple);
        state[k].v_join_id = _mm512_maskz_add_epi32(
            state[k].m_have_tuple, state[k].v_join_id, v_zero512);

        state[k].m_bucket_pass =
            _mm512_kandn(state[k].m_new_cells, state[k].m_bucket_pass);
        equal_num += _mm_popcnt_u32(m_done);
#if RESULTS
        addr_offset = (uint32_t*)&state[k].v_addr_offset;
        for (int i = 0; m_done && i < vector_scale;
             ++i, m_done = (m_done >> 1)) {
          if (m_done & 1) {
            int output_off = 0;
            for (int j = 0; j < ht_num; ++j) {
              memcpy(payloads + cur_payloads + output_off,
                     state[k].temp_payloads[i][j] + ht[0]->global_addr,
                     PAYLOADSIZE);
              output_off += PAYLOADSIZE;
            }
            memcpy(payloads + cur_payloads + output_off,
                   pb->start + addr_offset[i] + WORDSIZE * ht_num, PAYLOADSIZE);
#if OUTPUT
            for (int j = 0; j < ht_num; ++j) {
              fprintf(fp, "%d,", *((uint32_t*)(payloads + cur_payloads +
                                               j * PAYLOADSIZE)));
            }
            fprintf(fp, "%d\n", *((uint32_t*)(payloads + cur_payloads +
                                              ht_num * PAYLOADSIZE)));
#endif
            cur_payloads += result_size;
          }
        }
#endif
      }
    }
    ++k;
  }
// cout << "SIMD512Probe qualified tuples = " << equal_num << endl;
#if OUTPUT
  fclose(fp);
#endif
  return equal_num;
}
uint64_t Linear512Probe(Table* pb, HashTable** ht, int ht_num, char* payloads) {
  uint16_t vector_scale = 16, new_add;
  __mmask16 m_bucket_pass = 0, m_done = 0, m_match = 0, m_abort = 0,
            m_have_tuple = 0, m_new_cells = -1;
  __m512i v_offset = _mm512_set1_epi32(0), v_addr_offset = _mm512_set1_epi32(0),
          v_join_num = _mm512_set1_epi32(ht_num),
          v_base_offset_upper =
              _mm512_set1_epi32(pb->tuple_num * pb->tuple_size),
          v_tuple_cell = _mm512_set1_epi32(0), v_right_index, v_base_offset,
          v_left_size = _mm512_set1_epi32(8), v_bucket_offset, ht_cell,
          v_factor = _mm512_set1_epi32(table_factor),
          v_ht_cell_offset = _mm512_setzero_epi32(), v_shift, v_buckets_minus_1,
          v_cell_hash, v_ht_pos, v_neg_one512 = _mm512_set1_epi32(-1),
          v_zero512 = _mm512_set1_epi32(0), v_join_id = _mm512_set1_epi32(0),
          v_one = _mm512_set1_epi32(1), v_payloads, v_payloads_off,
          v_ht_global_addr_offset, v_word_size = _mm512_set1_epi32(WORDSIZE);
  __attribute__((aligned(64)))
  uint32_t cur_offset = 0,
           cur_payloads = 0, global_addr_offset[16] = {0}, tmp_cell[16],
           *addr_offset, temp_payloads[16][6] = {0}, *ht_pos, *join_id,
           base_off[32], tuple_cell_offset[16] = {0}, left_size[16] = {0},
           ht_cell_offset[16] = {0}, payloads_off[32],
           result_size = (ht_num + 1) * PAYLOADSIZE, buckets_minus_1[16] = {0},
           shift[16] = {0};
  __attribute__((aligned(64))) uint64_t htp[16] = {0};
  for (int i = 0; i <= vector_scale; ++i) {
    base_off[i] = i * pb->tuple_size;
    payloads_off[i] = i * 6;  // it is related to temp_payloads
  }
  for (int i = 0; i < ht_num; ++i) {
    tuple_cell_offset[i] = ht[i]->probe_offset;
    left_size[i] = ht[i]->tuple_size;
    ht_cell_offset[i] = ht[i]->key_offset;
    shift[i] = ht[i]->shift;
    buckets_minus_1[i] = ht[i]->slot_num - 1;
    htp[i] = (uint64_t)ht[i]->addr;
    global_addr_offset[i] = ht[i]->global_addr_offset;
  }
  ht_pos = (uint32_t*)&v_ht_pos;
  join_id = (uint32_t*)&v_join_id;
  addr_offset = (uint32_t*)&v_addr_offset;

  uint64_t equal_num = 0;
  v_base_offset = _mm512_load_epi32(base_off);
  v_payloads_off = _mm512_load_epi32(payloads_off);
#if OUTPUT
  FILE* fp;
  stringstream sstr;
  sstr << pthread_self();
  fp = fopen(string(("simd512.csv") + sstr.str()).c_str(), "wr");
  if (fp == 0) {
    assert(false && "can not open file!");
  }
#endif
  for (uint64_t cur = 0; cur < pb->tuple_num || m_have_tuple;) {
///////// step 1: load new tuples' address offsets
// the offset should be within MAX_32INT_
// the tail depends on the number of joins and tuples in each bucket
#if SEQPREFETCH
    _mm_prefetch((char*)(pb->start + cur_offset + PDIS), _MM_HINT_T0);
    _mm_prefetch((char*)(pb->start + cur_offset + PDIS + 64), _MM_HINT_T0);
    _mm_prefetch((char*)(pb->start + cur_offset + PDIS + 128), _MM_HINT_T0);
    _mm_prefetch((char*)(pb->start + cur_offset + PDIS + 192), _MM_HINT_T0);
    _mm_prefetch((char*)(pb->start + cur_offset + PDIS + 256), _MM_HINT_T0);
#endif
    v_offset = _mm512_add_epi32(_mm512_set1_epi32(cur_offset), v_base_offset);
    v_addr_offset = _mm512_mask_expand_epi32(
        v_addr_offset, _mm512_knot(m_have_tuple), v_offset);
    // count the number of empty tuples
    new_add = _mm_popcnt_u32(_mm512_knot(m_have_tuple));
    cur_offset = cur_offset + base_off[new_add];
    cur = cur + new_add;
    m_have_tuple = _mm512_cmpgt_epi32_mask(v_base_offset_upper, v_addr_offset);
    /////// step 2: load new cells from tuples
    /*
     * the cases that need to load new cells
     * (1) new tuples -> v_have_tuple
     * (2) last cells are matched for all cells in corresponding buckets ->
     * v_next_cells
     */

    v_right_index = _mm512_i32gather_epi32(v_join_id, tuple_cell_offset, 4);
    m_new_cells = _mm512_kand(m_new_cells, m_have_tuple);
    v_tuple_cell = _mm512_mask_i32gather_epi32(
        v_tuple_cell, m_new_cells,
        _mm512_add_epi32(v_addr_offset, v_right_index), pb->start, 1);

    ////// step 3: load new values in hash tables
    v_left_size = _mm512_i32gather_epi32(v_join_id, left_size, 4);
    // v_ht_cell_offset = _mm512_i32gather_epi32(v_join_id, ht_cell_offset,
    // 4);

    v_shift = _mm512_i32gather_epi32(v_join_id, shift, 4);
    v_buckets_minus_1 = _mm512_i32gather_epi32(v_join_id, buckets_minus_1, 4);
    // hash the cell values
    v_cell_hash = _mm512_mullo_epi32(v_tuple_cell, v_factor);
    v_cell_hash = _mm512_srlv_epi32(v_cell_hash, v_shift);
    // set 0 for new cells, but add 1 for old cells
    v_bucket_offset = _mm512_maskz_add_epi32(_mm512_knot(m_new_cells),
                                             v_bucket_offset, v_one);

    v_cell_hash = _mm512_add_epi32(v_cell_hash, v_bucket_offset);
    // avoid overflow
    v_cell_hash = _mm512_and_si512(v_cell_hash, v_buckets_minus_1);

    v_ht_pos = _mm512_mullo_epi32(v_cell_hash, v_left_size);

    // global address
    v_ht_global_addr_offset =
        _mm512_i32gather_epi32(v_join_id, global_addr_offset, 4);
    v_ht_pos = _mm512_add_epi32(v_ht_pos, v_ht_global_addr_offset);
#if PREFETCH
    //_mm512_mask_prefetch_i32gather_ps(v_ht_pos, m_have_tuple,
    //                               ht[0]->global_addr, 1, _MM_HINT_T0);
    for (int j = 0; j < 16; ++j) {
      _mm_prefetch(ht[0]->global_addr + ht_pos[j], _MM_HINT_T0);
    }
#endif
    ht_cell = _mm512_mask_i32gather_epi32(v_neg_one512, m_have_tuple, v_ht_pos,
                                          ht[0]->global_addr, 1);

    //// step 4: compare
    // load raw cell data, then judge whether they are equal ? the AND get
    // rid of invalid keys
    m_match = _mm512_cmpeq_epi32_mask(v_tuple_cell, ht_cell);
    m_match = _mm512_kand(m_match, m_have_tuple);

    // store the global address offset of payloads
    v_ht_pos = _mm512_add_epi32(v_ht_pos, v_word_size);
    v_offset = _mm512_add_epi32(v_join_id, v_payloads_off);
    _mm512_mask_i32scatter_epi32(temp_payloads, m_match, v_offset, v_ht_pos, 4);

    // the bucket is over if ht cells =-1 or early break due to match
    // so need to process new cells
    // then process next buckets (increase join_id and load new cells)
    // or process next tuples
    // or abort
    m_new_cells = _mm512_cmpeq_epi32_mask(ht_cell, v_neg_one512);
#if EARLYBREAK
    m_new_cells = _mm512_kor(m_new_cells, m_match);
#endif
    // the bucket is over, so it is necessary to increase join_id by one
    v_join_id = _mm512_mask_add_epi32(v_join_id, m_new_cells, v_join_id, v_one);

    m_bucket_pass = _mm512_kor(m_bucket_pass, m_match);
    m_done = _mm512_kand(m_bucket_pass,
                         _mm512_cmpeq_epi32_mask(v_join_id, v_join_num));
    // the bucket is over but it isn't passed
    m_abort = _mm512_kandn(m_bucket_pass, m_new_cells);
    m_have_tuple = _mm512_kandn(_mm512_kor(m_done, m_abort), m_have_tuple);
    v_join_id = _mm512_maskz_add_epi32(m_have_tuple, v_join_id, v_zero512);

    m_bucket_pass = _mm512_kandn(m_new_cells, m_bucket_pass);
    equal_num += _mm_popcnt_u32(m_done);
#if RESULTS
    for (int i = 0; m_done && i < vector_scale; ++i, m_done = (m_done >> 1)) {
      if (m_done & 1) {
        int output_off = 0;
        for (int j = 0; j < ht_num; ++j) {
          memcpy(payloads + cur_payloads + output_off,
                 temp_payloads[i][j] + ht[0]->global_addr, PAYLOADSIZE);
          output_off += PAYLOADSIZE;
        }
        memcpy(payloads + cur_payloads + output_off,
               pb->start + addr_offset[i] + WORDSIZE * ht_num, PAYLOADSIZE);
#if OUTPUT
        for (int j = 0; j < ht_num; ++j) {
          fprintf(fp, "%d,",
                  *((uint32_t*)(payloads + cur_payloads + j * PAYLOADSIZE)));
        }
        fprintf(fp, "%d\n",
                *((uint32_t*)(payloads + cur_payloads + ht_num * PAYLOADSIZE)));
#endif
        cur_payloads += result_size;
      }
    }
#endif
  }
// cout << "SIMD512Probe qualified tuples = " << equal_num << endl;
#if OUTPUT
  fclose(fp);
#endif
  return equal_num;
}

uint64_t Linear512ProbeHor(Table* pb, HashTable** ht, int ht_num,
                           char* payloads) {
  uint16_t vector_scale = 16, new_add;
  __mmask16 m_bucket_pass, m_done, m_match, m_abort, m_have_tuple = 0,
                                                     m_new_cells = -1;
  __m512i v_offset = _mm512_set1_epi32(0), v_addr_offset = _mm512_set1_epi32(0),
          v_join_num = _mm512_set1_epi32(ht_num),
          v_base_offset_upper =
              _mm512_set1_epi32(pb->tuple_num * pb->tuple_size),
          v_tuple_cell = _mm512_set1_epi32(0), v_right_index, v_base_offset,
          v_left_size = _mm512_set1_epi32(8), v_bucket_offset, ht_cell,
          v_factor = _mm512_set1_epi32(table_factor), v_ht_cell_offset, v_shift,
          v_buckets_minus_1, v_cell_hash, v_ht_pos, v_ht_global_addr_offset,
          v_neg_one512 = _mm512_set1_epi32(-1), v_tuple_off, v_payloads_off,
          v_zero512 = _mm512_set1_epi32(0), v_join_id = _mm512_set1_epi32(0),
          v_one = _mm512_set1_epi32(1);
  __attribute__((aligned(64)))
  uint32_t cur_offset = 0,
           cur_payloads = 0, global_addr_offset[16] = {0}, tmp_cell[16],
           *addr_offset, payloads_addr[16][6] = {0}, *ht_pos, *tuple_cell,
           *join_id, base_off[32], tuple_cell_offset[16] = {0},
           hor_probe_step = 16 * ht[0]->tuple_size, left_size[16] = {0},
           ht_cell_offset[16] = {0}, tuple_off[16] = {0}, *payloads_off,
           result_size = (ht_num + 1) * PAYLOADSIZE, buckets_minus_1[16] = {0},
           shift[16] = {0}, h_buf[16] = {0}, bucket_upper[16] = {0};
  __attribute__((aligned(64))) uint64_t htp[16] = {0};

  for (int i = 0; i <= vector_scale; ++i) {
    base_off[i] = i * pb->tuple_size;
  }
  for (int i = 0; i < ht_num; ++i) {
    tuple_cell_offset[i] = ht[i]->probe_offset;
    left_size[i] = ht[i]->tuple_size;
    ht_cell_offset[i] = ht[i]->key_offset;
    shift[i] = ht[i]->shift;
    buckets_minus_1[i] = ht[i]->slot_num - 1;
    htp[i] = (uint64_t)ht[i]->addr;
    bucket_upper[i] = ht[i]->slot_num * left_size[i];
    global_addr_offset[i] = ht[i]->global_addr_offset;
  }
  for (int i = 0; i < vector_scale; ++i) {
    tuple_off[i] = i * ht[0]->tuple_size;
  }
  ht_pos = (uint32_t*)&v_ht_pos;
  join_id = (uint32_t*)&v_join_id;

  addr_offset = (uint32_t*)&v_addr_offset;
  tuple_cell = (uint32_t*)&v_tuple_cell;
  payloads_off = (uint32_t*)&v_payloads_off;
  uint64_t equal_num = 0;
  v_base_offset = _mm512_load_si512(base_off);
  v_tuple_off = _mm512_load_si512(tuple_off);

#if OUTPUT
  FILE* fp;
  stringstream sstr;
  sstr << pthread_self();
  fp = fopen(string(("simd512hor.csv") + sstr.str()).c_str(), "wr");
  if (fp == 0) {
    assert(false && "can not open file!");
  }
#endif
  int times = 0;
  for (uint64_t cur = 0; cur < pb->tuple_num || m_have_tuple;) {
///////// step 1: load new tuples' address offsets
// the offset should be within MAX_32INT_
// the tail depends on the number of joins and tuples in each bucket
#if SEQPREFETCH
    _mm_prefetch((char*)(pb->start + cur_offset + PDIS), _MM_HINT_T0);
    _mm_prefetch((char*)(pb->start + cur_offset + PDIS + 64), _MM_HINT_T0);
    _mm_prefetch((char*)(pb->start + cur_offset + PDIS + 128), _MM_HINT_T0);
    _mm_prefetch((char*)(pb->start + cur_offset + PDIS + 192), _MM_HINT_T0);
    _mm_prefetch((char*)(pb->start + cur_offset + PDIS + 256), _MM_HINT_T0);
#endif
    v_offset = _mm512_add_epi32(_mm512_set1_epi32(cur_offset), v_base_offset);
    v_addr_offset = _mm512_mask_expand_epi32(
        v_addr_offset, _mm512_knot(m_have_tuple), v_offset);
    // count the number of empty tuples
    new_add = _mm_popcnt_u32(_mm512_knot(m_have_tuple));
    cur_offset = cur_offset + base_off[new_add];
    cur = cur + new_add;
    m_have_tuple = _mm512_cmpgt_epi32_mask(v_base_offset_upper, v_addr_offset);
    /////// step 2: load new cells from tuples
    /*
     * the cases that need to load new cells
     * (1) new tuples -> v_have_tuple
     * (2) last cells are matched for all cells in corresponding buckets ->
     * v_next_cells
     */

    v_right_index = _mm512_i32gather_epi32(v_join_id, tuple_cell_offset, 4);
    m_new_cells = _mm512_kand(m_new_cells, m_have_tuple);
    v_tuple_cell = _mm512_mask_i32gather_epi32(
        v_tuple_cell, m_new_cells,
        _mm512_add_epi32(v_addr_offset, v_right_index), pb->start, 1);

    ////// step 3: load new values in hash tables

    v_left_size = _mm512_i32gather_epi32(v_join_id, left_size, 4);
    // v_ht_cell_offset = _mm512_i32gather_epi32(v_join_id, ht_cell_offset,
    // 4);

    v_shift = _mm512_i32gather_epi32(v_join_id, shift, 4);
    v_buckets_minus_1 = _mm512_i32gather_epi32(v_join_id, buckets_minus_1, 4);
    // hash the cell values
    v_cell_hash = _mm512_mullo_epi32(v_tuple_cell, v_factor);
    v_cell_hash = _mm512_srlv_epi32(v_cell_hash, v_shift);
    // set 0 for new cells, but add 1 for old cells
    v_bucket_offset = _mm512_maskz_add_epi32(_mm512_knot(m_new_cells),
                                             v_bucket_offset, v_one);

    v_cell_hash = _mm512_add_epi32(v_cell_hash, v_bucket_offset);
    // avoid overflow
    v_cell_hash = _mm512_and_si512(v_cell_hash, v_buckets_minus_1);

    v_ht_pos = _mm512_mullo_epi32(v_cell_hash, v_left_size);
    // global address
    v_ht_global_addr_offset =
        _mm512_i32gather_epi32(v_join_id, global_addr_offset, 4);
    v_ht_pos = _mm512_add_epi32(v_ht_pos, v_ht_global_addr_offset);

    m_match = 0;
#if PREFETCH
    for (int j = 0; j < vector_scale; ++j) {
      _mm_prefetch((char*)(ht_pos[j] + ht[0]->global_addr), _MM_HINT_T0);
    }
#endif
    // probe buckets horizontally for each key
    for (int j = 0; j != 16; ++j) {
      if (((m_have_tuple >> j) & 1) == 0) {
        continue;
      }
      uint32_t global_off = ht_pos[j];
      __m512i key_x16 = _mm512_set1_epi32(tuple_cell[j]);
      short flag = 0;
      // pay attention to hor_probe_step
      for (;;) {
#if SEQPREFETCH
        _mm_prefetch((char*)(ht[0]->global_addr + global_off + PDIS),
                     _MM_HINT_T0);
        _mm_prefetch((char*)(ht[0]->global_addr + global_off + PDIS + 64),
                     _MM_HINT_T0);
#endif
        v_payloads_off =
            _mm512_add_epi32(_mm512_set1_epi32(global_off), v_tuple_off);
        // avoid overflow
        __m512i v_global_upper = _mm512_set1_epi32(
            global_addr_offset[join_id[j]] + bucket_upper[join_id[j]]);
        __mmask16 greater =
            _mm512_cmpge_epi32_mask(v_payloads_off, v_global_upper);
        v_payloads_off =
            _mm512_mask_sub_epi32(v_payloads_off, greater, v_payloads_off,
                                  _mm512_set1_epi32(bucket_upper[join_id[j]]));
        __m512i tab =
            _mm512_i32gather_epi32(v_payloads_off, ht[0]->global_addr, 1);
        __mmask16 out = _mm512_cmpeq_epi32_mask(tab, key_x16);
        if (out > 0) {
          int znum = _tzcnt_u32(out);
          payloads_addr[j][join_id[j]] = (payloads_off[znum] + WORDSIZE);
          flag = 1;
          break;
        }
        out = _mm512_cmpeq_epi32_mask(tab, v_neg_one512);
        if (out > 0) break;
        global_off =
            (global_off + hor_probe_step) >=
                    (global_addr_offset[join_id[j]] + bucket_upper[join_id[j]])
                ? global_off + hor_probe_step - bucket_upper[join_id[j]]
                : global_off + hor_probe_step;
      }

      m_match = _mm512_kor(m_match, (flag << j));
    }
    m_match = _mm512_kand(m_match, m_have_tuple);
    v_join_id = _mm512_add_epi32(v_join_id, v_one);
    m_done =
        _mm512_kand(m_match, _mm512_cmpeq_epi32_mask(v_join_id, v_join_num));
    // if any one dismatch
    m_abort = _mm512_kandn(m_match, -1);
    m_have_tuple = _mm512_kandn(_mm512_kor(m_done, m_abort), m_have_tuple);
    v_join_id = _mm512_maskz_add_epi32(m_have_tuple, v_join_id, v_zero512);

    equal_num += _mm_popcnt_u32(m_done);
#if RESULTS
    for (int i = 0; m_done && i < vector_scale; ++i, m_done = (m_done >> 1)) {
      if (m_done & 1) {
        int tmp = 0;
        for (int j = 0; j < ht_num; ++j) {
          memcpy(payloads + cur_payloads + tmp,
                 payloads_addr[i][j] + ht[0]->global_addr, PAYLOADSIZE);
          tmp += PAYLOADSIZE;
        }
        memcpy(payloads + cur_payloads + tmp,
               pb->start + addr_offset[i] + WORDSIZE * ht_num, PAYLOADSIZE);
#if OUTPUT
        for (int j = 0; j < ht_num; ++j) {
          fprintf(fp, "%d,",
                  *((uint32_t*)(payloads + cur_payloads + j * PAYLOADSIZE)));
        }
        fprintf(fp, "%d\n",
                *((uint32_t*)(payloads + cur_payloads + ht_num * PAYLOADSIZE)));
#endif
        cur_payloads += result_size;
      }
    }
#endif
  }
// cout << "SIMD512Probe qualified tuples = " << equal_num << endl;
#if OUTPUT
  fclose(fp);
#endif
  return equal_num;
}
uint64_t PrefetchLinear512ProbeHor(Table* pb, HashTable** ht, int ht_num,
                                   char* payloads) {
  uint16_t vector_scale = 16, new_add, k = 0, done = 0;
  __mmask16 m_done, m_match, m_abort;
  __m512i v_offset = _mm512_set1_epi32(0),
          v_join_num = _mm512_set1_epi32(ht_num),
          v_base_offset_upper =
              _mm512_set1_epi32(pb->tuple_num * pb->tuple_size),
          v_right_index, v_base_offset,
          v_left_size = _mm512_set1_epi32(ht[0]->tuple_size), ht_cell,
          v_factor = _mm512_set1_epi32(table_factor), v_ht_cell_offset, v_shift,
          v_buckets_minus_1, v_cell_hash, v_ht_global_addr_offset,
          v_neg_one512 = _mm512_set1_epi32(-1), v_tuple_off, v_payloads_off,
          v_zero512 = _mm512_set1_epi32(0), v_one = _mm512_set1_epi32(1);
  __attribute__((aligned(64)))
  uint32_t cur_offset = 0,
           cur_payloads = 0, global_addr_offset[16] = {0}, tmp_cell[16],
           *addr_offset, *ht_pos, *tuple_cell, *join_id, base_off[32],
           tuple_cell_offset[16] = {0}, hor_probe_step = 16 * ht[0]->tuple_size,
           left_size[16] = {0}, ht_cell_offset[16] = {0}, tuple_off[16] = {0},
           *payloads_off, result_size = (ht_num + 1) * PAYLOADSIZE,
           buckets_minus_1[16] = {0}, shift[16] = {0}, h_buf[16] = {0},
           bucket_upper[16] = {0};

  for (int i = 0; i <= vector_scale; ++i) {
    base_off[i] = i * pb->tuple_size;
  }
  for (int i = 0; i < ht_num; ++i) {
    tuple_cell_offset[i] = ht[i]->probe_offset;
    left_size[i] = ht[i]->tuple_size;
    ht_cell_offset[i] = ht[i]->key_offset;
    shift[i] = ht[i]->shift;
    buckets_minus_1[i] = ht[i]->slot_num - 1;
    bucket_upper[i] = ht[i]->slot_num * left_size[i];
    global_addr_offset[i] = ht[i]->global_addr_offset;
  }
  for (int i = 0; i < vector_scale; ++i) {
    tuple_off[i] = i * ht[0]->tuple_size;
  }
  uint64_t equal_num = 0;
  v_base_offset = _mm512_load_si512(base_off);
  v_tuple_off = _mm512_load_si512(tuple_off);

#if OUTPUT
  FILE* fp;
  stringstream sstr;
  sstr << pthread_self();
  fp = fopen(string(("simd512hor.csv") + sstr.str()).c_str(), "wr");
  if (fp == 0) {
    assert(false && "can not open file!");
  }
#endif
  SIMDMState state[SIMDStateSize];
  for (int i = 0; i < SIMDStateSize; ++i) {
    state[i].m_bucket_pass = 0;
    state[i].m_have_tuple = 0;
    state[i].m_new_cells = -1;
    state[i].stage = 1;
    state[i].v_addr_offset = _mm512_set1_epi32(0);
    state[i].v_bucket_offset = _mm512_set1_epi32(0);
    state[i].v_ht_pos = _mm512_set1_epi32(0);
    state[i].v_join_id = _mm512_set1_epi32(0);
    state[i].v_tuple_cell = _mm512_set1_epi32(0);
    memset(state[i].temp_payloads, 0, sizeof(state[i].temp_payloads));
  }
  k = 0;
  for (uint64_t cur = 0; (cur < pb->tuple_num) || (done < SIMDStateSize);) {
    k = (k >= SIMDStateSize) ? 0 : k;
    if (cur >= pb->tuple_num) {
      if (state[k].m_have_tuple == 0 && state[k].stage != 3) {
        ++done;
        state[k].stage = 3;
        ++k;
        continue;
      }
    }
    switch (state[k].stage) {
      case 1: {
        state[k].stage = 0;
///////// step 1: load new tuples' address offsets
// the offset should be within MAX_32INT_
// the tail depends on the number of joins and tuples in each bucket
#if !SEQPREFETCH
        _mm_prefetch((char*)(pb->start + cur_offset + PDIS), _MM_HINT_T0);
        _mm_prefetch((char*)(pb->start + cur_offset + PDIS + 64), _MM_HINT_T0);
        _mm_prefetch((char*)(pb->start + cur_offset + PDIS + 128), _MM_HINT_T0);
        _mm_prefetch((char*)(pb->start + cur_offset + PDIS + 192), _MM_HINT_T1);
        _mm_prefetch((char*)(pb->start + cur_offset + PDIS + 256), _MM_HINT_T1);
#endif
        v_offset =
            _mm512_add_epi32(_mm512_set1_epi32(cur_offset), v_base_offset);
        state[k].v_addr_offset = _mm512_mask_expand_epi32(
            state[k].v_addr_offset, _mm512_knot(state[k].m_have_tuple),
            v_offset);
        // count the number of empty tuples
        new_add = _mm_popcnt_u32(_mm512_knot(state[k].m_have_tuple));
        cur_offset = cur_offset + base_off[new_add];
        cur = cur + new_add;
        state[k].m_have_tuple = _mm512_cmpgt_epi32_mask(v_base_offset_upper,
                                                        state[k].v_addr_offset);
        /////// step 2: load new cells from tuples
        /*
         * the cases that need to load new cells
         * (1) new tuples -> v_have_tuple
         * (2) last cells are matched for all cells in corresponding buckets ->
         * v_next_cells
         */

        v_right_index =
            _mm512_i32gather_epi32(state[k].v_join_id, tuple_cell_offset, 4);
        state[k].m_new_cells =
            _mm512_kand(state[k].m_new_cells, state[k].m_have_tuple);
        state[k].v_tuple_cell = _mm512_mask_i32gather_epi32(
            state[k].v_tuple_cell, state[k].m_new_cells,
            _mm512_add_epi32(state[k].v_addr_offset, v_right_index), pb->start,
            1);

        ////// step 3: load new values in hash tables

        v_left_size = _mm512_i32gather_epi32(state[k].v_join_id, left_size, 4);
        // v_ht_cell_offset = _mm512_i32gather_epi32(v_join_id, ht_cell_offset,
        // 4);

        v_shift = _mm512_i32gather_epi32(state[k].v_join_id, shift, 4);
        v_buckets_minus_1 =
            _mm512_i32gather_epi32(state[k].v_join_id, buckets_minus_1, 4);
        // hash the cell values
        v_cell_hash = _mm512_mullo_epi32(state[k].v_tuple_cell, v_factor);
        v_cell_hash = _mm512_srlv_epi32(v_cell_hash, v_shift);
        // set 0 for new cells, but add 1 for old cells
        state[k].v_bucket_offset = _mm512_maskz_add_epi32(
            _mm512_knot(state[k].m_new_cells), state[k].v_bucket_offset, v_one);

        v_cell_hash = _mm512_add_epi32(v_cell_hash, state[k].v_bucket_offset);
        // avoid overflow
        v_cell_hash = _mm512_and_si512(v_cell_hash, v_buckets_minus_1);

        state[k].v_ht_pos = _mm512_mullo_epi32(v_cell_hash, v_left_size);
        // global address
        v_ht_global_addr_offset =
            _mm512_i32gather_epi32(state[k].v_join_id, global_addr_offset, 4);
        state[k].v_ht_pos =
            _mm512_add_epi32(state[k].v_ht_pos, v_ht_global_addr_offset);

#if !PREFETCH
        ht_pos = (uint32_t*)&state[k].v_ht_pos;
        for (int j = 0; j < vector_scale; ++j) {
          _mm_prefetch((char*)(ht_pos[j] + ht[0]->global_addr), _MM_HINT_T0);
        }
#endif
      } break;
      case 0: {
        m_match = 0;
        state[k].stage = 1;
        join_id = (uint32_t*)&state[k].v_join_id;
        ht_pos = (uint32_t*)&state[k].v_ht_pos;
        addr_offset = (uint32_t*)&state[k].v_addr_offset;
        tuple_cell = (uint32_t*)&state[k].v_tuple_cell;
        payloads_off = (uint32_t*)&v_payloads_off;
        // probe buckets horizontally for each key
        for (int j = 0; j < vector_scale; ++j) {
          if (((state[k].m_have_tuple >> j) & 1) == 0) {
            continue;
          }
          uint32_t global_off = ht_pos[j];
          __m512i key_x16 = _mm512_set1_epi32(tuple_cell[j]);
          short flag = 0;
          // pay attention to hor_probe_step
          for (;;) {
#if !SEQPREFETCH
            _mm_prefetch((char*)(ht[0]->global_addr + global_off + PDIS),
                         _MM_HINT_T0);
            _mm_prefetch((char*)(ht[0]->global_addr + global_off + PDIS + 64),
                         _MM_HINT_T0);
#endif
            v_payloads_off =
                _mm512_add_epi32(_mm512_set1_epi32(global_off), v_tuple_off);
            // avoid overflow
            __m512i v_global_upper = _mm512_set1_epi32(
                global_addr_offset[join_id[j]] + bucket_upper[join_id[j]]);
            __mmask16 greater =
                _mm512_cmpge_epi32_mask(v_payloads_off, v_global_upper);
            v_payloads_off = _mm512_mask_sub_epi32(
                v_payloads_off, greater, v_payloads_off,
                _mm512_set1_epi32(bucket_upper[join_id[j]]));
            __m512i tab =
                _mm512_i32gather_epi32(v_payloads_off, ht[0]->global_addr, 1);
            __mmask16 out = _mm512_cmpeq_epi32_mask(tab, key_x16);
            if (out > 0) {
              int znum = _tzcnt_u32(out);
              state[k].temp_payloads[j][join_id[j]] =
                  (payloads_off[znum] + WORDSIZE);
              flag = 1;
              break;
            }
            out = _mm512_cmpeq_epi32_mask(tab, v_neg_one512);
            if (out > 0) break;
            global_off =
                (global_off + hor_probe_step) >=
                        (global_addr_offset[join_id[j]] +
                         bucket_upper[join_id[j]])
                    ? global_off + hor_probe_step - bucket_upper[join_id[j]]
                    : global_off + hor_probe_step;
          }

          m_match = _mm512_kor(m_match, (flag << j));
        }
        m_match = _mm512_kand(m_match, state[k].m_have_tuple);
        state[k].v_join_id = _mm512_add_epi32(state[k].v_join_id, v_one);
        m_done = _mm512_kand(
            m_match, _mm512_cmpeq_epi32_mask(state[k].v_join_id, v_join_num));
        // if any one dismatch
        m_abort = _mm512_kandn(m_match, -1);
        state[k].m_have_tuple =
            _mm512_kandn(_mm512_kor(m_done, m_abort), state[k].m_have_tuple);
        state[k].v_join_id = _mm512_maskz_add_epi32(
            state[k].m_have_tuple, state[k].v_join_id, v_zero512);

        equal_num += _mm_popcnt_u32(m_done);
#if RESULTS
        for (int i = 0; m_done && i < vector_scale;
             ++i, m_done = (m_done >> 1)) {
          if (m_done & 1) {
            int tmp = 0;
            for (int j = 0; j < ht_num; ++j) {
              memcpy(payloads + cur_payloads + tmp,
                     state[k].temp_payloads[i][j] + ht[0]->global_addr,
                     PAYLOADSIZE);
              tmp += PAYLOADSIZE;
            }
            memcpy(payloads + cur_payloads + tmp,
                   pb->start + addr_offset[i] + WORDSIZE * ht_num, PAYLOADSIZE);
#if OUTPUT
            for (int j = 0; j < ht_num; ++j) {
              fprintf(fp, "%d,", *((uint32_t*)(payloads + cur_payloads +
                                               j * PAYLOADSIZE)));
            }
            fprintf(fp, "%d\n", *((uint32_t*)(payloads + cur_payloads +
                                              ht_num * PAYLOADSIZE)));
#endif
            cur_payloads += result_size;
          }
        }
#endif
      } break;
    }
    ++k;
  }
// cout << "SIMD512Probe qualified tuples = " << equal_num << endl;
#if OUTPUT
  fclose(fp);
#endif
  return equal_num;
}

uint64_t LinearSIMDProbe(Table* pb, HashTable** ht, int ht_num,
                         char* payloads) {
  __m256i v_join_num = _mm256_set1_epi32(ht_num), v_one = _mm256_set1_epi32(1),
          v_tuple_cell, ht_cell, v_next_cell, v_matches, v_abort, v_shift,
          v_ht_pos, v_ht_cell_offset = _mm256_set1_epi32(0), v_ht_addr4,
          v_ht_addr, v_left_size = _mm256_set1_epi32(8), v_done,
          v_have_tuple = _mm256_set1_epi32(0), zero256 = _mm256_set1_epi32(0),
          v_factor = _mm256_set1_epi32(table_factor), v_cell_hash,
          // need to load new cells when the bucket is over
      v_new_cells = _mm256_set1_epi32(-1), v_bucket_pass = _mm256_set1_epi32(0),
          // travel each tuple in the bucket
      v_bucket_offset = _mm256_set1_epi32(0), v_right_index,
          // join id +1 if the bucket is over; join id =0 if new tuples are
      // loaded
      v_join_id = _mm256_set1_epi32(0),
          v_base_offset_uppper =
              _mm256_set1_epi32(pb->tuple_num * pb->tuple_size),
          v_buckets_minus_1 = _mm256_set1_epi32(0), v_base_offset, v_offset,
          v_addr_offset;
  const uint32_t offset_index = 0x76543210, tuple_scale = 8;
  __attribute__((aligned(64))) int32_t base_off[10], *ht_pos, *join_id,
      *tuple_cell, *addr_offset, *done, *have_tuple, *offset,
      tuple_cell_offset[10] = {0}, left_size[8] = {0}, ht_cell_offset[8] = {0},
      buckets_minus_1[8] = {0}, mask[16] = {0}, shift[8] = {0}, tmp_cell[8];
  for (int i = 0; i <= tuple_scale; ++i) {
    base_off[i] = i * pb->tuple_size;
  }
#if MEMOUTPUT
  __attribute__((aligned(64))) uint32_t output_buffer[output_buffer_size];
#endif
  __attribute__((aligned(64))) uint64_t htp[8] = {0};
  __attribute__((aligned(64))) void * *ht_addr, **ht_addr4, *tmp_ht_addr[8];
  for (int i = tuple_scale; i < 16; ++i) {
    mask[i] = -1;
  }
  // probe offset for the tuple in probing table
  for (int i = 0; i < ht_num; ++i) {
    tuple_cell_offset[i] = ht[i]->probe_offset;
    left_size[i] = ht[i]->tuple_size;
    ht_cell_offset[i] = ht[i]->key_offset;
    htp[i] = (uint64_t)ht[i]->addr;
    shift[i] = ht[i]->shift;
    buckets_minus_1[i] = ht[i]->slot_num - 1;
  }
  left_size[ht_num] = pb->tuple_size;
  __m128i zero128 = _mm_set1_epi32(0);
  __m256i neg_one = _mm256_set1_epi32(-1);

  have_tuple = (int32_t*)&v_have_tuple;
  ht_pos = (int32_t*)&v_ht_pos;
  join_id = (int32_t*)&v_join_id;
  ht_addr = (void**)&v_ht_addr;
  ht_addr4 = (void**)&v_ht_addr4;
  done = (int32_t*)&v_done;
  tuple_cell = (int32_t*)&v_tuple_cell;
  addr_offset = (int32_t*)&v_addr_offset;
  offset = (int32_t*)&v_offset;
  uint64_t equal_num = 0;
  void* temp_result[8][8], *result_tuple;
  uint32_t temp_payloads[8][8] = {0};
  result_tuple = malloc(128);
  __m256i null_int = _mm256_set1_epi32(NULL_INT);
  v_base_offset = _mm256_load_si256((__m256i*)(&base_off));
  uint32_t cur_offset = 0, new_add = 0, continue_mask = pb->tuple_num;
  void* start_addr = pb->start;
#if OUTPUT
  FILE* fp;
  fp = fopen("simd256.csv", "wr");
#endif
  /*
   * when travel the tuples in the base table, pay attention to that the
   * offset
   * maybe overflow u32int
   */

  for (uint64_t cur = 0; cur < pb->tuple_num || continue_mask;) {
    ///////// step 1: load new tuples' address offsets
    v_next_cell = _mm256_cmpeq_epi32(v_have_tuple, zero256);
    uint32_t res32 = _mm256_movemask_epi8(v_next_cell);
    // load new address offsets according to the mask from v_have_tuple
    uint32_t mk = _pdep_u32(offset_index, res32);  // deposit contiguous masks
    uint64_t wi = _pdep_u64(mk, 0x0f0f0f0f0f0f0f0f);  // 4b->8b
    __m128i by = _mm_cvtsi64_si128(wi);
    __m256i sm = _mm256_cvtepu8_epi32(by);  // cast the mask the 256i
    // update offset
    v_offset = _mm256_add_epi32(_mm256_set1_epi32(cur_offset), v_base_offset);

    // count zero bytes in v_have_tuple
    new_add = (_mm_popcnt_u32(res32) >> 2);
    // update cursor in the probe_table
    cur = cur + new_add;
    cur_offset += base_off[new_add];
    __m256i rs = _mm256_permutevar8x32_epi32(v_offset, sm);
    // merge latest loaded address offsets with olds
    v_addr_offset = _mm256_blendv_epi8(rs, v_addr_offset, v_have_tuple);
    // valid case: addr_offset < base_offset_uppper
    v_have_tuple = _mm256_cmpgt_epi32(v_base_offset_uppper, v_addr_offset);

    // if all loaded tuples are finished, then mask = 0, so to stop this
    // loop
    continue_mask = _mm256_movemask_epi8(v_have_tuple);

/////// step 2: load new cells from tuples
/*
 * the cases that need to load new cells
 * (1) new tuples -> v_have_tuple
 * (2) last cells are matched for all cells in corresponding buckets ->
 * v_next_cells
 */
#if 1
    v_right_index =
        _mm256_i32gather_epi32(tuple_cell_offset, v_join_id, 4);  ////
#else
    v_right_index = _mm256_permutevar8x32_epi32(
        _mm256_load_si256((__m256i*)(&tuple_cell_offset)), v_join_id);
#endif
    // guarantee valid cells from valid tuples
    v_new_cells = _mm256_and_si256(v_new_cells, v_have_tuple);
    // gather cell values in new tuples
    v_tuple_cell = _mm256_mask_i32gather_epi32(
        v_tuple_cell, (int*)start_addr,
        _mm256_add_epi32(v_addr_offset, v_right_index), v_new_cells, 1);

////// step 3: load new values in hash tables
// get rid of invalid cells
//__m256i v_invalid = _mm256_cmpeq_epi32(v_tuple_cell, null_int);
// v_have_tuple = _mm256_andnot_si256(v_invalid, v_have_tuple);
// get the position of tuple values at the hash table
#if 1
    // v_left_size = _mm256_i32gather_epi32(left_size, v_join_id, 4);
    // v_ht_cell_offset =
    //    _mm256_i32gather_epi32(ht_cell_offset, v_join_id, 4);
    v_shift = _mm256_i32gather_epi32(shift, v_join_id, 4);  //////
    v_buckets_minus_1 =
        _mm256_i32gather_epi32(buckets_minus_1, v_join_id, 4);  //
#else
    v_shift = _mm256_permutevar8x32_epi32(_mm256_load_si256((__m256i*)(&shift)),
                                          v_join_id);
    v_buckets_minus_1 = _mm256_permutevar8x32_epi32(
        _mm256_load_si256((__m256i*)(&buckets_minus_1)), v_join_id);
#endif
    // hash the cell values
    v_cell_hash = _mm256_mullo_epi32(v_tuple_cell, v_factor);
    v_cell_hash = _mm256_srlv_epi32(v_cell_hash, v_shift);

    // get the position in the hash table according to hash values
    ////// enum each cell in the bucket of the hash table
    v_bucket_offset = _mm256_add_epi32(v_bucket_offset, v_one);
    // set to 0 for new cells
    v_bucket_offset = _mm256_andnot_si256(v_new_cells, v_bucket_offset);

    v_cell_hash = _mm256_add_epi32(v_cell_hash, v_bucket_offset);
    // overflow different hash tables
    v_cell_hash = _mm256_and_si256(v_cell_hash, v_buckets_minus_1);
    v_ht_pos = _mm256_mullo_epi32(v_cell_hash, v_left_size);
#if GATHERHT
    v_ht_pos = _mm256_add_epi32(v_ht_pos, v_ht_cell_offset);

    __m256i v_ht_pos_644 = _mm256_cvtepu32_epi64(
        _mm_load_si128(reinterpret_cast<__m128i*>(&ht_pos[4])));
    __m256i v_ht_pos_64 = _mm256_cvtepu32_epi64(
        _mm_load_si128(reinterpret_cast<__m128i*>(ht_pos)));

    v_ht_addr4 = _mm256_i32gather_epi64(
        (const long long*)htp,
        _mm_load_si128(reinterpret_cast<__m128i*>(&join_id[4])), 8);
    v_ht_addr = _mm256_i32gather_epi64(
        (const long long*)htp,
        _mm_load_si128(reinterpret_cast<__m128i*>(join_id)), 8);

    v_ht_addr4 = _mm256_add_epi64(v_ht_pos_644, v_ht_addr4);
    v_ht_addr = _mm256_add_epi64(v_ht_pos_64, v_ht_addr);
    ht_cell = _mm256_set_m128i(
        _mm256_mask_i64gather_epi32(zero128, 0, v_ht_addr4,
                                    _mm_load_si128((__m128i*)(&have_tuple[4])),
                                    1),
        _mm256_mask_i64gather_epi32(zero128, 0, v_ht_addr,
                                    _mm_load_si128((__m128i*)(have_tuple)), 1));
#else
#if PREFETCH

    for (int i = 0; i < tuple_scale; ++i) {
      _mm_prefetch(ht_pos[i] + htp[join_id[i]], _MM_HINT_T0);
    }
#endif
    for (int i = 0; i < tuple_scale; ++i) {
      tmp_ht_addr[i] = ht_pos[i] + htp[join_id[i]];
      tmp_cell[i] = *(uint32_t*)(tmp_ht_addr[i]);
    }
    ht_cell = _mm256_load_si256((__m256i*)tmp_cell);
#endif
    //// step 4: compare
    // load raw cell data, then judge whether they are equal ? the AND get
    // rid of invalid keys
    v_matches = _mm256_and_si256(_mm256_cmpeq_epi32(v_tuple_cell, ht_cell),
                                 v_have_tuple);

    // the bucket is over if ht cells =-1 or early break due to match
    // so need to process new cells
    // then process next buckets (increase join_id and load new cells)
    // or process next tuples
    // or abort
    v_new_cells = _mm256_cmpeq_epi32(ht_cell, neg_one);
#if EARLYBREAK
    v_new_cells = _mm256_or_si256(v_new_cells, v_matches);
#endif
    v_join_id =
        _mm256_add_epi32(v_join_id, _mm256_and_si256(v_new_cells, v_one));
    // the bucket is passed once there is a match
    v_bucket_pass = _mm256_or_si256(v_bucket_pass, v_matches);
    //   v_join_id = _mm256_add_epi32(v_join_id, v_one);
    // once there is a match, this join is passed
    // v_pass = _mm256_or_si256(v_pass, v_matches);
    // normally done = cell is empty & join_id == join_num
    v_done = _mm256_and_si256(v_bucket_pass,
                              _mm256_cmpeq_epi32(v_join_id, v_join_num));
    // the bucket is over but not passed
    v_abort = _mm256_andnot_si256(v_bucket_pass, v_new_cells);

    // have_tuple = !need_tuple & have_tuple
    v_have_tuple =
        _mm256_andnot_si256(_mm256_or_si256(v_done, v_abort), v_have_tuple);
    // initialize controlling parameters
    v_join_id = _mm256_and_si256(v_join_id, v_have_tuple);
    // initialize parameters for new buckets
    v_bucket_pass = _mm256_andnot_si256(v_new_cells, v_bucket_pass);
    equal_num += (_mm_popcnt_u32(_mm256_movemask_epi8(v_done)) >> 2);
// generate point pairs, note to use ==-1 or 0, otherwise the special
// case may lead to errors (i.e. invalid block tuples)
#if RESULTS
    int32_t* res = (int32_t*)&v_matches;
#if GATHERHT
    int mid = (tuple_scale >> 1);
    for (int i = 0; i < mid; ++i) {
      if (res[i] == -1) {
        //  temp_result[i][join_id[i]] = ht_addr[i];
        temp_payloads[i][join_id[i]] = *(uint32_t*)(WORDSIZE + ht_addr[i]);
      }
    }
    for (int i = mid; i < tuple_scale; ++i) {
      if (res[i] == -1) {
        //  temp_result[i][join_id[i]] = ht_addr4[i - mid];
        temp_payloads[i][join_id[i]] =
            *(uint32_t*)(WORDSIZE + ht_addr4[i - mid]);
      }
    }
#else
    for (int i = 0; (i < tuple_scale); ++i) {
      //      if (m_match & 1) {
      temp_payloads[i][join_id[i]] = *(uint32_t*)(WORDSIZE + tmp_ht_addr[i]);
      //    }
    }
#endif
    for (int i = 0; i < tuple_scale; ++i) {
      if (done[i]) {
        temp_payloads[i][ht_num] =
            *(uint32_t*)(start_addr + addr_offset[i] + WORDSIZE * ht_num);
#if MEMOUTPUT
        _mm256_store_si256((__m256i*)output_buffer,
                           _mm256_load_si256((__m256i*)&temp_payloads[i][0]));
#endif
#if OUTPUT
        for (int j = 0; j < ht_num; ++j) {
          fprintf(fp, "%d,", temp_payloads[i][j]);
        }
        fprintf(fp, "%d\n", temp_payloads[i][ht_num]);
#endif
      }
    }
#endif
  }
// cout << "SIMDProbe qualified tuples = " << equal_num << endl;
#if OUTPUT
  fclose(fp);
#endif
  return equal_num;
}

uint64_t LinearSIMDProbeHor(Table* pb, HashTable** ht, int ht_num,
                            char* payloads) {
  __m256i v_pass,
      v_join_num = _mm256_set1_epi32(ht_num), v_one = _mm256_set1_epi32(1),
      v_tuple_cell, ht_cell, v_next_cell, v_matches, v_abort, v_shift, v_ht_pos,
      v_ht_cell_offset = _mm256_set1_epi32(0), v_ht_addr4, v_ht_addr,
      v_left_size = _mm256_set1_epi32(8), v_done,
      v_have_tuple = _mm256_set1_epi32(0), zero256 = _mm256_set1_epi32(0),
      v_factor = _mm256_set1_epi32(table_factor), v_cell_hash,
      // need to load new cells when the bucket is over
      v_new_cells = _mm256_set1_epi32(-1), v_bucket_pass = _mm256_set1_epi32(0),
      // travel each tuple in the bucket
      v_bucket_offset = _mm256_set1_epi32(0), v_right_index,
      // join id +1 if the bucket is over; join id =0 if new tuples are
      // loaded
      v_join_id = _mm256_set1_epi32(0),
      v_base_offset_uppper = _mm256_set1_epi32(pb->tuple_num * pb->tuple_size),
      v_buckets_minus_1 = _mm256_set1_epi32(0), v_base_offset, v_offset,
      v_addr_offset, v_10mask = _mm256_set_epi32(0, -1, 0, -1, 0, -1, 0, -1);
  const uint32_t offset_index = 0x76543210, tuple_scale = 8;
  __attribute__((aligned(64))) int32_t base_off[10], *ht_pos, *join_id,
      *tuple_cell, *addr_offset, *done, *have_tuple, *offset,
      tuple_cell_offset[10] = {0}, left_size[8] = {0}, ht_cell_offset[8] = {0},
      buckets_minus_1[8] = {0}, mask[16] = {0}, shift[8] = {0}, tmp_cell[8];
  for (int i = 0; i <= tuple_scale; ++i) {
    base_off[i] = i * pb->tuple_size;
  }
#if MEMOUTPUT
  __attribute__((aligned(64))) uint32_t output_buffer[output_buffer_size];
#endif
  __attribute__((aligned(64))) uint64_t htp[8] = {0}, bucket_upper[8] = {0};
  __attribute__((aligned(64))) void * *ht_addr, **ht_addr4, *tmp_ht_addr[8];
  for (int i = tuple_scale; i < 16; ++i) {
    mask[i] = -1;
  }
  // probe offset for the tuple in probing table
  for (int i = 0; i < ht_num; ++i) {
    tuple_cell_offset[i] = ht[i]->probe_offset;
    left_size[i] = ht[i]->tuple_size;
    ht_cell_offset[i] = ht[i]->key_offset;
    htp[i] = (uint64_t)ht[i]->addr;
    shift[i] = ht[i]->shift;
    buckets_minus_1[i] = ht[i]->slot_num - 1;
    bucket_upper[i] = buckets_minus_1[i] * left_size[i];
  }
  left_size[ht_num] = pb->tuple_size;
  __m128i zero128 = _mm_set1_epi32(0);
  __m256i neg_one = _mm256_set1_epi32(-1);

  have_tuple = (int32_t*)&v_have_tuple;
  ht_pos = (int32_t*)&v_ht_pos;
  join_id = (int32_t*)&v_join_id;
  ht_addr = (void**)&v_ht_addr;
  ht_addr4 = (void**)&v_ht_addr4;
  done = (int32_t*)&v_done;
  tuple_cell = (int32_t*)&v_tuple_cell;
  addr_offset = (int32_t*)&v_addr_offset;
  offset = (int32_t*)&v_offset;
  uint64_t equal_num = 0;
  void* temp_result[8][8], *result_tuple;
  uint32_t temp_payloads[8][8] = {0};
  result_tuple = malloc(128);
  __m256i null_int = _mm256_set1_epi32(NULL_INT);
  v_base_offset = _mm256_load_si256((__m256i*)(&base_off));
  uint32_t cur_offset = 0, new_add = 0, continue_mask = pb->tuple_num;
  void* start_addr = pb->start;
#if OUTPUT
  FILE* fp;
  fp = fopen("simd256.csv", "wr");
#endif
  /*
   * when travel the tuples in the base table, pay attention to that the
   * offset
   * maybe overflow u32int
   */

  for (uint64_t cur = 0; cur < pb->tuple_num || continue_mask;) {
    ///////// step 1: load new tuples' address offsets
    v_next_cell = _mm256_cmpeq_epi32(v_have_tuple, zero256);
    uint32_t res32 = _mm256_movemask_epi8(v_next_cell);
    // load new address offsets according to the mask from v_have_tuple
    uint32_t mk = _pdep_u32(offset_index, res32);  // deposit contiguous masks
    uint64_t wi = _pdep_u64(mk, 0x0f0f0f0f0f0f0f0f);  // 4b->8b
    __m128i by = _mm_cvtsi64_si128(wi);
    __m256i sm = _mm256_cvtepu8_epi32(by);  // cast the mask the 256i
    // update offset
    v_offset = _mm256_add_epi32(_mm256_set1_epi32(cur_offset), v_base_offset);

    // count zero bytes in v_have_tuple
    new_add = (_mm_popcnt_u32(res32) >> 2);
    // update cursor in the probe_table
    cur = cur + new_add;
    cur_offset += base_off[new_add];
    __m256i rs = _mm256_permutevar8x32_epi32(v_offset, sm);
    // merge latest loaded address offsets with olds
    v_addr_offset = _mm256_blendv_epi8(rs, v_addr_offset, v_have_tuple);
    // valid case: addr_offset < base_offset_uppper
    v_have_tuple = _mm256_cmpgt_epi32(v_base_offset_uppper, v_addr_offset);

    // if all loaded tuples are finished, then mask = 0, so to stop this
    // loop
    continue_mask = _mm256_movemask_epi8(v_have_tuple);

/////// step 2: load new cells from tuples
/*
 * the cases that need to load new cells
 * (1) new tuples -> v_have_tuple
 * (2) last cells are matched for all cells in corresponding buckets ->
 * v_next_cells
 */
#if 1
    v_right_index =
        _mm256_i32gather_epi32(tuple_cell_offset, v_join_id, 4);  ////
#else
    v_right_index = _mm256_permutevar8x32_epi32(
        _mm256_load_si256((__m256i*)(&tuple_cell_offset)), v_join_id);
#endif
    // guarantee valid cells from valid tuples
    v_new_cells = _mm256_and_si256(v_new_cells, v_have_tuple);
    // gather cell values in new tuples
    v_tuple_cell = _mm256_mask_i32gather_epi32(
        v_tuple_cell, (const int*)start_addr,
        _mm256_add_epi32(v_addr_offset, v_right_index), v_new_cells, 1);

////// step 3: load new values in hash tables
// get rid of invalid cells
//__m256i v_invalid = _mm256_cmpeq_epi32(v_tuple_cell, null_int);
// v_have_tuple = _mm256_andnot_si256(v_invalid, v_have_tuple);
// get the position of tuple values at the hash table
#if 1
    // v_left_size = _mm256_i32gather_epi32(left_size, v_join_id, 4);
    // v_ht_cell_offset =
    //    _mm256_i32gather_epi32(ht_cell_offset, v_join_id, 4);
    v_shift = _mm256_i32gather_epi32(shift, v_join_id, 4);  //////
    v_buckets_minus_1 =
        _mm256_i32gather_epi32(buckets_minus_1, v_join_id, 4);  //
#else
    v_shift = _mm256_permutevar8x32_epi32(_mm256_load_si256((__m256i*)(&shift)),
                                          v_join_id);
    v_buckets_minus_1 = _mm256_permutevar8x32_epi32(
        _mm256_load_si256((__m256i*)(&buckets_minus_1)), v_join_id);
#endif
    // hash the cell values
    v_cell_hash = _mm256_mullo_epi32(v_tuple_cell, v_factor);
    v_cell_hash = _mm256_srlv_epi32(v_cell_hash, v_shift);

    // get the position in the hash table according to hash values
    ////// enum each cell in the bucket of the hash table
    v_bucket_offset = _mm256_add_epi32(v_bucket_offset, v_one);
    // set to 0 for new cells
    v_bucket_offset = _mm256_andnot_si256(v_new_cells, v_bucket_offset);

    v_cell_hash = _mm256_add_epi32(v_cell_hash, v_bucket_offset);
    // overflow different hash tables
    v_cell_hash = _mm256_and_si256(v_cell_hash, v_buckets_minus_1);
    v_ht_pos = _mm256_mullo_epi32(v_cell_hash, v_left_size);
    v_matches = zero256;
    for (int j = 0; j < tuple_scale; ++j) {
      uint32_t p = ht_pos[j];
      __m256i key_x8 = _mm256_set1_epi32(tuple_cell[j]);
#if PREFETCH
      if (j < 15) {
        _mm_prefetch((const void*)(ht_pos[j + 1] + htp[join_id[j + 1]]),
                     _MM_HINT_T0);
      }
#endif
      short flag = 0;
      for (;;) {
        __m256i tab =
            _mm256_loadu_si256((__m256i*)((void*)(htp[join_id[j]] + p)));
        __m256i out = _mm256_cmpeq_epi32(tab, key_x8);
        out = _mm256_and_si256(v_10mask, out);
        if (!_mm256_testz_si256(out, out)) {
          int znum = (_tzcnt_u32(_mm256_movemask_epi8(out)) >> 2);
          // avoid overflowing
          if (znum * left_size[join_id[j]] + p <= bucket_upper[join_id[j]]) {
            temp_payloads[j][join_id[j]] =
                *(uint32_t*)(htp[join_id[j]] + p +
                             znum * left_size[join_id[j]] + WORDSIZE);
            flag = 1;
#if EARLYBREAK
            break;
#endif
          }
        }
        out = _mm256_cmpeq_epi32(tab, neg_one);
        out = _mm256_and_si256(v_10mask, out);

        if (!_mm256_testz_si256(out, out)) break;
        p = (p + 32) >= (bucket_upper[join_id[j]])
                ? p + 32 - bucket_upper[join_id[j]]
                : p + 32;
      }
      v_matches = _mm256_mask_set1_epi32(v_matches, (flag << j), -1);
    }
    v_matches = _mm256_and_si256(v_matches, v_have_tuple);
    v_join_id = _mm256_add_epi32(v_join_id, v_one);
    // the bucket is passed once there is a match
    //   v_join_id = _mm256_add_epi32(v_join_id, v_one);
    // once there is a match, this join is passed
    // v_pass = _mm256_or_si256(v_pass, v_matches);
    // normally done = cell is empty & join_id == join_num
    v_done =
        _mm256_and_si256(v_matches, _mm256_cmpeq_epi32(v_join_id, v_join_num));
    // the bucket is over but not passed
    v_abort = _mm256_andnot_si256(v_matches, neg_one);

    // have_tuple = !need_tuple & have_tuple
    v_have_tuple =
        _mm256_andnot_si256(_mm256_or_si256(v_done, v_abort), v_have_tuple);
    // initialize controlling parameters
    v_join_id = _mm256_and_si256(v_join_id, v_have_tuple);

    equal_num += (_mm_popcnt_u32(_mm256_movemask_epi8(v_done)) >> 2);
// generate point pairs, note to use ==-1 or 0, otherwise the special
// case may lead to errors (i.e. invalid block tuples)
#if RESULTS
    int32_t* res = (int32_t*)&v_matches;
    for (int i = 0; i < tuple_scale; ++i) {
      if (done[i]) {
        temp_payloads[i][ht_num] =
            *(uint32_t*)(start_addr + addr_offset[i] + WORDSIZE * ht_num);
#if MEMOUTPUT
        _mm256_store_si256((__m256i*)output_buffer,
                           _mm256_load_si256((__m256i*)&temp_payloads[i][0]));
#endif
#if OUTPUT
        for (int j = 0; j < ht_num; ++j) {
          fprintf(fp, "%d,", temp_payloads[i][j]);
        }
        fprintf(fp, "%d\n", temp_payloads[i][ht_num]);
#endif
      }
    }
#endif
  }
// cout << "SIMDProbe qualified tuples = " << equal_num << endl;
#if OUTPUT
  fclose(fp);
#endif
  return equal_num;
}

uint64_t SIMDProbe(Table* pb, HashTable** ht, int ht_num, char* payloads) {
  __m256i v_pass,
      v_join_num = _mm256_set1_epi32(ht_num), v_one = _mm256_set1_epi32(1),
      v_tuple_cell, ht_cell, v_next_cell, v_matches, v_abort, v_shift, v_ht_pos,
      v_ht_cell_offset, v_ht_addr4, v_ht_addr, v_left_size, v_done,
      v_have_tuple = _mm256_set1_epi32(0), zero256 = _mm256_set1_epi32(0),
      v_join_id = _mm256_set1_epi32(0), v_base_offset, v_offset, v_addr_offset;
  const uint32_t offset_index = 0x76543210, tuple_scale = 8;
  __attribute__((aligned(64))) int32_t base_off[10], *ht_pos, *join_id,
      *tuple_cell, *addr_offset, *done, *have_tuple, *offset,
      tuple_cell_offset[10] = {0}, left_size[8] = {0}, ht_cell_offset[8] = {0},
      mask[16] = {0};
  for (int i = 0; i <= tuple_scale; ++i) {
    base_off[i] = i * pb->tuple_size;
  }
  __attribute__((aligned(64))) uint64_t htp[8] = {0};
  __attribute__((aligned(64))) void * *ht_addr, **ht_addr4;
  for (int i = tuple_scale; i < 16; ++i) {
    mask[i] = -1;
  }
  // probe offset for the tuple in probing table
  for (int i = 0; i < ht_num; ++i) {
    tuple_cell_offset[i] = ht[i]->probe_offset;
    left_size[i] = ht[i]->tuple_size;
    ht_cell_offset[i] = ht[i]->key_offset;
    htp[i] = (uint64_t)ht[i]->addr;
  }
  left_size[ht_num] = pb->tuple_size;
  __m128i zero128 = _mm_set1_epi32(0);
  __m256i neg_one = _mm256_set1_epi32(-1);

  have_tuple = (int32_t*)&v_have_tuple;
  ht_pos = (int32_t*)&v_ht_pos;
  join_id = (int32_t*)&v_join_id;
  ht_addr = (void**)&v_ht_addr;
  ht_addr4 = (void**)&v_ht_addr4;
  done = (int32_t*)&v_done;
  tuple_cell = (int32_t*)&v_tuple_cell;
  addr_offset = (int32_t*)&v_addr_offset;
  offset = (int32_t*)&v_offset;
  uint64_t equal_num = 0;
  void* temp_result[8][8], *result_tuple;
  result_tuple = malloc(128);
  __m256i null_int = _mm256_set1_epi32(NULL_INT);
  v_base_offset = _mm256_load_si256((__m256i*)(&base_off));
  uint32_t cur_offset = 0, new_add = 0, continue_mask = pb->tuple_num;
  void* start_addr = pb->start;
  for (uint64_t cur = 0; true;) {
    if (cur < pb->tuple_num) {
      ///////// step 1: load new tuples' address offsets
      v_next_cell = _mm256_cmpeq_epi32(v_have_tuple, zero256);
      uint32_t res32 = _mm256_movemask_epi8(v_next_cell);
      // load new address offsets according to the mask from v_have_tuple
      uint32_t mk = _pdep_u32(offset_index, res32);  // deposit contiguous masks
      uint64_t wi = _pdep_u64(mk, 0x0f0f0f0f0f0f0f0f);  // 4b->8b
      __m128i by = _mm_cvtsi64_si128(wi);
      __m256i sm = _mm256_cvtepu8_epi32(by);  // cast the mask the 256i
      // update offset
      v_offset = _mm256_add_epi32(_mm256_set1_epi32(cur_offset), v_base_offset);

      // count zero bytes in v_have_tuple
      new_add = (_mm_popcnt_u32(res32) >> 2);
      // update cursor in the probe_table
      int temp = cur + new_add;
      if (temp <= pb->tuple_num) {
        cur = temp;
        cur_offset += base_off[new_add];
#if 0
      if (cur_offset > OVER_FLOW) {
        start_addr += OVER;
        cur_offset -= OVER;
        cout << "cur = " << cur << endl;
        for (int i = 0; i < tuple_scale; ++i) {
          if (addr_offset[i] > 0 && addr_offset[i] < OVER) {
            assert(false && "invalid offset");
          } else if (addr_offset[i] > 0 && addr_offset[i] >= OVER) {
            addr_offset[i] -= OVER;
          }
          cout << i << " " << addr_offset[i] << endl;
        }
        v_addr_offset =
            _mm256_sub_epi32(v_addr_offset, _mm256_set1_epi32(OVER));
      }
#endif
      } else {
        for (int i = 0; i < pb->tuple_num - cur; ++i) {
          mask[i] = 0;
        }
        for (int i = pb->tuple_num - cur; i < tuple_scale; ++i) {
          mask[i] = -1;
        }
        __m256i v_mask = _mm256_load_si256((__m256i*)mask);
        v_offset = _mm256_or_si256(v_offset, v_mask);

#if 0
        cout << "tuple_num = " << pb->tuple_num << " - cur = " << cur
             << " == " << pb->tuple_num - cur << endl;
        for (int i = 0; i < tuple_scale; ++i) {
          cout << "offset " << i << "  " << offset[i] << endl;
        }
        int* tmp = (int*)&v_have_tuple;
        for (int i = 0; i < tuple_scale; ++i) {
          cout << "have tuple " << i << "  " << tmp[i] << endl;
        }
        tmp = (int*)&sm;
        for (int i = 0; i < tuple_scale; ++i) {
          cout << "sm " << i << "  " << tmp[i] << endl;
        }
        for (int i = 0; i < tuple_scale; ++i) {
          cout << "before addr " << i << "  " << addr_offset[i] << endl;
        }
#endif
        cur = pb->tuple_num;
      }
      __m256i rs = _mm256_permutevar8x32_epi32(v_offset, sm);

      // merge latest loaded address offsets with olds
      v_addr_offset = _mm256_blendv_epi8(rs, v_addr_offset, v_have_tuple);

      // invalid case: add_offset == -1, so all cells > -1
      v_have_tuple = _mm256_cmpgt_epi32(v_addr_offset, neg_one);
#if 0
      if (cur == pb->tuple_num) {
        int* tmp = (int*)&v_have_tuple;
        for (int i = 0; i < tuple_scale; ++i) {
          cout << "after have tuple " << i << "  " << tmp[i] << endl;
        }
        for (int i = 0; i < tuple_scale; ++i) {
          cout << "after addr " << i << "  " << addr_offset[i] << endl;
        }
      }
#endif
    } else if (continue_mask == 0) {
      break;
    }

    // if all loaded tuples are finished, then mask = 0, so to stop this
    // loop
    continue_mask = _mm256_movemask_epi8(v_have_tuple);
    /////// step 2: load new cells from tuples
    /*
     * the cases that need to load new cells
     * (1) new tuples -> v_have_tuple
     * (2) last cells are matched for all cells in corresponding buckets ->
     * v_next_cells
     *  but for ARRAYHASHTUPLE is special, all cells just match once
     */
    __m256i right_index =
        _mm256_i32gather_epi32(tuple_cell_offset, v_join_id, 4);

    // gather cell values in new tuples
    v_tuple_cell = _mm256_mask_i32gather_epi32(
        v_tuple_cell, (const int*)start_addr,
        _mm256_add_epi32(v_addr_offset, right_index), v_have_tuple, 1);
    ////// step 3: load new values in hash tables
    // get rid of invalid cells
    __m256i v_invalid = _mm256_cmpeq_epi32(v_tuple_cell, null_int);
    v_have_tuple = _mm256_andnot_si256(v_invalid, v_have_tuple);
    // get the position of tuple values at the hash table
    v_left_size = _mm256_i32gather_epi32(left_size, v_join_id, 4);
    v_ht_cell_offset = _mm256_i32gather_epi32(ht_cell_offset, v_join_id, 4);
    v_ht_pos = _mm256_mullo_epi32(v_tuple_cell, v_left_size);
    v_ht_pos = _mm256_add_epi32(v_ht_pos, v_ht_cell_offset);

    __m256i v_ht_pos_644 = _mm256_cvtepu32_epi64(
        _mm_load_si128(reinterpret_cast<__m128i*>(&ht_pos[4])));
    __m256i v_ht_pos_64 = _mm256_cvtepu32_epi64(
        _mm_load_si128(reinterpret_cast<__m128i*>(ht_pos)));

    v_ht_addr4 = _mm256_i32gather_epi64(
        (const long long*)htp,
        _mm_load_si128(reinterpret_cast<__m128i*>(&join_id[4])), 8);
    v_ht_addr = _mm256_i32gather_epi64(
        (const long long*)htp,
        _mm_load_si128(reinterpret_cast<__m128i*>(join_id)), 8);

    ht_cell = _mm256_set_m128i(
        _mm256_mask_i64gather_epi32(
            zero128, 0, _mm256_add_epi64(v_ht_pos_644, v_ht_addr4),
            _mm_load_si128((__m128i*)(&have_tuple[4])), 1),
        _mm256_mask_i64gather_epi32(zero128, 0,
                                    _mm256_add_epi64(v_ht_pos_64, v_ht_addr),
                                    _mm_load_si128((__m128i*)(have_tuple)), 1));

    //// step 4: compare
    // load raw cell data, then judge whether they are equal ? the AND get
    // rid of invalid keys
    v_matches = _mm256_and_si256(_mm256_cmpeq_epi32(v_tuple_cell, ht_cell),
                                 v_have_tuple);

// generate point pairs, note to use ==-1 or 0, otherwise the special
// case may lead to errors (i.e. invalid block tuples)
#if RESULTS
    int32_t* res = (int32_t*)&v_matches;
    int mid = tuple_scale >> 1;
    for (int i = 0; i < mid; ++i) {
      if (res[i] == -1) {
        temp_result[i][join_id[i]] = ht_addr[i];
      }
    }
    for (int i = mid; i < tuple_scale; ++i) {
      if (res[i] == -1) {
        temp_result[i][join_id[i]] = ht_addr4[i - mid];
      }
    }
#endif
    v_join_id = _mm256_add_epi32(v_join_id, v_one);
    // once there is a match, this join is passed
    // v_pass = _mm256_or_si256(v_pass, v_matches);
    // normally done = cell is empty & join_id == join_num
    v_done =
        _mm256_and_si256(v_matches, _mm256_cmpeq_epi32(v_join_id, v_join_num));
    // abort = cell is empty & !pass
    v_abort = _mm256_andnot_si256(v_matches, neg_one);

    // have_tuple = !need_tuple & have_tuple
    v_have_tuple =
        _mm256_andnot_si256(_mm256_or_si256(v_done, v_abort), v_have_tuple);
    // initialize controlling parameters
    v_join_id = _mm256_and_si256(v_join_id, v_have_tuple);
#if RESULTS
    for (int i = 0; i < tuple_scale; ++i) {
      if (done[i]) {
        ++equal_num;
        temp_result[i][ht_num] = start_addr + addr_offset[i];
        int copyed_bytes = 0;
        for (int id = 0; id <= ht_num; ++id) {
          memcpy(result_tuple + copyed_bytes, temp_result[i][id],
                 left_size[id]);
          copyed_bytes += left_size[id];
        }
      }
    }
#else
    equal_num += (_mm_popcnt_u32(_mm256_movemask_epi8(v_done)) >> 2);
#endif
  }
  // cout << "SIMDProbe qualified tuples = " << equal_num << endl;

  return equal_num;
}

#endif
