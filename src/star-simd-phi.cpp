#ifndef __SIMDPHITEST__
#define __SIMDPHITEST__
#include "star-simd.h"

// use short mask to avoid long mask
uint64_t Linear512Probe(Table* pb, HashTable** ht, int ht_num, char* payloads) {
  uint16_t vector_scale = 16, new_add = 0;
  __mmask16 m_bucket_pass = 0, m_done = 0, m_match = 0, m_abort = 0,
            m_have_tuple = 0, m_new_cells = -1;
  __m512i v_offset = _mm512_set1_epi32(0), v_addr_offset = _mm512_set1_epi32(0),
          v_join_num = _mm512_set1_epi32(ht_num),
          v_base_offset_upper =
              _mm512_set1_epi32(pb->tuple_num * pb->tuple_size),
          v_tuple_cell = _mm512_set1_epi32(0), v_right_index,
          v_left_size = _mm512_set1_epi32(ht[0]->tuple_size), v_bucket_offset,
          v_ht_cell, v_factor = _mm512_set1_epi32(table_factor),
          v_ht_cell_offset, v_shift, v_buckets_minus_1, v_cell_hash, v_ht_pos,
          v_neg_one512 = _mm512_set1_epi32(-1), v_ht_cell_upper,
          v_ht_cell_lower, v_zero512 = _mm512_set1_epi32(0),
          v_join_id = _mm512_set1_epi32(0), v_ht_global_addr_offset,
          v_one = _mm512_set1_epi32(1), v_payloads, v_payloads_off,
          v_word_size = _mm512_set1_epi32(WORDSIZE);
  __attribute__((aligned(64)))
  uint32_t cur_offset = 0,
           cur_payloads = 0, global_addr_offset[16] = {0}, *addr_offset,
           payloads_addr[16][ht_num + 1], *ht_pos, *join_id, base_off[32],
           tuple_cell_offset[16] = {0}, left_size[16] = {0},
           ht_cell_offset[16] = {0}, payloads_off[32],
           result_size = (ht_num + 1) * PAYLOADSIZE, buckets_minus_1[16] = {0},
           shift[16] = {0};

  for (int i = 0; i <= vector_scale; ++i) {
    base_off[i] = i * pb->tuple_size;
    // subject to the second size of temp_payloads
    payloads_off[i] = i * (ht_num + 1);
  }
  for (int i = 0; i < ht_num; ++i) {
    tuple_cell_offset[i] = ht[i]->probe_offset;
    left_size[i] = ht[i]->tuple_size;
    ht_cell_offset[i] = ht[i]->key_offset;
    shift[i] = ht[i]->shift;
    buckets_minus_1[i] = ht[i]->slot_num - 1;
    global_addr_offset[i] = ht[i]->global_addr_offset;
  }
  ht_pos = (uint32_t*)&v_ht_pos;
  join_id = (uint32_t*)&v_join_id;
  addr_offset = (uint32_t*)&v_addr_offset;

  uint64_t equal_num = 0;
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

    v_addr_offset = _mm512_mask_loadunpacklo_epi32(
        v_addr_offset, _mm512_knot(m_have_tuple), base_off);
    // v_addr_offset = _mm512_mask_loadunpackhi_epi32(
    // v_addr_offset, _mm512_knot(m_have_tuple), &base_off[16]);
    v_addr_offset =
        _mm512_mask_add_epi32(v_addr_offset, _mm512_knot(m_have_tuple),
                              v_addr_offset, _mm512_set1_epi32(cur_offset));
    new_add = _mm_countbits_32(_mm512_mask2int(_mm512_knot(m_have_tuple)));
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

#if 1
    v_left_size = _mm512_i32gather_epi32(v_join_id, left_size, 4);
    v_ht_cell_offset = _mm512_i32gather_epi32(v_join_id, ht_cell_offset, 4);
#endif
    v_shift = _mm512_i32gather_epi32(v_join_id, shift, 4);
    v_buckets_minus_1 = _mm512_i32gather_epi32(v_join_id, buckets_minus_1, 4);
    // hash the cell values
    v_cell_hash = _mm512_mullo_epi32(v_tuple_cell, v_factor);
    v_cell_hash = _mm512_srlv_epi32(v_cell_hash, v_shift);
    // set 0 for new cells, but add 1 for old cells
    v_bucket_offset = _mm512_mask_add_epi32(v_zero512, _mm512_knot(m_new_cells),
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
    _mm512_mask_prefetch_i32gather_ps(v_ht_pos, m_have_tuple,
                                      ht[0]->global_addr, 1, _MM_HINT_T0);
#endif
    v_ht_cell = _mm512_mask_i32gather_epi32(v_neg_one512, m_have_tuple,
                                            v_ht_pos, ht[0]->global_addr, 1);

    //// step 4: compare
    // load raw cell data, then judge whether they are equal ? the AND get
    // rid of invalid keys
    m_match = _mm512_cmpeq_epi32_mask(v_tuple_cell, v_ht_cell);
    m_match = _mm512_kand(m_match, m_have_tuple);

    // store the global address offset of payloads
    v_ht_pos = _mm512_add_epi32(v_ht_pos, v_word_size);
    v_offset = _mm512_add_epi32(v_join_id, v_payloads_off);
    _mm512_mask_i32scatter_epi32(payloads_addr, m_match, v_offset, v_ht_pos, 4);

    // the bucket is over if ht cells =-1 or early break due to match
    // so need to process new cells
    // then process next buckets (increase join_id and load new cells)
    // or process next tuples
    // or abort
    m_new_cells = _mm512_cmpeq_epi32_mask(v_ht_cell, v_neg_one512);
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
    v_join_id = _mm512_mask_blend_epi32(m_have_tuple, v_zero512, v_join_id);
    m_bucket_pass = _mm512_kandn(m_new_cells, m_bucket_pass);
    equal_num += _mm_countbits_32(_mm512_mask2int(m_done));

#if RESULTS
    for (int i = 0; m_done && i < vector_scale; ++i, m_done = (m_done >> 1)) {
      if (m_done & 1) {
        int output_off = 0;
        for (int j = 0; j < ht_num; ++j) {
          memcpy(payloads + cur_payloads + output_off,
                 payloads_addr[i][j] + ht[0]->global_addr, PAYLOADSIZE);
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
inline uint16_t zero_count(uint16_t a) {
  uint16_t num = 0;
  while ((a & 1) == 0 && num < 16) {
    ++num;
    a = (a >> 1);
  }
  return num;
}

uint64_t Linear512ProbeHor(Table* pb, HashTable** ht, int ht_num,
                           char* payloads) {
  uint16_t vector_scale = 16, new_add = 0;
  __mmask16 m_bucket_pass = 0, m_done = 0, m_match = 0, m_abort = 0,
            m_have_tuple = 0;
  __m512i v_offset = _mm512_set1_epi32(0), v_addr_offset = _mm512_set1_epi32(0),
          v_join_num = _mm512_set1_epi32(ht_num),
          v_base_offset_upper =
              _mm512_set1_epi32(pb->tuple_num * pb->tuple_size),
          v_tuple_cell = _mm512_set1_epi32(0), v_right_index, v_base_offset,
          v_left_size = _mm512_set1_epi32(ht[0]->tuple_size), v_bucket_offset,
          ht_cell, v_factor = _mm512_set1_epi32(table_factor), v_ht_cell_offset,
          v_shift, v_buckets_minus_1, v_cell_hash, v_ht_pos, v_tuple_off,
          v_neg_one512 = _mm512_set1_epi32(-1), v_ht_global_addr_offset,
          v_zero512 = _mm512_set1_epi32(0), v_join_id = _mm512_set1_epi32(0),
          v_one = _mm512_set1_epi32(1), v_payloads_off;
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
    v_addr_offset = _mm512_mask_loadunpacklo_epi32(
        v_addr_offset, _mm512_knot(m_have_tuple), base_off);
    // v_addr_offset = _mm512_mask_loadunpackhi_epi32(
    //     v_addr_offset, _mm512_knot(m_have_tuple), &base_off[16]);
    v_addr_offset =
        _mm512_mask_add_epi32(v_addr_offset, _mm512_knot(m_have_tuple),
                              v_addr_offset, _mm512_set1_epi32(cur_offset));
    new_add = _mm_countbits_32(_mm512_mask2int(_mm512_knot(m_have_tuple)));

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

    v_right_index =
        _mm512_i32gather_epi32(v_join_id, tuple_cell_offset, 4);  ///////
    v_tuple_cell = _mm512_mask_i32gather_epi32(
        v_tuple_cell, m_have_tuple,
        _mm512_add_epi32(v_addr_offset, v_right_index), pb->start, 1);

////// step 3: load new values in hash tables
#if 1
    v_left_size =
        _mm512_i32gather_epi32(v_join_id, left_size, 4);  /////////////
    v_ht_cell_offset =
        _mm512_i32gather_epi32(v_join_id, ht_cell_offset, 4);  /////
#endif
    v_shift = _mm512_i32gather_epi32(v_join_id, shift, 4);  ////////////////////
    v_buckets_minus_1 =
        _mm512_i32gather_epi32(v_join_id, buckets_minus_1, 4);  ////
    // hash the cell values
    v_cell_hash = _mm512_mullo_epi32(v_tuple_cell, v_factor);
    v_cell_hash = _mm512_srlv_epi32(v_cell_hash, v_shift);
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
      uint32_t global_off = ht_pos[j];
      __m512i key_x16 = _mm512_set1_epi32(tuple_cell[j]);
      short flag = 0;
      for (;;) {
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
          int znum = zero_count(out);
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
    v_join_id = _mm512_mask_blend_epi32(m_have_tuple, v_zero512, v_join_id);
    equal_num += _mm_countbits_32(_mm512_mask2int(m_done));
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

#endif
