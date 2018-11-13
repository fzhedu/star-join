#ifndef __SINGLEPROBE__
#define __SINGLEPROBE__
#include "star-simd.h"
#define _mm256_set_m128i(v0, v1) \
  _mm256_insertf128_si256(_mm256_castsi128_si256(v1), (v0), 1)
uint64_t SIMDSingleProbe(Table* pb, HashTable** ht, int ht_num,
                         char* payloads) {
  uint64_t equal_num = 0;
  uint16_t vector_scale = 16, new_add;
  __mmask16 m_match = 0, m_have_tuple = 0, m_new_cells = -1;
  __m512i v_offset = _mm512_set1_epi32(0), v_addr_offset = _mm512_set1_epi32(0),
          v_base_offset_upper =
              _mm512_set1_epi32(pb->tuple_num * pb->tuple_size),
          v_tuple_cell = _mm512_set1_epi32(0), v_base_offset,
          v_left_size = _mm512_set1_epi32(8),
          v_bucket_offset = _mm512_set1_epi32(0), ht_cell,
          v_factor = _mm512_set1_epi32(table_factor), v_shift,
          v_buckets_minus_1, v_cell_hash, v_ht_pos = _mm512_set1_epi32(0),
          v_neg_one512 = _mm512_set1_epi32(-1), v_ht_upper,
          v_zero512 = _mm512_set1_epi32(0), v_one = _mm512_set1_epi32(1),
          v_word_size = _mm512_set1_epi32(WORDSIZE);
  __attribute__((aligned(64))) uint32_t cur_offset = 0, cur_payloads = 0,
                                        *addr_offset, *ht_pos, base_off[32],
                                        buckets_minus_1[16] = {0};
  for (int i = 0; i <= vector_scale; ++i) {
    base_off[i] = i * pb->tuple_size;
  }
  v_base_offset = _mm512_load_epi32(base_off);
  v_left_size = _mm512_set1_epi32(ht[0]->tuple_size);
  v_shift = _mm512_set1_epi32(ht[0]->shift);
  v_buckets_minus_1 = _mm512_set1_epi32(ht[0]->slot_num - 1);
  v_ht_upper = _mm512_set1_epi32(ht[0]->slot_num * ht[0]->tuple_size);
  ht_pos = (uint32_t*)&v_ht_pos;
  addr_offset = (uint32_t*)&v_addr_offset;

  for (uint64_t cur = 0; cur < pb->tuple_num || m_have_tuple;) {
///////// step 1: load new tuples' address offsets
// the offset should be within MAX_32INT_
// the tail depends on the number of joins and tuples in each bucket
#if !SEQPREFETCH
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
    m_new_cells = _mm512_knot(m_have_tuple);
    new_add = _mm_popcnt_u32(_mm512_knot(m_have_tuple));
    cur_offset = cur_offset + base_off[new_add];
    cur = cur + new_add;
    m_have_tuple = _mm512_cmpgt_epi32_mask(v_base_offset_upper, v_addr_offset);
    ///// step 2: load new cells from right tuples;
    m_new_cells = _mm512_kand(m_new_cells, m_have_tuple);
    // maybe need offset within a tuple
    v_tuple_cell = _mm512_mask_i32gather_epi32(v_tuple_cell, m_new_cells,
                                               v_addr_offset, pb->start, 1);
    ///// step 3: load new values from hash tables;
    // hash the cell values
    v_cell_hash = _mm512_mullo_epi32(v_tuple_cell, v_factor);
    v_cell_hash = _mm512_srlv_epi32(v_cell_hash, v_shift);
#if 0
    // set 0 for new cells, but add 1 for old cells
    v_bucket_offset = _mm512_maskz_add_epi32(_mm512_knot(m_new_cells),
                                             v_bucket_offset, v_one);

    v_cell_hash = _mm512_add_epi32(v_cell_hash, v_bucket_offset);
    // avoid overflow
    v_cell_hash = _mm512_and_si512(v_cell_hash, v_buckets_minus_1);

    v_ht_pos = _mm512_mullo_epi32(v_cell_hash, v_left_size);
#else
    // new hash
    v_cell_hash = _mm512_mullo_epi32(v_cell_hash, v_left_size);
    // old_hash = old_hash+left_size
    v_ht_pos = _mm512_add_epi32(v_ht_pos, v_left_size);
    // old_hash = old_hash >= upper ? 0 : old_hash;
    v_ht_pos = _mm512_maskz_mov_epi32(
        _mm512_cmplt_epi32_mask(v_ht_pos, v_ht_upper), v_ht_pos);
    // combine new hash value with old hash value
    v_ht_pos = _mm512_mask_mov_epi32(v_ht_pos, m_new_cells, v_cell_hash);

#endif
    ht_cell = _mm512_mask_i32gather_epi32(v_neg_one512, m_have_tuple, v_ht_pos,
                                          ht[0]->global_addr, 1);

    ///// step 4: compare;
    m_match = _mm512_cmpeq_epi32_mask(v_tuple_cell, ht_cell);
    m_match = _mm512_kand(m_match, m_have_tuple);
    equal_num += _mm_popcnt_u32(m_match);
    m_new_cells = _mm512_cmpeq_epi32_mask(ht_cell, v_neg_one512);
#if EARLYBREAK
    m_new_cells = _mm512_kor(m_new_cells, m_match);
#endif
    m_have_tuple = _mm512_kandn(m_new_cells, m_have_tuple);

///// step 5: generate results;
#if RESULTS
    for (int i = 0; (i < vector_scale) && m_match;
         ++i, m_match = (m_match >> 1)) {
      if (m_match & 1) {
        memcpy(payloads + cur_payloads,
               ht[0]->global_addr + ht_pos[i] + WORDSIZE, PAYLOADSIZE);
        memcpy(payloads + cur_payloads + PAYLOADSIZE,
               pb->start + addr_offset[i] + WORDSIZE, PAYLOADSIZE);
        // cur_payloads += PAYLOADSIZE + PAYLOADSIZE;
      }
    }
#endif
  }
  return equal_num;
}

#endif
