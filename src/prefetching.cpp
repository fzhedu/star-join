#include "star-simd.h"
#include <stdlib.h>
// 128 for multi-stage

struct State {
  uint32_t key;
  uint32_t pb_off;
  uint32_t ht_off;
  char stage;
};
struct StateSIMD {
  __m512i key;
  __m512i pb_off;
  __m512i ht_off;
  __mmask16 m_have_tuple;
  char stage;
};

uint64_t SIMDAMACProbe(Table* pb, HashTable** ht, int ht_num, char* payloads) {
  uint64_t equal_num = 0;
  uint16_t vector_scale = 16, new_add, k = 0, done = 0;
  __mmask16 m_match = 0, m_new_cells = -1;
  __m512i v_offset = _mm512_set1_epi32(0),
          v_base_offset_upper =
              _mm512_set1_epi32(pb->tuple_num * pb->tuple_size),
          v_tuple_cell = _mm512_set1_epi32(0), v_base_offset,
          v_left_size = _mm512_set1_epi32(8), ht_cell,
          v_ht_upper = _mm512_set1_epi32(ht[0]->slot_num * ht[0]->tuple_size),
          v_factor = _mm512_set1_epi32(table_factor), v_shift, v_cell_hash,
          v_ht_pos, v_neg_one512 = _mm512_set1_epi32(-1),
          v_zero512 = _mm512_set1_epi32(0), v_one = _mm512_set1_epi32(1),
          v_word_size = _mm512_set1_epi32(WORDSIZE);
  __attribute__((aligned(64))) uint32_t cur_offset = 0, cur_payloads = 0,
                                        *addr_offset, *ht_pos, base_off[32];
  for (int i = 0; i <= vector_scale; ++i) {
    base_off[i] = i * pb->tuple_size;
  }
  v_base_offset = _mm512_load_epi32(base_off);
  v_left_size = _mm512_set1_epi32(ht[0]->tuple_size);
  v_shift = _mm512_set1_epi32(ht[0]->shift);

  StateSIMD state[SIMDStateSize];
  // init # of the state
  for (int i = 0; i < SIMDStateSize; ++i) {
    state[i].stage = 1;
    state[i].m_have_tuple = 0;
    state[i].ht_off = _mm512_set1_epi32(0);
    // state[i].pb_off = _mm512_set1_epi32(0);
    // state[i].key = _mm512_set1_epi32(0);
  }
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
      case 1: {  // init state for each tuple
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

        v_offset =
            _mm512_add_epi32(_mm512_set1_epi32(cur_offset), v_base_offset);
        state[k].pb_off = _mm512_mask_expand_epi32(
            state[k].pb_off, _mm512_knot(state[k].m_have_tuple), v_offset);
        // count the number of empty tuples
        m_new_cells = _mm512_knot(state[k].m_have_tuple);
        new_add = _mm_popcnt_u32(_mm512_knot(state[k].m_have_tuple));
        cur_offset = cur_offset + base_off[new_add];
        cur = cur + new_add;
        state[k].m_have_tuple =
            _mm512_cmpgt_epi32_mask(v_base_offset_upper, state[k].pb_off);

        ///// step 2: load new cells from right tuples;
        m_new_cells = _mm512_kand(m_new_cells, state[k].m_have_tuple);
        // maybe need offset within a tuple
        state[k].key = _mm512_mask_i32gather_epi32(
            state[k].key, m_new_cells, state[k].pb_off, pb->start, 1);
        ///// step 3: load new values from hash tables;
        // hash the cell values
        v_cell_hash = _mm512_mullo_epi32(state[k].key, v_factor);
        v_cell_hash = _mm512_srlv_epi32(v_cell_hash, v_shift);

        // new hash
        v_cell_hash = _mm512_mullo_epi32(v_cell_hash, v_left_size);
        // old_hash = old_hash+left_size
        state[k].ht_off = _mm512_add_epi32(state[k].ht_off, v_left_size);
        // old_hash = old_hash >= upper ? 0 : old_hash;
        state[k].ht_off = _mm512_maskz_mov_epi32(
            _mm512_cmplt_epi32_mask(state[k].ht_off, v_ht_upper),
            state[k].ht_off);
        // combine new hash value with old hash value
        state[k].ht_off =
            _mm512_mask_mov_epi32(state[k].ht_off, m_new_cells, v_cell_hash);
        state[k].stage = 0;
#if MultiPrefetch
        ht_pos = (uint32_t*)&state[k].ht_off;
        for (int i = 0; i < vector_scale; ++i) {
          _mm_prefetch((char*)(ht[0]->addr + ht_pos[i]), _MM_HINT_T1);
        }
#else
        ht_pos = (uint32_t*)&state[k].ht_off;
        for (int i = 0; i < vector_scale; ++i) {
          _mm_prefetch((char*)(ht[0]->addr + ht_pos[i]), _MM_HINT_T0);
        }
#endif
      } break;
      case 0: {
#if MultiPrefetch  ///
        if ((k + SIMDStep < SIMDStateSize) && state[k + SIMDStep].stage == 2) {
          ht_pos = (uint32_t*)&state[k + SIMDStep].ht_off;
          for (int i = 0; i < vector_scale; ++i) {
            _mm_prefetch((char*)(ht[0]->addr + ht_pos[i]), _MM_HINT_T0);
          }
        }
#endif
#if RESULTS
        ht_cell =
            _mm512_mask_i32gather_epi32(v_neg_one512, state[k].m_have_tuple,
                                        state[k].ht_off, ht[0]->global_addr, 1);

        ///// step 4: compare;
        m_match = _mm512_cmpeq_epi32_mask(state[k].key, ht_cell);
        m_match = _mm512_kand(m_match, state[k].m_have_tuple);
        equal_num += _mm_popcnt_u32(m_match);
        m_new_cells = _mm512_cmpeq_epi32_mask(ht_cell, v_neg_one512);
        m_new_cells = _mm512_kor(m_new_cells, m_match);
        state[k].m_have_tuple =
            _mm512_kandn(m_new_cells, state[k].m_have_tuple);

        ///// step 5: generate results;
        addr_offset = (uint32_t*)&state[k].pb_off;
        ht_pos = (uint32_t*)&state[k].ht_off;

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
        state[k].stage = 1;
#elif 1
        ht_cell =
            _mm512_mask_i32gather_epi32(v_neg_one512, state[k].m_have_tuple,
                                        state[k].ht_off, ht[0]->global_addr, 1);
        ///// step 4: compare;
        m_match = _mm512_cmpeq_epi32_mask(state[k].key, ht_cell);
        // m_match = _mm512_kand(m_match, state[k].m_have_tuple);
        m_new_cells = _mm512_cmpeq_epi32_mask(ht_cell, v_neg_one512);
        m_new_cells = _mm512_kor(m_new_cells, m_match);
        m_new_cells = _mm512_kand(m_new_cells, state[k].m_have_tuple);
        if (m_new_cells != state[k].m_have_tuple) {
          // old_hash = old_hash+left_size,
          state[k].ht_off =
              _mm512_mask_add_epi32(state[k].ht_off, _mm512_knot(m_new_cells),
                                    state[k].ht_off, v_left_size);
          // old_hash = old_hash >= upper ? 0 : old_hash;
          state[k].ht_off = _mm512_maskz_mov_epi32(
              _mm512_cmplt_epi32_mask(state[k].ht_off, v_ht_upper),
              state[k].ht_off);
          state[k].stage = 0;

        } else {
          equal_num += _mm_popcnt_u32(m_match);
          state[k].m_have_tuple =
              _mm512_kandn(m_new_cells, state[k].m_have_tuple);
          state[k].stage = 1;
          addr_offset = (uint32_t*)&state[k].pb_off;
          ht_pos = (uint32_t*)&state[k].ht_off;

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
        }
#else
        m_new_cells = 0;
        while (m_new_cells != state[k].m_have_tuple) {
          ht_cell = _mm512_mask_i32gather_epi32(
              v_neg_one512, state[k].m_have_tuple, state[k].ht_off,
              ht[0]->global_addr, 1);
          ///// step 4: compare;
          m_match = _mm512_cmpeq_epi32_mask(state[k].key, ht_cell);
          // m_match = _mm512_kand(m_match, state[k].m_have_tuple);
          m_new_cells = _mm512_cmpeq_epi32_mask(ht_cell, v_neg_one512);
          m_new_cells = _mm512_kor(m_new_cells, m_match);
          m_new_cells = _mm512_kand(m_new_cells, state[k].m_have_tuple);

          // old_hash = old_hash+left_size,
          state[k].ht_off =
              _mm512_mask_add_epi32(state[k].ht_off, _mm512_knot(m_new_cells),
                                    state[k].ht_off, v_left_size);
          // old_hash = old_hash >= upper ? 0 : old_hash;
          state[k].ht_off = _mm512_maskz_mov_epi32(
              _mm512_cmplt_epi32_mask(state[k].ht_off, v_ht_upper),
              state[k].ht_off);
        }
        equal_num += _mm_popcnt_u32(m_match);
        state[k].m_have_tuple =
            _mm512_kandn(m_new_cells, state[k].m_have_tuple);
        addr_offset = (uint32_t*)&state[k].pb_off;
        ht_pos = (uint32_t*)&state[k].ht_off;

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
        state[k].stage = 1;

#endif
      } break;
    }
    ++k;
  }
  return equal_num;
}
uint64_t SIMDGPProbe(Table* pb, HashTable** ht, int ht_num, char* payloads) {
  uint64_t equal_num = 0;
  uint16_t vector_scale = 16, new_add, k = 0, done = 0;
  __mmask16 m_match = 0, m_new_cells = -1;
  __m512i v_offset = _mm512_set1_epi32(0), v_addr_offset = _mm512_set1_epi32(0),
          v_base_offset_upper =
              _mm512_set1_epi32(pb->tuple_num * pb->tuple_size),
          v_tuple_cell = _mm512_set1_epi32(0), v_base_offset,
          v_left_size = _mm512_set1_epi32(8), ht_cell,
          v_ht_upper = _mm512_set1_epi32(ht[0]->slot_num * ht[0]->tuple_size),
          v_factor = _mm512_set1_epi32(table_factor), v_shift, v_cell_hash,
          v_neg_one512 = _mm512_set1_epi32(-1),
          v_zero512 = _mm512_set1_epi32(0), v_one = _mm512_set1_epi32(1),
          v_word_size = _mm512_set1_epi32(WORDSIZE);
  __attribute__((aligned(64))) uint32_t cur_offset = 0, cur_payloads = 0,
                                        *addr_offset, *ht_pos, base_off[32];
  for (int i = 0; i <= vector_scale; ++i) {
    base_off[i] = i * pb->tuple_size;
  }
  v_base_offset = _mm512_load_epi32(base_off);
  v_left_size = _mm512_set1_epi32(ht[0]->tuple_size);
  v_shift = _mm512_set1_epi32(ht[0]->shift);

  StateSIMD state[SIMDStateSize];
  // init # of the state
  for (int i = 0; i < SIMDStateSize; ++i) {
    state[i].stage = 1;
    state[i].m_have_tuple = 0;
    state[i].ht_off = _mm512_set1_epi32(0);
    // state[i].pb_off = _mm512_set1_epi32(0);
    // state[i].key = _mm512_set1_epi32(0);
  }
  for (uint64_t cur = 0; (done < SIMDStateSize);) {
    for (k = 0; k < SIMDStateSize; ++k) {  // init state for each tuple
      if (cur >= pb->tuple_num) {
        if (state[k].m_have_tuple == 0) {
          ++done;
          continue;
        }
      }
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
      state[k].pb_off = _mm512_mask_expand_epi32(
          state[k].pb_off, _mm512_knot(state[k].m_have_tuple), v_offset);
      // count the number of empty tuples
      m_new_cells = _mm512_knot(state[k].m_have_tuple);
      new_add = _mm_popcnt_u32(_mm512_knot(state[k].m_have_tuple));
      cur_offset = cur_offset + base_off[new_add];
      cur = cur + new_add;
      state[k].m_have_tuple =
          _mm512_cmpgt_epi32_mask(v_base_offset_upper, state[k].pb_off);

      ///// step 2: load new cells from right tuples;
      m_new_cells = _mm512_kand(m_new_cells, state[k].m_have_tuple);
      // maybe need offset within a tuple
      state[k].key = _mm512_mask_i32gather_epi32(state[k].key, m_new_cells,
                                                 state[k].pb_off, pb->start, 1);
      ///// step 3: load new values from hash tables;
      // hash the cell values
      v_cell_hash = _mm512_mullo_epi32(state[k].key, v_factor);
      v_cell_hash = _mm512_srlv_epi32(v_cell_hash, v_shift);

      // new hash
      v_cell_hash = _mm512_mullo_epi32(v_cell_hash, v_left_size);
      // old_hash = old_hash+left_size
      state[k].ht_off = _mm512_add_epi32(state[k].ht_off, v_left_size);
      // old_hash = old_hash >= upper ? 0 : old_hash;
      state[k].ht_off = _mm512_maskz_mov_epi32(
          _mm512_cmplt_epi32_mask(state[k].ht_off, v_ht_upper),
          state[k].ht_off);
      // combine new hash value with old hash value
      state[k].ht_off =
          _mm512_mask_mov_epi32(state[k].ht_off, m_new_cells, v_cell_hash);

//  state[k].stage = 0;
#if MultiPrefetch
      ht_pos = (uint32_t*)&state[k].ht_off;
      for (int i = 0; i < vector_scale; ++i) {
        _mm_prefetch((char*)(ht[0]->addr + ht_pos[i]), _MM_HINT_T1);
      }
#else
      ht_pos = (uint32_t*)&state[k].ht_off;
      for (int i = 0; i < vector_scale; ++i) {
        _mm_prefetch((char*)(ht[0]->addr + ht_pos[i]), _MM_HINT_T0);
      }
#endif
    }
    for (k = 0; (k < SIMDStateSize); ++k) {
#if MultiPrefetch
      if (k + SIMDStep < SIMDStateSize) {
        ht_pos = (uint32_t*)&state[k + SIMDStep].ht_off;
        for (int i = 0; i < vector_scale; ++i) {
          _mm_prefetch((char*)(ht[0]->addr + ht_pos[i]), _MM_HINT_T0);
        }
      }
#endif
      if (state[k].m_have_tuple == 0) {
        continue;
      }
#if RESULTS1
      ht_cell =
          _mm512_mask_i32gather_epi32(v_neg_one512, state[k].m_have_tuple,
                                      state[k].ht_off, ht[0]->global_addr, 1);

      ///// step 4: compare;
      m_match = _mm512_cmpeq_epi32_mask(state[k].key, ht_cell);
      m_match = _mm512_kand(m_match, state[k].m_have_tuple);
      equal_num += _mm_popcnt_u32(m_match);
      m_new_cells = _mm512_cmpeq_epi32_mask(ht_cell, v_neg_one512);
      m_new_cells = _mm512_kor(m_new_cells, m_match);
      state[k].m_have_tuple = _mm512_kandn(m_new_cells, state[k].m_have_tuple);

      ///// step 5: generate results;
      addr_offset = (uint32_t*)&state[k].pb_off;
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
#else
      m_new_cells = 0;
      while (m_new_cells != state[k].m_have_tuple) {
        ht_cell =
            _mm512_mask_i32gather_epi32(v_neg_one512, state[k].m_have_tuple,
                                        state[k].ht_off, ht[0]->global_addr, 1);
        ///// step 4: compare;
        m_match = _mm512_cmpeq_epi32_mask(state[k].key, ht_cell);
        // m_match = _mm512_kand(m_match, state[k].m_have_tuple);
        m_new_cells = _mm512_cmpeq_epi32_mask(ht_cell, v_neg_one512);
        m_new_cells = _mm512_kor(m_new_cells, m_match);
        m_new_cells = _mm512_kand(m_new_cells, state[k].m_have_tuple);

        // old_hash = old_hash+left_size,
        state[k].ht_off =
            _mm512_mask_add_epi32(state[k].ht_off, _mm512_knot(m_new_cells),
                                  state[k].ht_off, v_left_size);
        // old_hash = old_hash >= upper ? 0 : old_hash;
        state[k].ht_off = _mm512_maskz_mov_epi32(
            _mm512_cmplt_epi32_mask(state[k].ht_off, v_ht_upper),
            state[k].ht_off);
      }
      equal_num += _mm_popcnt_u32(m_match);
      state[k].m_have_tuple = _mm512_kandn(m_new_cells, state[k].m_have_tuple);
      addr_offset = (uint32_t*)&state[k].pb_off;
      ht_pos = (uint32_t*)&state[k].ht_off;

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
      // state[k].stage = 1;
    }
  }

  return equal_num;
}

uint64_t AMACProbe(Table* pb, HashTable** ht, int ht_num, char* payloads) {
  uint32_t hash = 0, ht_off, num = 0, k = 0, done = 0, kk = 0;
  void* probe_tuple_start = NULL, *payloads_addr;  // attention!!!
  uint32_t cur_payloads = 0, result_size = PAYLOADSIZE * (ht_num + 1);
  uint64_t upper = pb->tuple_num * pb->tuple_size;
  const int j = 0;

  State state[StateSize];
  // init # of the state
  for (int i = 0; i < StateSize; ++i) {
    state[i].stage = 1;
  }
  for (uint64_t pb_off = 0; (pb_off < upper) || (done < StateSize);) {
    k = (k == StateSize) ? 0 : k;
    switch (state[k].stage) {
      case 1: {  // init state for each tuple
        if (pb_off >= upper) {
          ++done;
          break;
        }
        _mm_prefetch((char*)(pb->start + pb_off + PDIS), _MM_HINT_T0);
        state[k].key = *(uint32_t*)(pb->start + pb_off + ht[j]->probe_offset);
        hash = ((uint32_t)(state[k].key * table_factor)) >> ht[j]->shift;
        state[k].ht_off = hash * ht[j]->tuple_size;
// prefetch
#if MultiPrefetch
        _mm_prefetch((char*)(ht[j]->addr + state[k].ht_off), _MM_HINT_T2);
#else
        _mm_prefetch((char*)(ht[j]->addr + state[k].ht_off), _MM_HINT_T0);
#endif
        state[k].pb_off = pb_off;
        state[k].stage = 0;
        pb_off += pb->tuple_size;
      } break;
      case 0: {
#if MultiPrefetch
        if (k + Step < StateSize) {
          _mm_prefetch((char*)(ht[j]->addr + state[k + Step].ht_off),
                       _MM_HINT_T0);
        }
#endif
        ht_off = state[k].ht_off;
        while (*(int*)(ht[j]->addr + ht_off + ht[j]->key_offset) != -1) {
          if (state[k].key ==
              *(uint32_t*)(ht[j]->addr + ht_off + ht[j]->key_offset)) {
            payloads_addr =
                (void*)(ht[j]->addr + ht_off + ht[j]->key_offset + WORDSIZE);
            memcpy(payloads + cur_payloads, (char*)payloads_addr, PAYLOADSIZE);
            memcpy(payloads + cur_payloads + PAYLOADSIZE,
                   (char*)(pb->start + state[k].pb_off + WORDSIZE * ht_num),
                   PAYLOADSIZE);
            //__builtin___clear_cache(
            //    (char*)(pb->start + state[k].pb_off),
            //   (char*)(pb->start + state[k].pb_off + pb->tuple_size));
            // cur_payloads += PAYLOADSIZE + PAYLOADSIZE;
            ++num;
            break;
          }
          ht_off += ht[j]->tuple_size;
          ht_off = ht_off >= ht[j]->ht_size ? ht_off - ht[j]->ht_size : ht_off;
          /*if ((ht_off & 63) == 0) {
            state[k].ht_off = ht_off;
            _mm_prefetch((char*)(ht[j]->addr + state[k].ht_off),
          _MM_HINT_T0);
            break;
          }*/
        }
        state[k].stage = 1;
      } break;
    }
    ++k;
  }
  return num;
}
uint64_t GPProbe(Table* pb, HashTable** ht, int ht_num, char* payloads) {
  uint32_t hash = 0, ht_off, num = 0, k = 0, done = 0;
  void* probe_tuple_start = NULL, *payloads_addr;  // attention!!!
  uint32_t cur_payloads = 0, result_size = PAYLOADSIZE * (ht_num + 1);
  uint64_t upper = pb->tuple_num * pb->tuple_size;
  const int j = 0;

  State state[StateSize];
  // init # of the state
  for (int i = 0; i < StateSize; ++i) {
    state[i].stage = 1;
  }
  for (uint64_t pb_off = 0; (pb_off < upper) || (done < StateSize);) {
    for (k = 0; k < StateSize; ++k) {  // init state for each tuple
      if (pb_off >= upper) {
        ++done;
        break;
      }
      _mm_prefetch((char*)(pb->start + pb_off + PDIS), _MM_HINT_T0);
      state[k].key = *(uint32_t*)(pb->start + pb_off + ht[j]->probe_offset);
      hash = ((uint32_t)(state[k].key * table_factor)) >> ht[j]->shift;
      state[k].ht_off = hash * ht[j]->tuple_size;
// prefetch
#if MultiPrefetch
      _mm_prefetch((char*)(ht[j]->addr + state[k].ht_off), _MM_HINT_T2);
#else
      _mm_prefetch((char*)(ht[j]->addr + state[k].ht_off), _MM_HINT_T0);
#endif
      state[k].pb_off = pb_off;
      state[k].stage = 0;
      pb_off += pb->tuple_size;
    }
    for (int kk = 0; (kk < k); ++kk) {
#if MultiPrefetch
      if (kk + Step < k) {
        _mm_prefetch((char*)(ht[j]->addr + state[kk + Step].ht_off),
                     _MM_HINT_T0);
      }
#endif
      ht_off = state[kk].ht_off;
      while (*(int*)(ht[j]->addr + ht_off + ht[j]->key_offset) != -1) {
        if (state[kk].key ==
            *(uint32_t*)(ht[j]->addr + ht_off + ht[j]->key_offset)) {
          payloads_addr =
              (void*)(ht[j]->addr + ht_off + ht[j]->key_offset + WORDSIZE);
          memcpy(payloads + cur_payloads, (char*)payloads_addr, PAYLOADSIZE);
          memcpy(payloads + cur_payloads + PAYLOADSIZE,
                 (char*)(pb->start + state[kk].pb_off + WORDSIZE * ht_num),
                 PAYLOADSIZE);
          //__builtin___clear_cache(
          //    (char*)(pb->start + state[k].pb_off),
          //   (char*)(pb->start + state[k].pb_off + pb->tuple_size));
          // cur_payloads += PAYLOADSIZE + PAYLOADSIZE;
          ++num;
          break;
        }
        ht_off += ht[j]->tuple_size;
        ht_off = ht_off >= ht[j]->ht_size ? ht_off - ht[j]->ht_size : ht_off;
        /*if ((ht_off & 63) == 0) {
          state[k].ht_off = ht_off;
          _mm_prefetch((char*)(ht[j]->addr + state[k].ht_off), _MM_HINT_T0);
          break;
        }*/
      }
      state[kk].stage = 1;
    }
  }
  return num;
}

uint64_t SingleProbe(Table* pb, HashTable** ht, int ht_num, char* payloads) {
  uint32_t tuple_key, hash = 0, ht_off, num = 0;
  void* probe_tuple_start = NULL, *payloads_addr;  // attention!!!
  uint32_t cur_payloads = 0;
  const int j = 0;

  for (uint64_t pb_off = 0; pb_off < pb->tuple_num * pb->tuple_size;
       pb_off += pb->tuple_size) {
    probe_tuple_start = (pb->start + pb_off);
#if SEQPREFETCH
    _mm_prefetch((char*)(probe_tuple_start + PDIS), _MM_HINT_T0);
#endif
    tuple_key = *(uint32_t*)(probe_tuple_start + ht[j]->probe_offset);
    hash = ((uint32_t)(tuple_key * table_factor)) >> ht[j]->shift;
    // lay at the hash table
    // assert(hash <= ht[j]->slot_num);
    ht_off = hash * ht[j]->tuple_size;
#if PREFETCH
    _mm_prefetch((char*)(ht[j]->addr + ht_off), _MM_HINT_T0);
#endif
    // probe each bucket
    while (*(int*)(ht[j]->addr + ht_off + ht[j]->key_offset) != -1) {
      if (tuple_key == *(uint32_t*)(ht[j]->addr + ht_off + ht[j]->key_offset)) {
        // record the payload address
        payloads_addr =
            (void*)(ht[j]->addr + ht_off + ht[j]->key_offset + WORDSIZE);

        memcpy(payloads + cur_payloads, (char*)payloads_addr, PAYLOADSIZE);
        memcpy(payloads + cur_payloads + PAYLOADSIZE,
               (char*)(probe_tuple_start + WORDSIZE * ht_num), PAYLOADSIZE);
        // cur_payloads += PAYLOADSIZE + PAYLOADSIZE;
        ++num;
#if EARLYBREAK
        break;
#endif
      }
      ht_off += ht[j]->tuple_size;
      ht_off = ht_off >= ht[j]->ht_size ? ht_off - ht[j]->ht_size : ht_off;
    }
  }

  return num;
}
