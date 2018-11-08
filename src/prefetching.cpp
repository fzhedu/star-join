#include "star-simd.h"
#include <stdlib.h>
#define StateSize 16
#define Step 6
struct State {
  uint32_t key;
  uint32_t pb_off;
  uint32_t ht_off;
  char stage;
};

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
        _mm_prefetch((char*)(ht[j]->addr + state[k].ht_off), _MM_HINT_T1);
        state[k].pb_off = pb_off;
        state[k].stage = 0;
        pb_off += pb->tuple_size;
      } break;
      case 0: {
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
            _mm_prefetch((char*)(ht[j]->addr + state[k].ht_off), _MM_HINT_T0);
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
      _mm_prefetch((char*)(ht[j]->addr + state[k].ht_off), _MM_HINT_T0);
      state[k].pb_off = pb_off;
      state[k].stage = 0;
      pb_off += pb->tuple_size;
    }
    for (int kk = 0; (kk < k); ++kk) {
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
