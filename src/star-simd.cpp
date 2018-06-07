#ifndef __SIMDTEST__
#define __SIMDTEST__
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <algorithm>
#include <unistd.h>
#include <iostream>
#include <assert.h>
#include <string>
#include <vector>
#include <sys/time.h>
#include <immintrin.h>
#include <cmath>
#include <map>
using namespace std;
typedef unsigned long long lld;
typedef unsigned int uint;
#define NULL_INT 2147483647
#define BLOCK_SIZE 65536
#define RESULTS 1
#define OUTPUT 0
#define MEMOUTPUT 0
#define GATHERHT 0
#define PREFETCH 1
#define EARLYBREAK 1
// filter out
float selectity = 0;
#define INVALID 2147483647
#define _mm256_set_m128i(v0, v1) \
  _mm256_insertf128_si256(_mm256_castsi128_si256(v1), (v0), 1)

struct Table {
  string name;
  string path;
  lld raw_tuple_size;
  lld tuple_size;
  lld tuple_num;
  int upper;
  void* start;
  vector<int> offset, size;
}* tb[10];
struct HashTable {
  int tuple_size;
  int key_offset;
  int tuple_num;
  int slot_num;
  void* addr;
  int shift;
  int probe_offset;
}* ht[100];
typedef lld (*ProbeFunc)(Table* pb, HashTable** ht, int ht_num);
struct ThreadArgs {
  Table* tb;
  ProbeFunc func;
};
#define WORDSIZE 4
unsigned int table_factor = 1;
typedef unsigned int uint32_t;
pthread_mutex_t mutex;
lld global_probe_corsur = 0, global_matched = 0;
#define probe_step 1024 * 1024  // 1048576
int thread_num = 32;
int ht_num = 4;
int times = 3;
int output_buffer_size = 32;
// typedef unsigned long long uint64_t;
// b
/*
 dbgen_version
 customer_address
 customer_demographics
 date_dim
 warehouse
 ship_mode
 time_dim                86400    124
 reason
 income_band
 item
 store                   402     788
 call_center
 customer
 web_site
 store_returns
 household_demographics  7200    40
 web_page
 promotion
 catalog_page
 inventory
 catalog_returns
 web_returns
 web_sales
 catalog_sales
 store_sales             287997024   100
 store4                  402
 */
bool build_ht(HashTable& ht, Table& table, int key_offset, int probe_offset) {
  ht.tuple_num = table.tuple_num;
  ht.tuple_size = table.tuple_size;
  ht.key_offset = key_offset;
  ht.probe_offset = probe_offset;
  ht.addr = malloc(table.upper * table.tuple_size);
  for (int i = 0; i < table.tuple_num; ++i) {
    int key = *(int*)(table.start + i * table.tuple_size + key_offset);
    memcpy(ht.addr + key * ht.tuple_size, table.start + i * table.tuple_size,
           table.tuple_size);
  }
  return true;
}
int upper_log2(int num) {
  int p = log2(num);
  return (1 << (p + 2));
}
bool build_linear_ht(HashTable& ht, Table& table, int key_offset,
                     int probe_offset, float selectity = 0) {
  int bound = table.tuple_num * selectity;
  ht.tuple_num = table.tuple_num - bound;
  ht.slot_num = upper_log2(ht.tuple_num);
  ht.tuple_size = table.tuple_size;
  ht.key_offset = key_offset;
  ht.probe_offset = probe_offset;
  lld ht_size = ht.slot_num * ht.tuple_size;
  ht.addr = malloc(ht_size);
  ht.shift = 32 - log2(ht.slot_num);
  memset(ht.addr, -1, ht_size);
  uint key, hash = 0;
  lld offset = 0;
  for (uint i = 0; i < (table.tuple_num - bound) * table.tuple_size;
       i += table.tuple_size) {
    // enum the key in the table
    key = *(uint*)(table.start + i + key_offset);
    // set filter
    // if (key < bound) continue;
    // compute the hash value
    hash = ((uint)(key * table_factor)) >> ht.shift;
    offset = hash * ht.tuple_size;
    // travel corresponding position in the hash table
    while (*(uint*)(ht.addr + offset + key_offset) != -1) {
      offset += ht.tuple_size;
      offset = offset > ht_size ? offset - ht_size : offset;
    }
    memcpy(ht.addr + offset, table.start + i, table.tuple_size);
  }
  return true;
}
bool travel_linear_ht(HashTable& ht) {
  unsigned ht_size = ht.slot_num * ht.tuple_size;
  map<int, int> count;
  int tmp = 0, id = 0;
  for (int offset = 0; offset < ht_size; offset += ht.tuple_size) {
    if (*(int*)(ht.addr + offset + ht.key_offset) != -1) {
      ++tmp;
    } else {
      if (tmp > 0) {
        count[tmp]++;
        tmp = 0;
      }
    }
  }
  if (tmp) {
    count[tmp]++;
  }
  tmp = 0;
  cout << "-------------the cardinality of hash table---------- " << endl;
  for (map<int, int>::iterator it = count.begin(); it != count.end(); ++it) {
    cout << "val = " << it->first << " count = " << it->second << endl;
    tmp += it->first * it->second;
  }
  cout << "tmp = " << tmp << " tuple_num = " << ht.tuple_num << endl;
  //  assert(tmp == ht.tuple_num);
  return true;
}

bool read_data_in_memory(Table* table) {
  void* addr = NULL, *start = NULL;
  struct stat f_stat;
  int fd = -1;
  if ((fd = open(table->path.c_str(), O_RDONLY)) == -1) return false;
  if (fstat(fd, &f_stat)) return false;
  // int len = lseek(fd, 0, SEEK_END);
  lld size = table->tuple_num * table->tuple_size, tuple_num = 0;
  addr = malloc(size);
  assert(addr != NULL);
  if ((start = mmap(0, f_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0)) ==
      MAP_FAILED) {
    return false;
  }
  int num = 0;
  // read each BLOCK, then get each tuple in a block
  for (lld k = 0, j = 0; k < f_stat.st_size; k += BLOCK_SIZE) {
    num = *(int*)(start + k + BLOCK_SIZE - 4);
    tuple_num += num;
    // copy needed cells from each row
    for (int i = 0; i < num; ++i) {
      for (int m = 0; m < table->offset.size(); ++m) {
        memcpy(addr + j,
               (start + k + i * table->raw_tuple_size + table->offset[m]),
               table->size[m]);
        j += table->size[m];
      }
    }
  }

  if (munmap(start, f_stat.st_size)) {
    return false;
  }
  close(fd);
  table->start = addr;
  table->upper = table->tuple_num;  // special
  cout << table->name << " tuple num =" << table->tuple_num << " but read "
       << tuple_num << endl;
  assert(table->tuple_num == tuple_num);
  return true;
}
int SumOfVector(vector<int>& vec) {
  int sum = 0;
  for (int i = 0; i < vec.size(); ++i) {
    sum += vec[i];
  }
  return sum;
}
void SetVectorValue(int* array, int num, vector<int>& vec) {
  for (int i = 0; i < num; ++i) {
    vec.push_back(array[i]);
  }
  return;
}
void test(Table* test_table, void* addr, int off) {
  printf("%d %d\n", test_table->tuple_num, test_table->tuple_size);
  int count = 0, t = 0;
  for (int i = 0; i < test_table->tuple_num; ++i) {
    int k = *(int*)(addr + (i * test_table->tuple_size) + off);
    if (k < 100) {
      cout << k << "  ";
      t++;
    }
    count++;
  }
  cout << endl << count << " has " << t << endl;
}

lld LinearHandProbe(Table* pb, HashTable** ht, int ht_num) {
  bool tmp = true, flag;
  uint tuple_key[10], hash = 0, ht_off, cell_equal[16], num = 0, result[16];
  void* probe_tuple_start = NULL;  // attention!!!
  memset(cell_equal, 0, 16);
#if MEMOUTPUT
  uint32_t output_buffer[output_buffer_size];
#endif
#if OUTPUT
  FILE* fp;
  fp = fopen("hand.csv", "wr");
#endif
  for (lld pb_off = 0; pb_off < pb->tuple_num * pb->tuple_size;
       pb_off += pb->tuple_size) {
    tmp = true;
    probe_tuple_start = (pb->start + pb_off);
    for (int j = 0; j < ht_num && tmp; ++j) {
      tuple_key[j] = *(uint*)(probe_tuple_start + ht[j]->probe_offset);
      hash = ((uint)(tuple_key[j] * table_factor)) >> ht[j]->shift;
      // lay at the hash table
      // assert(hash <= ht[j]->slot_num);
      ht_off = hash * ht[j]->tuple_size;
      flag = false;

      // probe each bucket
      while (*(int*)(ht[j]->addr + ht_off + ht[j]->key_offset) != -1) {
        if (tuple_key[j] ==
            *(uint*)(ht[j]->addr + ht_off + ht[j]->key_offset)) {
          cell_equal[j] =
              *(int*)(ht[j]->addr + ht_off + ht[j]->key_offset + WORDSIZE);
          flag = true;
#if EARLYBREAK
          break;
#endif
        }
        ht_off += ht[j]->tuple_size;
      }

      tmp = tmp & flag;
    }
#if RESULTS
    if (tmp) {
      cell_equal[ht_num] = *(uint*)(probe_tuple_start + WORDSIZE * ht_num);
#if MEMOUTPUT
      memcpy(output_buffer, cell_equal, WORDSIZE * (1 + ht_num));
#endif
#if OUTPUT
      for (int i = 0; i < ht_num; ++i) {
        fprintf(fp, "%d,", cell_equal[i][1]);
      }
      fprintf(fp, "%d\n", cell_equal[ht_num][1]);
#endif
    }
#endif
    num += tmp;
  }
// cout << "LinearHandProbe result = " << num << endl;
#if OUTPUT
  fclose(fp);
#endif
  return num;
}
lld HandProbe(Table* pb, HashTable** ht, int ht_num) {
  bool tmp = true;
  lld num = 0;
  int offset = 0, key[10];
  void* probe_tuple_start = NULL, *ht_tuple_start[10];
#if RESULTS
  void* output = malloc(128);
#endif
  for (lld i = 0; i < pb->tuple_num; ++i) {
    tmp = true;
    probe_tuple_start = pb->start + i * pb->tuple_size;
    for (int j = 0; j < ht_num && tmp; ++j) {
      key[j] = *(int*)(probe_tuple_start + ht[j]->probe_offset);
      if (key[j] > -1) {
        ht_tuple_start[j] = ht[j]->addr + key[j] * ht[j]->tuple_size;
        tmp = key[j] == (*(int*)(ht_tuple_start[j] + ht[j]->key_offset));
      }
    }
    num += tmp;
#if RESULTS
    if (tmp) {
      offset = 0;
      memcpy(output, probe_tuple_start, pb->tuple_size);
      offset += pb->tuple_size;
      for (int j = 0; j < ht_num; ++j) {
        memcpy(output + offset, ht_tuple_start[j], ht[j]->tuple_size);
        offset += ht[j]->tuple_size;
      }
    }
#endif
  }
  // cout << "HandProbe qualified tuples = " << num << endl;
  return num;
}
char* ProbeTuple(void* src, int src_size, HashTable** ht, int ht_num, int cur,
                 int& ret_size, char output[][128]) {
  char* res;
  void* start;
  if (ht_num == cur + 1) {
    int key = *(int*)(src + ht[cur]->probe_offset);
    if (key > -1) {
      start = ht[cur]->addr + key * ht[cur]->tuple_size;
      if (key == *(int*)(start + ht[cur]->key_offset)) {
#if RESULTS
        memcpy(output[cur], src, src_size);
        memcpy(output[cur] + src_size, start, ht[cur]->tuple_size);
#endif
        ret_size = src_size + ht[cur]->tuple_size;
        return output[cur];
      }
    }
  } else {
    res = ProbeTuple(src, src_size, ht, ht_num, cur + 1, ret_size, output);
    if (res == NULL) {
      return NULL;
    }
    int key = *(int*)(src + ht[cur]->probe_offset);
    if (key > -1) {
      start = ht[cur]->addr + key * ht[cur]->tuple_size;
      if (key == *(int*)(start + ht[cur]->key_offset)) {
#if RESULTS
        memcpy(output[cur], res, ret_size);
        memcpy(output[cur] + ret_size, start, ht[cur]->tuple_size);
#endif
        ret_size = ret_size + ht[cur]->tuple_size;
        return output[cur];
      }
    }
  }
  return NULL;
}

uint32_t* LinearProbeTuple(void* src, int src_size, HashTable** ht, int ht_num,
                           int cur, int& ret_size, uint32_t output[][16]) {
  uint32_t* res;
  void* start;
  bool tmp = false;
  if (cur == 0) {
    uint key = *(uint*)(src + ht[cur]->probe_offset);
    uint hash = (uint)(key * table_factor) >> ht[cur]->shift;
    uint ht_off = hash * ht[cur]->tuple_size;
    while (*(int*)(ht[cur]->addr + ht_off + ht[cur]->key_offset) != -1) {
      if (key == *(uint*)(ht[cur]->addr + ht_off + ht[cur]->key_offset)) {
#if RESULTS
#if MEMCPY
        memcpy(output[cur], src, src_size);
        memcpy(output[cur] + src_size, ht[cur]->addr + ht_off,
               ht[cur]->tuple_size);
        ret_size = src_size + ht[cur]->tuple_size;
        return output[cur];
#else
        output[0][ht_num] = *(uint*)(src + WORDSIZE * ht_num);
        output[0][cur] =
            *(uint*)(ht[cur]->addr + ht_off + ht[cur]->key_offset + WORDSIZE);
        tmp = true;
#if EARLYBREAK
        return output[0];
#endif
#endif
#endif
      }
      ht_off += ht[cur]->tuple_size;
    }

  } else {
    res =
        LinearProbeTuple(src, src_size, ht, ht_num, cur - 1, ret_size, output);
    if (res == NULL) {
      return NULL;
    }
    uint key = *(uint*)(src + ht[cur]->probe_offset);
    uint hash = (uint)(key * table_factor) >> ht[cur]->shift;
    uint ht_off = hash * ht[cur]->tuple_size;
    while (*(int*)(ht[cur]->addr + ht_off + ht[cur]->key_offset) != -1) {
      if (key == *(uint*)(ht[cur]->addr + ht_off + ht[cur]->key_offset)) {
#if RESULTS
#if MEMCPY
        memcpy(output[cur], res, ret_size);
        memcpy(output[cur] + ret_size, ht[cur]->addr + ht_off,
               ht[cur]->tuple_size);
        ret_size = ret_size + ht[cur]->tuple_size;
        return output[cur];
#else
        output[0][cur] =
            *(uint*)(ht[cur]->addr + ht_off + ht[cur]->key_offset + WORDSIZE);
        tmp = true;
#if EARLYBREAK
        return output[0];
#endif
#endif
#endif
      }
      ht_off += ht[cur]->tuple_size;
    }
  }
#if !EARLYBREAK
  if (tmp) {
    return output[0];
  }
#endif
  return NULL;
}
lld TupleAtATimeProbe(Table* pb, HashTable** ht, int ht_num) {
  int ret_size, len = 128;
  uint32_t* res;
  lld num = 0;
  uint32_t output[8][16];
#if MEMOUTPUT
  uint32_t output_buffer[output_buffer_size];
#endif
#if OUTPUT
  FILE* fp;
  fp = fopen("tuple.csv", "wr");
#endif
  for (lld i = 0, off = 0; i < pb->tuple_num; ++i, off += pb->tuple_size) {
    ret_size = 0;
    //    res = ProbeTuple(pb->start + off, pb->tuple_size, ht, ht_num, 0,
    //    ret_size,output);

    res = LinearProbeTuple(pb->start + off, pb->tuple_size, ht, ht_num,
                           ht_num - 1, ret_size, output);
    if (res) {
      ++num;
#if MEMOUTPUT
      memcpy(output_buffer, output[0], WORDSIZE * (1 + ht_num));
#endif
#if OUTPUT

      for (int j = 0; j < ht_num; ++j) {
        fprintf(fp, "%d,", output[0][j]);
      }
      fprintf(fp, "%d\n", output[0][ht_num]);
#endif
    }
  }
// cout << "TupleAtATimeProbe qualified tuples = " << num << endl;
#if OUTPUT
  fclose(fp);
#endif
  return num;
}
// use short mask to avoid long mask
lld Linear512Probe(Table* pb, HashTable** ht, int ht_num) {
  uint16_t vector_scale = 16, new_add;
  __mmask16 m_bucket_pass, m_done, m_match, m_abort, m_have_tuple = 0,
                                                     m_new_cells = -1;
  __m512i v_offset = _mm512_set1_epi32(0), v_addr_offset = _mm512_set1_epi32(0),
          v_join_num = _mm512_set1_epi32(ht_num),
          v_base_offset_upper =
              _mm512_set1_epi32(pb->tuple_num * pb->tuple_size),
          v_ht_addr4, v_ht_addr, v_tuple_cell = _mm512_set1_epi32(0),
          v_right_index, v_base_offset, v_left_size = _mm512_set1_epi32(8),
          v_bucket_offset, ht_cell, v_factor = _mm512_set1_epi32(table_factor),
          v_ht_cell_offset, v_shift, v_buckets_minus_1, v_cell_hash, v_ht_pos,
          v_neg_one512 = _mm512_set1_epi32(-1),
          v_zero512 = _mm512_set1_epi32(0), v_join_id = _mm512_set1_epi32(0),
          v_one = _mm512_set1_epi32(1), v_payloads, v_payloads_off,
          v_word_size = _mm512_set1_epi64(WORDSIZE);
  __attribute__((aligned(32)))
  uint32_t cur_offset = 0,
           tmp_cell[16], *addr_offset, temp_payloads[16][16] = {0}, *ht_pos,
           *join_id, base_off[32], tuple_cell_offset[16] = {0},
           left_size[16] = {0}, ht_cell_offset[16] = {0}, payloads_off[32],
           buckets_minus_1[16] = {0}, shift[16] = {0};
  __attribute__((aligned(32))) void * *ht_addr, **ht_addr4, *tmp_ht_addr[16];
  __attribute__((aligned(32))) uint64_t htp[16] = {0};
#if MEMOUTPUT
  __attribute__((aligned(64))) uint32_t output_buffer[output_buffer_size];
#endif
  for (int i = 0; i <= vector_scale; ++i) {
    base_off[i] = i * pb->tuple_size;
    payloads_off[i] = i * 16;  // it is related to temp_payloads
  }
  for (int i = 0; i < ht_num; ++i) {
    tuple_cell_offset[i] = ht[i]->probe_offset;
    left_size[i] = ht[i]->tuple_size;
    ht_cell_offset[i] = ht[i]->key_offset;
    shift[i] = ht[i]->shift;
    buckets_minus_1[i] = ht[i]->slot_num - 1;
    htp[i] = (uint64_t)ht[i]->addr;
  }
  ht_pos = (uint32_t*)&v_ht_pos;
  join_id = (uint32_t*)&v_join_id;
  ht_addr = (void**)&v_ht_addr;
  ht_addr4 = (void**)&v_ht_addr4;
  addr_offset = (uint32_t*)&v_addr_offset;

  __m256i v_zero256 = _mm256_set1_epi32(0), v256_upper, v256_lower;
  lld equal_num = 0;
  v_base_offset = _mm512_load_si512((__m512i*)(&base_off));
  v_payloads_off = _mm512_load_epi32(payloads_off);
#if OUTPUT
  FILE* fp;
  fp = fopen("simd512.csv", "wr");
#endif
  int times = 0;
  for (lld cur = 0; cur < pb->tuple_num || m_have_tuple;) {
    ///////// step 1: load new tuples' address offsets
    // the offset should be within MAX_32INT_
    // the tail depends on the number of joins and tuples in each bucket
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

    v_right_index =
        _mm512_i32gather_epi32(v_join_id, tuple_cell_offset, 4);  ///////
    m_new_cells = _mm512_kand(m_new_cells, m_have_tuple);
    v_tuple_cell = _mm512_mask_i32gather_epi32(
        v_tuple_cell, m_new_cells,
        _mm512_add_epi32(v_addr_offset, v_right_index), pb->start, 1);

////// step 3: load new values in hash tables
#if 0
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
    // set 0 for new cells, but add 1 for old cells
    v_bucket_offset = _mm512_maskz_add_epi32(_mm512_knot(m_new_cells),
                                             v_bucket_offset, v_one);

    v_cell_hash = _mm512_add_epi32(v_cell_hash, v_bucket_offset);
    // avoid overflow
    v_cell_hash = _mm512_and_si512(v_cell_hash, v_buckets_minus_1);

    v_ht_pos = _mm512_mullo_epi32(v_cell_hash, v_left_size);
#if GATHERHT
    v_ht_pos = _mm512_add_epi32(v_ht_pos, v_ht_cell_offset);
    __m512i v_ht_pos_644 = _mm512_cvtepu32_epi64(
        _mm256_load_si256(reinterpret_cast<__m256i*>(&ht_pos[8])));
    __m512i v_ht_pos_64 = _mm512_cvtepu32_epi64(
        _mm256_load_si256(reinterpret_cast<__m256i*>(ht_pos)));

    v_ht_addr4 = _mm512_i32gather_epi64(
        _mm256_load_si256(reinterpret_cast<__m256i*>(&join_id[8])), htp,
        8);  //////
    v_ht_addr = _mm512_i32gather_epi64(
        _mm256_load_si256(reinterpret_cast<__m256i*>(join_id)), htp, 8);  /////
    v_ht_addr4 = _mm512_add_epi64(v_ht_pos_644, v_ht_addr4);
    v_ht_addr = _mm512_add_epi64(v_ht_pos_64, v_ht_addr);
    // attention!!! should (m_have_tuple>>8)
    v256_upper = _mm512_mask_i64gather_epi32(v_zero256, (m_have_tuple >> 8),
                                             v_ht_addr4, 0, 1);
    v256_lower =
        _mm512_mask_i64gather_epi32(v_zero256, m_have_tuple, v_ht_addr, 0, 1);
    // the first parameter should be at the lower position, and the second one
    // should lay at the upper position, and the third must be 1
    ht_cell =
        _mm512_inserti32x8(_mm512_castsi256_si512(v256_lower), v256_upper, 1);
#else
// no improvement, but it is worse when judging the m_have_tuple
// PREFETCHNTA
#if PREFETCH
    for (int i = 0; i < vector_scale; ++i) {
      _mm_prefetch(ht_pos[i] + htp[join_id[i]], _MM_HINT_T0);
    }
#endif
    for (int i = 0; i < vector_scale; ++i) {
      tmp_ht_addr[i] = ht_pos[i] + htp[join_id[i]];
      tmp_cell[i] = *(uint32_t*)(tmp_ht_addr[i]);
    }

    ht_cell = _mm512_load_epi32(tmp_cell);
#endif

    //// step 4: compare
    // load raw cell data, then judge whether they are equal ? the AND get
    // rid of invalid keys
    m_match = _mm512_cmpeq_epi32_mask(v_tuple_cell, ht_cell);
    m_match = _mm512_kand(m_match, m_have_tuple);
#ifdef SCATTER
    // it is slower than salar code
    ///// step 5: store matched payloads
    // load payloads
    v_ht_addr4 = _mm512_add_epi64(v_word_size, v_ht_addr4);
    v_ht_addr = _mm512_add_epi64(v_word_size, v_ht_addr);
    // attention!!! should (m_have_tuple>>8)
    v256_upper = _mm512_mask_i64gather_epi32(v_zero256, (m_match >> 8),
                                             v_ht_addr4, 0, 1);
    v256_lower =
        _mm512_mask_i64gather_epi32(v_zero256, m_match, v_ht_addr, 0, 1);
    // the first parameter should be at the lower position, and the second one
    // should lay at the upper position, and the third must be 1
    v_payloads =
        _mm512_inserti32x8(_mm512_castsi256_si512(v256_lower), v256_upper, 1);
    v_offset = _mm512_add_epi32(v_join_id, v_payloads_off);
    _mm512_mask_i32scatter_epi32(temp_payloads, m_match, v_offset, v_payloads,
                                 4);
#endif
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
#if GATHERHT
    int mid = vector_scale >> 1;
    for (int i = 0; m_match && i < mid; ++i, m_match = (m_match >> 1)) {
      if (m_match & 1) {
        temp_payloads[i][join_id[i]] = *(uint32_t*)(WORDSIZE + ht_addr[i]);
      }
    }
    for (int i = mid; m_match && i < vector_scale;
         ++i, m_match = (m_match >> 1)) {
      if (m_match & 1) {
        temp_payloads[i][join_id[i]] =
            *(uint32_t*)(WORDSIZE + ht_addr4[i - mid]);
      }
    }

#else
    for (int i = 0; m_match && (i < vector_scale);
         ++i, m_match = m_match >> 1) {
      //      if (m_match & 1) {
      temp_payloads[i][join_id[i]] = *(uint32_t*)(WORDSIZE + tmp_ht_addr[i]);
      //      }
    }
#endif
    for (int i = 0; m_done && i < vector_scale; ++i, m_done = (m_done >> 1)) {
      if (m_done & 1) {
        temp_payloads[i][ht_num] =
            *(uint32_t*)(pb->start + addr_offset[i] + WORDSIZE * ht_num);
#if MEMOUTPUT
        _mm512_store_epi32(output_buffer, _mm512_load_epi32(temp_payloads[i]));
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
// cout << "SIMD512Probe qualified tuples = " << equal_num << endl;
#if OUTPUT
  fclose(fp);
#endif
  return equal_num;
}
lld Linear512ProbeHor(Table* pb, HashTable** ht, int ht_num) {
  uint16_t vector_scale = 16, new_add;
  __mmask16 m_bucket_pass, m_done, m_match, m_abort, m_have_tuple = 0,
                                                     m_new_cells = -1;
  __m512i v_offset = _mm512_set1_epi32(0), v_addr_offset = _mm512_set1_epi32(0),
          v_join_num = _mm512_set1_epi32(ht_num),
          v_base_offset_upper =
              _mm512_set1_epi32(pb->tuple_num * pb->tuple_size),
          v_ht_addr4, v_ht_addr, v_tuple_cell = _mm512_set1_epi32(0),
          v_right_index, v_base_offset, v_left_size = _mm512_set1_epi32(8),
          v_bucket_offset, ht_cell, v_factor = _mm512_set1_epi32(table_factor),
          v_ht_cell_offset, v_shift, v_buckets_minus_1, v_cell_hash, v_ht_pos,
          v_neg_one512 = _mm512_set1_epi32(-1),
          v_zero512 = _mm512_set1_epi32(0), v_join_id = _mm512_set1_epi32(0),
          v_one = _mm512_set1_epi32(1);
  __attribute__((aligned(32)))
  uint32_t cur_offset = 0,
           tmp_cell[16], *addr_offset, temp_payloads[16][16] = {0}, *ht_pos,
           *tuple_cell, *join_id, base_off[32], tuple_cell_offset[16] = {0},
           left_size[16] = {0}, ht_cell_offset[16] = {0},
           buckets_minus_1[16] = {0}, shift[16] = {0}, h_buf[16] = {0};
  __attribute__((aligned(32))) void * *ht_addr, **ht_addr4, *tmp_ht_addr[16];
  __attribute__((aligned(32))) uint64_t htp[16] = {0}, bucket_upper[16] = {0};
#if MEMOUTPUT
  __attribute__((aligned(64))) uint32_t output_buffer[output_buffer_size];
#endif
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
    bucket_upper[i] = buckets_minus_1[i] * left_size[i];
  }
  ht_pos = (uint32_t*)&v_ht_pos;
  join_id = (uint32_t*)&v_join_id;
  ht_addr = (void**)&v_ht_addr;
  ht_addr4 = (void**)&v_ht_addr4;
  addr_offset = (uint32_t*)&v_addr_offset;
  tuple_cell = (uint32_t*)&v_tuple_cell;

  __m256i v_zero256 = _mm256_set1_epi32(0), v256_upper, v256_lower;
  lld equal_num = 0;
  v_base_offset = _mm512_load_si512((__m512i*)(&base_off));
#if OUTPUT
  FILE* fp;
  fp = fopen("simd512.csv", "wr");
#endif
  int times = 0;
  for (lld cur = 0; cur < pb->tuple_num || m_have_tuple;) {
    ///////// step 1: load new tuples' address offsets
    // the offset should be within MAX_32INT_
    // the tail depends on the number of joins and tuples in each bucket
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

    v_right_index =
        _mm512_i32gather_epi32(v_join_id, tuple_cell_offset, 4);  ///////
    m_new_cells = _mm512_kand(m_new_cells, m_have_tuple);
    v_tuple_cell = _mm512_mask_i32gather_epi32(
        v_tuple_cell, m_new_cells,
        _mm512_add_epi32(v_addr_offset, v_right_index), pb->start, 1);

////// step 3: load new values in hash tables
#if 0
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
    // set 0 for new cells, but add 1 for old cells
    v_bucket_offset = _mm512_maskz_add_epi32(_mm512_knot(m_new_cells),
                                             v_bucket_offset, v_one);

    v_cell_hash = _mm512_add_epi32(v_cell_hash, v_bucket_offset);
    // avoid overflow
    v_cell_hash = _mm512_and_si512(v_cell_hash, v_buckets_minus_1);

    v_ht_pos = _mm512_mullo_epi32(v_cell_hash, v_left_size);
    m_match = 0;
    // probe buckets horizationally for each key
    for (int j = 0; j != 16; ++j) {
      size_t p = ht_pos[j];
      __m512i key_x16 = _mm512_set1_epi32(tuple_cell[j]);
#if PREFETCH
      if (j < 15) {
        _mm_prefetch(ht_pos[j + 1] + htp[join_id[j + 1]], _MM_HINT_T0);
      }
#endif
      short flag = 0;
      for (;;) {
        __m512i tab = _mm512_loadu_si512((void*)(htp[join_id[j]] + p));
        __mmask16 out = _mm512_cmpeq_epi32_mask(tab, key_x16);
        out = _mm512_kand(out, 0x5555);
        if (out > 0) {
          int znum = (_tzcnt_u32(out) >> 1);
          // avoid overflowing
          if (znum * left_size[join_id[j]] + p <= bucket_upper[join_id[j]]) {
            temp_payloads[j][join_id[j]] =
                *(uint32_t*)(htp[join_id[j]] + p +
                             znum * left_size[join_id[j]]);
            flag = 1;
            break;
          }
        }
        out = _mm512_cmpeq_epi32_mask(tab, v_neg_one512);
        out = _mm512_kand(out, 0x5555);

        if (out > 0) break;
        p = (p + 64) >= (bucket_upper[join_id[j]])
                ? p + 64 - bucket_upper[join_id[j]]
                : p + 64;
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
        temp_payloads[i][ht_num] =
            *(uint32_t*)(pb->start + addr_offset[i] + WORDSIZE * ht_num);
#if MEMOUTPUT
        _mm512_store_epi32(output_buffer, _mm512_load_epi32(temp_payloads[i]));
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
// cout << "SIMD512Probe qualified tuples = " << equal_num << endl;
#if OUTPUT
  fclose(fp);
#endif
  return equal_num;
}

lld LinearSIMDProbe(Table* pb, HashTable** ht, int ht_num) {
  __m256i
      v_join_num = _mm256_set1_epi32(ht_num),
      v_one = _mm256_set1_epi32(1), v_tuple_cell, ht_cell, v_next_cell,
      v_matches, v_abort, v_shift, v_ht_pos,
      v_ht_cell_offset = _mm256_set1_epi32(0), v_ht_addr4, v_ht_addr,
      v_left_size = _mm256_set1_epi32(8), v_done,
      v_have_tuple = _mm256_set1_epi32(0), zero256 = _mm256_set1_epi32(0),
      v_factor = _mm256_set1_epi32(table_factor), v_cell_hash,
      // need to load new cells when the bucket is over
      v_new_cells = _mm256_set1_epi32(-1), v_bucket_pass = _mm256_set1_epi32(0),
      // travel each tuple in the bucket
      v_bucket_offset = _mm256_set1_epi32(0), v_right_index,
      // join id +1 if the bucket is over; join id =0 if new tuples are loaded
      v_join_id = _mm256_set1_epi32(0),
      v_base_offset_uppper = _mm256_set1_epi32(pb->tuple_num * pb->tuple_size),
      v_buckets_minus_1 = _mm256_set1_epi32(0), v_base_offset, v_offset,
      v_addr_offset;
  const uint32_t offset_index = 0x76543210, tuple_scale = 8;
  __attribute__((aligned(32))) int32_t base_off[10], *ht_pos, *join_id,
      *tuple_cell, *addr_offset, *done, *have_tuple, *offset,
      tuple_cell_offset[10] = {0}, left_size[8] = {0}, ht_cell_offset[8] = {0},
      buckets_minus_1[8] = {0}, mask[16] = {0}, shift[8] = {0}, tmp_cell[8];
  for (int i = 0; i <= tuple_scale; ++i) {
    base_off[i] = i * pb->tuple_size;
  }
#if MEMOUTPUT
  __attribute__((aligned(32))) uint32_t output_buffer[output_buffer_size];
#endif
  __attribute__((aligned(32))) uint64_t htp[8] = {0};
  __attribute__((aligned(32))) void * *ht_addr, **ht_addr4, *tmp_ht_addr[8];
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
  lld equal_num = 0;
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
   * when travel the tuples in the base table, pay attention to that the offset
   * maybe overflow u32int
   */

  for (lld cur = 0; cur < pb->tuple_num || continue_mask;) {
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
        v_tuple_cell, start_addr,
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
        htp, _mm_load_si128(reinterpret_cast<__m128i*>(&join_id[4])), 8);
    v_ht_addr = _mm256_i32gather_epi64(
        htp, _mm_load_si128(reinterpret_cast<__m128i*>(join_id)), 8);

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

lld LinearSIMDProbeHor(Table* pb, HashTable** ht, int ht_num) {
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
      // join id +1 if the bucket is over; join id =0 if new tuples are loaded
      v_join_id = _mm256_set1_epi32(0),
      v_base_offset_uppper = _mm256_set1_epi32(pb->tuple_num * pb->tuple_size),
      v_buckets_minus_1 = _mm256_set1_epi32(0), v_base_offset, v_offset,
      v_addr_offset, v_10mask = _mm256_set_epi32(0, -1, 0, -1, 0, -1, 0, -1);
  const uint32_t offset_index = 0x76543210, tuple_scale = 8;
  __attribute__((aligned(32))) int32_t base_off[10], *ht_pos, *join_id,
      *tuple_cell, *addr_offset, *done, *have_tuple, *offset,
      tuple_cell_offset[10] = {0}, left_size[8] = {0}, ht_cell_offset[8] = {0},
      buckets_minus_1[8] = {0}, mask[16] = {0}, shift[8] = {0}, tmp_cell[8];
  for (int i = 0; i <= tuple_scale; ++i) {
    base_off[i] = i * pb->tuple_size;
  }
#if MEMOUTPUT
  __attribute__((aligned(32))) uint32_t output_buffer[output_buffer_size];
#endif
  __attribute__((aligned(32))) uint64_t htp[8] = {0}, bucket_upper[8] = {0};
  __attribute__((aligned(32))) void * *ht_addr, **ht_addr4, *tmp_ht_addr[8];
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
  lld equal_num = 0;
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
   * when travel the tuples in the base table, pay attention to that the offset
   * maybe overflow u32int
   */

  for (lld cur = 0; cur < pb->tuple_num || continue_mask;) {
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
        v_tuple_cell, start_addr,
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
        _mm_prefetch(ht_pos[j + 1] + htp[join_id[j + 1]], _MM_HINT_T0);
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
                             znum * left_size[join_id[j]]);
            flag = 1;
            break;
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

lld SIMDProbe(Table* pb, HashTable** ht, int ht_num) {
  __m256i v_pass,
      v_join_num = _mm256_set1_epi32(ht_num), v_one = _mm256_set1_epi32(1),
      v_tuple_cell, ht_cell, v_next_cell, v_matches, v_abort, v_shift, v_ht_pos,
      v_ht_cell_offset, v_ht_addr4, v_ht_addr, v_left_size, v_done,
      v_have_tuple = _mm256_set1_epi32(0), zero256 = _mm256_set1_epi32(0),
      v_join_id = _mm256_set1_epi32(0), v_base_offset, v_offset, v_addr_offset;
  const uint32_t offset_index = 0x76543210, tuple_scale = 8;
  __attribute__((aligned(32))) int32_t base_off[10], *ht_pos, *join_id,
      *tuple_cell, *addr_offset, *done, *have_tuple, *offset,
      tuple_cell_offset[10] = {0}, left_size[8] = {0}, ht_cell_offset[8] = {0},
      mask[16] = {0};
  for (int i = 0; i <= tuple_scale; ++i) {
    base_off[i] = i * pb->tuple_size;
  }
  __attribute__((aligned(32))) uint64_t htp[8] = {0};
  __attribute__((aligned(32))) void * *ht_addr, **ht_addr4;
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
  lld equal_num = 0;
  void* temp_result[8][8], *result_tuple;
  result_tuple = malloc(128);
  __m256i null_int = _mm256_set1_epi32(NULL_INT);
  v_base_offset = _mm256_load_si256((__m256i*)(&base_off));
  uint32_t cur_offset = 0, new_add = 0, continue_mask = pb->tuple_num;
  void* start_addr = pb->start;
  for (lld cur = 0; true;) {
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
        v_tuple_cell, start_addr, _mm256_add_epi32(v_addr_offset, right_index),
        v_have_tuple, 1);
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
        htp, _mm_load_si128(reinterpret_cast<__m128i*>(&join_id[4])), 8);
    v_ht_addr = _mm256_i32gather_epi64(
        htp, _mm_load_si128(reinterpret_cast<__m128i*>(join_id)), 8);

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
void* ThreadProbe(void* args) {
  ThreadArgs* arg = (ThreadArgs*)args;
  lld upper = 0, lower = 0, matched = 0;
  Table* pb = new Table();
  pb->tuple_size = arg->tb->tuple_size;
  while (true) {
    pthread_mutex_lock(&mutex);
    global_probe_corsur = global_probe_corsur + probe_step;
    upper = global_probe_corsur;
    pthread_mutex_unlock(&mutex);
    if (upper <= arg->tb->tuple_num) {
      lower = upper - probe_step;
    } else {
      lower = upper - probe_step;
      if (lower < arg->tb->tuple_num) {
        upper = arg->tb->tuple_num;
      } else {
        break;
      }
    }
    pb->tuple_num = upper - lower;
    pb->start = arg->tb->start + lower * arg->tb->tuple_size;
    matched += arg->func(pb, ht, ht_num);
  }
  pthread_mutex_lock(&mutex);
  global_matched += matched;
  pthread_mutex_unlock(&mutex);
  return NULL;
}
void TestSet(Table* tb, string fun_name, int times, ProbeFunc func,
             int thread_num) {
  struct timeval t1, t2;
  pthread_t id[40];
  int deltaT = 0;
  ThreadArgs args;
  args.tb = tb;
  args.func = func;
  for (int t = 0; t < times; ++t) {
    gettimeofday(&t1, NULL);
    global_probe_corsur = 0;
    global_matched = 0;
    for (int j = 0; j < thread_num; ++j) {
      int ret = pthread_create(&id[j], NULL, ThreadProbe, (void*)&args);
    }
    for (int j = 0; j < thread_num; ++j) {
      pthread_join(id[j], NULL);
    }
    cout << fun_name << " qualified tuples = " << global_matched << endl;

    //   Linear512Probe(&lineorder, ht, ht_num);
    gettimeofday(&t2, NULL);
    deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
    printf("****** probing costs time (ms) = %lf\n", deltaT * 1.0 / 1000);
  }
}

#endif
