
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
}* ht[10];

unsigned int table_factor = 1;
#define OVER_FLOW 2147479552  // 2^31 - 4096
#define OVER 2147475456       // =OVER_FLOW- 4096
typedef unsigned int uint32_t;

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
                     int probe_offset) {
  ht.slot_num = upper_log2(table.tuple_num);
  ht.tuple_num = table.tuple_num;
  ht.tuple_size = table.tuple_size;
  ht.key_offset = key_offset;
  ht.probe_offset = probe_offset;
  lld ht_size = ht.slot_num * ht.tuple_size;
  ht.addr = malloc(ht_size);
  ht.shift = 32 - log2(ht.slot_num);
  memset(ht.addr, -1, ht_size);
  uint key, hash = 0;
  lld offset = 0;
  for (uint i = 0; i < table.tuple_num * table.tuple_size;
       i += table.tuple_size) {
    // enum the key in the table
    key = *(uint*)(table.start + i + key_offset);
    // set filter
    // if (key >= 10000) continue;
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
      cout << k << endl;
      t++;
    }
    count++;
  }
  cout << endl << count << " has " << t << endl;
}

bool LinearHandProbe(Table* pb, HashTable** ht, int ht_num) {
  bool tmp = true;
  uint key[10], hash = 0, ht_off, cell_equal[10][10], num = 0;
  void* probe_tuple_start = NULL;  // attention!!!
  memset(cell_equal, 0, 100);

  for (lld pb_off = 0; pb_off < pb->tuple_num * pb->tuple_size;
       pb_off += pb->tuple_size) {
    tmp = true;
    probe_tuple_start = (uint*)(pb->start + pb_off);
    for (int j = 0; j < ht_num && tmp; ++j) {
      key[j] = *(uint*)(probe_tuple_start + ht[j]->probe_offset);
      hash = ((uint)(key[j] * table_factor)) >> ht[j]->shift;
      // lay at the hash table
      assert(hash <= ht[j]->slot_num);
      ht_off = hash * ht[j]->tuple_size;
      cell_equal[j][0] = 0;

      // probe each bucket
      while (*(int*)(ht[j]->addr + ht_off + ht[j]->key_offset) != -1) {
        if (key[j] == *(uint*)(ht[j]->addr + ht_off + ht[j]->key_offset)) {
          ++cell_equal[j][0];
          cell_equal[j][cell_equal[j][0]] = ht_off;
        }
        ht_off += ht[j]->tuple_size;
      }
      // if the cell does not match anyone
      if (cell_equal[j][0] == 0) {
        tmp = false;
      }
    }

    num += tmp;
  }
  cout << "probe result = " << num << endl;
  return true;
}
bool HandProbe(Table* pb, HashTable** ht, int ht_num) {
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
  cout << "HandProbe qualified tuples = " << num << endl;
  return true;
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

char* LinearProbeTuple(void* src, int src_size, HashTable** ht, int ht_num,
                       int cur, int& ret_size, char output[][128]) {
  char* res;
  void* start;
  if (ht_num == cur + 1) {
    uint key = *(uint*)(src + ht[cur]->probe_offset);
    uint hash = (uint)(key * table_factor) >> ht[cur]->shift;
    uint ht_off = hash * ht[cur]->tuple_size;
    while (*(int*)(ht[cur]->addr + ht_off + ht[cur]->key_offset) != -1) {
      if (key == *(uint*)(ht[cur]->addr + ht_off + ht[cur]->key_offset)) {
#if RESULTS
        memcpy(output[cur], src, src_size);
        memcpy(output[cur] + src_size, ht[cur]->addr + ht_off,
               ht[cur]->tuple_size);
#endif
        ret_size = src_size + ht[cur]->tuple_size;
        return output[cur];
      }
      ht_off += ht[cur]->tuple_size;
    }

  } else {
    res =
        LinearProbeTuple(src, src_size, ht, ht_num, cur + 1, ret_size, output);
    if (res == NULL) {
      return NULL;
    }
    uint key = *(uint*)(src + ht[cur]->probe_offset);
    uint hash = (uint)(key * table_factor) >> ht[cur]->shift;
    uint ht_off = hash * ht[cur]->tuple_size;
    while (*(int*)(ht[cur]->addr + ht_off + ht[cur]->key_offset) != -1) {
      if (key == *(uint*)(ht[cur]->addr + ht_off + ht[cur]->key_offset)) {
#if RESULTS
        memcpy(output[cur], res, ret_size);
        memcpy(output[cur] + ret_size, ht[cur]->addr + ht_off,
               ht[cur]->tuple_size);
#endif
        ret_size = ret_size + ht[cur]->tuple_size;
        return output[cur];
      }
      ht_off += ht[cur]->tuple_size;
    }
  }
  return NULL;
}
bool TupleAtATimeProbe(Table* pb, HashTable** ht, int ht_num) {
  int ret_size, len = 128;
  char* res;
  lld num = 0;
  char output[10][128];
  for (lld i = 0, off = 0; i < pb->tuple_num; ++i, off += pb->tuple_size) {
    ret_size = 0;
    //    res = ProbeTuple(pb->start + off, pb->tuple_size, ht, ht_num, 0,
    //    ret_size,output);

    res = LinearProbeTuple(pb->start + off, pb->tuple_size, ht, ht_num, 0,
                           ret_size, output);
    if (res) {
      ++num;
    }
  }
  cout << "TupleAtATimeProbe qualified tuples = " << num << endl;

  return true;
}
// use short mask to avoid long mask

bool Linear512Probe(Table* pb, HashTable** ht, int ht_num) {
  uint16_t vector_scale = 16, new_add;
  __mmask16 m_bucket_pass, m_done, m_match, m_abort, m_have_tuple = 0,
                                                     m_new_cells = -1;
  __m512i v_offset = _mm512_set1_epi32(0), v_addr_offset,
          v_join_num = _mm512_set1_epi32(ht_num),
          v_base_offset_upper =
              _mm512_set1_epi32(pb->tuple_num * pb->tuple_size),
          v_ht_addr4, v_ht_addr, v_tuple_cell, right_index, v_base_offset,
          v_left_size, v_bucket_offset, ht_cell,
          v_factor = _mm512_set1_epi32(table_factor), v_ht_cell_offset, v_shift,
          v_buckets_minus_1, v_cell_hash, v_ht_pos,
          v_neg_one512 = _mm512_set1_epi32(-1),
          v_zero512 = _mm512_set1_epi32(0), v_join_id = _mm512_set1_epi32(0),
          v_one = _mm512_set1_epi32(1);
  __attribute__((aligned(32)))
  uint32_t cur_offset = 0,
           *ht_pos, *join_id, base_off[32], tuple_cell_offset[16] = {0},
           left_size[16] = {0}, ht_cell_offset[16] = {0},
           buckets_minus_1[16] = {0}, shift[16] = {0};
  __attribute__((aligned(32))) void * *ht_addr, **ht_addr4;
  __attribute__((aligned(32))) uint64_t htp[16] = {0};

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
  }
  ht_pos = (uint32_t*)&v_ht_pos;
  join_id = (int32_t*)&v_join_id;
  ht_addr = (void**)&v_ht_addr;
  ht_addr4 = (void**)&v_ht_addr4;
  __m256i v_zero256 = _mm256_set1_epi32(0);
  lld equal_num = 0;
  v_base_offset = _mm512_load_si512((__m512i*)(&base_off));

  for (lld cur = 0; cur < pb->tuple_num || m_have_tuple;) {
    ///////// step 1: load new tuples' address offsets
    v_offset = _mm512_add_epi32(_mm512_set1_epi32(cur_offset), v_base_offset);
    v_addr_offset = _mm512_mask_expand_epi32(
        v_addr_offset, _mm512_knot(m_have_tuple), v_offset);
    // count the number of empty tuples
    new_add = _mm_popcnt_u32(_mm512_knot(m_have_tuple));
    cur = cur + new_add;
    cur_offset = cur_offset + base_off[new_add];
    m_have_tuple = _mm512_cmpgt_epi32_mask(v_base_offset_upper, v_addr_offset);
    /////// step 2: load new cells from tuples
    /*
     * the cases that need to load new cells
     * (1) new tuples -> v_have_tuple
     * (2) last cells are matched for all cells in corresponding buckets ->
     * v_next_cells
     */

    right_index = _mm512_i32gather_epi32(v_join_id, tuple_cell_offset, 4);
    m_new_cells = _mm512_kand(m_new_cells, m_have_tuple);
    v_tuple_cell = _mm512_mask_i32gather_epi32(
        v_tuple_cell, m_new_cells, _mm512_add_epi32(v_addr_offset, right_index),
        pb->start, 1);

    ////// step 3: load new values in hash tables
    v_left_size = _mm512_i32gather_epi32(v_join_id, left_size, 4);
    v_ht_cell_offset = _mm512_i32gather_epi32(v_join_id, ht_cell_offset, 4);
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
    v_ht_pos = _mm512_add_epi32(v_ht_pos, v_ht_cell_offset);
    __m512i v_ht_pos_644 = _mm512_cvtepu32_epi64(
        _mm256_load_si256(reinterpret_cast<__m256i*>(&ht_pos[8])));
    __m512i v_ht_pos_64 = _mm512_cvtepu32_epi64(
        _mm256_load_si256(reinterpret_cast<__m256i*>(ht_pos)));

    v_ht_addr4 = _mm512_i32gather_epi64(
        _mm256_load_si256(reinterpret_cast<__m256i*>(&join_id[8])), htp, 8);
    v_ht_addr = _mm512_i32gather_epi64(
        _mm256_load_si256(reinterpret_cast<__m256i*>(join_id)), htp, 8);

    __m256i a = _mm512_mask_i64gather_epi32(
        v_zero256, m_have_tuple >> 8,
        _mm512_add_epi64(v_ht_pos_644, v_ht_addr4), 0, 1);
    __m256i b = _mm512_mask_i64gather_epi32(
        v_zero256, m_have_tuple, _mm512_add_epi64(v_ht_pos_64, v_ht_addr), 0,
        1);
    // the first parameter should be at the lower position, and the second one
    // should lay at the upper position, and the third must be 1
    ht_cell = _mm512_inserti32x8(_mm512_castsi256_si512(b), a, 1);

    m_match = _mm512_cmpeq_epi32_mask(v_tuple_cell, ht_cell);
    m_match = _mm512_kand(m_match, m_have_tuple);
    // the bucket is over if ht cells =-1 or early break due to match
    // so need to process new cells
    // then process next buckets (increase join_id and load new cells)
    // or process next tuples
    // or abort
    m_new_cells = _mm512_cmpeq_epi32_mask(ht_cell, v_neg_one512);
    m_new_cells = _mm512_kor(m_new_cells, m_match);
    v_join_id = _mm512_mask_add_epi32(v_join_id, m_new_cells, v_join_id, v_one);

    m_bucket_pass = _mm512_kor(m_bucket_pass, m_match);
    m_done = _mm512_kand(m_bucket_pass,
                         _mm512_cmpeq_epi32_mask(v_join_id, v_join_num));
    m_abort = _mm512_kandn(m_bucket_pass, m_new_cells);
    m_have_tuple = _mm512_kandn(_mm512_kor(m_done, m_abort), m_have_tuple);
    v_join_id = _mm512_maskz_add_epi32(m_have_tuple, v_join_id, v_zero512);
    m_bucket_pass = _mm512_kandn(m_new_cells, m_bucket_pass);
    equal_num += _mm_popcnt_u32(m_done);
  }
  cout << "SIMD512Probe qualified tuples = " << equal_num << endl;

  return true;
}
bool LinearSIMDProbe(Table* pb, HashTable** ht, int ht_num) {
  __m256i v_pass,
      v_join_num = _mm256_set1_epi32(ht_num), v_one = _mm256_set1_epi32(1),
      v_tuple_cell, ht_cell, v_next_cell, v_matches, v_abort, v_shift, v_ht_pos,
      v_ht_cell_offset, v_ht_addr4, v_ht_addr, v_left_size, v_done,
      v_have_tuple = _mm256_set1_epi32(0), zero256 = _mm256_set1_epi32(0),
      v_factor = _mm256_set1_epi32(table_factor), v_cell_hash,
      // need to load new cells when the bucket is over
      v_new_cells = _mm256_set1_epi32(-1), v_bucket_pass = _mm256_set1_epi32(0),
      // travel each tuple in the bucket
      v_bucket_offset = _mm256_set1_epi32(0),
      // join id +1 if the bucket is over; join id =0 if new tuples are loaded
      v_join_id = _mm256_set1_epi32(0),
      v_base_offset_uppper = _mm256_set1_epi32(pb->tuple_num * pb->tuple_size),
      v_buckets_minus_1 = _mm256_set1_epi32(0), v_base_offset, v_offset,
      v_addr_offset;
  const uint32_t offset_index = 0x76543210, tuple_scale = 8;
  __attribute__((aligned(32))) int32_t base_off[10], *ht_pos, *join_id,
      *tuple_cell, *addr_offset, *done, *have_tuple, *offset,
      tuple_cell_offset[10] = {0}, left_size[8] = {0}, ht_cell_offset[8] = {0},
      buckets_minus_1[8] = {0}, mask[16] = {0}, shift[8] = {0};
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
  result_tuple = malloc(128);
  __m256i null_int = _mm256_set1_epi32(NULL_INT);
  v_base_offset = _mm256_load_si256((__m256i*)(&base_off));
  uint32_t cur_offset = 0, new_add = 0, continue_mask = pb->tuple_num;
  void* start_addr = pb->start;
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
    __m256i right_index =
        _mm256_i32gather_epi32(tuple_cell_offset, v_join_id, 4);
    // guarantee valid cells from valid tuples
    v_new_cells = _mm256_and_si256(v_new_cells, v_have_tuple);
    // gather cell values in new tuples
    v_tuple_cell = _mm256_mask_i32gather_epi32(
        v_tuple_cell, start_addr, _mm256_add_epi32(v_addr_offset, right_index),
        v_new_cells, 1);

    ////// step 3: load new values in hash tables
    // get rid of invalid cells
    //__m256i v_invalid = _mm256_cmpeq_epi32(v_tuple_cell, null_int);
    // v_have_tuple = _mm256_andnot_si256(v_invalid, v_have_tuple);
    // get the position of tuple values at the hash table
    v_left_size = _mm256_i32gather_epi32(left_size, v_join_id, 4);
    v_ht_cell_offset = _mm256_i32gather_epi32(ht_cell_offset, v_join_id, 4);
    v_shift = _mm256_i32gather_epi32(shift, v_join_id, 4);
    v_buckets_minus_1 = _mm256_i32gather_epi32(buckets_minus_1, v_join_id, 4);
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
#if !RESULTS
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
    // the bucket is over if ht cells =-1 or early break due to match
    // so need to process new cells
    // then process next buckets (increase join_id and load new cells)
    // or process next tuples
    // or abort
    v_new_cells =
        _mm256_or_si256(_mm256_cmpeq_epi32(ht_cell, neg_one), v_matches);
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
#if !RESULTS
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
  cout << "SIMDProbe qualified tuples = " << equal_num << endl;

  return true;
}

bool SIMDProbe(Table* pb, HashTable** ht, int ht_num) {
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
  cout << "SIMDProbe qualified tuples = " << equal_num << endl;

  return true;
}
int main() {
  struct timeval t1, t2;
  int deltaT = 0;
  gettimeofday(&t1, NULL);
  table_factor = (rand() << 1) | 1;
  cout << "table_factor = " << table_factor << endl;
//  table_factor = (rand() << 1) | 1;
//  cout << "table_factor = " << table_factor << endl;
//  table_factor = (rand() << 1) | 1;
//  cout << "table_factor = " << table_factor << endl;

#if TPCDS
  Table store_sales;
  store_sales.name = "store_sales";
  store_sales.tuple_num = 287997024;
  store_sales.path = "/home/claims/data/tpc-ds/sf100/1partition/T48G0P0";
  store_sales.raw_tuple_size = 100;
  int array[10] = {12, 28, 36};
  SetVectorValue(array, 3, store_sales.offset);
  int array1[10] = {4, 4, 4};
  SetVectorValue(array1, 3, store_sales.size);
  store_sales.tuple_size = SumOfVector(store_sales.size);
  tb[0] = &store_sales;
  read_data_in_memory(&store_sales);

  Table time_dim;
  time_dim.name = "time_dim";
  time_dim.tuple_num = 86400;
  time_dim.path = "/home/claims/data/tpc-ds/sf100/1partition/T12G0P0";
  time_dim.raw_tuple_size = 124;
  int array3[10] = {8};
  SetVectorValue(array3, 1, time_dim.offset);
  int array4[10] = {4};
  SetVectorValue(array4, 1, time_dim.size);
  time_dim.tuple_size = SumOfVector(time_dim.size);
  tb[1] = &time_dim;
  read_data_in_memory(&time_dim);

  Table store;
  store.name = "store";
  store.tuple_num = 402;
  store.path = "/home/claims/data/tpc-ds/sf100/1partition/T20G0P0";
  store.raw_tuple_size = 788;
  int array5[10] = {8};
  SetVectorValue(array5, 1, store.offset);
  int array6[10] = {4};
  SetVectorValue(array6, 1, store.size);
  store.tuple_size = SumOfVector(store.size);
  tb[2] = &store;
  read_data_in_memory(&store);

  Table household_demographics;
  household_demographics.name = "household_demographics";
  household_demographics.tuple_num = 7200;
  household_demographics.path =
      "/home/claims/data/tpc-ds/sf100/1partition/T30G0P0";
  household_demographics.raw_tuple_size = 40;
  int array7[10] = {8};
  SetVectorValue(array7, 1, household_demographics.offset);
  int array8[10] = {4};
  SetVectorValue(array8, 1, household_demographics.size);
  household_demographics.tuple_size = SumOfVector(household_demographics.size);
  tb[3] = &household_demographics;
  read_data_in_memory(&household_demographics);

  Table household_demographics;
  household_demographics.name = "household_demographics";
  household_demographics.tuple_num = 7200;
  household_demographics.path =
      "/home/claims/data/tpc-ds/sf100/1partition/T30G0P0";
  household_demographics.raw_tuple_size = 40;
  int array7[10] = {8};
  SetVectorValue(array7, 1, household_demographics.offset);
  int array8[10] = {4};
  SetVectorValue(array8, 1, household_demographics.size);
  household_demographics.tuple_size = SumOfVector(household_demographics.size);
  tb[3] = &household_demographics;
  read_data_in_memory(&household_demographics);
#else
  Table part;
  part.name = "part";
  part.tuple_num = 2000000;
  part.path = "/home/claims/data/tpc-h/sf10/1partition/T0G0P0";
  part.raw_tuple_size = 8;
  int array5[10] = {0, 4};
  SetVectorValue(array5, 2, part.offset);
  int array6[10] = {4, 4};
  SetVectorValue(array6, 2, part.size);
  part.tuple_size = SumOfVector(part.size);
  tb[0] = &part;
  read_data_in_memory(&part);
  //  test(&part, part.start, 0);

  Table supplier;
  supplier.name = "supplier";
  supplier.tuple_num = 100000;
  supplier.path = "/home/claims/data/tpc-h/sf10/1partition/T2G0P0";
  supplier.raw_tuple_size = 8;
  int array7[10] = {0, 4};
  SetVectorValue(array7, 2, supplier.offset);
  int array8[10] = {4, 4};
  SetVectorValue(array8, 2, supplier.size);
  supplier.tuple_size = SumOfVector(supplier.size);
  tb[1] = &supplier;
  read_data_in_memory(&supplier);
  //  test(&supplier, supplier.start, 0);
  /*
  L_ORDERKEY,
  L_PARTKEY,
  L_SUPPKEY,
  L_LINENUMBER
   */
  Table lineitem;
  lineitem.name = "lineitem";
  lineitem.tuple_num = 59986052;
  lineitem.path = "/home/claims/data/tpc-h/sf10/1partition/T6G0P0";
  lineitem.raw_tuple_size = 16;
  int array9[10] = {4, 8, 12};
  SetVectorValue(array9, 3, lineitem.offset);
  int array99[10] = {4, 4, 4};
  SetVectorValue(array99, 3, lineitem.size);
  lineitem.tuple_size = SumOfVector(lineitem.size);
  tb[2] = &lineitem;
  read_data_in_memory(&lineitem);
// test(&lineitem, lineitem.start, 0);
#endif
  gettimeofday(&t2, NULL);
  deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
  printf("--------load table is over costs time (ms) = %lf\n",
         deltaT * 1.0 / 1000);
  gettimeofday(&t1, NULL);
#if TPCDS
  HashTable ht_time_dim;
  build_ht(ht_time_dim, time_dim, 0, 0);
  test(&time_dim, ht_time_dim.addr, 0);
  ht[0] = &ht_time_dim;

  HashTable ht_household_demographics;
  build_ht(ht_household_demographics, household_demographics, 0, 4);
  test(&household_demographics, ht_household_demographics.addr, 0);
  ht[1] = &ht_household_demographics;

  HashTable ht_store;
  build_ht(ht_store, store, 0, 8);
  test(&store, ht_store.addr, 0);
  ht[2] = &ht_store;
  store_sales.tuple_num = 178956000;
// store_sales.tuple_num = 100000;
#else
  HashTable ht_part;
  build_linear_ht(ht_part, part, 0, 0);
  travel_linear_ht(ht_part);
  ht[0] = &ht_part;

  HashTable ht_supplier;
  build_linear_ht(ht_supplier, supplier, 0, 4);
  travel_linear_ht(ht_supplier);
  ht[1] = &ht_supplier;
#endif
  gettimeofday(&t2, NULL);
  deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
  printf("++++ build hashtable costs time (ms) = %lf\n", deltaT * 1.0 / 1000);
#if TPCDS
  int times = 10;
  for (int t = 0; t < times; ++t) {
    gettimeofday(&t1, NULL);

    HandProbe(&store_sales, ht, 3);
    // TupleAtATimeProbe(&store_sales, ht, 3);
    // SIMDProbe(&store_sales, ht, 3);
    gettimeofday(&t2, NULL);
    deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
    printf("++++ build hashtable costs time (ms) = %lf\n", deltaT * 1.0 / 1000);
  }
  for (int t = 0; t < times; ++t) {
    gettimeofday(&t1, NULL);

    // HandProbe(&store_sales, ht, 3);
    TupleAtATimeProbe(&store_sales, ht, 3);
    // SIMDProbe(&store_sales, ht, 3);
    gettimeofday(&t2, NULL);
    deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
    printf("++++ build hashtable costs time (ms) = %lf\n", deltaT * 1.0 / 1000);
  }
  for (int t = 0; t < times; ++t) {
    gettimeofday(&t1, NULL);

    // HandProbe(&store_sales, ht, 3);
    // TupleAtATimeProbe(&store_sales, ht, 3);
    SIMDProbe(&store_sales, ht, 3);

    gettimeofday(&t2, NULL);
    deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
    printf("++++ build hashtable costs time (ms) = %lf\n", deltaT * 1.0 / 1000);
  }
#else
  int times = 5;
  for (int t = 0; t < times; ++t) {
    gettimeofday(&t1, NULL);
    // LinearHandProbe(&lineitem, ht, 2);
    // TupleAtATimeProbe(&lineitem, ht, 2);
    Linear512Probe(&lineitem, ht, 2);
    gettimeofday(&t2, NULL);
    deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
    printf("****** probing costs time (ms) = %lf\n", deltaT * 1.0 / 1000);
  }
  for (int t = 0; t < times; ++t) {
    gettimeofday(&t1, NULL);
    // LinearHandProbe(&lineitem, ht, 2);
    // TupleAtATimeProbe(&lineitem, ht, 2);
    LinearSIMDProbe(&lineitem, ht, 2);
    gettimeofday(&t2, NULL);
    deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
    printf("****** probing costs time (ms) = %lf\n", deltaT * 1.0 / 1000);
  }
  for (int t = 0; t < times; ++t) {
    gettimeofday(&t1, NULL);
    // LinearHandProbe(&lineitem, ht, 2);
    TupleAtATimeProbe(&lineitem, ht, 2);
    // SIMDProbe(&store_sales, ht, 3);
    gettimeofday(&t2, NULL);
    deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
    printf("****** probing costs time (ms) = %lf\n", deltaT * 1.0 / 1000);
  }
  for (int t = 0; t < times; ++t) {
    gettimeofday(&t1, NULL);
    LinearHandProbe(&lineitem, ht, 2);
    // TupleAtATimeProbe(&store_sales, ht, 3);
    // SIMDProbe(&store_sales, ht, 3);
    gettimeofday(&t2, NULL);
    deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
    printf("****** probing costs time (ms) = %lf\n", deltaT * 1.0 / 1000);
  }
#endif
  return 0;
}
