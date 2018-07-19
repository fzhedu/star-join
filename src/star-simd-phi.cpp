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
#include <sstream>
#include <map>
using namespace std;
typedef unsigned long long lld;
typedef unsigned int uint;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
#define NULL_INT 2147483647
#define BLOCK_SIZE 65536
#define RESULTS 1
#define OUTPUT 1
#define MEMOUTPUT 0
#define GATHERHT 1
#define PREFETCH 0
#define EARLYBREAK 1
// filter out
float selectity = 0.5;
#define INVALID 2147483647

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
  int max;
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
int thread_num = 2;
int ht_num = 4;
int times = 1;
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
  ht.addr = aligned_alloc(64, table.upper * table.tuple_size);
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
  int bound = table.tuple_num * selectity;  // bound?
  ht.tuple_num = table.tuple_num - bound;
  ht.slot_num = upper_log2(ht.tuple_num);  // slot_num    2^(p+2)
  ht.tuple_size = table.tuple_size;        //
  ht.key_offset = key_offset;              // 0
  ht.probe_offset = probe_offset;
  lld ht_size = ht.slot_num * ht.tuple_size;
  ht.addr = aligned_alloc(64, ht_size);
  ht.shift = 32 - log2(ht.slot_num);
  ht.max = 0;
  memset(ht.addr, -1, ht_size);
  uint key, hash = 0;
  lld offset = 0;
  for (uint i = 0; i < (table.tuple_num - bound) * table.tuple_size;
       i += table.tuple_size) {
    // enum the key in the table
    key = *(uint*)(table.start + i + key_offset);
    ht.max = (key > ht.max ? key : ht.max);
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
    // find the bucket which is not null
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
    tmp += it->first * it->second;  // what is tmp
  }
  cout << "max= " << ht.max << endl;
  cout << "tmp = " << tmp << " tuple_num = " << ht.tuple_num << endl;
  //  assert(tmp == ht.tuple_num);
  return true;
}

bool read_data_in_memory(Table* table, int repeats = 1) {
  void* addr = NULL, *start = NULL;
  struct stat f_stat;
  int fd = -1;
  if ((fd = open(table->path.c_str(), O_RDONLY)) == -1) return false;
  if (fstat(fd, &f_stat)) return false;
  // int len = lseek(fd, 0, SEEK_END);
  lld size = table->tuple_num * table->tuple_size, tuple_num = 0;
  addr = aligned_alloc(64, size * repeats);
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
  for (int j = 1; j < repeats; ++j) {
    memcpy(addr + size * j, addr, size);
  }
  if (munmap(start, f_stat.st_size)) {
    return false;
  }
  close(fd);
  table->start = addr;
  table->tuple_num = table->tuple_num * repeats;
  table->upper = table->tuple_num;  // special
  cout << table->name << " tuple num = " << table->tuple_num << " but read "
       << tuple_num << " repeats= " << repeats << endl;
  assert(table->tuple_num == tuple_num * repeats);
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
  stringstream sstr;
  sstr << pthread_self();
  fp = fopen(string(("hand.csv") + sstr.str()).c_str(), "wr");
  if (fp == 0) {
    assert(false && "can not open file!");
  }
#endif
  for (lld pb_off = 0; pb_off < pb->tuple_num * pb->tuple_size;
       pb_off += pb->tuple_size) {
    tmp = true;
    probe_tuple_start = (pb->start + pb_off);
    for (int j = 0; j < ht_num && tmp; ++j) {
      tuple_key[j] = *(uint*)(probe_tuple_start + ht[j]->probe_offset);
      //      if ((tuple_key[j] > ht[j]->max)) {
      //        tmp = false;
      //        break;
      //      }
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
        fprintf(fp, "%d,", cell_equal[i]);
      }
      fprintf(fp, "%d\n", cell_equal[ht_num]);
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

  stringstream sstr;
  sstr << pthread_self();
  fp = fopen(string(("tuple.csv") + sstr.str()).c_str(), "wr");
  if (fp == 0) {
    assert(false && "can not open file!");
  }
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
void output_vector(char vname[], __m512i* vector) {
  uint32_t* array = (uint32_t*)vector;
  printf("%s ", vname);
  for (int i = 0; i < 16; ++i) {
    printf("%8d", array[i]);
  }
  puts("");
}
void output_vector64(char vname[], __m512i* vector) {
  uint64_t* array = (uint64_t*)vector;
  printf("%s ", vname);
  for (int i = 0; i < 8; ++i) {
    printf("%10x", array[i]);
  }
  puts("");
}
#define _mm512_32bcvt64b(idx, val32)                          \
  _mm512_mask_blend_epi32(                                    \
      _mm512_int2mask(0xAAAA),                                \
      _mm512_permutevar_epi32(_mm512_load_si512(idx), val32), \
      _mm512_set1_epi32(0));

// use short mask to avoid long mask
lld Linear512Probe(Table* pb, HashTable** ht, int ht_num) {
  if (pb->tuple_num <= 0) {
    return 0;
  }
  uint16_t vector_scale = 16, new_add = 0;
  __mmask16 m_bucket_pass = 0, m_done = 0, m_match = 0, m_abort = 0,
            m_have_tuple = 0, m_new_cells = -1;
  __m512i v_offset = _mm512_set1_epi32(0), v_addr_offset = _mm512_set1_epi32(0),
          v_join_num = _mm512_set1_epi32(ht_num),
          v_base_offset_upper =
              _mm512_set1_epi32(pb->tuple_num * pb->tuple_size),
          v_ht_addr_hi, v_ht_addr_lo, v_tuple_cell = _mm512_set1_epi32(0),
          v_right_index, v_base_offset, v_left_size = _mm512_set1_epi32(8),
          v_bucket_offset, v_ht_cell,
          v_factor = _mm512_set1_epi32(table_factor), v_ht_cell_offset, v_shift,
          v_buckets_minus_1, v_cell_hash, v_ht_pos,
          v_neg_one512 = _mm512_set1_epi32(-1), v_ht_cell_upper,
          v_ht_cell_lower, v_zero512 = _mm512_set1_epi32(0), v_ht_pos_lo,
          v_ht_pos_hi, v_join_id = _mm512_set1_epi32(0),
          v_one = _mm512_set1_epi32(1), v_payloads, v_payloads_off,
          v_word_size = _mm512_set1_epi64(WORDSIZE);
  __attribute__((aligned(64)))
  uint32_t cur_offset = 0,
           tmp_cell[16], *addr_offset, temp_payloads[16][5] = {0}, *ht_pos,
           *join_id, base_off[32], tuple_cell_offset[16] = {0},
           left_size[16] = {0}, ht_cell_offset[16] = {0}, payloads_off[32],
           buckets_minus_1[16] = {0}, shift[16] = {0};
  __attribute__((aligned(64))) void * *ht_addr, **ht_addr4, *tmp_ht_addr[16];
  __attribute__((aligned(64))) uint64_t htp[16] = {0};
  __attribute__((aligned(64))) int id[32] = {
      0, 0, 1, 1, 2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,
      8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15};
  __attribute__((aligned(64))) uint32_t tmp_addr_offset[32];
#if MEMOUTPUT
  __attribute__((aligned(64))) uint32_t output_buffer[output_buffer_size];
#endif
  for (int i = 0; i <= vector_scale; ++i) {
    base_off[i] = i * pb->tuple_size;
    payloads_off[i] = i * 5;  // subject to the second size of temp_payloads
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
  ht_addr = (void**)&v_ht_addr_lo;
  ht_addr4 = (void**)&v_ht_addr_hi;
  addr_offset = (uint32_t*)&v_addr_offset;

  lld equal_num = 0;
  v_base_offset = _mm512_load_si512((__m512i*)(&base_off));
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
  int times = 0;
  for (lld cur = 0; cur < pb->tuple_num || m_have_tuple;) {
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
    v_bucket_offset = _mm512_mask_add_epi32(v_zero512, _mm512_knot(m_new_cells),
                                            v_bucket_offset, v_one);

    v_cell_hash = _mm512_add_epi32(v_cell_hash, v_bucket_offset);
    // avoid overflow
    v_cell_hash = _mm512_and_si512(v_cell_hash, v_buckets_minus_1);

    v_ht_pos = _mm512_mullo_epi32(v_cell_hash, v_left_size);
#if GATHERHT
    v_ht_pos_lo = _mm512_32bcvt64b(id, v_ht_pos);
    v_ht_pos_hi = _mm512_32bcvt64b(&id[16], v_ht_pos);
    v_ht_addr_lo = _mm512_i32logather_epi64(v_join_id, htp, 8);
    v_join_id = _mm512_permute4f128_epi32(v_join_id, _MM_PERM_BADC);
    v_ht_addr_hi = _mm512_i32logather_epi64(v_join_id, htp, 8);
    v_join_id = _mm512_permute4f128_epi32(v_join_id, _MM_PERM_BADC);
    v_ht_addr_hi = _mm512_add_epi64(v_ht_pos_hi, v_ht_addr_hi);
    v_ht_addr_lo = _mm512_add_epi64(v_ht_pos_lo, v_ht_addr_lo);
    v_ht_cell_upper = _mm512_mask_i64gather_epi32lo(
        v_zero512, (m_have_tuple >> 8), v_ht_addr_hi, 0, 1);
    v_ht_cell_lower = _mm512_mask_i64gather_epi32lo(v_zero512, m_have_tuple,
                                                    v_ht_addr_lo, 0, 1);
    v_ht_cell_upper = _mm512_permute4f128_epi32(v_ht_cell_upper, _MM_PERM_BADC);
    v_ht_cell = _mm512_or_epi32(v_ht_cell_lower, v_ht_cell_upper);

#else
// no improvement, but it is worse when judging the m_have_tuple
// PREFETCHNTA
#if PREFETCH
    for (int i = 0; i < vector_scale; ++i) {
      _mm_prefetch((char*)(ht_pos[i] + htp[join_id[i]]), _MM_HINT_T0);
    }
#endif
    for (int i = 0; i < vector_scale; ++i) {
      tmp_ht_addr[i] = (void*)(ht_pos[i] + htp[join_id[i]]);
      tmp_cell[i] = *(uint32_t*)(tmp_ht_addr[i]);
    }

    v_ht_cell = _mm512_load_epi32(tmp_cell);
#endif

    //// step 4: compare
    // load raw cell data, then judge whether they are equal ? the AND get
    // rid of invalid keys
    m_match = _mm512_cmpeq_epi32_mask(v_tuple_cell, v_ht_cell);
    m_match = _mm512_kand(m_match, m_have_tuple);
#if RESULTS
#if !SCATTER
    // it is slower than salar code
    ///// step 5: store matched payloads
    // load payloads
    v_ht_addr_hi = _mm512_add_epi64(v_word_size, v_ht_addr_hi);
    v_ht_addr_lo = _mm512_add_epi64(v_word_size, v_ht_addr_lo);

    v_ht_cell_upper = _mm512_mask_i64gather_epi32lo(v_zero512, (m_match >> 8),
                                                    v_ht_addr_hi, 0, 1);
    v_ht_cell_lower =
        _mm512_mask_i64gather_epi32lo(v_zero512, m_match, v_ht_addr_lo, 0, 1);
    v_ht_cell_upper = _mm512_permute4f128_epi32(v_ht_cell_upper, _MM_PERM_BADC);
    v_payloads = _mm512_or_epi32(v_ht_cell_lower, v_ht_cell_upper);

    v_offset = _mm512_add_epi32(v_join_id, v_payloads_off);
    _mm512_mask_i32scatter_epi32(temp_payloads, m_match, v_offset, v_payloads,
                                 4);
#else
    __mmask16 m_match_copy = m_match;
#if GATHERHT
    int mid = vector_scale >> 1;
    for (int i = 0; m_match_copy && i < mid;
         ++i, m_match_copy = (m_match_copy >> 1)) {
      if (m_match_copy & 1) {
        temp_payloads[i][join_id[i]] = *(uint32_t*)(WORDSIZE + ht_addr[i]);
      }
    }
    for (int i = mid; m_match_copy && i < vector_scale;
         ++i, m_match_copy = (m_match_copy >> 1)) {
      if (m_match_copy & 1) {
        temp_payloads[i][join_id[i]] =
            *(uint32_t*)(WORDSIZE + ht_addr4[i - mid]);
      }
    }

#else
    for (int i = 0; m_match_copy && (i < vector_scale);
         ++i, m_match_copy = m_match_copy >> 1) {
      //      if (m_match_copy & 1) {
      temp_payloads[i][join_id[i]] = *(uint32_t*)(WORDSIZE + tmp_ht_addr[i]);
      //      }
    }
#endif
#endif
#endif
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
uint16_t zero_count(uint16_t a) {
  uint16_t num = 0;
  while ((a & 1) == 0 && num < 16) {
    ++num;
    a = (a >> 1);
  }
  return num;
}

lld Linear512ProbeHor(Table* pb, HashTable** ht, int ht_num) {
  if (pb->tuple_num <= 0) {
    return 0;
  }
  uint16_t vector_scale = 16, new_add = 0;
  __mmask16 m_bucket_pass = 0, m_done = 0, m_match = 0, m_abort = 0,
            m_have_tuple = 0;
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
  __attribute__((aligned(64)))
  uint32_t cur_offset = 0,
           tmp_cell[16], *addr_offset, temp_payloads[16][16] = {0}, *ht_pos,
           *tuple_cell, *join_id, base_off[32], tuple_cell_offset[16] = {0},
           left_size[16] = {0}, ht_cell_offset[16] = {0},
           buckets_minus_1[16] = {0}, shift[16] = {0}, h_buf[16] = {0};
  __attribute__((aligned(64))) void * *ht_addr, **ht_addr4, *tmp_ht_addr[16];
  __attribute__((aligned(64))) uint64_t htp[16] = {0}, bucket_upper[16] = {0};
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

  lld equal_num = 0;
  v_base_offset = _mm512_load_si512((__m512i*)(&base_off));
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
  for (lld cur = 0; cur < pb->tuple_num || m_have_tuple;) {
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
        __m512i tab =
            _mm512_loadunpacklo_epi32(v_zero512, (void*)(htp[join_id[j]] + p));
        tab = _mm512_loadunpackhi_epi32(tab, (void*)(htp[join_id[j]] + p + 64));
        __mmask16 out = _mm512_cmpeq_epi32_mask(tab, key_x16);
        out = _mm512_kand(out, 0x5555);
        if (out > 0) {
          int znum = (zero_count(out) >> 1);
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
    v_join_id = _mm512_mask_blend_epi32(m_have_tuple, v_zero512, v_join_id);
    equal_num += _mm_countbits_32(_mm512_mask2int(m_done));
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
  pthread_t id[300];
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
