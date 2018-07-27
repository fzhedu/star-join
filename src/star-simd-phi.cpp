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
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
#define NULL_INT 2147483647
#define BLOCK_SIZE 65536
#define RESULTS 1
#define OUTPUT 0
#define MEMOUTPUT 1
#define GATHERHT 1
// if adopt global address
#define GLOBALADDR 1
#define PAYLOADSIZE 4
#define PREFETCH 1
#define EARLYBREAK 1
// filter out
float selectity = 0.5;
#define INVALID 2147483647
#define up64(a) (a - (a % 64) + 64)
struct Table {
  string name;
  string path;
  uint64_t raw_tuple_size;
  uint64_t tuple_size;
  uint64_t tuple_num;
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
  void* global_addr;
  int global_addr_offset;
  int shift;
  int max;
  int probe_offset;
}* ht[100];
typedef uint64_t (*ProbeFunc)(Table* pb, HashTable** ht, int ht_num,
                              char* payloads);
struct ThreadArgs {
  Table* tb;
  ProbeFunc func;
};
#define WORDSIZE 4
unsigned int table_factor = 1;
typedef unsigned int uint32_t;
pthread_mutex_t mutex;
uint64_t global_probe_corsur = 0, global_matched = 0;
#define probe_step 1024 * 1024  // 1048576
int thread_num = 2;
int ht_num = 4;
int times = 1;
int output_buffer_size = 32;
// typedef unsigned long long uint64_t;

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
  uint64_t ht_size = ht.slot_num * ht.tuple_size;
  ht.addr = (void*)(ht.global_addr + ht.global_addr_offset);
  ht.shift = 32 - log2(ht.slot_num);
  ht.max = 0;
  memset(ht.addr, -1, ht_size);
  uint32_t key, hash = 0;
  uint64_t offset = 0;
  for (uint32_t i = 0; i < (table.tuple_num - bound) * table.tuple_size;
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
  uint64_t size = table->tuple_num * table->tuple_size, tuple_num = 0;
  addr = aligned_alloc(64, size * repeats);
  assert(addr != NULL);
  if ((start = mmap(0, f_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0)) ==
      MAP_FAILED) {
    return false;
  }
  int num = 0;
  // read each BLOCK, then get each tuple in a block
  for (uint64_t k = 0, j = 0; k < f_stat.st_size; k += BLOCK_SIZE) {
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

uint64_t LinearHandProbe(Table* pb, HashTable** ht, int ht_num,
                         char* payloads) {
  bool tmp = true, flag;
  uint32_t tuple_key[10], hash = 0, ht_off, num = 0;
  void* probe_tuple_start = NULL, *payloads_addr[32];  // attention!!!
  uint32_t cur_payloads = 0, result_size = PAYLOADSIZE * (ht_num + 1);
#if OUTPUT
  FILE* fp;
  stringstream sstr;
  sstr << pthread_self();
  fp = fopen(string(("hand.csv") + sstr.str()).c_str(), "wr");
  if (fp == 0) {
    assert(false && "can not open file!");
  }
#endif
  for (uint64_t pb_off = 0; pb_off < pb->tuple_num * pb->tuple_size;
       pb_off += pb->tuple_size) {
    tmp = true;
    probe_tuple_start = (pb->start + pb_off);
    for (int j = 0; j < ht_num && tmp; ++j) {
      tuple_key[j] = *(uint32_t*)(probe_tuple_start + ht[j]->probe_offset);
      hash = ((uint32_t)(tuple_key[j] * table_factor)) >> ht[j]->shift;
      // lay at the hash table
      // assert(hash <= ht[j]->slot_num);
      ht_off = hash * ht[j]->tuple_size;
      flag = false;
#if PREFETCH
      _mm_prefetch((char*)(ht[j]->addr + ht_off), _MM_HINT_T0);
#endif
      // probe each bucket
      while (*(int*)(ht[j]->addr + ht_off + ht[j]->key_offset) != -1) {
        if (tuple_key[j] ==
            *(uint32_t*)(ht[j]->addr + ht_off + ht[j]->key_offset)) {
          // record the payload address
          payloads_addr[j] =
              (void*)(ht[j]->addr + ht_off + ht[j]->key_offset + WORDSIZE);
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
      int output_off = 0;
      for (int i = 0; i < ht_num; ++i) {
        memcpy(payloads + cur_payloads + output_off, (char*)payloads_addr[i],
               PAYLOADSIZE);
        output_off += PAYLOADSIZE;
      }
      memcpy(payloads + cur_payloads + output_off,
             (char*)(probe_tuple_start + WORDSIZE * ht_num), PAYLOADSIZE);
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
#endif
    num += tmp;
  }
// cout << "LinearHandProbe result = " << num << endl;
#if OUTPUT
  fclose(fp);
#endif
  return num;
}

char* LinearProbeTuple(void* src, int src_size, HashTable** ht, int ht_num,
                       int cur, int& ret_size,
                       char output[][PAYLOADSIZE * (ht_num + 1)]) {
  char* res = NULL;
  void* start;
  bool tmp = false;
  if (cur > 0) {
    res =
        LinearProbeTuple(src, src_size, ht, ht_num, cur - 1, ret_size, output);
    if (res == NULL) {
      return NULL;
    }
  }
  uint32_t key = *(uint32_t*)(src + ht[cur]->probe_offset);
  uint32_t hash = (uint32_t)(key * table_factor) >> ht[cur]->shift;
  uint32_t ht_off = hash * ht[cur]->tuple_size;
#if PREFETCH
  _mm_prefetch((char*)(ht[cur]->addr + ht_off), _MM_HINT_T0);
#endif
  while (*(int*)(ht[cur]->addr + ht_off + ht[cur]->key_offset) != -1) {
    if (key == *(uint32_t*)(ht[cur]->addr + ht_off + ht[cur]->key_offset)) {
#if RESULTS
      if (cur == 0) {
        memcpy(output[cur], (char*)(src + WORDSIZE * ht_num), PAYLOADSIZE);
        ret_size = PAYLOADSIZE;
      } else {
        memcpy(output[cur], res, ret_size);
      }
      memcpy(output[cur] + ret_size, ht[cur]->addr + ht_off + WORDSIZE,
             PAYLOADSIZE);
      ret_size = ret_size + PAYLOADSIZE;
      tmp = true;
#if EARLYBREAK
      return output[cur];
#endif

#endif
    }
    ht_off += ht[cur]->tuple_size;
  }

#if !EARLYBREAK
  if (tmp) {
    return output[cur]
  }
#endif
  return NULL;
}

uint64_t TupleAtATimeProbe(Table* pb, HashTable** ht, int ht_num,
                           char* payloads) {
  int ret_size, len = 128;
  char* res;
  uint64_t num = 0;
  uint32_t cur_payloads = 0, result_size = PAYLOADSIZE * (ht_num + 1);

  char output[8][result_size];

#if OUTPUT
  FILE* fp;

  stringstream sstr;
  sstr << pthread_self();
  fp = fopen(string(("tuple.csv") + sstr.str()).c_str(), "wr");
  if (fp == 0) {
    assert(false && "can not open file!");
  }
#endif
  for (uint64_t i = 0, off = 0; i < pb->tuple_num; ++i, off += pb->tuple_size) {
    ret_size = 0;
    //    res = ProbeTuple(pb->start + off, pb->tuple_size, ht, ht_num, 0,
    //    ret_size,output);

    res = LinearProbeTuple(pb->start + off, pb->tuple_size - WORDSIZE, ht,
                           ht_num, ht_num - 1, ret_size, output);
    if (res) {
      ++num;
#if MEMOUTPUT
      memcpy(payloads + cur_payloads, res, result_size);
      cur_payloads += result_size;
#endif
#if OUTPUT
      // payloads: right,left0,left1...
      for (int i = 1; i <= ht_num; ++i) {
        fprintf(fp, "%d,", *((uint32_t*)(res + i * PAYLOADSIZE)));
      }
      fprintf(fp, "%d\n", *((uint32_t*)(res)));
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

#if 0
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
uint16_t zero_count(uint16_t a) {
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
          v_one = _mm512_set1_epi32(1);
  __attribute__((aligned(64)))
  uint32_t cur_offset = 0,
           cur_payloads = 0, global_addr_offset[16] = {0}, tmp_cell[16],
           *addr_offset, payloads_addr[16][6] = {0}, *ht_pos, *tuple_cell,
           *join_id, base_off[32], tuple_cell_offset[16] = {0},
           hor_probe_step = 16 * ht[0]->tuple_size, left_size[16] = {0},
           ht_cell_offset[16] = {0}, tuple_off[16] = {0},
           result_size = (ht_num + 1) * PAYLOADSIZE, buckets_minus_1[16] = {0},
           shift[16] = {0}, h_buf[16] = {0};
  __attribute__((aligned(64))) uint64_t htp[16] = {0}, bucket_upper[16] = {0};

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
    global_addr_offset[i] = ht[i]->global_addr_offset;
    tuple_off[i] = i * ht[0]->tuple_size;
  }
  ht_pos = (uint32_t*)&v_ht_pos;
  join_id = (uint32_t*)&v_join_id;
  addr_offset = (uint32_t*)&v_addr_offset;
  tuple_cell = (uint32_t*)&v_tuple_cell;

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
        __m512i v_payloads_off =
            _mm512_add_epi32(_mm512_set1_epi32(global_off), v_tuple_off);
        __m512i tab =
            _mm512_i32gather_epi32(v_payloads_off, ht[0]->global_addr, 1);

        __mmask16 out = _mm512_cmpeq_epi32_mask(tab, key_x16);
        if (out > 0) {
          int znum = zero_count(out);
          // avoid overflowing
          if (znum * left_size[join_id[j]] + global_off <=
              global_addr_offset[join_id[j]] + bucket_upper[join_id[j]]) {
            payloads_addr[j][join_id[j]] =
                (global_off + znum * left_size[join_id[j]] + WORDSIZE);
            flag = 1;
#if EARLYBREAK
            break;
#endif
          }
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

void* ThreadProbe(void* args) {
  ThreadArgs* arg = (ThreadArgs*)args;
  uint64_t upper = 0, lower = 0, matched = 0;
  Table* pb = new Table();
  pb->tuple_size = arg->tb->tuple_size;
  // stores the results after probes
  char* payloads =
      (char*)aligned_alloc(64, (probe_step + 32) * PAYLOADSIZE * (ht_num + 1));
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
    matched += arg->func(pb, ht, ht_num, payloads);
  }
  pthread_mutex_lock(&mutex);
  global_matched += matched;
  pthread_mutex_unlock(&mutex);
  free(payloads);
  return NULL;
}
void TestSet(Table* tb, string fun_name, int times, ProbeFunc func,
             int thread_num) {
  struct timeval t1, t2;
  pthread_t id[250];
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
