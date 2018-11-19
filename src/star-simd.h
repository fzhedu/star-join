
#ifndef SRC_STAR_SIMD_H_
#define SRC_STAR_SIMD_H_

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
#include <fstream>
#include <map>
using namespace std;
// CPU or PHI
#define FAVX512 1
#if !FAVX512
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
#endif
#define DEBUGINFO 1
#define NULL_INT 2147483647
#define BLOCK_SIZE 65536
#define RESULTS 1
#define OUTPUT 0
#define GATHERHT 1
// if adopt global address
#define PAYLOADSIZE 4
// for random access
#define PREFETCH 0
// for sequencal access
#define SEQPREFETCH 0
// +64
#define PDIS 128
#define EARLYBREAK 1
// filter out
float selectity = 0.5;
#define INVALID 2147483647
#define up64(a) (a - (a % 64) + 64)

#define StateSize 30
#define SIMDStateSize 4
#define Step 6
#define SIMDStep 4
#define MultiPrefetch 0
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
  uint64_t ht_size;
}* ht[10];
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
int thread_num = 1;
int ht_num = 4;
int times = 1;
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
  ht.ht_size = ht.slot_num * ht.tuple_size;
  ht.addr = (void*)(ht.global_addr + ht.global_addr_offset);
  ht.shift = 32 - log2(ht.slot_num);
  ht.max = 0;
  // memset(ht.addr, -1, ht_size);
  uint32_t key, hash = 0;
  uint64_t offset = 0;
  for (uint32_t i = 0; i < (ht.tuple_num) * table.tuple_size;
       i += table.tuple_size) {
    // enum the key in the table
    key = *(uint32_t*)(table.start + i + key_offset);
    ht.max = (key > ht.max ? key : ht.max);
    // set filter
    // if (key < bound) continue;
    // compute the hash value
    hash = ((uint32_t)(key * table_factor)) >> ht.shift;
    offset = hash * ht.tuple_size;
    // travel corresponding position in the hash table
    while (*(uint32_t*)(ht.addr + offset + key_offset) != -1) {
      offset += ht.tuple_size;
      offset = offset >= ht.ht_size ? offset - ht.ht_size : offset;
    }
    memcpy(ht.addr + offset, table.start + i, table.tuple_size);
  }
  return true;
}
bool travel_linear_ht(HashTable& ht) {
  map<int, int> count;
  int tmp = 0, id = 0;
  for (uint64_t offset = 0; offset < ht.ht_size; offset += ht.tuple_size) {
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
#if DEBUGINFO
  tmp = 0;
  cout << "-------------the cardinality of hash table---------- " << endl;
  for (map<int, int>::iterator it = count.begin(); it != count.end(); ++it) {
    cout << "val = " << it->first << " count = " << it->second << endl;
    tmp += it->first * it->second;  // what is tmp
  }
  cout << "max= " << ht.max << endl;
  cout << "tmp = " << tmp << " tuple_num = " << ht.tuple_num << endl;
  assert(tmp == ht.tuple_num);
#endif
  return true;
}
bool read_data_in_memory(Table* table, int repeats = 1) {
  fstream iofile(table->path.c_str(), ios::in | ios::out | ios::binary);
  if (!iofile) {
    cerr << "open error!" << endl;
    abort();
  }
  char start[BLOCK_SIZE];
  void* addr = NULL;
  uint64_t size = table->tuple_num * table->tuple_size, tuple_num = 0;
  addr = aligned_alloc(64, size * repeats);
  assert(addr != NULL && "limited memory");
  uint64_t dest_off = 0;
  iofile.seekg(0, ios::beg);
  while (iofile.read(start, BLOCK_SIZE)) {
    int num = *(int*)(start + BLOCK_SIZE - 4);
    tuple_num += num;
    // copy needed cells from each row
    for (int i = 0; i < num; ++i) {
      for (int m = 0; m < table->offset.size(); ++m) {
        memcpy(addr + dest_off,
               (start + i * table->raw_tuple_size + table->offset[m]),
               table->size[m]);
        dest_off += table->size[m];
      }
    }
  }
  assert(dest_off == size);
  for (int r = 1; r < repeats; ++r) {
    memcpy(addr + size * r, addr, size);
  }
  iofile.close();
  table->start = addr;
  table->tuple_num = table->tuple_num * repeats;
  table->upper = table->tuple_num;  // special
#if DEBUGINFO
  cout << table->name << " tuple num = " << table->tuple_num << " but read "
       << tuple_num << " repeats= " << repeats << endl;
#endif
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
#if SEQPREFETCH
    _mm_prefetch((char*)(probe_tuple_start + PDIS), _MM_HINT_T0);
#endif
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
        ht_off = ht_off >= ht[j]->ht_size ? ht_off - ht[j]->ht_size : ht_off;
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

char* LinearProbeTuple(void* src, int src_size, HashTable** ht, int ht_nu,
                       int cur, int& ret_size, char output[][PAYLOADSIZE * 5]) {
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
    ht_off = ht_off >= ht[cur]->ht_size ? ht_off - ht[cur]->ht_size : ht_off;
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

  char output[8][PAYLOADSIZE * 5];

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
#if SEQPREFETCH
    _mm_prefetch((char*)(pb->start + off + PDIS), _MM_HINT_T0);
#endif
    res = LinearProbeTuple(pb->start + off, pb->tuple_size - WORDSIZE, ht,
                           ht_num, ht_num - 1, ret_size, output);
    if (res) {
      ++num;
#if RESULTS
      memcpy(payloads + cur_payloads, res, result_size);
      cur_payloads += result_size;
#if OUTPUT
      // payloads: right,left0,left1...
      for (int i = 1; i <= ht_num; ++i) {
        fprintf(fp, "%d,", *((uint32_t*)(res + i * PAYLOADSIZE)));
      }
      fprintf(fp, "%d\n", *((uint32_t*)(res)));
#endif
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
  delete pb;
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

#endif  //  SRC_STAR_SIMD_H_
