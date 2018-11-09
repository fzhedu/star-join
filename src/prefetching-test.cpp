#ifndef __PREFETCHINGTEST__
#define __PREFETCHINGTEST__
#include "star-simd.h"
#include "prefetching.cpp"
int PrefetchingTest(int argc, char** argv) {
  struct timeval t1, t2;
  int times = 3;
  if (argc > 1) {
    times = atoi(argv[1]);
  }
  int deltaT = 0;
  gettimeofday(&t1, NULL);
  // table_factor = (rand() << 1) | 1;
  // cout << "table_factor = " << table_factor << endl;
  //  table_factor = (rand() << 1) | 1;
  //  cout << "table_factor = " << table_factor << endl;
  //  table_factor = (rand() << 1) | 1;
  //  cout << "table_factor = " << table_factor << endl;
  /*
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
  */
  Table orders;
  orders.name = "orders";
  orders.tuple_num = 15000000;
  orders.path = "/home/claims/data/tpc-h/sf10/1partition/T4G0P0";
  orders.raw_tuple_size = 8;
  int array17[10] = {0, 4};
  SetVectorValue(array17, 2, orders.offset);
  int array18[10] = {4, 4};
  SetVectorValue(array18, 2, orders.size);
  orders.tuple_size = SumOfVector(orders.size);
  tb[0] = &orders;
  read_data_in_memory(&orders);
  //  test(&orders, orders.start, 0);
  /* raw data schema
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
  int array9[10] = {0, 4, 8, 0, 12};
  SetVectorValue(array9, 5, lineitem.offset);
  int array99[10] = {4, 4, 4, 4, 4};
  SetVectorValue(array99, 5, lineitem.size);
  lineitem.tuple_size = SumOfVector(lineitem.size);
  tb[3] = &lineitem;
  read_data_in_memory(&lineitem, 10);
  // test(&lineitem, lineitem.start, 0);
  gettimeofday(&t2, NULL);
  deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
  printf("--------load table is over costs time (ms) = %lf\n",
         deltaT * 1.0 / 1000);
  gettimeofday(&t1, NULL);

  // orders.tuple_num = 500000;
  // part.tuple_num = 100000;
  // supplier.tuple_num = 50000;
  // lineitem.tuple_num = 10000000;
  unsigned int global_size =
      upper_log2(orders.tuple_num) * orders.tuple_size + 64 * 8;
  void* global_addr = aligned_alloc(64, global_size);
  memset(global_addr, -1, global_size);

  cout << "global size = " << global_size << endl;

  HashTable ht_orders;
  ht_orders.global_addr = global_addr;
  ht_orders.global_addr_offset = 0;
  build_linear_ht(ht_orders, orders, 0, 0, selectity);
  travel_linear_ht(ht_orders);
  ht[0] = &ht_orders;
  /*
    HashTable ht_part;
    build_linear_ht(ht_part, part, 0, 4);
    travel_linear_ht(ht_part);
    ht[2] = &ht_part;

    HashTable ht_supplier;
    build_linear_ht(ht_supplier, supplier, 0, 8);
    travel_linear_ht(ht_supplier);
    ht[0] = &ht_supplier;

    HashTable ht_orders1;
    build_linear_ht(ht_orders1, orders, 0, 12);
    travel_linear_ht(ht_orders1);
    ht[3] = &ht_orders1;
  */
  gettimeofday(&t2, NULL);
  deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
  printf("++++ build hashtable costs time (ms) = %lf\n", deltaT * 1.0 / 1000);
  int id = 0;
  puts("input choice:");
  while (scanf("%d", &id) != EOF) {
    if (id == 0) {
      TestSet(&lineitem, "AMACProbe", times, AMACProbe, thread_num);
    } else if (id == 1) {
      TestSet(&lineitem, "GPProbe", times, GPProbe, thread_num);
    } else {
      TestSet(&lineitem, "SingleProbe", times, SingleProbe, thread_num);
    }
    puts("input choice again:");
  }
  // TestSet(&lineitem, "tupleAtTime", times, TupleAtATimeProbe, thread_num);

  free(orders.start);
  free(lineitem.start);
  free(global_addr);
  return 0;
}
/*
 * GPProbe (1.8s)< AMACProbe(2.1s) < SingleProbe(2.7s)
 * AMAC introduce more cache misses
 */

#endif
