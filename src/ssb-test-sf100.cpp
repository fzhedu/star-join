#ifndef __SSBTESTSF100__
#define __SSBTESTSF100__
#include "star-simd.h"

#if FAVX512
#include "star-simd.cpp"
#else
#include "star-simd-phi.cpp"
#endif
int SsbTestSf100(int argc, char** argv) {
  cout << "----------------SSB=100-------------------" << endl;

  struct timeval t1, t2;
  if (argc > 1) {
    times = atoi(argv[1]);
  }
  if (argc > 2) {
    thread_num = atoi(argv[2]);
  }
  int deltaT = 0;
  gettimeofday(&t1, NULL);
  table_factor = (rand() << 1) | 1;
  // cout << "table_factor = " << table_factor << endl;

  table_factor = (rand() << 1) | 1;
#if DEBUGINFO
  cout << "table_factor = " << table_factor << endl;
#endif

  Table dim_date;
  dim_date.name = "dim_date";
  dim_date.tuple_num = 2556;
#if FAVX512
  dim_date.path = "/home/claims/data/ssb/sf100/T0G0P0";
#else
  dim_date.path = "/tmp/share_nfs/sf100/T0G0P0";
#endif
  dim_date.raw_tuple_size = 116;
  int array3[10] = {0, 4};
  SetVectorValue(array3, 2, dim_date.offset);  // dim_date.offset:{0,4}
  int array33[10] = {4, PAYLOADSIZE};
  SetVectorValue(array33, 2, dim_date.size);         // dim_date.size:{4,4}
  dim_date.tuple_size = SumOfVector(dim_date.size);  // tuple_size=8
  tb[3] = &dim_date;
  read_data_in_memory(&dim_date);
  //  test(&dim_date, dim_date.start, 0);

  Table customer;
  customer.name = "customer";
  customer.tuple_num = 3000000;
#if FAVX512
  customer.path = "/home/claims/data/ssb/sf100/T2G0P0";
#else
  customer.path = "/tmp/share_nfs/sf100/T2G0P0";
#endif
  customer.raw_tuple_size = 132;
  int array4[10] = {0, 4};
  SetVectorValue(array4, 2, customer.offset);
  int array44[10] = {4, PAYLOADSIZE};
  SetVectorValue(array44, 2, customer.size);
  customer.tuple_size = SumOfVector(customer.size);
  tb[0] = &customer;
  read_data_in_memory(&customer);
  // test(&customer, customer.start, 0);

  Table part;
  part.name = "part";
  part.tuple_num = 1400000;
#if FAVX512
  part.path = "/home/claims/data/ssb/sf100/T4G0P0";
#else
  part.path = "/tmp/share_nfs/sf100/T4G0P0";
#endif
  part.raw_tuple_size = 112;
  int array5[10] = {0, 96};
  SetVectorValue(array5, 2, part.offset);
  int array6[10] = {4, PAYLOADSIZE};
  SetVectorValue(array6, 2, part.size);
  part.tuple_size = SumOfVector(part.size);
  tb[2] = &part;
  read_data_in_memory(&part);
  //  test(&part, part.start, 0);

  Table supplier;
  supplier.name = "supplier";
  supplier.tuple_num = 200000;
#if FAVX512
  supplier.path = "/home/claims/data/ssb/sf100/T6G0P0";
#else
  supplier.path = "/tmp/share_nfs/sf100/T6G0P0";
#endif
  supplier.raw_tuple_size = 120;
  int array2[10] = {0, 4};
  SetVectorValue(array2, 2, supplier.offset);
  int array22[10] = {4, PAYLOADSIZE};
  SetVectorValue(array22, 2, supplier.size);
  supplier.tuple_size = SumOfVector(supplier.size);
  tb[1] = &supplier;
  read_data_in_memory(&supplier);
  // test(&supplier, supplier.start, 0);

  /* raw data schema
lo_orderkey,0
lo_linenumber,4
lo_custkey,8
lo_partkey,12
lo_suppkey,16
lo_orderdate
   */
  Table lineorder;
  lineorder.name = "lineorder";
  lineorder.tuple_num = 600038145;
#if FAVX512
  lineorder.path = "/home/claims/data/ssb/sf100/T8G0P0";
  lineorder.raw_tuple_size = 24;
  int array9[10] = {12, 16, 8, 20, 0};
#else
  lineorder.path = "/tmp/share_nfs/sf100/T8G0P0";
  lineorder.raw_tuple_size = 24;
  int array9[10] = {12, 16, 8, 20, 0};
#endif
  SetVectorValue(array9, 5, lineorder.offset);
  int array99[10] = {4, 4, 4, 4, 4};
  SetVectorValue(array99, 5, lineorder.size);
  lineorder.tuple_size = SumOfVector(lineorder.size);
  tb[9] = &lineorder;
  read_data_in_memory(&lineorder);
  //  test(&lineorder, lineorder.start, 0);
  gettimeofday(&t2, NULL);
  deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
  printf("--------load table is over costs time (ms) = %lf\n",
         deltaT * 1.0 / 1000);
  gettimeofday(&t1, NULL);
  // orders.tuple_num = 500000;
  // part.tuple_num = 100000;
  // supplier.tuple_num = 50000;
  // lineorder.tuple_num = 10000000;

  unsigned int global_size =
      upper_log2(dim_date.tuple_num) * dim_date.tuple_size +
      upper_log2(customer.tuple_num) * customer.tuple_size +
      upper_log2(part.tuple_num) * part.tuple_size +
      upper_log2(supplier.tuple_num) * supplier.tuple_size + 64 * 8;
  void* global_addr = aligned_alloc(64, global_size);
  memset(global_addr, -1, global_size);

  cout << "global size = " << global_size << endl;
  HashTable ht_dim_date;
  ht_dim_date.global_addr = global_addr;
  ht_dim_date.global_addr_offset = 0;
  build_linear_ht(ht_dim_date, dim_date, 0, 12, selectity);
  travel_linear_ht(ht_dim_date);
  ht[3] = &ht_dim_date;

  HashTable ht_customer;
  ht_customer.global_addr = global_addr;
  ht_customer.global_addr_offset =
      up64(ht_dim_date.slot_num * ht_dim_date.tuple_size);
  build_linear_ht(ht_customer, customer, 0, 8, selectity);
  travel_linear_ht(ht_customer);
  ht[2] = &ht_customer;

  HashTable ht_part;
  ht_part.global_addr = global_addr;
  ht_part.global_addr_offset =
      ht_customer.global_addr_offset +
      up64(ht_customer.slot_num * ht_customer.tuple_size);
  build_linear_ht(ht_part, part, 0, 0, selectity);
  travel_linear_ht(ht_part);
  ht[0] = &ht_part;

  HashTable ht_supplier;
  ht_supplier.global_addr = global_addr;
  ht_supplier.global_addr_offset =
      ht_part.global_addr_offset + up64(ht_part.slot_num * ht_part.tuple_size);
  build_linear_ht(ht_supplier, supplier, 0, 4, selectity);
  travel_linear_ht(ht_supplier);
  ht[1] = &ht_supplier;

  gettimeofday(&t2, NULL);
  deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
  printf("++++ build hashtable costs time (ms) = %lf\n", deltaT * 1.0 / 1000);

  TestSet(&lineorder, "handprobe", times, LinearHandProbe, thread_num);
  TestSet(&lineorder, "tupleAtTime", times, TupleAtATimeProbe, thread_num);
  TestSet(&lineorder, "SIMD512Hor", times, Linear512ProbeHor, thread_num);
  TestSet(&lineorder, "SIMD512", times, Linear512Probe, thread_num);
  // TestSet(&lineorder, "SIMD256Hor", times, LinearSIMDProbeHor, thread_num);
  // TestSet(&lineorder, "SIMD256", times, LinearSIMDProbe, thread_num);

  free(global_addr);
  free(dim_date.start);
  free(customer.start);
  free(supplier.start);
  free(lineorder.start);
  free(part.start);
  return 0;
}
#endif
