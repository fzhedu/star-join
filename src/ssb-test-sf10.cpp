#ifndef __SSBTESTSF10__
#define __SSBTESTSF10__
#include <string>

#include "star-simd.cpp"

int SsbTestSf10(int argc, char** argv) {
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
  cout << "table_factor = " << table_factor << endl;
  table_factor = (rand() << 1) | 1;
  cout << "table_factor = " << table_factor << endl;

  Table dim_date;
  dim_date.name = "dim_date";
  dim_date.tuple_num = 2556;
  dim_date.path = "/home/claims/data/ssb/sf10/T0G0P0";
  dim_date.raw_tuple_size = 116;
  int array3[10] = {0, 4};
  SetVectorValue(array3, 2, dim_date.offset);
  int array33[10] = {4, 4};
  SetVectorValue(array33, 2, dim_date.size);
  dim_date.tuple_size = SumOfVector(dim_date.size);
  tb[0] = &dim_date;
  read_data_in_memory(&dim_date);
  //  test(&dim_date, dim_date.start, 0);

  Table customer;
  customer.name = "customer";
  customer.tuple_num = 300000;
  customer.path = "/home/claims/data/ssb/sf10/T2G0P0";
  customer.raw_tuple_size = 132;
  int array4[10] = {0, 4};
  SetVectorValue(array4, 2, customer.offset);
  int array44[10] = {4, 4};
  SetVectorValue(array44, 2, customer.size);
  customer.tuple_size = SumOfVector(customer.size);
  tb[1] = &customer;
  read_data_in_memory(&customer);
  // test(&customer, customer.start, 0);

  Table part;
  part.name = "part";
  part.tuple_num = 800000;
  part.path = "/home/claims/data/ssb/sf10/T4G0P0";
  part.raw_tuple_size = 112;
  int array5[10] = {0, 96};
  SetVectorValue(array5, 2, part.offset);
  int array6[10] = {4, 4};
  SetVectorValue(array6, 2, part.size);
  part.tuple_size = SumOfVector(part.size);
  tb[2] = &part;
  read_data_in_memory(&part);
  //  test(&part, part.start, 0);

  Table supplier;
  supplier.name = "supplier";
  supplier.tuple_num = 20000;
  supplier.path = "/home/claims/data/ssb/sf10/T6G0P0";
  supplier.raw_tuple_size = 120;
  int array2[10] = {0, 4};
  SetVectorValue(array2, 2, supplier.offset);
  int array22[10] = {4, 4};
  SetVectorValue(array22, 2, supplier.size);
  supplier.tuple_size = SumOfVector(supplier.size);
  tb[3] = &supplier;
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
  lineorder.tuple_num = 59986214;
  lineorder.path = "/home/claims/data/ssb/sf10/T12G0P0";
  lineorder.raw_tuple_size = 24;
  int array9[10] = {12, 16, 8, 20, 0};
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

  HashTable ht_dim_date;
  build_linear_ht(ht_dim_date, dim_date, 0, 12, selectity);
  travel_linear_ht(ht_dim_date);
  ht[3] = &ht_dim_date;

  HashTable ht_customer;
  build_linear_ht(ht_customer, customer, 0, 8, selectity);
  travel_linear_ht(ht_customer);
  ht[0] = &ht_customer;

  HashTable ht_part;
  build_linear_ht(ht_part, part, 0, 0, selectity);
  travel_linear_ht(ht_part);
  ht[2] = &ht_part;

  HashTable ht_supplier;
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
  TestSet(&lineorder, "SIMD256Hor", times, LinearSIMDProbeHor, thread_num);
  TestSet(&lineorder, "SIMD256", times, LinearSIMDProbe, thread_num);
  return 0;
}
#endif
