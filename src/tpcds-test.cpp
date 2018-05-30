#ifndef __TPCDSTEST__
#define __TPCDSTEST__
#include "star-simd.cpp"
int TpcdsTest(int argc, char** argv) {
  struct timeval t1, t2;
  int times = 3;
  if (argc > 1) {
    times = atoi(argv[1]);
  }
  int ht_num = 4;
  int deltaT = 0;
  gettimeofday(&t1, NULL);
  table_factor = (rand() << 1) | 1;
  cout << "table_factor = " << table_factor << endl;
  //  table_factor = (rand() << 1) | 1;
  //  cout << "table_factor = " << table_factor << endl;
  //  table_factor = (rand() << 1) | 1;
  //  cout << "table_factor = " << table_factor << endl;

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

  gettimeofday(&t2, NULL);
  deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
  printf("--------load table is over costs time (ms) = %lf\n",
         deltaT * 1.0 / 1000);
  gettimeofday(&t1, NULL);
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

  gettimeofday(&t2, NULL);
  deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
  printf("++++ build hashtable costs time (ms) = %lf\n", deltaT * 1.0 / 1000);
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
  return 0;
}

#endif
