#ifndef __TPCHTEST__
#define __TPCHTEST__
#include "star-simd.cpp"
int TpchTest(int argc, char** argv) {
  struct timeval t1, t2;
  int times = 3;
  if (argc > 1) {
    times = atoi(argv[1]);
  }
  int ht_num = 2;
  int deltaT = 0;
  gettimeofday(&t1, NULL);
  table_factor = (rand() << 1) | 1;
  cout << "table_factor = " << table_factor << endl;
  //  table_factor = (rand() << 1) | 1;
  //  cout << "table_factor = " << table_factor << endl;
  //  table_factor = (rand() << 1) | 1;
  //  cout << "table_factor = " << table_factor << endl;

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
  tb[2] = &orders;
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
  read_data_in_memory(&lineitem);
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
  HashTable ht_orders;
  build_linear_ht(ht_orders, orders, 0, 0);
  travel_linear_ht(ht_orders);
  ht[1] = &ht_orders;

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

  gettimeofday(&t2, NULL);
  deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
  printf("++++ build hashtable costs time (ms) = %lf\n", deltaT * 1.0 / 1000);

#if 1
  for (int t = 0; t < times; ++t) {
    gettimeofday(&t1, NULL);
    // LinearHandProbe(&lineitem, ht, 2);
    // TupleAtATimeProbe(&lineitem, ht, 2);
    Linear512Probe(&lineitem, ht, ht_num);
    gettimeofday(&t2, NULL);
    deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
    printf("****** probing costs time (ms) = %lf\n", deltaT * 1.0 / 1000);
  }
  for (int t = 0; t < times; ++t) {
    gettimeofday(&t1, NULL);
    // LinearHandProbe(&lineitem, ht, 2);
    // TupleAtATimeProbe(&lineitem, ht, 2);
    LinearSIMDProbe(&lineitem, ht, ht_num);
    gettimeofday(&t2, NULL);
    deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
    printf("****** probing costs time (ms) = %lf\n", deltaT * 1.0 / 1000);
  }
#endif
  for (int t = 0; t < times; ++t) {
    gettimeofday(&t1, NULL);
    // LinearHandProbe(&lineitem, ht, 2);
    TupleAtATimeProbe(&lineitem, ht, ht_num);
    // SIMDProbe(&store_sales, ht, 3);
    gettimeofday(&t2, NULL);
    deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
    printf("****** probing costs time (ms) = %lf\n", deltaT * 1.0 / 1000);
  }
#if 1
  for (int t = 0; t < times; ++t) {
    gettimeofday(&t1, NULL);
    LinearHandProbe(&lineitem, ht, ht_num);
    // TupleAtATimeProbe(&store_sales, ht, 3);
    // SIMDProbe(&store_sales, ht, 3);
    gettimeofday(&t2, NULL);
    deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
    printf("****** probing costs time (ms) = %lf\n", deltaT * 1.0 / 1000);
  }
#endif
  return 0;
}

#endif
