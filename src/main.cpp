// g++ -g main.cpp -o 2star.out -mavx512f -mavx512cd -mavx512bw -mavx512dq
// -mavx512vl -mavx2 -mbmi2 -w -fpermissive
#include "star-simd.h"
#include "ssb-test-sf1.cpp"
//#include "ssb-test-sf2.cpp"
#include "ssb-test-sf10.cpp"
//#include "tpcds-test.cpp"
//#include "tpch-test.cpp"
#include "ssb-test-sf100.cpp"
#include "ssb-test-sf50.cpp"

int main(int argc, char** argv) {
  if (argc > 1) {
    times = atoi(argv[1]);
  }
  if (argc > 2) {
    thread_num = atoi(argv[2]);
  }
  table_factor = (rand() << 1) | 1;
  // cout << "table_factor = " << table_factor << endl;
  table_factor = (rand() << 1) | 1;
#if DEBUGINFO
  cout << "table_factor = " << table_factor << endl;
#endif

  // TpcdsTest(argc, argv);
  // TpchTest(argc, argv);
  SsbTestSf100(argc, argv);
  SsbTestSf50(argc, argv);
  SsbTestSf10(argc, argv);
  SsbTestSf1(argc, argv);
  // SsbTestSf2(argc, argv);

  return 0;
}
