// g++ -g main.cpp -o 2star.out -mavx512f -mavx512cd -mavx512bw -mavx512dq
// -mavx512vl -mavx2 -mbmi2 -w -fpermissive

#include "ssb-test.cpp"
#include "tpcds-test.cpp"
#include "tpch-test.cpp"
#include "ssb-test-sf100.cpp"
int main(int argc, char** argv) {
  // TpcdsTest(argc, argv);
  // TpchTest(argc, argv);
  SsbTestSf100(argc, argv);

  return 0;
}
