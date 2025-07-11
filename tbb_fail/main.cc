#include <tbb/tbb.h>

#include <iostream>

int main()
{
  std::cout << tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism)
            << " threads are max used for TBB execution" << std::endl;
  std::cout << tbb::this_task_arena::max_concurrency() << " threads are TBB concurrency execution" << std::endl;

  tbb::task_arena ar(32);
  ar.initialize();

  ar.execute([&]() {
    tbb::parallel_for(
        0, 100,
        [](int i) {
          while (true) {
            i += 1;
          }
        },
        tbb::auto_partitioner());
  });
  return 0;
}
