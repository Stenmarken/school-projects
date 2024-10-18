#include <vector>
#include <thread>
#include <iostream>
#include <memory>
#include <unistd.h> 
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <atomic>

using namespace std::chrono_literals;
using namespace std;

class WaterManager
{
    public:
        bool should_trace;
        std::condition_variable cv;
        std::atomic<int> hyenasInside;
        std::atomic<int> gnusInside;
        std::mutex lock;
        
        WaterManager();
        WaterManager(bool trace_option);
        void hyenaEnters();
        void gnuEnters();
        void hyenaLeaves();
        void gnuLeaves();
};

void gnu(WaterManager &watermanager, int gnu_number);
void hyena(WaterManager &watermanager, int hyena_number);
