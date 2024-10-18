#include <vector>
#include <thread>
#include <iostream>
#include <memory>
#include <unistd.h> 
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include "main.h"


using namespace std::chrono_literals;
using namespace std;


std::unordered_map<std::thread::id, std::string> gnu_map;
std::unordered_map<std::thread::id, std::string> hyena_map;



WaterManager::WaterManager(bool trace_option)
{
    should_trace = trace_option;
    gnusInside = 0;
    hyenasInside = 0;
}

WaterManager::WaterManager()
{
    should_trace = true;
    gnusInside = 0;
    hyenasInside = 0;
}

void WaterManager::hyenaEnters()
{     
    cout << "Hyena " << gnusInside + 1 << " is thirsty" << endl;
    
    //const std::lock_guard<std::mutex> lock_guard(lock);
    std::unique_lock<std::mutex> lock_guard(lock);
    cv.wait(lock_guard, [this] {return gnusInside == 0;});
	hyenasInside += 1;
        
    if (should_trace)
        cout << "A gnu enters the water cave    hyenas = " << hyenasInside << "     gnus = " << gnusInside << endl;
}

void WaterManager::gnuEnters()
{
    cout << "Gnu " << gnusInside + 1 << " is thirsty" << endl;
    
    std::unique_lock<std::mutex> lock_guard(lock);
    cv.wait(lock_guard, [this] {return hyenasInside == 0;});
    gnusInside += 1;

    if (should_trace)
        cout << "A hyena enters the water cave    hyenas = " << hyenasInside << "     gnus = " << gnusInside << endl;
}

void WaterManager::hyenaLeaves()
{       
    lock.lock();
    if (should_trace)
        cout << "Hyena " << hyenasInside << " finished drinking and exits the water cave" << endl;
    hyenasInside -= 1;
    cv.notify_all();
}

void WaterManager::gnuLeaves()
{
    lock.lock();    
    if (should_trace) 
        cout << "Gnu " << hyenasInside << " finished drinking and exits the water cave" << endl;

    gnusInside -= 1;
    cv.notify_all();
}



void gnu(WaterManager &watermanager, int gnu_number)
{
    std::string name = "Gnu " + gnu_number;
    gnu_map.insert(std::make_pair(std::this_thread::get_id(), name));  

    int i = 0;
    while (i < 10)
    {
        std::this_thread::sleep_for(300ms);
	    watermanager.gnuEnters();            // see monitoring class below
        std::this_thread::sleep_for(100ms);
	    watermanager.gnuLeaves();
        i++;
    }
}

void hyena(WaterManager &watermanager, int hyena_number)
{
    std::string name = "Hyena " + hyena_number;
    gnu_map.insert(std::make_pair(std::this_thread::get_id(), name));

    int i = 0;
    while (i < 10)
    {
        std::this_thread::sleep_for(300ms);
        watermanager.hyenaEnters();
        std::this_thread::sleep_for(100ms);
        watermanager.hyenaLeaves();
        i++;
    }
}



int main()
{
    WaterManager watermanager = new WaterManager(true);

    int nrGnus = 100;
    int nrHyenas = 100;

    vector<thread> threadvec;

    for (int i = 0; i < nrGnus; i++)
        threadvec.push_back(thread(gnu, std::ref(watermanager), i+1));
    for (int i = 0; i < nrHyenas; i++)
    threadvec.push_back(thread(hyena, std::ref(watermanager), i+1));

    for (auto& thread : threadvec)
	    thread.join();
}
