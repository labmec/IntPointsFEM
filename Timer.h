//
// Created by natalia on 10/05/19.
//

#ifndef INTPOINTSFEM_TIMER_H
#define INTPOINTSFEM_TIMER_H

#include <chrono>
using namespace std::chrono;

class Timer {
protected:
    high_resolution_clock::time_point t1;
    high_resolution_clock::time_point t2;
    duration<double> time_span;

public:
    void Start() {
        t1 = high_resolution_clock::now();
    }

    void End() {
        t2 = high_resolution_clock::now();
    }

    REAL ElapsedTime() {
        time_span = duration_cast<duration<double>>(t2 - t1);
        return time_span.count()*1000;
    }
};


#endif //INTPOINTSFEM_TIMER_H
