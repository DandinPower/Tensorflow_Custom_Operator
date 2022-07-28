#include <iostream>
#include <random>
#include <climits>
#include <string.h>
#include <string>
#include <time.h>
#include <cmath>
#include <vector>
#define FLOAT_BITS_LENGTH 32
using namespace std;

int history[32];
random_device rd;
default_random_engine eng(rd());
long zero = 0;
long mantissa = 8388607;
long full = 4294967295;

//根據要翻轉的bits數取得rate
long GetFlipRate(int num){
    long result = 0;
    for (int i=0; i<num; i++) result += pow(2, i);
    return result;
}

//根據給定的rate來翻轉bits
int FlipBits(float* x, long rate){
    int flip_nums = 0;
    int* address = (int *)x; 
    *address ^= rate;
    for (int i = 0; i < FLOAT_BITS_LENGTH; ++i) flip_nums += ((1 << i) & rate) != 0 ? 1 : 0;
    return flip_nums;
}

//根據範圍取得rate
long GetRateByRange(long start, long end){
    uniform_int_distribution<long> distr(start, end);
    return distr(eng);
}

int main(){
    float x = 1.2321;
    long start = GetFlipRate(25);
    long end = GetFlipRate(32);
    long rate = GetRateByRange(start,end);
    cout << rate << endl;
    cout << FlipBits(&x, rate) << endl;
    cout << x << endl;
    /*
    for (int i=0; i< 32; i++) history[i] = 0;
    for (int i=0; i< 10000000; i++){
        long rate = GetRateByRange(zero, mantissa);
        history[FlipBits(&x, rate)-1] += 1;
    }
    for (int i=0; i< 32; i++) cout << history[i] << endl;*/
}