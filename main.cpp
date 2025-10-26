// variant 12: inclusive_scan

#include <iostream>
#include <vector>
#include <numeric>
#include <execution>
#include <random>
#include <chrono>
#include <thread>
#include <iomanip>
#include <algorithm>

using namespace std;
using namespace std::chrono;

vector<int> generateRandomData(size_t size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(1, 100);

    vector<int> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

template<typename Func>
double measureTime(Func&& func) {
    auto start = high_resolution_clock::now();
    func();
    auto end = high_resolution_clock::now();
    return duration<double, milli>(end - start).count();
}

template<typename T>
vector<T> parallelInclusiveScan(const vector<T>& input, int numThreads) {
    size_t n = input.size();
    vector<T> result(n);

    if (numThreads <= 1 || n < 1000) {
        inclusive_scan(input.begin(), input.end(), result.begin());
        return result;
    }

    size_t chunkSize = (n + numThreads - 1) / numThreads;
    vector<thread> threads;
    vector<T> lastValues(numThreads);

    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            size_t start = t * chunkSize;
            size_t end = min(start + chunkSize, n);

            if (start >= n) return;

            inclusive_scan(input.begin() + start, input.begin() + end,
                result.begin() + start);

            lastValues[t] = result[end - 1];
            });
    }

    for (auto& th : threads) {
        th.join();
    }
    threads.clear();

    vector<T> prefixSums(numThreads);
    inclusive_scan(lastValues.begin(), lastValues.begin() + numThreads,
        prefixSums.begin());

    for (int t = 1; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            size_t start = t * chunkSize;
            size_t end = min(start + chunkSize, n);

            if (start >= n) return;

            T offset = prefixSums[t - 1];
            for (size_t i = start; i < end; ++i) {
                result[i] += offset;
            }
            });
    }

    for (auto& th : threads) {
        th.join();
    }

    return result;
}

void experiment1(const vector<int>& data) {
    cout << "\n=== Experiment 1: Sequential Algorithm ===" << endl;
    cout << "Data size: " << data.size() << endl;

    vector<int> result(data.size());
    double time = measureTime([&]() {
        inclusive_scan(data.begin(), data.end(), result.begin());
        });

    cout << "Execution time: " << fixed << setprecision(3) << time << " ms" << endl;
}

void experiment2(const vector<int>& data) {
    cout << "\n=== Experiment 2: Algorithms with different policies ===" << endl;
    cout << "ata size: " << data.size() << endl;

    vector<int> result(data.size());

    double time_seq = measureTime([&]() {
        inclusive_scan(execution::seq, data.begin(), data.end(), result.begin());
        });
    cout << "execution::seq: " << fixed << setprecision(3) << time_seq << " ms" << endl;

    double time_par = measureTime([&]() {
        inclusive_scan(execution::par, data.begin(), data.end(), result.begin());
        });
    cout << "execution::par: " << fixed << setprecision(3) << time_par << " ms" << endl;

    double time_par_unseq = measureTime([&]() {
        inclusive_scan(execution::par_unseq, data.begin(), data.end(), result.begin());
        });
    cout << "execution::par_unseq: " << fixed << setprecision(3) << time_par_unseq << " ms" << endl;

    cout << "\nAcceleration (par): " << fixed << setprecision(2)
        << time_seq / time_par << "x" << endl;
    cout << "Acceleration (par_unseq): " << fixed << setprecision(2)
        << time_seq / time_par_unseq << "x" << endl;
}

void experiment3(const vector<int>& data) {
    cout << "\n=== Experiment 3: Own parallel algorithm ===" << endl;
    cout << "Data size: " << data.size() << endl;

    unsigned int hwThreads = thread::hardware_concurrency();
    cout << "Number of hardware streams: " << hwThreads << endl;

    int maxK = min(32, (int)hwThreads * 4);

    cout << "\n" << setw(5) << "K" << setw(15) << "Time (ms)"
        << setw(20) << "Acceleration" << endl;
    cout << string(40, '-') << endl;

    vector<int> result_base(data.size());
    double baseTime = measureTime([&]() {
        inclusive_scan(data.begin(), data.end(), result_base.begin());
        });

    double bestTime = numeric_limits<double>::max();
    int bestK = 1;

    vector<pair<int, double>> results;

    for (int k = 1; k <= maxK; ++k) {
        vector<int> result;
        double time = measureTime([&]() {
            result = parallelInclusiveScan(data, k);
            });

        results.push_back({ k, time });

        double speedup = baseTime / time;
        cout << setw(5) << k
            << setw(15) << fixed << setprecision(3) << time
            << setw(20) << fixed << setprecision(2) << speedup << "x" << endl;

        if (time < bestTime) {
            bestTime = time;
            bestK = k;
        }
    }

    cout << "\n=== Results ===" << endl;
    cout << "Best K: " << bestK << endl;
    cout << "Time at best K: " << fixed << setprecision(3) << bestTime << " ms" << endl;
    cout << "Maximum acceleration: " << fixed << setprecision(2)
        << baseTime / bestTime << "x" << endl;
    cout << "Ratio of K to number of streams: "
        << fixed << setprecision(2) << (double)bestK / hwThreads << endl;

    cout << "\n=== Time growth analysis ===" << endl;
    if (bestK < maxK / 2) {
        cout << "After reaching the optimal K=" << bestK
            << ", time begins to increase due to overhead" << endl;
        cout << "creating and synchronizing streams." << endl;
    }
}

void optimizationLevelInfo() {
    cout << "\n=== Compilation information ===" << endl;
    cout << "To study optimization levels:" << endl;
    cout << "1. Without optimization: g++ -std=c++20 -O0 -pthread main.cpp -o lab2_O0" << endl;
    cout << "2. With optimization: g++ -std=c++20 -O3 -pthread main.cpp -o lab2_O3" << endl;
    cout << "Compare the execution time for both options." << endl;
}

int main() {
    cout << "lab 2: inclusive_scan" << endl;
    cout << "variant 12" << endl;
    cout << "=================================================" << endl;

    vector<size_t> sizes = { 100'000, 1'000'000, 10'000'000 };

    for (size_t size : sizes) {
        cout << "\n\n";
        cout << "### Testing on size: " << size << " elements ###" << endl;

        auto data = generateRandomData(size);

        experiment1(data);
        experiment2(data);
        experiment3(data);
    }

    optimizationLevelInfo();

    return 0;
}