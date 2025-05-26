// matmul_mpi.cpp — Distributed matmul benchmark using MPI + MKL
// Build: mpicxx -std=c++17 code/matmul_mpi.cpp -lmkl_rt -o bin/matmul_mpi
#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <cstring>
#include <cmath>       // for std::sqrt
#include <algorithm>   // for std::min, std::max
#include <tuple>       // for std::make_tuple
#include <mkl.h>


int main(int argc, char** argv) {
    // default params
    int N = 1024, count = 10, warmup = 2;
    std::string outfile = "results.csv";

    // parse flags
    for(int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--size") == 0 && i+1<argc)        N      = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--count") == 0 && i+1<argc)  count  = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i+1<argc) warmup = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--outfile") == 0 && i+1<argc) outfile = argv[++i];
    }

    // 1. MPI init
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // 2. allocate buffers
    std::vector<float> A(N*N), B(N*N), localC(N*N), finalC(N*N);
    if (world_rank == 0) {
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(0.f,1.f);
        for (int i = 0; i < N*N; ++i) {
            A[i] = dist(rng);
            B[i] = dist(rng);
        }
    }
    MPI_Bcast(A.data(), N*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), N*N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 3. partition K-dimension
    int base = N / world_size;
    int rem  = N % world_size;
    int k_start = world_rank < rem
                ? world_rank * (base + 1)
                : rem * (base + 1) + (world_rank - rem) * base;
    int k_count = (world_rank < rem ? base + 1 : base);
    int K       = k_count;

    // 4. warmup + timed iterations
    std::vector<double> times_mm, times_ar;
    for (int iter = 0; iter < warmup + count; ++iter) {
        std::fill(localC.begin(), localC.end(), 0.f);

        auto t0 = std::chrono::high_resolution_clock::now();
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            N, N, K,
            1.0f,
            A.data() + k_start, N,
            B.data() + k_start * N, N,
            0.0f,
            localC.data(), N
        );
        auto t1 = std::chrono::high_resolution_clock::now();

        MPI_Allreduce(
            localC.data(), finalC.data(),
            N*N, MPI_FLOAT, MPI_SUM,
            MPI_COMM_WORLD
        );
        auto t2 = std::chrono::high_resolution_clock::now();

        if (iter >= warmup) {
            times_mm.push_back(std::chrono::duration<double>(t1 - t0).count());
            times_ar.push_back(std::chrono::duration<double>(t2 - t1).count());
        }
    }

    // 5. compute stats
    auto stats = [&](const std::vector<double>& v) {
        double sum = 0, mn = v[0], mx = v[0];
        for (double x : v) { sum += x; mn = std::min(mn, x); mx = std::max(mx, x); }
        double avg = sum / v.size();
        double sd = 0;
        for (double x : v) sd += (x - avg) * (x - avg);
        sd = std::sqrt(sd / v.size()) / avg * 100.0;
        return std::make_tuple(mn, mx, avg, sd);
    };
    auto [mm_min, mm_max, mm_avg, mm_sd] = stats(times_mm);
    auto [ar_min, ar_max, ar_avg, ar_sd] = stats(times_ar);

    // 6. validate & write CSV on rank 0
    if (world_rank == 0) {
        // reference C
        std::vector<float> Cref(N*N, 0.f);
        for (int i = 0; i < N; ++i)
            for (int k = 0; k < N; ++k)
                for (int j = 0; j < N; ++j)
                    Cref[i*N+j] += A[i*N+k] * B[k*N+j];

        bool ok = true;
        for (int i = 0; i < N*N; ++i)
            if (std::abs(Cref[i] - finalC[i]) > 1e-3f) { ok = false; break; }

        std::cout << (ok ? "✅ result matches reference\n" : "❌ result mismatch\n");

        std::ofstream out(outfile);
        out << "Implementation,CPU_Count,Matrix_Size,Iterations,"
               "Matmul_t_min(s),Matmul_t_max(s),Matmul_t_avg(s),Matmul_stddev(%),"
               "Allreduce_t_min(s),Allreduce_t_max(s),Allreduce_t_avg(s),Allreduce_stddev(%),"
               "Total_t_min(s),Total_t_max(s),Total_t_avg(s),Total_stddev(%),"
               "GPU_Count,Accelerator,CCL,Launcher,Backend\n";

        double tt_min = mm_min + ar_min;
        double tt_max = mm_max + ar_max;
        double tt_avg = mm_avg + ar_avg;
        double tt_sd  = std::sqrt(mm_sd*mm_sd + ar_sd*ar_sd);

        out << "MPI," << world_size << "," << N << "," << count << ","
            << mm_min  << "," << mm_max  << "," << mm_avg  << "," << mm_sd  << ","
            << ar_min  << "," << ar_max  << "," << ar_avg  << "," << ar_sd  << ","
            << tt_min  << "," << tt_max  << "," << tt_avg  << "," << tt_sd  << ","
            << "0,CPU,false,mpirun,mpi\n";
        out.close();

        std::cout << "Wrote " << outfile << "\n";
    }

    MPI_Finalize();
    return 0;
}
