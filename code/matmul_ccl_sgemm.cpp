// matmul_ccl.cpp — Distributed matmul benchmark using MPI + oneCCL
// Build: mpiicpc -std=c++17 matmul_ccl.cpp -lccl -o matmul_ccl

#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <cstring>
#include <mkl.h>    // at the top of your file


int main(int argc, char** argv) {
    // Default params
    int N = 1024, count = 10, warmup = 2;
    std::string outfile = "results.csv";

    // Simple flag parsing
    for(int i = 1; i < argc; ++i) {
        if(std::strcmp(argv[i], "--size") == 0 && i+1<argc)          N = std::atoi(argv[++i]);
        else if(std::strcmp(argv[i], "--count") == 0 && i+1<argc)    count = std::atoi(argv[++i]);
        else if(std::strcmp(argv[i], "--warmup") == 0 && i+1<argc)   warmup = std::atoi(argv[++i]);
        else if(std::strcmp(argv[i], "--outfile") == 0 && i+1<argc)  outfile = argv[++i];
    }

    // 1. MPI init
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // 2. oneCCL init + KVS handshake
    ccl::init();
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type addr{};
    if (world_rank == 0) {
        kvs  = ccl::create_main_kvs();
        addr = kvs->get_address();
    }
    MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (world_rank != 0) {
        kvs = ccl::create_kvs(addr);
    }
    auto comm = ccl::create_communicator(world_size, world_rank, kvs);

    // 3. Allocate & broadcast A, B
    std::vector<float> A(N*N), B(N*N), localC(N*N), finalC(N*N);
    if (world_rank == 0) {
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(0.f,1.f);
        for(int i=0;i<N*N;++i) {
            A[i]=dist(rng);
            B[i]=dist(rng);
        }
    }
    MPI_Bcast(A.data(), N*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), N*N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 4. Partition K-dimension
    int base = N / world_size, rem = N % world_size;
    int k_start = world_rank < rem
                ? world_rank * (base+1)
                : rem*(base+1) + (world_rank-rem)*base;
    int k_count = world_rank < rem ? base+1 : base;
    int k_end   = k_start + k_count;

    // 5. Warmup + timed iters
    std::vector<double> times_mm, times_ar;
    for(int iter=0; iter < count+warmup; ++iter) {
        // local matmul
        auto t0 = std::chrono::high_resolution_clock::now();
        std::fill(localC.begin(), localC.end(), 0.f);
        // for(int i=0;i<N;++i)
        //     for(int k=k_start;k<k_end;++k)
        //         for(int j=0;j<N;++j)
        //             localC[i*N+j] += A[i*N+k] * B[k*N+j];
        int K = k_end - k_start;   // number of columns in this block
        // A_block is &A[k_start] for each row, B_block is &B[k_start*N]
        cblas_sgemm(
            CblasRowMajor,         // row‐major ordering
            CblasNoTrans,          // A not transposed
            CblasNoTrans,          // B not transposed
            N,                     // M = rows of A_block and C
            N,                     // N = cols of B_block and C
            K,                     // K = inner dimension
            1.0f,                  // α
            A.data() + k_start,    // pointer to A[0][k_start]
            N,                     // leading dim of A (full row length)
            B.data() + k_start*N,  // pointer to B[k_start][0]
            N,                     // leading dim of B
            0.0f,                  // β
            localC.data(),         // output buffer
            N                      // leading dim of C
        );
        auto t1 = std::chrono::high_resolution_clock::now();

        // all-reduce sum into finalC
        auto req = ccl::allreduce(localC.data(), finalC.data(),
                                   N*N, ccl::reduction::sum, comm);
        req.wait();
        auto t2 = std::chrono::high_resolution_clock::now();

        if(iter >= warmup) {
            times_mm.push_back(std::chrono::duration<double>(t1-t0).count());
            times_ar.push_back(std::chrono::duration<double>(t2-t1).count());
        }
    }

    // 6. Stats lambda
    auto stats = [&](const std::vector<double>& v){
        double sum=0, mn=v[0], mx=v[0];
        for(auto x:v){ sum+=x; mn=std::min(mn,x); mx=std::max(mx,x); }
        double avg = sum/v.size();
        double sd=0; for(auto x:v) sd += (x-avg)*(x-avg);
        sd = std::sqrt(sd/v.size())/avg*100;
        return std::tuple<double,double,double,double>{mn,mx,avg,sd};
    };
    auto [mm_min,mm_max,mm_avg,mm_sd] = stats(times_mm);
    auto [ar_min,ar_max,ar_avg,ar_sd] = stats(times_ar);

    // 7. Validation & CSV output on rank 0
    if(world_rank==0) {
        // validate
        // regenerate reference C
        std::vector<float> Cref(N*N,0.f);
        for(int i=0;i<N;++i)
            for(int k=0;k<N;++k)
                for(int j=0;j<N;++j)
                    Cref[i*N+j] += A[i*N+k]*B[k*N+j];
        bool ok=true;
        for(int i=0;i<N*N;++i)
            if(std::abs(Cref[i]-finalC[i]) > 1e-3f) { ok=false; break; }
        if(!ok) {
            std::cerr<<"❌ result mismatch\n";
        } else {
            std::cout<<"✅ result matches reference\n";
        }

        // write CSV
        std::ofstream out(outfile);
        out << "Implementation,CPU_Count,Matrix_Size,Iterations,"
               "Matmul_t_min(s),Matmul_t_max(s),Matmul_t_avg(s),Matmul_stddev(%),"
               "Allreduce_t_min(s),Allreduce_t_max(s),Allreduce_t_avg(s),Allreduce_stddev(%),"
               "Total_t_min(s),Total_t_max(s),Total_t_avg(s),Total_stddev(%),"
               "GPU_Count,Accelerator,CCL,Launcher,Backend\n";

        // total = mm + ar
        double tt_min=mm_min+ar_min, tt_max=mm_max+ar_max, tt_avg=mm_avg+ar_avg, tt_sd=0;
        // rough estimate: sd_total ≈ sqrt(sd_mm^2 + sd_ar^2)
        tt_sd = std::sqrt(mm_sd*mm_sd + ar_sd*ar_sd);

        out << "oneCCL," << world_size << "," << N << "," << count << ","
            << mm_min<<","<<mm_max<<","<<mm_avg<<","<<mm_sd<<","
            << ar_min<<","<<ar_max<<","<<ar_avg<<","<<ar_sd<<","
            << tt_min<<","<<tt_max<<","<<tt_avg<<","<<tt_sd<<","
            << "0,CPU,true,mpirun,ccl\n";
        out.close();

        std::cout<<"Wrote "<<outfile<<"\n";
    }

    MPI_Finalize();
    return 0;
}
