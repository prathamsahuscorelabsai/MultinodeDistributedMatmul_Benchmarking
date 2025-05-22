For installation of the libraries and benchmarks, please view the following:

## To make the binaries
```
./scripts/build.sh
```


## Running the deepspeed code

With mpi and gloo:
- First we switch of oneCCL, that is we uninstall it from the system. This is because the deepspeed code is such that it will default always to oneCCL implementation if it's environment variables exist.
    ```
    pip uninstall -y oneccl-bind-pt oneccl_bind_pt oneccl_bindings_for_pytorch
    unset CCL_ATL_TRANSPORT
    unset CCL_WORKER_COUNT
    unset OMP_NUM_THREADS   
    unset CCL_ATL_TRANSPORT CCL_WORKER_COUNT CCL_LOG_LEVEL CCL_ROOT
    unset CCL_ROOT

    pip uninstall -y oneccl-bind-pt oneccl_bind_pt oneccl_bindings_for_pytorch
    pip uninstall ~/bkm_method/torch-ccl/dist/oneccl_bind_pt-2.5.0+cpu-cp310-cp310-linux_x86_64.whl

    ```

With oneCCL:
- We need to set the variables here to use oneCCL:
    ```
    source $HOME/bkm_method/oneCCL/build/_install/env/setvars.sh 
    export PT_VER_OVERRIDE=2.5.0          # matches your PyTorch tag
    pip install --force-reinstall \
    ~/bkm_method/torch-ccl/dist/oneccl_bind_pt-2.5.0+cpu-cp310-cp310-linux_x86_64.whl

    ```


## Running the oneCCL vanilla code:
We aim to run the oneCCL code for distributed matmul with multiple algorithms on two nodes with different configurations.

How it works

Environment:

Exports CCL_KVS_IFACE=ens1f0np0 as requested.

Iterates CCL_ATL_TRANSPORT over mpi and ofi.

Iterates CCL_ALLREDUCE over each algorithm (applies to entire size range via :0-max).

Inner loops:

For each slot‐count (1–32) and each matrix size, runs your matmul_ccl binary under mpirun.

Captures per‐run CSV in results/tmp_…\.csv and the stdout/stderr in logs/…\.log.

Merges all the per‐size CSVs into a single master CSV per transport+algorithm, preserving the header exactly once.

Output:

results/sweep_<transport>_<alg>.csv for each combination.

Logs under logs/<transport>_<alg>_N<SIZE>_S<SLOTS>.log.

Feel free to tweak the naming or add more environment variables (e.g. CCL_RS_CHUNK_COUNT, CCL_RS_MIN_CHUNK_SIZE) in the same pattern.