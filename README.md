For installation of the libraries and benchmarks, please view the following:


## To make the binaries
```
./scripts/build.sh
```


## Running the first principles mpi code.
To run the first principles code

```
./scripts/sweep_mpi.sh
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
    Comment line 54 in code/deepspeed_matmul.py and uncomment line 53.

With oneCCL:
- We need to set the variables here to use oneCCL:
    ```
    source $HOME/bkm_method/oneCCL/build/_install/env/setvars.sh 
    export PT_VER_OVERRIDE=2.5.0          # matches your PyTorch tag
    pip install --force-reinstall \
    ~/bkm_method/torch-ccl/dist/oneccl_bind_pt-2.5.0+cpu-cp310-cp310-linux_x86_64.whl

    ```

    Comment line 53 in code/deepspeed_matmul.py and uncomment line 54.


## Running the oneCCL vanilla code:
We aim to run the oneCCL code for distributed matmul with multiple algorithms on two nodes with different configurations.

Feel free to tweak the naming or add more environment variables (e.g. CCL_RS_CHUNK_COUNT, CCL_RS_MIN_CHUNK_SIZE) in the same pattern.

#### Can we replace the oneCCL vanilla raw matmul with sgemm?
mpiicpc -std=c++17 matmul_ccl.cpp \
    -lccl -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -o matmul_ccl

Use the same scripts/sweep_ccl.sh to run the code. We are still facing some inefficiencies in matmul as we are using CBLAS sgemm and not MKL DNN Sgemm where .


## Some generic patterns:
- The net time taken to initialise deepspeed etc, takes much longer than that of oneCCL.(cold overhead is too high, even though CCL has KVS setup, etc.)
- however matmul used by deepspeed is very optimised, leading to much lesser time net for bigger size matrices. We can figure out how to solve this for ccl



### SETUP DOC:

https://docs.google.com/document/d/16OXTz6EIEryOPK5kUGN_6cTtXSb1maPrUjJ40wJqLZc/edit?tab=t.0

### Understanding the multinode setup:
https://docs.google.com/document/d/16CH82ZIJ0I9Zr04Ni-lgYmf-CRc480HrzT1vkU0_CGc/edit?pli=1&tab=t.0

### Peeling matmul:
https://docs.google.com/document/d/13z7U5v_Bdyv83qUcI0qZtU8b93zjR-6zx56LVaBXUR4/edit?tab=t.0#heading=h.jtjyoeraeqow

### Understanding KVS in OneCCL
https://docs.google.com/document/d/1yHz6oeacdBfFrscEi8JqRMLBecjILpkk4DlsOxmCFK8/edit?tab=t.0#heading=h.hs1pr9xgogf8