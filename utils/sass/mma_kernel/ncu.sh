#!/bin/bash

ncu --kernel-name-base "function" --kernel-name "flash_attention_forward_kernel" \
--metrics "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__throughput.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.per_cycle_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread,\
launch__shared_mem_per_block,\
launch__occupancy_limit_warps,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem" \
./forward_kernel