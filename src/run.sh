#!/bin/bash

sleep 1800
nohup ./query_top_k_self_random.sh > test_object_llama3.1-instruct_ft_1024.log 2>&1 &