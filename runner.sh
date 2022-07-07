#!/usr/bin/bash

data_seeds=(
  77295
  40865
  82635
  86576
  84055
  15487
  60076
  39714
  60063
  46530
)

# python main.py "normal" "normal" 0.2 3
# python main.py "bernoulli" "normal" 0.2 3
# python main.py "normal" "bernoulli" 0.2 3
# python main.py "bernoulli" "bernoulli" 0.2 3

echo "Starting run..."

distribution="normal"
model_dist="normal"
proxy_noise=0.2
#latent_dims=(1 2 3 5 10 25 50 75 100 150 200)
sample_sizes=(1000 2000 4000 8000 16000)

data_latent_dims=3
results_batch="lat_dim_3}9INCSAMPLE-1LD-complex"
description="3 latent to 9 proxy"
model_latent_dim=1

for i in {1..2}
do
  echo "Running loop $i"
  for sample in "${sample_sizes[@]}"
  do
    echo "Using seed: ${data_seeds[$i - 1]}"
    python main.py "$distribution" "$model_dist" "$proxy_noise" "$model_latent_dim" "$data_latent_dims" "$results_batch" "$description" "${data_seeds[$i - 1]}" "$sample" 0 0
  done
done
echo "Finished run"

data_latent_dims=3
results_batch="lat_dim_3}9INCSAMPLE-200LD-complex"
description="3 latent to 9 proxy"
model_latent_dim=200

for i in {1..2}
do
  echo "Running loop $i"
  for sample in "${sample_sizes[@]}"
  do
    echo "Using seed: ${data_seeds[$i - 1]}"
    python main.py "$distribution" "$model_dist" "$proxy_noise" "$model_latent_dim" "$data_latent_dims" "$results_batch" "$description" "${data_seeds[$i - 1]}" "$sample" 0 0
  done
done
echo "Finished run"

#data_latent_dims=1
#results_batch="lat_dim_1}9-MOREMOREcomplex-larger-sample"
#description="1 latent to 9 proxy"
#
#for i in {1..2}
#do
#  echo "Running loop $i"
#  for dim in "${latent_dims[@]}"
#  do
#    echo "Using seed: ${data_seeds[$i - 1]}"
#    python main.py "$distribution" "$model_dist" "$proxy_noise" "$dim" "$data_latent_dims" "$results_batch" "$description" "${data_seeds[$i - 1]}"
#  done
#done
#echo "Finished run"

#data_latent_dims=0
#results_batch="lat_dim_0}9-simple-larger-sample"
#description="0 latent to 9 proxy"
#
#for i in {1..2}
#do
#  echo "Running loop $i"
#  for dim in "${latent_dims[@]}"
#  do
#    echo "Using seed: ${data_seeds[$i - 1]}"
#    python main.py "$distribution" "$model_dist" "$proxy_noise" "$dim" "$data_latent_dims" "$results_batch" "$description" "${data_seeds[$i - 1]}"
#  done
#done
#echo "Finished run"
