gpu=0
model=punet
extra_tag=punet_with_uni_pugan_dataset

mkdir logs/${extra_tag}

nohup python -u train_with_uni_pugan_dataset.py \
    --model ${model} \
    --log_dir logs/${extra_tag} \
    --gpu ${gpu} \
    >> logs/${extra_tag}/nohup.log 2>&1 &
