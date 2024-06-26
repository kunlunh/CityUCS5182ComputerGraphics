gpu=0
model=punet
extra_tag=punet_with_uni_pugan_dataset
epoch=99

mkdir outputs/${extra_tag}

python -u test.py \
    --model ${model} \
    --save_dir outputs/${extra_tag} \
    --gpu ${gpu} \
    --resume logs/${extra_tag}/punet_epoch_${epoch}.pth
