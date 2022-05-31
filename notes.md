## training
1. prepare data: `python ./model/stylegan/prepare_data.py --out ./data/cartoon/lmdb/ --n_worker 4 --size 1024 ./data/cartoon/images/`
2. Fine-tune StyleGAN: `python finetune_stylegan.py --iter 600 --batch 4 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --style cartoon --augment ./data/cartoon/lmdb/ --wandb`
3. Destylize artistic portraits: `python destylize.py --model_name fintune-000600.pt --batch 1 --iter 300 cartoon`

1. Pretrain DualStyleGAN on FFHQ: `python pretrain_dualstylegan.py --iter 3000 --batch 4 ./data/ffhq/lmdb/`
2. Fine-Tune DualStyleGAN on Target Domain: `python finetune_dualstylegan.py --iter 1500 --batch 4 --ckpt ./checkpoint/generator-pretrain.pt --style_loss 0.25 --CX_loss 0.25 --perc_loss 1 --id_loss 1 --L2_reg_loss 0.015 --augment cartoon`

1. Refine extrinsic style code: `python refine_exstyle.py --lr_color 0.1 --lr_structure 0.005 --ckpt ./checkpoint/cartoon/generator-001400.pt cartoon`
2. Training sampling network: `python train_sampler.py cartoon`

## prepare
### prepare data
`python ./model/stylegan/prepare_data.py --out LMDB_PATH --n_worker N_WORKER --size SIZE1,SIZE2,SIZE3,... DATASET_PATH`  
将DATASET_PATH中的图片存到LMDB数据库中。

### Fine-tune StyleGAN
`python finetune_stylegan.py --batch BATCH_SIZE --ckpt FFHQ_MODEL_PATH --iter ITERATIONS --style DATASET_NAME --augment LMDB_PATH`
微调后的模型保存在`./checkpoint/cartoon/fintune-000600.pt`.中间结果保存在 `./log/cartoon/`.

### Destylize artistic portraits
`python destylize.py --model_name FINETUNED_MODEL_NAME --batch BATCH_SIZE --iter ITERATIONS DATASET_NAME`  
内部和外部风格编码保存在 `./checkpoint/cartoon/instyle_code.npy` 和 `./checkpoint/cartoon/exstyle_code.npy`。要加速处理过程，将batch_size设为16。如果风格和真实人脸相差很大，将`--truncation`设小一点，如0.5。

## Progressive fine-tune
### Pretrain DualStyleGAN on FFHQ
`python pretrain_dualstylegan.py --iter 3000 --batch 4 ./data/ffhq/lmdb/`  
或直接用作者预训练好的模型: generator-pretrain.pt 

### Fine-Tune DualStyleGAN on Target Domain
`python finetune_stylegan.py --iter ITERATIONS --batch BATCH_SIZE --ckpt PRETRAINED_MODEL_PATH --augment DATASET_NAME`  

## Optimize
### Refine extrinsic style code
`python refine_exstyle.py --lr_color COLOR_LEARNING_RATE --lr_structure STRUCTURE_LEARNING_RATE DATASET_NAME`

### Training sampling network
`python train_sampler.py DATASET_NAME`