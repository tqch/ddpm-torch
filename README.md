# PyTorch Implementation of Denoising Diffusion Probabilistic Models [[paper]](https://arxiv.org/abs/2006.11239) [[official repo]](https://github.com/hojonathanho/diffusion)

## Code usage

### Toy data

<details>
<summary>Expand</summary>
<pre><code>
usage: train_toy.py [-h] [--dataset {gaussian8,gaussian25,swissroll}]
                    [--size SIZE] [--root ROOT] [--epochs EPOCHS] [--lr LR]
                    [--beta1 BETA1] [--beta2 BETA2] [--lr-warmup LR_WARMUP]
                    [--batch-size BATCH_SIZE] [--timesteps TIMESTEPS]
                    [--beta-schedule {quad,linear,warmup10,warmup50,jsd}]
                    [--beta-start BETA_START] [--beta-end BETA_END]
                    [--model-mean-type {mean,x_0,eps}]
                    [--model-var-type {learned,fixed-small,fixed-large}]
                    [--loss-type {kl,mse}] [--image-dir IMAGE_DIR]
                    [--chkpt-dir CHKPT_DIR] [--chkpt-intv CHKPT_INTV]
                    [--eval-intv EVAL_INTV] [--seed SEED] [--resume]
                    [--device DEVICE] [--mid-features MID_FEATURES]
                    [--num-temporal-layers NUM_TEMPORAL_LAYERS]
optional arguments:
  -h, --help            show this help message and exit
  --dataset {gaussian8,gaussian25,swissroll}
  --size SIZE
  --root ROOT           root directory of datasets
  --epochs EPOCHS       total number of training epochs
  --lr LR               learning rate
  --beta1 BETA1         beta_1 in Adam
  --beta2 BETA2         beta_2 in Adam
  --lr-warmup LR_WARMUP
                        number of warming-up epochs
  --batch-size BATCH_SIZE
  --timesteps TIMESTEPS
                        number of diffusion steps
  --beta-schedule {quad,linear,warmup10,warmup50,jsd}
  --beta-start BETA_START
  --beta-end BETA_END
  --model-mean-type {mean,x_0,eps}
  --model-var-type {learned,fixed-small,fixed-large}
  --loss-type {kl,mse}
  --image-dir IMAGE_DIR
  --chkpt-dir CHKPT_DIR
  --chkpt-intv CHKPT_INTV
                        frequency of saving a checkpoint
  --eval-intv EVAL_INTV
  --seed SEED           random seed
  --resume              to resume from a checkpoint
  --device DEVICE
  --mid-features MID_FEATURES
  --num-temporal-layers NUM_TEMPORAL_LAYERS
</code></pre>
</details>

### Real-world data

<details><summary>Expand</summary>
<pre><code>
usage: train.py [-h] [--dataset {mnist,cifar10,celeba}] [--root ROOT]
                [--epochs EPOCHS] [--lr LR] [--beta1 BETA1] [--beta2 BETA2]
                [--batch-size BATCH_SIZE] [--timesteps TIMESTEPS]
                [--beta-schedule {quad,linear,warmup10,warmup50,jsd}]
                [--beta-start BETA_START] [--beta-end BETA_END]
                [--model-mean-type {mean,x_0,eps}]
                [--model-var-type {learned,fixed-small,fixed-large}]
                [--loss-type {kl,mse}] [--num-workers NUM_WORKERS]
                [--train-device TRAIN_DEVICE] [--eval-device EVAL_DEVICE]
                [--image-dir IMAGE_DIR] [--num-save-images NUM_SAVE_IMAGES]
                [--config-dir CONFIG_DIR] [--chkpt-dir CHKPT_DIR]
                [--chkpt-intv CHKPT_INTV] [--seed SEED] [--resume] [--eval]
                [--use-ema] [--ema-decay EMA_DECAY] [--distributed]
optional arguments:
  -h, --help            show this help message and exit
  --dataset {mnist,cifar10,celeba}
  --root ROOT           root directory of datasets
  --epochs EPOCHS       total number of training epochs
  --lr LR               learning rate
  --beta1 BETA1         beta_1 in Adam
  --beta2 BETA2         beta_2 in Adam
  --batch-size BATCH_SIZE
  --timesteps TIMESTEPS
                        number of diffusion steps
  --beta-schedule {quad,linear,warmup10,warmup50,jsd}
  --beta-start BETA_START
  --beta-end BETA_END
  --model-mean-type {mean,x_0,eps}
  --model-var-type {learned,fixed-small,fixed-large}
  --loss-type {kl,mse}
  --num-workers NUM_WORKERS
                        number of workers for data loading
  --train-device TRAIN_DEVICE
  --eval-device EVAL_DEVICE
  --image-dir IMAGE_DIR
  --num-save-images NUM_SAVE_IMAGES
                        number of images to generate & save
  --config-dir CONFIG_DIR
  --chkpt-dir CHKPT_DIR
  --chkpt-intv CHKPT_INTV
                        frequency of saving a checkpoint
  --seed SEED           random seed
  --resume              to resume from a checkpoint
  --eval                whether to evaluate fid during training
  --use-ema             whether to use exponential moving average
  --ema-decay EMA_DECAY
                        decay factor of ema
  --distributed         whether to use distributed training
</code></pre>
</details>

### Examples
```shell
# train a 25-Gaussian toy model with single gpu for a total of 100 epochs
python train_toy.py --dataset gaussian8 --device cuda:0 --epochs 100

# train a cifar10 model with single gpu for a total of 50 epochs
python train.py --dataset cifar10 --train-device cuda:0 --epochs 50

# train a celeba model with 2 gpus and an effective batch-size of 64 x 2 = 128
export CUDA_VISIBLE_DEVICES=0,1&&torchrun --standalone --nproc_per_node 2 --rdzv_backend c10d train.py --dataset celeba --use-ema --distributed
```

## Experiment results

### Toy data

#### 8 Gaussian
<p align="center"> <img alt="gaussian8" src="./assets/gaussian8.gif" /> </p>

#### 25 Gaussian
<p align="center"> <img alt="gaussian25" src="./assets/gaussian25.gif" /> </p>

#### Swiss Roll
<p align="center"> <img alt="swissroll" src="./assets/swissroll.gif" /> </p>

### Real-world data

*Table of evaluation metrics*

|Dataset|FID (↓)|Precision (↑)|Recall (↑)|Training steps|Training loss|
|:---:|:---:|:---:|:---:|:---:|:---:|
|CIFAR-10|11.11|0.738|0.421|46.8k|0.0296|
|\|__|6.45|0.727|0.480|93.6k|0.0294|
|\|__|4.99|0.727|0.503|140.4k|0.0287|
|\|__|4.48|0.730|0.517|187.2k|0.0289|
|\|__|4.07|0.731|**0.524**|234.0k|0.0286|
|\|__|**4.01**|**0.733**|0.520|280.8k|**0.0285**|
|CelebA|4.45|0.778|0.478|237.3k|0.0154|

#### CIFAR-10 [[checkpoint]](https://github.com/tqch/ddpm-torch/releases/download/checkpoints/ddpm_cifar10_720.pt)

##### Training samples (720 epochs) 
<p align="center"> <img alt="cifar10_train_100" src="./assets/cifar10_train_720.webp" /> </p>

##### Denoising process
<p align="center"> <img alt="cifar10_denoise_100" src="./assets/cifar10_denoise_100.webp" /></p>

#### CelebA [[checkpoint]](https://github.com/tqch/ddpm-torch/releases/download/checkpoints/ddpm_celeba_150.pt)

##### Training samples (150 epochs)
<p align="center"> <img alt="celeba_train_100" src="./assets/celeba_train_150.webp" /> </p>

##### Denoising process
<p align="center"> <img alt="celeba_denoise_100" src="./assets/celeba_denoise_100.webp" /> </p>

## Reference formulae

### Posterior mean and variance

- (Predict $x\_{t-1}$ from $x\_t, x\_0$) 

$$ x\_{t-1} \mid x\_t, x\_0 \sim \text{N}\left(\frac{\sqrt{\alpha\_t}(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}x\_t+\frac{\sqrt{\alpha\_{t-1}}\beta\_t}{1-\bar{\alpha}\_t}x\_0, \sigma\_t^2\right) $$

- (Predict $x\_{t-1}$ from $x\_t, \epsilon\_t$) 


$$ x\_{t-1} \mid x\_t, x\_0 \sim \text{N}\left(\frac{1}{\sqrt{\bar{\alpha}\_t}}\left(x\_t-\frac{\beta\_t}{\sqrt{1-\bar{\alpha}\_t}}\epsilon\_t\right), \sigma\_t^2\right) $$

where 

$$\sigma\_t^2 = \frac{\beta\_t(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}$$
