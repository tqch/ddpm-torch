# PyTorch Implementation of Denoising Diffusion Probabilistic Models [[paper]](https://arxiv.org/abs/2006.11239) [[official repo]](https://github.com/hojonathanho/diffusion)


## Reference formula

### Posterior mean and variance

- (Predict $x\_{t-1}$ from $x\_t, x\_0$) 

$$ x\_{t-1} \mid x\_t, x\_0 \sim \text{N}\left(\frac{\sqrt{\alpha\_t}(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}x\_t+\frac{\sqrt{\alpha\_{t-1}}\beta\_t}{1-\bar{\alpha}\_t}x\_0, \sigma\_t^2\right) $$

- (Predict $x\_{t-1}$ from $x\_t, \epsilon\_t$) 


$$ x\_{t-1} \mid x\_t, x\_0 \sim \text{N}\left(\frac{1}{\sqrt{\bar{\alpha}\_t}}\left(x\_t-\frac{\beta\_t}{\sqrt{1-\bar{\alpha}\_t}}\epsilon\_t\right), \sigma\_t^2\right) $$

where $\sigma\_t^2 = \frac{\beta\_t(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}$

## Toy Example

### 8 Gaussian
![8 Gaussian](./assets/gaussian8.gif)

### 25 Gaussian
![25 Gaussian](./assets/gaussian25.gif)

### Swiss Roll
![Swiss Roll](./assets/swissroll.gif)

## Celeb-A Example

### Training samples (50 epochs)
![sample-50](./assets/celeba-sample-50.gif)

### Reverse sampling path
![reverse-process](./assets/celeba-reverse-process.gif)

