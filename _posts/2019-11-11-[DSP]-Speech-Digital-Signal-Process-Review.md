---
layout:     post
date:       2019-11-11
tag:        note
author:     BY Zhi-kai Yang
---

> For I have few oppotunities to study DSP(Digital Signal Processing) systematically, I intend to gather some of key knowledge and processing tricks here for the following research.

- Where do I need DSP method?

    Recently I am ongoing the studies of Detecting Depression via a Multi-Media model. I need to process the digital signal like video and sound which need some basic knowlege about Speech Processing and Compuer Vision. And DSP lay the base for them. For I am oriented by practicing, I will introduce the programming tricks firstly, then codes and mathmatics will combine. Only main concepts in DSP of course. This article focus on **speech signal** first.
    
- What I will cover

    - Signals, Wave, Spectrum
    - Analytics Method (DCT, FFT)
    - Spectrum Research (Convolution, Filter, Autocorrelation, Defferentiation and Integration)
    - Cepstrum
    
- Programming Tools

    Though `Matlab` is a greet tool for DSP, I will use more light tool `Scipy.signal` and `Scipy.fftpack` in `Python`. `Scipy` includes `numpuy`, `matplotlib`.
    
    The majority of the following codes are reffered to [ThinkDSP](https://github.com/AllenDowney/ThinkDSP/blob/master/code/thinkdsp.py). However, the origin book's code are highly wrapped so that the author can explain the objects more comprehensively. I try to extract the main idear of the code and display it linearly for a convinient review.

## Signals, Wave and Spectrum
First research object includes signals, wave and spectrum.

A **signal** represents a qunatity that varies in time. Its processes are systhesizing, transforming and analyzing.

A **Wave** represents a signal evaluated at a sequence of points in time `ts` (also called**frame**), and computing the corresponding values of signals `ys`.


```python
import numpy as np
PI2 = np.pi * 2
# make a signal
freq = 440
amp = 1.0
offset = 0,
func = np.sin
```

In fact, the aboved parameters can simulate a sine and cosine wave first like $ys = sin(2\pi ft)$, but it is continous in time. We have to evaluate the signal into **Wave** form to plot it.


```python
duration = 1
start = 0
framerate = 11025

# transform to wave
n = round(duration * framerate) # total number of frames
ts = start + np.arange(n) / framerate # sampling time point corresponding to each frame
phases = PI2 * freq * ts + offset
ys = amp * func(phases)
```

Then, we plot the `Wave`.


```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(ts[:100], ys[:100], linewidth=3, alpha=0.7)
plt.xlabel('time(s)');plt.ylabel('amplitude')
```




    Text(0, 0.5, 'amplitude')




![png](/Users/yangzhikai/Documents/louisyzk.github.io/_posts/DSP/output_6_1.png)



```python
# signsesizing a sin and cos
phases_cos = PI2 * 880 * ts
ys_cos = amp * np.cos(phases_cos)
ys += ys_cos
plt.plot(ts[:100], ys[:100], linewidth=3, alpha=0.7)
plt.xlabel('time(s)');plt.ylabel('amplitude')
```

![png](https://pic.superbed.cn/item/5dc951168e0e2e3ee96524b4.png)

**Spectral decomposition** is the idea that any signal can be expressed as the sum of sinusoids with different frequencies, which usually uses the method of **DFT**, **FFT**.

Next, we transfrom a **segment** of wave into spectrum.


```python
seg_ts = ts[:100]
seg_ys = ys[:100]
hs = np.fft.rfft(seg_ys)
fs = np.fft.rfftfreq(len(seg_ys), 1 / framerate)
plt.plot(fs, hs, linewidth=3, alpha=0.7)
plt.xlabel('Hz');plt.ylabel('amplitude')
```


![png](https://pic.superbed.cn/item/5dc951408e0e2e3ee9654175.jpg)


Signal and Wave are in the perspective of time domain and Spectrum lies on the frequency domain. Another graph can combine them:

We can break the wave into segments and plot the spectrum of each segment. The result is called a **short-time Fourier transform (STFT).** or **Spectrogram**.


```python
# make a chirp signal
start=220; end=440
freqs = np.linspace(start, end, len(ts) - 1)
dts = np.diff(ts)
dps = PI2 * freqs * dts
phases = np.cumsum(dps)
phases = np.insert(phases, 0, 0)
ys_chirp = amp * np.cos(phases)
# trans to spectrum
hs = np.fft.rfft(ys_chirp)
fs = np.fft.rfftfreq(len(ys_chirp), 1 / framerate)
ind = (fs <= 700)
plt.plot(fs[ind], hs[ind], linewidth=3, alpha=0.7)
plt.xlabel('Hz');plt.ylabel('amplitude')
```

```python
# implement the spectriogram and plot it!
import matplotlib
seg_length = 512
i, j = 0, seg_length
step = int(seg_length // 2)
# map from time to Spectrum
spec_map = []
while j < len(ys_chirp):
    segment = ys_chirp[i:j]
    # the nominal time for this segment is the midpoint
    t = (i + j) / 2
    hs = np.fft.rfft(segment)
    fs = np.fft.rfftfreq(len(segment), 1 / framerate)
    spec_map.append(hs)
    i += step
    j += step

spec_map = np.array(spec_map).real
X, Y = np.meshgrid(np.arange(x), np.arange(y))

plt.pcolormesh(X.T, Y.T, spec_map, cmap=matplotlib.cm.Blues)
```




![png](https://pic.superbed.cn/item/5dc951d58e0e2e3ee965a0fb.jpg)


## DCT and DFT
The above-mentioned **spectrum decomposition** have many ways to implement. DCT is similar in many ways to the Discrete Fourier Transform (DFT), which we have been using for spectral analysis. Once we learn how DCT works, it will be easier to explain DFT.

### DCT
We can write the signal synthesizing process in linear algebra:

$$\begin{aligned} M &=\cos (2 \pi t \otimes f) \\ y &=M a \end{aligned}$$

where $a$ is a vector of amplitudes, $t$ is a vector of times, $f$ is a vector of frequencies, and $\otimes$ is the symbol for the outer produdct of two vectors.

We want to find a so that $y=M a$ ,in other words, we want to solve a linear system. Numpy provides `linalg.solve` which can do it in $O(n^3)$ because we need to derive matrix inverse.

But if $M$ is orthogonal, the inverse of $M$ is just the transpose of $M$. So we could choose $t$ and $f$ carefully so that $M$ is orthogonal. There are several ways to do it, which is why there are several versions of the discrete
cosine transform (DCT). One simple option is to shift ts and fs by a half unit.


```python
from scipy.fftpack import fft, dct
fft(np.array([4., 3., 5., 10., 5., 3.])).real
```


    array([30., -8.,  6., -2.,  6., -8.])


```python
dct(np.array([4., 3., 5., 10.]), 1)、
```


    array([30., -8.,  6., -2.])

### DFT and FFT
The only difference is that instead of using the cosine function, we’ll use the complex exponential function.

$$
e^{i 2 \pi f t}=\cos 2 \pi f t+i \sin 2 \pi f t
$$

We can also use the linear system method or conjucate method to solve it in $O(n^2)$. However, **FFT** can solve the DFT problem in $O(nlogn)$. The basic idea is the factorizatio of Fourier Matrix.
$$
F_{2 n}=\left[\begin{array}{cc}{I} & {D} \\ {I} & {-D}\end{array}\right]\left[\begin{array}{cc}{F_{n}} & {0} \\ {0} & {F_{n}}\end{array}\right] P
$$

## Filtering and Convolution

The definition of convolution:
$$
(f * g)[n]=\sum_{m=0}^{N-1} f[m] g[n-m]
$$

### The convolution theorem
$$
\operatorname{DFT}(f * g)=\operatorname{DFT}(f) \cdot \operatorname{DFT}(g)
$$

where $f$ is a wave array and $g$ is a **window**. In words, the convolution theorem says that if we convolve $f$ and $g$, and then compute the DFT, we get the same answer as computing the DFT of $f$ and $g$, and then multiplying
the results element-wise. More concisely, convolution in the time domain corresponds to multiplication in the frequency domain.

There are my window functions in `scipy.signal`


```python
import scipy.signal as signal
window = signal.gaussian(51, std=7)
plt.plot(window)
plt.title(r"Gaussian window ($\sigma$=7)")
plt.ylabel("Amplitude")
plt.xlabel("Sample")
```


    Text(0.5, 0, 'Sample')


In this context, the DFT of a window is called a **filter**. The following figure depicts the convolution theorem.

![](https://pic.superbed.cn/item/5dc942d08e0e2e3ee95af67d.jpg)

## Differenciation and Integration
**Diff and Derivate**

- Diff window(operator): one-degree differentiation, like convovle with window [1, -1]

- Derivate window: 

$$
E_{f}(t)=e^{2 \pi i f t}
$$

$$
\frac{d}{d t} E_{f}(t)=2 \pi i f e^{2 \pi i f t}
$$

$$
\frac{d}{d t} E_{f}(t)=2 \pi i f E_{f}(t)
$$

- Computing the difference between successive values in a signal can be expressed as convolution with a simple window. The result is an approximation of the first derivative.

- Differentiation in the time domain corresponds to a simple filter in the frequency domain. For periodic signals, the result is the first derivative, exactly. For some non-periodic signals, it can approximate the derivative.

![](https://pic.superbed.cn/item/5dc948528e0e2e3ee95e06d7.jpg)

### Integration
**Cumsum and Integrate**
The cumulative sum is a good approximation of integration except at the highest frequencies, where it drops off a little faster.

![](https://pic.superbed.cn/item/5dc9495b8e0e2e3ee95f0f30.jpg)

## Cepstrum
The cepstrum is defined as the inverse DFT of the log magnitude of the
DFT of a signal

$$
c[n]=\mathcal{F}^{-1}\{\log |\mathcal{F}\{x[n]\}|\}
$$

- The cepstrum calculation is different in two ways
    - First, we only use magnitude information, and throw away the phase
    - Second, we take the IDFT of the log-magnitude which is already very different since the log operation emphasizes the “periodicity” of the harmonics
    
- The cepstrum is useful because it separates source and filter
    - If we are interested in the glottal excitation, we keep the high coefficients
    - If we are interested in the vocal tract, we keep the low coefficients
    - Truncating the cepstrum at different quefrency values allows us to preserve different amounts of spectral detail (see next slide)
    
### Mel Frequency Cepstral Coefficients (MFCC)
<img src="https://pic.superbed.cn/item/5dc94f0f8e0e2e3ee963cc9d.jpg" style="width:67%">



## Reference

- [Think DSP Digital Signal Processing in Python. Allen B. Downey ](http://greenteapress.com/thinkdsp/thinkdsp.pdf)
- [Complex matrices; fast Fourier transform](https://ocw.mit.edu/courses/mathematics/18-06sc-linear-algebra-fall-2011/positive-definite-matrices-and-applications/complex-matrices-fast-fourier-transform-fft/MIT18_06SCF11_Ses3.2sum.pdf)
- [L9: Cepstral analysis](http://research.cs.tamu.edu/prism/lectures/sp/l9.pdf)
- [Scipy Reference](https://docs.scipy.org/doc/scipy-0.18.1/reference/signal.html#module-scipy.signal)
