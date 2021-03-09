---
layout: post
title: Rejection & Importance Sampling Explained in Code
comments: True
---

<center>
<a href="https://colab.research.google.com/github/n2cholas/dsc-workshops/blob/master/Bayesian_Data_Analysis/rejection_importance_sampling_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
</center>

Lecture 4 of [Bayesian Data Analysis](https://avehtari.github.io/BDA_course_Aalto) by Aki Vehtari covered sampling techniques.
Essentially, we want to infer some properties of a (possibly unnormalized) density function $q$ (our target distribution) that we can't sample directly from.
In this course, $q$ is usually is the product of our likelihood and prior, i.e., the unnormalized posterior.
Rejection sampling and importance sampling are techniques that allow us to transform observations from a proposal distribution $g$ (from which we can draw samples) into observations from our target distribution $q$.

In this post, we will look at how to implement these algorithms in practice.
This supplements the course content as well as the [demo](https://github.com/avehtari/BDA_py_demos/blob/master/demos_ch10/demo10_1.ipynb).
Our example objective is to estimate the variance of $q$.
Suppose we _could_ sample from $q$ directly.
Then, given $n$ samples $\theta_1, \theta_2, ..., \theta_n$, we would compute the sample variance $\hat \sigma^2$ via:

$$\hat \mu = \frac{1}{n} \sum_{i=1}^n \theta_i$$

$$\hat \sigma^2 = \frac{1}{n} \sum_{i=1}^n (\theta_i - \hat \mu)^2$$

For simplicity we're using a _biased_ estimator for the sample variance (to have an unbiased estimator, you'd need to divide by $n-1$ instead of $n$).
In code:

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def estimate_variance(thetas):
    mu = np.mean(thetas)
    return np.mean((thetas - mu)**2)
```

But as mentioned, we don't know how to sample from $q$.

## Set-up

Before getting into the algorithms, we'll set up some functions we need.

First, the (unnormalized) density for $q$. 
We don't care what the function is, only that we can evaluate it at various points.
So, we'll use the same function Professor Vehtari provides in his [demo](https://github.com/avehtari/BDA_py_demos/blob/master/demos_ch10/demo10_1.ipynb).

```python
def q_density(theta):
    r = np.array([1.1, 1.3,-0.1,-0.7, 0.2, -0.4 , 0.06,-1.7,
                  1.7, 0.3, 0.7, 1.6,-2.06,-0.74, 0.2, 0.5 ])
    return stats.gaussian_kde(r, bw_method=0.48).evaluate(theta)
```

Let's plot it in the range $[-3, 3]$ to see how it looks:

```python
thetas = np.linspace(-3, 3, 100)
fig, ax = plt.subplots()
ax.plot(thetas, q_density(thetas))
ax.set(xlabel=r'$\theta$', ylabel=r'$q(\theta)$')
```
![](/assets/images/posts/rejection-importance-sampling/plot1.svg#center)

Next, we need to choose our proposal distribution $g$.
In practice, it is tough to choose a good $g$, particularly in higher dimensions.
Since we're working in one dimension, we can visualize our distribution to see that it's roughly bell shaped, and so the normal distribution seems like a reasonable proposal distribution.
We're cheating a little by stealing the parameters from the original demo---in practice this requires more work.

```python
def g_density(theta):
    return stats.norm(0.0, 1.1).pdf(theta)
```

Plotting $g$ with our $q$:

```python
fig, ax = plt.subplots()
ax.plot(thetas, q_density(thetas), label=r'$q(\theta)$')
ax.plot(thetas, g_density(thetas), label=r'$g(\theta)$')
ax.set(xlabel=r'$\theta$', ylabel=r'Density Value at $\theta$')
ax.legend()
```
![](/assets/images/posts/rejection-importance-sampling/plot2.svg#center)

We also want to be able to sample from $g$:

```python
def sample_from_g(n_samples):
    return stats.norm(0.0, 1.1).rvs(n_samples)

g_samples = sample_from_g(10_000)

fig, ax = plt.subplots(figsize=(8, 1))
# Outline to improve visibility of faint points
ax.scatter(g_samples, np.ones_like(g_samples), s=600, alpha=0.2,
           c='white', edgecolor='#1f77b4')
ax.scatter(g_samples, np.ones_like(g_samples), s=600, alpha=0.01)
ax.tick_params(axis='y', left=False, labelleft=False)
```
![](/assets/images/posts/rejection-importance-sampling/plot3.png#center)

We have a bunch of values from our sample, which we can see comes from our $g$ distribution through a histogram:

```python
fig, ax = plt.subplots()
ax.hist(g_samples)
```
![](/assets/images/posts/rejection-importance-sampling/plot4.svg#center)

## Rejection Sampling

If we use `estimate_variance` with `g_samples`, we'll get an approximation of the variance of $g$ (which is $1.1^2 = 1.21$)

```python
estimate_variance(g_samples)
```
<div class="output_block">
<pre class="output">
1.1934573460327211
</pre>
</div>

If we can transform `g_samples` into samples that follow the $q$ distribution, we could estimate $q$'s variance.
As we learned, in rejection sampling, we must first choose an $M$ such that $Mg(\theta) > q(\theta)$ for all $\theta$.
Let's do this visually, since in our one dimensional case, this is easy:

```python
fig, ax = plt.subplots()
ax.plot(thetas, q_density(thetas), label=r'$q(\theta)$')
for M in [1., 1.5, 2.0, 2.5, 3.0]:
    ax.plot(thetas, M*g_density(thetas), label=f'${M}g(\\theta)$')
ax.set(xlabel=r'$\theta$', ylabel=r'Density Value at $\theta$')
ax.legend()
```
![](/assets/images/posts/rejection-importance-sampling/plot5.svg#center)

Looks like $M \approx 2.5$ is sufficient.

Recall the plot above gives us the _density_ of our probability distribution, which roughly tells how likely it is to draw a sample from that area.
Since $g$ and $q$ have different densities, we want to essentially resample points that we drew from $g$ to better match $q$.
We do this by accepting some points and rejecting others.

If the density $q(\theta)$ is relatively large when $g(\theta)$ is relatively large, we should accept more points around that $\theta$.
If the density $q(\theta)$ is relatively small when $g(\theta)$ is relatively large, we should reject more points around that $\theta$.
The ratio $\frac{q(\theta)}{M g(\theta)}$ gives us this information.
We only need $M$ to ensure $Mg(\theta) > q(\theta)$ for all $\theta$, so $0 \le \frac{q(\theta)}{M g(\theta)} \le 1$, allowing the ratio to be used as the probability of acceptance.
Let's encapsulate this in a function:

```python
M = 2.5
def acceptance_probability(theta):
    return q_density(theta) / (M*g_density(theta))

fig, ax = plt.subplots()
ax.plot(acceptance_probability(np.linspace(-3, 3, 100)))
```
![](/assets/images/posts/rejection-importance-sampling/plot6.svg#center)

We observe we have a higher chance of accepting points near the tails, since $q$ has more density in the tails compared to $g$.
Let's look at the acceptance probability of each individual point in our sample:

```python
acceptance_probabilities = acceptance_probability(g_samples)

fig, ax = plt.subplots()
ax.scatter(g_samples, acceptance_probabilities)
ax.scatter(g_samples[5], acceptance_probabilities[5], c='red')
```
![](/assets/images/posts/rejection-importance-sampling/plot7.svg#center)

To pull one example, sample 0.6045 (in red) has an acceptance probability of 41.89%.
Also, as expected, this has the same shape as the previous plot, since we're showing the same function with our samples as inputs instead of arbitrary values.

Now, onto the actual rejection sampling. 
To randomly accept and reject points based on their probability, we'll draw a (uniformly) random number between 0 and 1 for each sample.
If this draw is less than the acceptance probability, we'll keep the point, otherwise, we'll get reject the point:

```python
rand_01 = np.random.uniform(size=len(g_samples))
to_keep = rand_01 < acceptance_probabilities
to_keep
```
<div class="output_block">
<pre class="output">
array([False, False, False, ..., False,  True,  True])
</pre>
</div>

The array is `True` for the indices we'll keep and `False` for the ones we reject.
We'll use _boolean masking_ to extract the points we're accepting from our `g_samples` array:

```python
q_samples = g_samples[to_keep]
print(f'Original number of samples: {len(g_samples)}')
print(f'Number of accepted samples: {len(q_samples)}')
```
<div class="output_block">
<pre class="output">
Original number of samples: 10000
Number of accepted samples: 3992
</pre>
</div>

We can see that in our case, rejection sampling was pretty inefficient---we threw away a whole bunch of points.
Our *effective sample size* is only 3958, which we can improve slightly by using a smaller $M$ (while still maintaining our inequality).

Now, we can use those samples to get the variance of $q$:

```python
estimate_variance(q_samples)
```
<div class="output_block">
<pre class="output">
1.3519654308471265
</pre>
</div>

## Importance Sampling

It can be challenging to find an efficient $g$ and $M$ such that $Mg(\theta) > q(\theta)$ for all $\theta$.
Since $Mg(\theta)$ will likely be quite different from $q(\theta)$, we'll end up rejecting a lot of points.

Importance sampling improves this with one key observation: often, the property of interest is an expectation.
If we have samples from $q$, we could just average those samples to estimate this property (for the variance, we average $(\theta_i - \hat \mu)^2$).
If we have samples from $g$, we can do a _weighted average_ where we downweight points that would be less likely to be sampled from $q$ (compared to $g$), and upweight points that would be more likely to be sampled from $q$ (compared to $g$).

These weights come from the ratio $\frac{q(\theta)}{g(\theta)}$:

```python
weights = q_density(g_samples)/g_density(g_samples)

fig, ax = plt.subplots()
ax.scatter(g_samples, weights)
```
![](/assets/images/posts/rejection-importance-sampling/plot8.svg#center)

As we can see, the weights are just a scaled version of our rejection probabilities!
We can use these weights along with `g_samples` to compute the variance of $q$ by using a weighted mean instead of a normal mean:

```python
def estimate_variance2(thetas, weights):
    normalizing_val = np.sum(weights)
    mu = np.sum(weights*thetas) / normalizing_val
    # Below, we are doing E[(\theta-\mu)^2]. 
    # Want to weight each item in the average, not each \theta_i.
    return np.sum(weights*(thetas - mu)**2) / normalizing_val
```

Notice this function with a weights of all 1s is the same as our original `estimate_variance`.
To check that, let's estimate the variance of $g$ from the sample:

```python
estimate_variance2(g_samples, np.ones_like(g_samples))
```
<div class="output_block">
<pre class="output">
1.1934573460327211
</pre>
</div>

Now, let's estimate the variance of our $q$ using these importance weights:

```python
estimate_variance2(g_samples, weights)
```
<div class="output_block">
<pre class="output">
1.3422838900127976
</pre>
</div>

Nice, we get (approximately) the same answer!

We can compute the _effective sample size_ for importance sampling using this formula:

$$\tilde w_i = \frac{w_i}{\sum_{j=1}^n w_j}$$

$$n_{ESS} = \frac{1}{\sum_{i=1}^n \tilde w_i^2} $$

$\tilde w_i$ is the normalized version of $w_i$ (so $\sum_{i=1}^n \tilde w_i = 1$).

```python
normalized_weights = weights / np.sum(weights)
n_ess = 1.0 / np.sum(normalized_weights**2)
n_ess
```
<div class="output_block">
<pre class="output">
9382.334536341743
</pre>
</div>

Our effective sample size is much better than with rejection sampling, since $g$ was pretty similar to $q$ everywhere (i.e. was a good proposal distribution).