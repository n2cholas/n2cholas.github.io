---
layout: post
title: Optimizing k-Means in NumPy & SciPy
comments: True
---

<center>
<a href="https://colab.research.google.com/drive/1aLFM6K-ZJ7QkZoQFJ4QQ5xOvbN3OOndo?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
</center>

In this article, we'll analyze and optimize the runtime of a basic implementation of the k-means algorithm using techniques like vectorization, broadcasting, sparse matrices, unbuffered operations, and more.
We'll focus on generally applicable techniques for writing fast NumPy/SciPy and stay away from arcane tricks (no promises, though).

## k-Means Crash Course

Suppose you run a smartphone company and plan to sell a new phone in three sizes: small, medium, and large.
To figure out each size, you do a survey on your customers' preferences.
Given this data on preferred sizes, how can you deduce the best phone sizes?

One approach is to use _clustering_: a procedure that discovers groups within data.
You can find 3 clusters in your data, then cater your phones to the average customer in each cluster.
One of the most popular algorithms for doing so is called _k-means_.

As the name implies, this algorithm aims to find $k$ clusters in your data.
Initially, k-means chooses $k$ random points in your data, called centroids.
Then, each point is assigned to the closest centroid, where "closeness" is measured by Euclidean distance.
Next, each centroid is updated to the average of all the points assigned to that cluster.
The algorithm repeats the previous two steps until convergence.
This post won't detail the math behind this algorithm, as there are many great resources online that go more in depth.

## Initial Implementation

To focus on optimizing the core logic (i.e. the inner loop), we'll make a few simplifying assumptions:

1. The data is provided as a $n \times d$ array ($n$ data points, each of dimension $d$).
3. The algorithm runs for a fixed number of iterations (instead of checking convergence criteria).
4. All inputs/outputs are NumPy arrays.

We'll focus on a moderately sized problem with $n=5000$, $d=26$ and $k=26$.
In practice, k-means doesn't work well when $d$ is too large, since the euclidean distance isn't a great measure in high dimensions.
$k$ can vary a lot depending on the problem, but for the sake of simplicity, we'll assume it's similar in scale to $d$.

Given these constraints, here's a reasonable first implementation:

```python
def kmeans(data, k, num_iter=50):
    n, d = data.shape
    centroids = data[np.random.choice(n, k, replace=False)]  # (k, d)
    labels = np.empty(n)  # (n,)

    for _ in range(num_iter):
        # ASSIGNMENT STEP
        for i, point in enumerate(data):
            # Compute euclidean distance to each centroid
            distances = [np.linalg.norm(point - c) for c in centroids]
            # Find the closest centroid
            labels[i] = np.argmin(distances)

        # UPDATE STEP
        # For each of the k groups, use boolean indexing to extract
        # the points that belong to that group.
        # Then, find the mean vector within that group.
        centroids = np.stack([data[labels==i].mean(axis=0)
                              for i in range(k)])  # (k, d)

    return centroids

n, k, d = 5000, 26, 26
data = np.random.uniform(size=(n, d))  # dummy data
%timeit kmeans(data, k)  # magic function from Jupyter Notebooks
```

<div class="output_block">
<pre class="output">
1 loop, best of 5: 51.7 s per loop
</pre>
</div>

The implementation is straightforward, except for the small optimization of using `np.empty` instead of `np.zeros`: since we know we will be assigning a value to every element in the array, it's a waste to initialize with 0s.
Instead, numpy will simply allocate a vector of appropriate size and return the uninitialized vector with garbage values.

We'll optimize the assignment step and update step seperately, so we'll refactor the method as follows.

```python
def kmeans(data, k, num_iter=50):
    n, d = data.shape
    centroids = data[np.random.choice(n, k, replace=False)]  # (k, d)

    for _ in range(num_iter):
        labels = assignment_step(data, centroids)  # (n,)
        centroids = update_step(data, labels, k)  # (k, d)

    return centroids
```

## Optimizing the Assignment Step

```python
def assignment_step_v1(data, centroids):
    labels = np.empty(data.shape[0])  # (n,)
    for i, point in enumerate(data):
        distances = [np.linalg.norm(point - c) for c in centroids]
        labels[i] = np.argmin(distances)
    return labels

centroids = data[np.random.choice(n, k, replace=False)]  # (k, d)
%timeit assignment_step_v1(data, centroids)
```

<div class="output_block">
<pre class="output">
1 loop, best of 5: 1.01 s per loop
</pre>
</div>

In general, the slowest thing you can do when processing data in Python is to use pure Python.
As a rule of thumb, you should avoid Python loops unless your code has some sequential dependence (i.e. the current iteration depends on the previous).
In our assignment step, we use two python loops that can both be _vectorized_, since each iteration of the loops don't depend on other iterations.

We do this via [_broadcasting_](https://numpy.org/doc/stable/user/basics.broadcasting.html).
Instead of finding the distance between each data point with each centroid, we find the distance between each data point and all the centroids at once.

```python
def assignment_step_v2(data, centroids):
    labels = np.empty(data.shape[0])  # (n, )
    for i, point in enumerate(data):
        distances = np.linalg.norm(point - centroids, axis=1)  # (k,)
        labels[i] = np.argmin(distances)
    return labels

%timeit assignment_step_v2(data, centroids)
```

<div class="output_block">
<pre class="output">
10 loops, best of 5: 72.6 ms per loop
</pre>
</div>

Recall that `point.shape == (d,)`, and `centroids.shape == (k, d)`.
When we do `point - centroids`, the NumPy *pretends* `point` is replicated `k` times into an array of shape `(k, d)` before doing the subtraction. Then, we compute the norm along the `axis=1`, to obtain `k` distances.

The key word is "pretending": actually materializing the larger array would waste space and time.
Broadcasting simply interprets the existing data in a different (usually larger) shape.

This simple change brings with it a significant speedup.
We can take this idea one step further: use broadcasting to vectorize computing the distance between every point and every centroid:

```python
def assignment_step_v3(data, centroids):
    diff = data[:, None] - centroids[None]  # (n, k, d)
    distances = np.linalg.norm(diff, axis=2)  # (n, k)
    labels = np.argmin(distances, axis=1)  # (n,)
    return labels

%timeit assignment_step_v3(data, centroids)
```

<div class="output_block">
<pre class="output">
100 loops, best of 5: 18.1 ms per loop
</pre>
</div>

When you index with `None` in NumPy, you are adding a new axis (equivalent to indexing with `np.newaxis`).

Thus, `centroids.shape == (k, d)` and `centroids[None].shape == (1, k, d)`.
For data, we insert a new axis in the second spot, so `data[:, None].shape == (n, 1, d)`.
Recall that indexing with `:` selects the entire axis.

Now, `diff.shape == (n, k, d)`.
Effectively, NumPy replicated `data` `k` times along the second axis, and replicated `centroids` `n` times along the first axis.
Therefore, `diff` contains all the pairwise differences.

Then, we compute the norm along `d` once again, then compute the argmin along `k` to get our final labels.

Eliminating these loops resulted in an order of magnitude improvement, though we can still do slightly better. The euclidean distance between two points is computed as follows:

$$d(x, y) = \sqrt{\sum_{i=1}^d (x_i - y_i)^2} $$

The square root here is a monotonic function, so removing it won't change the relative order of the centroids.
Let's get rid of it:

```python
def assignment_step_v4(data, centroids):
    diff = data[:, None] - centroids[None]  # (n, k, d)
    distances = (diff**2).sum(axis=2)  # (n, k)
    labels = np.argmin(distances, axis=1)  # (n,)
    return labels

%timeit assignment_step_v4(data, centroids)
```

<div class="output_block">
<pre class="output">
100 loops, best of 5: 17.8 ms per loop
</pre>
</div>

Though this brings a slight improvement, it's not entirely due to removing the relatively cheap square root operation.
The `np.linalg.norm` function has a lot of preamble code, thus the difference between `v3` and `v4` will get smaller as the inputs grow larger.

Notice, `(diff**2)` produces an intermediate `(n, k, d)` array before we reduce it via the `sum`.
`np.linalg.norm` on the other hand does not: the square and sum happen at the same time.
A line-by-line profile would reveal that it's not really a bottleneck for our current input size, but nonetheless, we can get rid of it without adding complexity.

`np.einsum` can do the square and reduction in one step, avoiding the intermediate array. `np.einsum` allows you to express products and sums in a concise syntax, a nice explanation can be found [here](https://rockt.github.io/2018/04/30/einsum).

```python
def assignment_step_v5(data, centroids):
    diff = data[:, None] - centroids[None]  # (n, k, d)
    distances = np.einsum('nkd,nkd->nk', diff, diff)  # (n, k)
    labels = np.argmin(distances, axis=1)  # (n,)
    return labels

%timeit assignment_step_v5(data, centroids)
```

<div class="output_block">
<pre class="output">
100 loops, best of 5: 9.94 ms per loop
</pre>
</div>

`nkd,nkd` indicates that we'll supply two arrays of shape `(n, k, d)` as input, and that the procedure should multiple them together elementwise (axes with the same letters are multiplied). `->nk` indicates that the result will have shape `(n, k)`, and thus `einsum` will sum across the missing dimension `d`.

`np.einsum` will also choose the best implementation for the given expresssion, which is why the time will always be similar to or better than `np.linalg.norm`.

Let's compare these algorithms across various values for `n`, `d`, and `k`:

![](/assets/images/posts/fast-k-means/assignment_cmp.svg#center)

Each `n, d, k` triplet is run 20 times.
The mean with a 95% confidence band is plotted above.
The methods are grouped into different rows to show trends more clearly, as the latter 3 are significantly faster than the former 2.

We can see the operations are linear with respect to all of the inputs.
For `v1` and `v2`, the relationship appears constant with respect to `d` because the loops over `k` and `n` take dominate the runtime.
When we remove the loop over `k` in 2, the linear relationship disappears in `k` because the time is dominated by the loop over `n`.
We see that for larger sized inputs, `v3` and `v4` are about the same because the overhead of `v3` becomes negligible.
`v5` is consistently the fastest due to superior implementation selection for the given inputs.

For the assignment step, our runtime cuts primarily came from removing Python loops via vectorization.
Optimizing the `update_step` will be a bit more tricky.

## Optimizing the Update Step

For the update step, we _group-by_ the label then _aggregate_ via a mean.
I wrote a blog post comparing ways to perform this operation in NumPy [here](https://nicholasvadivelu.com/2021/02/16/groupby-from-scratch/), which I recommend reading before proceeding.
This section will use some of those techniques, including boolean masking, onehot matrices, sparse matrices, unbuffered addition, and `np.bincount`.
Unlike the previous blog post, we're aggregating a vector-values (centroids) instead of scalars, thus the explanations will focus on the key changes needed for vectors.

Our goal is to optimize the inner loop of the computation.
So, any arrays that can be pre-computed (outside the loop) will be passed to `update_step` as arguments (i.e. not timed).

Our current `update_step` uses boolean indexing to select the data points in each group, which is suboptimal since we are using a python loop to index $k$ different times instead of vectorizing the operation:

```python
def update_step_original(data, labels, k):
    return np.stack([data[labels==i].mean(axis=0) for i in range(k)])

labels = np.random.randint(0, k, size=n)  # dummy labels
%timeit update_step_original(data, labels, k)
```

<div class="output_block">
<pre class="output">
1000 loops, best of 5: 1.15 ms per loop
</pre>
</div>

In terms of time complexity, creating a boolean mask takes $\mathcal O(n)$ time, due to `labels` having `n` elements.
Then, selecting the corresponding elements in `data` and performing the mean takes $\mathcal O(nd)$ time in the worst case (since at worst, `n` elements will be selected).
This operation happens `k` times in total, leading to a time complexity of $\mathcal O(k(nd+n)) = \mathcal O(ndk)$.

We can remove the loop and vectorize computing the boolean mask.
Concretely, we'll need a matrix of shape `(n, k)`, where column `i` has 1s in rows corresponding to data points in cluster `i` (and 0s elsewhere).
This means each row is filled with 0s except for a single spot spot (since each data point belongs to one cluster).
We'll call this a matrix of one _one-hot vectors_ (terminology from machine learning), and is explained in more detail in the [group-by post](https://nicholasvadivelu.com/2021/02/16/groupby-from-scratch/).
We'll use two different approaches to compute this matrix, both of which are benchmarked further in the appendix.
The time complexity for both approaches $\mathcal O(nk)$.

We can use this one-hot matrix to perform a _masked-mean_.
First, we use explicit broadcasting to replicate data and one-hot matrix into a `(n, k, d)` array (remember, no extra memory is needed, NumPy only pretends that the original matrices are duplicated).
Then, we compute a mean along the first axis, only considering values where the mask value (i.e. broadcasted one-hot matrix) is 1.

```python
def update_step_masked_mean(data, labels, centroid_labels):
    onehot_matrix = labels[:,None] == centroid_labels  # (n, k)
    b_data, b_oh = np.broadcast_arrays(  # (n, k, d), (n, k, d)
        data[:, None], onehot_matrix[:, :, None])
    return b_data.mean(axis=0, where=b_oh)  # (k, d)

centroid_labels = np.arange(k, dtype=np.int32)  # (k, )
%timeit update_step_masked_mean(data, labels, centroid_labels)
```

<div class="output_block">
<pre class="output">
100 loops, best of 5: 9.78 ms per loop
</pre>
</div>

The `mean` method still has to iterate over the entire `(n, k, d)` array when accumulating values, resulting in a time complexity of $\mathcal O(nk + nkd) = \mathcal O(nkd)$

Despite eliminating the loop through vectorization, this approach is _slower_ than our initial implementation.
Based on my experience, the `where` argument for NumPy reductions is not well optimized.

Our upcoming approaches sum up the vectors in each group, then divide by the number of elements in each group.
Recall from the group-by post, that `np.bincount` is the fastest way to compute the number of the elements in each group.
Also from that post, given the one-hot matrix, summing within each groups can be represented as a matrix multiplication, which we'll do via `np.dot`:

```python
def update_step_dense_matmul(data, labels, k, identity):
    onehot_labels = identity[labels]  # (n, k)
    group_counts = np.bincount(labels, minlength=k)[:, None]  # (k, 1)
    group_sums = onehot_labels.T.dot(data)  # (k, d)
    # equivalently,
    # group_sums =  np.einsum('nd,nk->kd', data, onehot_labels)
    return group_sums / group_counts

identity = np.eye(k)  # (k, k)
%timeit update_step_dense_matmul(data, labels, k, identity)
```

<div class="output_block">
<pre class="output">
1000 loops, best of 5: 1.03 ms per loop
</pre>
</div>

The time complexity of dense matrix multiplication between a `(n, d)` and `(k, d)` matrix is $\mathcal O(nkd)$. The `group_counts` allocates an array of size `k`, then simply does an increment for each value of `labels`, leading to a complexity of $\mathcal O(n + k)$. Combined with the construction of the `onehot_labels`, the overall complexity is $\mathcal O(nkd + (n+k) + nk) = \mathcal O(nkd)$.

However, this method does more work than the masked mean, since we are doing a matrix multiplication instead of a masked sum and divide.
A lot of these sums and products are redundant as they're with 0s, since our one-hot matrix has a sparse structure (and gets increasingly sparse as `k` grows larger).
For our current input sizes, this method still beats the innefficient masked mean, but is slower than our original.

We can avoid avoid constructing the dense one-hot matrix using `scipy.sparse`.


### Sparse Matrices

The one-hot matrix has a sparse structure--only one non-zero value per row--which we'll leverage to construct the matrix faster and do less work during the matrix multiplication.

Compressed Sparse Row (CSR) and Compressed Sparse Column (CSC) matrices are two formats which are designed for fast arithmetic (e.g. matrix multiplication).
The [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/sparse.html) explains how they work.
Practically speaking, CSR is more space-efficient for matrices with fewer rows; CSC for matrices with fewer columns.
It's nearly impossible to reason about which will be more time-efficient, so you should always benchmark to check.

```python
from scipy import sparse

def update_step_csr(data, labels, k, ones_vec, column_inds):
    # notice this onehot matrix is the transpose of our original
    onehot_matrix = sparse.csr_matrix(
        # constructor: (values, (row_indices, column_indices))
        (ones_vec, (labels, column_inds)), shape=(k, len(data)))
    group_counts = np.bincount(labels, minlength=k)[:, None]  # (k, 1)
    return onehot_matrix.dot(data) / group_counts

ones_vec = np.ones_like(labels)  # (n,)
column_inds = np.arange(n)  # (n,)
%timeit update_step_csr(data, labels, k, ones_vec, column_inds)
```

<div class="output_block">
<pre class="output">
1000 loops, best of 5: 514 µs per loop
</pre>
</div>

```python
def update_step_csc(data, labels, k, ones_vec, column_inds):
    # constructor: (values, (row_indices, column_indices))
    onehot_matrix = sparse.csc_matrix(
        (ones_vec, (labels, column_inds)), shape=(k, len(data)))
    group_counts = np.bincount(labels, minlength=k)[:, None]  # (k, 1)
    return onehot_matrix.dot(data) / group_counts

%timeit update_step_csc(data, labels, k, ones_vec, column_inds)
```

<div class="output_block">
<pre class="output">
1000 loops, best of 5: 491 µs per loop
</pre>
</div>

For the sake of analysis, we'll ignore the time needed to construct these sparse matrices (spoiler: it's negligible compared to the rest).
The sparse matrix multiplication is more efficient than the dense: it only multiplies the non-zero values, which we have `n` of. We must multiply with all the values in our dense data matrix, and accumulate the result in an `(k, d)` matrix, resulting in a time complexity of $\mathcal O(n + nd + kd) + \mathcal O(d(n+k))$.
Combined with the `group_counts` computation, this results in a time complexity of $\mathcal O(d(n+k) + (n+k)) = \mathcal O (d(n+k))$.

This is _much_ better than our previous approaches, but we still have some wasted work, since we are doing a matrix multiplication instead of a straight sum.
Our upcoming approaches will avoid boolean masks altogether.

### `update_step` without boolean masks

_Unbuffered addition_ is a simple way to sum values in each group, and works for vector-valued sums (unlike `np.bincount`).
Recall from the group-by post that `np.add.at` essentially does `array[indices] += values`, except `indices` is allowed to have duplicates.

```python
def update_step_add_at(data, labels, k):
    _, d = data.shape
    group_counts = np.bincount(labels, minlength=k)[:, None]  # (k, 1)
    group_sums = np.zeros((k, d))  # (k, d)
    np.add.at(group_sums, labels, data)  # unbuffered sum
    return group_sums / group_counts  # (k, d)

%timeit update_step_add_at(data, labels, k)
```

<div class="output_block">
<pre class="output">
100 loops, best of 5: 9.9 ms per loop
</pre>
</div>

As we saw before, creating `group_counts` takes $\mathcal O(n + k)$ time.
Creating the `group_sums` takes $\mathcal O(kd)$ time, then we sum `n` vectors of size `d`, which takes $\mathcal O(nd)$ (the index select takes constant time).
Adding these up, we get a time complexity of $\mathcal O((n+k) + kd + nd) = \mathcal O(d(n+k))$.

Unfortunately, as seen in the group-by post, unbuffered addition in NumPy is comparatively slow, making this method the second slowest we've tried so far.

`np.bincount` will do the exact same operation much faster for scalars.
Recall that using `weights=data` lets us sum the groups in the data corresponding to the `labels`.
Not providing weights is equivalent to `weights=np.ones_like(labels)`, which we use to compute the `group_counts`.
We can use `np.apply_along_axis` to run `np.bincount` `d` times to average each dimension individually.
However, `np.apply_along_axis` uses a python for-loop in the backend, instead of actually vectorizing the operation.

```python
def update_step_apply_bincount(data, labels, k):
    group_counts = np.bincount(labels, minlength=k)[:, None]  # (k, 1)
    fn = lambda w: np.bincount(labels, weights=w, minlength=k)
    return np.apply_along_axis(fn, 0, data) / group_counts

%timeit update_step_apply_bincount(data, labels, k)
```

<div class="output_block">
<pre class="output">
1000 loops, best of 5: 556 µs per loop
</pre>
</div>

The `apply_along_axis` does this same operations for each dimension in`d`, leading to $\mathcal O(d(n+k))$ time.
With the `group_counts`, the total time complexity is  $\mathcal O((n+k) + d(n+k)) = \mathcal O(d(n+k))$.

Although `np.apply_along_axis` is designed for convenience not speed, we observe a runtime improvement compared to our initial approaches.

Can we get rid of this Python loop altogether?

We currently have `k` bins: one for each centroid.
Then, we sum the elements for each centroid one dimension at a time.
To vectorize this operation and avoid the loop, we can instead pretend we have `k*d` groups.
Then, we can run `np.bincount` once on the flattened data, then reshape our result so we have `k` groups of dimension `d`.

```python
def update_step_flat_bincount(data, labels, k, extended_labels):
    _, d = data.shape
    group_counts = np.bincount(labels, minlength=k)  # (k,)
    label_matrix = extended_labels + labels[:, None]  # (n, d)
    group_sums = np.bincount(  # (k*d,)
        label_matrix.ravel(), weights=data.ravel(), minlength=k*d)
    return (group_sums.reshape((d, k)) / group_counts).T  # (k, d)

extended_labels = np.arange(start=0, stop=d*k, step=k)  # (d,)
%timeit update_step_flat_bincount(data, labels, k, extended_labels)
```

<div class="output_block">
<pre class="output">
1000 loops, best of 5: 547 µs per loop
</pre>
</div>

Constructing our `label_matrix` takes $\mathcal O(nd)$ time;  `group_sums` takes $\mathcal O(nd + dk)$ time.
Combined with the `group_counts` computation, this takes $\mathcal O(nd + nd + dk + (n+k)) = \mathcal O(d(n+k))$ time.

Despite the same time complexity, we're doing a non-trivial amount of extra work here: we need to generate the `label_matrix` of size `(n, d)`, which is why we don't see a win over the Python-loop `apply_along_axis` variant.

### Timing

We've been comparing these approaches for one set of `n, d, k`, so let's sweep these values.
For the below plot, we run each `n, d, k` triplet for 1000 iterations (with 95% confidence intervals, once again).

![](/assets/images/posts/fast-k-means/update_cmp.svg#center)

We can very clearly see the effect of the various time complexities at play here, as the $\mathcal O(nkd)$ approaches seem to scale much more poorly with respect to $k$ than the $\mathcal O(d(n+k)$ approaches.

It's quite interesting that our initial, unvectorized implementation beats a few of our vectorized ones.
One possible explanation is that our mean is often computed over a comparatively smaller matrix for each group (due to boolean indexing one group at a time), while the vectorized approaches always scan over the full `(n, k, d)` array.
It's also strange that the `dense_matmul` implementation varies so much as the size increases (despite the confidence intervals being quite tight).
This may be due to cache interactions.

Among the faster implementations, the CSR/CSC approaches are the best, with CSC being slightly better.
The bincount based approaches are faster at smaller dimensions, as they do less work (due to the lack of multiplication entirely) at the limit of `d=1`, while scaling poorly for larger `d`.

But this is only one view of the data, and we should investigate more closely how these methods interact with varying $k$ and $d$.
We'll look closer at the bincount based approaches and CSC.

![](/assets/images/posts/fast-k-means/update_step_hmap.svg#center)

Note that the times in the heatmap have been multiplied by 10,000 to more easily display the timings.
As we had anticipated, the culprit in our slowdowns for both bincounts is indeed `d`, without much influence from `k`.

These results indicate that a `bincount` based approach may be more suitable for small `d`, while CSC will be superiour for larger `d`.

## Putting it all together

Based on our results, we can have two potential implementations for k-means, depending on whether we are dealing with high dimensional vectors or low:

```python
def kmeans_small_d(data, k, num_iter=50):
    n, d = data.shape
    centroids = data[np.random.choice(n, k, replace=False)]  # (k, d)

    for _ in range(num_iter):
        # ASSIGNMENT STEP
        diff = data[:, None] - centroids[None]  # (n, k, d)
        labels = np.einsum('nkd,nkd->nk', diff, diff).argmin(1)

        # UPDATE STEP
        group_counts = np.bincount(labels, minlength=k)[:, None]
        fn = lambda w: np.bincount(labels, weights=w, minlength=k)
        centroids = np.apply_along_axis(fn, 0, data) / group_counts

    return centroids

def kmeans_large_d(data, k, num_iter=50):
    n, d = data.shape
    centroids = data[np.random.choice(n, k, replace=False)]  # (k, d)

    for _ in range(num_iter):
        # ASSIGNMENT STEP
        diff = data[:, None] - centroids[None]  # (n, k, d)
        labels = np.einsum('nkd,nkd->nk', diff, diff).argmin(axis=1)

        # UPDATE STEP
        onehot_labels = sparse.csc_matrix(
            (ones_vec, (labels, column_inds)), shape=(k, len(data)))
        group_counts = np.bincount(labels, minlength=k)[:, None]
        centroids = onehot_labels.dot(data) / group_counts  # (k, d)

    return centroids

%timeit kmeans_small_d(data, k)
```

<div class="output_block">
<pre class="output">
1 loop, best of 5: 649 ms per loop
</pre>
</div>

```python
%timeit kmeans_large_d(data, k)
```

<div class="output_block">
<pre class="output">
1 loop, best of 5: 553 ms per loop
</pre>
</div>

We opted for the `apply_along_axis` solution over the `flat_bincount` solution due to its simplicity.
With these changes, we're able to cut down the runtime from dozens of seconds to about half a second!

## Takeaways

Through this process, we've learned that the main priority when writing fast NumPy is to avoid loops and vectorize operations.
Sometimes, doing extra work with vectorization can be faster than doing less work with Python loops; however, as the problem scales, this becomes less true.
The only way to determine what will be fastest for your situation is to benchmark!

## Appendix

Earlier, we chose different methods for creating one-hot matrices of booleans (for masking) vs floats (for multiplication), both of which are detailed in the [group-by post](https://nicholasvadivelu.com/2021/02/16/groupby-from-scratch/).
Here, we'll show a detailed comparison of when each approach is suitable.

Below are the implementations:

```python
def eye_bool(labels, centroid_labels, bool_identity):
    return bool_identity[labels]  # (n, k)

bool_identity = np.eye(k, dtype=bool)  # (k, k)
%timeit eye_bool(labels, centroid_labels, bool_identity)
```

<div class="output_block">
<pre class="output">
10000 loops, best of 5: 85.9 µs per loop
</pre>
</div>

```python
def arange_bool(labels, centroid_labels):
    return labels[:,None] == centroid_labels  # (n, k)

%timeit arange_bool(labels, centroid_labels)
```

<div class="output_block">
<pre class="output">
10000 loops, best of 5: 184 µs per loop
</pre>
</div>

```python
def eye_float(labels, float_identity):
    return float_identity[labels]

float_identity = np.eye(k, dtype=np.float32)  # (k, k)
%timeit eye_float(labels, float_identity)
```

<div class="output_block">
<pre class="output">
10000 loops, best of 5: 134 µs per loop
</pre>
</div>

```python
def arange_float(labels, centroid_labels, out):
    return np.equal(labels[:,None], centroid_labels, out=out)

out = np.empty((n, k), dtype=np.float32)  # (n, k)
%timeit arange_float(labels, centroid_labels, out=out)
```

<div class="output_block">
<pre class="output">
1000 loops, best of 5: 308 µs per loop
</pre>
</div>

For the `arange_float`, we use a pre-allocated output array to give it a fighting chance against `eye_float`, as otherwise, we need to materialize the array then make a copy to conver the dtype.
Generally, I avoid using output arrays like this, as it's bug-prone and not idiomatic.

Let's compare these implementations across various `n` and `k`:

![](/assets/images/posts/fast-k-means/onehot_hmap.svg#center)

Above, we plot the ratio between `arange` and `eye` for `float` and `bool`.
We see in general, `eye` has the advantage, except for small `k` for bool.

The `arange`-based approach needs to do a comparison `n*k` times, whereas the `eye` based approach is merely an index select.
`arange` only beats out `eye` at smaller `k` beecause populating the final array with the comparison result ends up being faster than copying over the corresponding row from the identity matrix.
