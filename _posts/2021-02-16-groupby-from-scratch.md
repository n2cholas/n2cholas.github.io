---
layout: post
title: Groupby-by From Scratch "Part 2"
comments: True
---

The group-by (or split-apply-combine) pattern, illustrated below, is ubiquitous
in data analysis. Essentially, you have data with predefined groups and want to
compute some summarizing statistics for each of those groups. For example,
suppose you have a set of High School students and want to estimate the average
height by grade. You would need to _split_ the dataset by grade, _apply_ the
mean operation to each group, then _combine_ the means of all the groups
together.  The figure below is from the [Python Data Science
Handbook](https://github.com/jakevdp/PythonDataScienceHandbook).

![](/assets/images/posts/groupby/split-apply-combine.svg#center)

In Python, the Pandas DataFrame library provides a fast, general implementation
of this algorithm. Jake VanderPlas wrote an excellent [blog
post](https://jakevdp.github.io/blog/2017/03/22/group-by-from-scratch/) looking
at implementing this algorithm from scratch using NumPy (and SciPy). He
benchmarks the various approaches to help build intuition about writing fast
NumPy code.  This post extends that one (which is why this post is labelled
"Part 2"), so please read Jake's first then come back here.

Jake looks at the general case where you want to compute a sum within each
group, then return the results as a Python dictionary. The keys for the groups
can be any hashable type (strings, integers, etc.). In this post, I want to
relax a few of those requirements to optimize the implementations and see how
the results differ. Those relaxations are:

1. The return value does not have to be a dictionary. It can be any type you
   can index into using a key to retrieve the result (e.g. a list or
   array).
2. The inputs are always NumPy arrays.
3. The keys are integers from 0 to $k$.
4. The maximum key, $k$, is known in advance.

(1) seems reasonable, and should save us some time since we can avoid
converting the container types. (2) is realistic, since you almost always
want to manipulate numeric data in NumPy arrays instead of Python lists. (3) is
fine since we can always map our arbitrary labels to integers in advance. (4)
seems restrictive, but in practice, you often know how many groups you have in
your data. For the High School students example, you know there are 4 grades
(at least in Canada) apriori.  If you are using a group-by to perform
unsupervised clustering, many algorithms require you to decide the number of
clusters in advance.

We'll see that these relaxations allow us to simplify the implementations and
improve performance.

The rest of this post is structured as follows. First, we update the Jake's
implementations according to our new constraints. The pandas/Python
implementations see very little change, while the NumPy based solutions see large 
improvements. We do some big-O time complexity analysis to understand where we
gain efficiency. Next, we show some new implementations that these
relaxations afford us and go into more detail about these (as they're not in
Jake's post). Finally, we benchmark these implementations to see if the
conclusions from our analysis hold in practice.

All the code is available in
[this](https://colab.research.google.com/drive/1X0tIA_MmYVQVo--6SVa90mcnAMa55Bm1?usp=sharing)
Colab.  It's is similar to Jake's, except we use `time.perf_counter()` to time,
`seaborn` to plot, and include 95% confidence bands in the line-plots.
Notable, the data distributions are the same.

## Updated Implementations

We use the following arrays as our example data:

```python
import numpy as np

keys = np.asarray([0, 1, 2, 0, 1, 2])
vals = np.asarray([1, 2, 3, 4, 5, 6])
max_key = keys.max()
```

This is the same as Jake's running example, except our keys are integers and we
pre-compute the max key.

### Non-NumPy Approaches

Our non-NumPy approaches don't benefit much from our updated rules. First, the
most idiomatic and common solution in Python: Pandas.

```python
import pandas as pd

def pandas_groupby(keys, vals, max_key=None):
    return pd.Series(vals).groupby(keys).sum()

pandas_groupby(keys, vals, max_key)
```
<div class="output_block">
<pre class="output">
0    5
1    7
2    9
dtype: int64
</pre>
</div>

The code is nearly identical, except we remove the `.to_dict()`, so the result
is a Pandas `Series`.

Since we know that the keys are from 0 to `max_key` in advance, we don't have
to use a `defaultdict` for the dictionary-based approach, and can instead
prepopulate with 0s (shown below). In the end, we'll see this doesn't make a
big difference in performance.

```python
def dict_groupby(keys, vals, max_key=None):
    count = {k: 0 for k in range(max_key+1)}
    for key, val in zip(keys, vals):
        count[key] += val
    return count

dict_groupby(keys, vals, max_key)
```
<div class="output_block">
<pre class="output">
{0: 5, 1: 7, 2: 9}
</pre>
</div>

The itertools implementaton does not change, so I excluded the code (it's in
Jake's post). 

### NumPy-based Approaches

The additional information allows us to improve the _asymptotic behaviour_ of
these NumPy-based approaches. That is, the behaviour as the dataset size ($n$)
and number of groups ($k$) grows large.  This post assumes you're familiar with
this concept,
[here](https://www.freecodecamp.org/news/big-o-notation-simply-explained-with-illustrations-and-video-87d5a71c0174/)
is a great explanation if you're not. 

In the original problem, all these implementations require `np.unique` in
order to determine the keys. `np.unique` sorts the data, which is an $O(n \log
n)$ operation, where $n$ is the length of the dataset.  Knowing our keys are 0
to $k$ where $k$ is known in advance allows us avoid this sort to save time.

```python
def masking_groupby(keys, vals, max_key):
    return [vals[keys == key].sum() for key in range(max_key+1)]
    
masking_groupby(keys, vals, max_key)
```
<div class="output_block">
<pre class="output">
[5, 7, 9]
</pre>
</div>

Every boolean mask computation takes $O(n)$ time, which we do for each key,
resulting in $O(nk)$ time. This is more efficient than the original
implementation, which requires $O(n \log n + nk)$ time due to the sort.

The other small improvement is constructing a `list` instead of a `dict` due
to our keys being integer from 0 to `max_key`.

The `np.bincount` implementation becomes very sleek:

```python
def bincount_groupby(keys, vals, max_key=None):
    return np.bincount(keys, weights=vals, minlength=max_key+1)

bincount_groupby(keys, vals, max_key)
```
<div class="output_block">
<pre class="output">
array([5., 7., 9.])
</pre>
</div>

Since we have the `max_key`, we can tell bincount how much space to
pre-allocate for the array (via `minlength`), which saves time on potential
array resizing. This approach only takes $O(n)$ time, since under the hood, the
function just iterates through the array once and adds each value to the
appropriate bucket.

For the sparse implementation, we again we save time by skipping `np.unique`
(and conversion to `dict`).

```python
from scipy import sparse

def sparse_groupby(keys, vals, max_key=None):
    col = np.arange(len(keys))
    mat = sparse.coo_matrix((vals, (keys, col)))
    return mat.sum(1)

sparse_groupby(keys, vals, max_key)
```
<div class="output_block">
<pre class="output">
matrix([[5],
        [7],
        [9]])
</pre>
</div>

Since the sparse representation essentially stores a key-value lookup, the
time complexity here is once again $O(n)$. If this was a dense matrix instead
of sparse (i.e. if we summed up the 0s that this sparse matrix hides), it would
be more expensive (which we'll see below).

Our output seems a bit strange now, but this still gives us the desired
property of being able to index into it to get our sum:

```python
x = sparse_groupby(keys, vals, max_key)
x[0]
```
<div class="output_block">
<pre class="output">
matrix([[5]])
</pre>
</div>

But if you want an array, you can convert between NumPy matrices and arrays
without copying memory:

```python
np.asarray(x).ravel()
```
<div class="output_block">
<pre class="output">
array([5, 7, 9])
</pre>
</div>

## New Implementations

The convenient properties of our keys afford us some alternative
implementations. We'll cover these in more detail since they were not in Jake's
post. First:

```python
def arange_groupby(keys, vals, max_key):
    one_hot = keys == np.arange(max_key+1)[:,None]
    return np.dot(one_hot, vals)

arange_groupby(keys, vals, max_key)
```
<div class="output_block">
<pre class="output">
array([5, 7, 9])
</pre>
</div>

Let's break this one down. Conceptually, this is almost identical to the sparse
matrix approach. Instead of having integer keys, we convert to a "one hot"
representation (which you may be familiar with from other machine learning
tasks). Essentially, each label becomes a vector with `max_key` dimensions. All
the values of that vector are 0 except at the index of the label. For example,
if `max_key = 2` and our `key = 0`, our one hot vector would be `[1, 0, 0]` (1
only in the 0th position).

Note that `np.arange(max_key+1) = [0, 1, 2]`. If `key = 0`, then `(key == [0, 1,
2]) = [True, False, False]`, which can be interpreted as `[1, 0, 0]`, which is
our one hot representation. We want to obtain the one hot representation for
every key in our collection, which we achieve using a
[broadcasting](https://numpy.org/devdocs/user/theory.broadcasting.html) trick.
The documentation linked provides an excellent explanation which I highly
encourage you read. We use this trick to get:

```python
one_hot = keys == np.arange(max_key+1)[:,None]
one_hot.astype(int)
```
<div class="output_block">
<pre class="output">
array([[1, 0, 0, 1, 0, 0],
       [0, 1, 0, 0, 1, 0],
       [0, 0, 1, 0, 0, 1]])
</pre>
</div>

Here, each column in our matrix represents the one hot representation of each
key. Now, when we use `np.dot`, we are computing the inner product between each
row of this matrix and the entire values vector. This gives us the sum within
each group (I'll leave it as an exercise to you to verify that). It's
essentially the same as the summation in the sparse COO matrix case.

In terms of time complexity, the one_hot computation compares every value in
our data to every potential key, resulting in $O(nk)$ time. The matrix
multiplication is $O(n)$ for each row (since you're multiplying $n$ values and
summing), so it's a total of $O(nk)$ for the entire dot product. This result in
a total running time of $O(nk+nk) = O(nk)$.

Our next implementation leverages a one hot representation as well:

```python
def eye_groupby(keys, vals, max_key):
    one_hot = np.eye(max_key+1)[keys]
    return np.dot(one_hot.T, vals)

eye_groupby(keys, vals, max_key)
```

Here, we compute the identity matrix using `np.eye`. We know row $i$ of this
matrix is zero everywhere but the $i$th column, which is precisely our one hot
representation. We use [advanced
indexing](https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing)
to extract the row corresponding to each label. Concretely:

```python
np.eye(max_key+1)
```
<div class="output_block">
<pre class="output">
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
</pre>
</div>

Recall our keys are `[0, 1, 2, 0, 1, 2]`. With advanced indexing:

```python
np.eye(max_key+1)[keys]
```
<div class="output_block">
<pre class="output">
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
</pre>
</div>

This matrix is the transpose of what we had with `arange_groupby`, which is why
we transpose it before applying the dot product.

The time complexity is the same as the `arange` case. The advanced indexing
hides some of the operations, but you still have to materialize $nk$ values, then do the same
multiplication.

Finally, we can use another built-in function for performing a group-by, this
time in SciPy:

```python
from scipy import ndimage

def ndimage_groupby(keys, vals, max_key):
    return ndimage.sum(vals, labels=keys, 
                       index=np.arange(max_key+1))

ndimage_groupby(keys, vals, max_key)
```
<div class="output_block">
<pre class="output">
array([5., 7., 9.])
</pre>
</div>

I'm not sure how this is implemented under the hood, but (spoiler
alert!) the timings below indicate the time complexity is $O(n)$.

## Timings

The timings were measured in Python 3.8.5 on an Intel i5-8265U CPU @ 1.60GHz
with 8gb of RAM on Ubuntu 20.04. For a baseline, below are timings of the
implementations provided in Jake's blog post:

![](/assets/images/posts/groupby/original_timings.svg#center)

The colored bands around the lines are 95% confidence intervals (for most of
the implementations it's invisible). Notice both axes are log. As shown in
Jake's post, the Pandas implementation really starts to shine as we increase
the dataset size. At smaller sizes, the NumPy-based implementations all do
well, with the pure-NumPy (i.e. no Python `dict`s or `list`s) pulling ahead.
Masking slows down significantly with larger group sizes: at every iteration of
the masking loop (which is 1 per unique key), we have to evaluate a boolean
mask for the entire dataset, which gets expensive.

Below are timings for the updated implementations:

![](/assets/images/posts/groupby/new_timings.svg#center)

For the updated implementations, we observe some similar trends: the Pandas
implementation goes from slowest to among the fastest as our data size
increases, while the other increase with generally the same slope. When we
increase the number of groups, the masking and one-hot implementations suffer,
since their time complexity depends on the number of groups $k$.

The pure-numpy implementations benefit the most from our relaxations. All of
them used `np.unique`, and removing that call improved speed across the board
(as predicted by our big-O analysis).  In particular, the `np.bincount`-based
implementation is now the fastest at all dataset and group sizes.

Below, we show the relative speed-up:

![](/assets/images/posts/groupby/faster.svg#center)

As we saw in the other two plots, the `np.unique`-based solutions saw the
biggest benefit (bincount, sparse, and masking), while the others saw
negligible improvements. This is expected: as the data size grows large,
container and key conversions are just small overheads compared to the large
computation.

<br>

## Code

All the code can be found in the Colab
[here](https://colab.research.google.com/drive/1X0tIA_MmYVQVo--6SVa90mcnAMa55Bm1?usp=sharing).
The main benchmarking functions are shown below:

```python
def time_groupby(func, n_group, size, rseed=754389, n_iter=500):
    times = np.empty(n_iter)
    rand = np.random.RandomState(rseed)
    for n in range(n_iter):
        keys = rand.randint(0, n_group, size)
        vals = rand.rand(size)

        start = time.perf_counter()
        _ = func(keys, vals, n_group)
        end = time.perf_counter()
        times[n] = end - start

    return times


def bench(funcs, n_groups, sizes, n_iter=500):
    """Run a set of benchmarks and return as a dataframe"""    
    n_groups, sizes = np.broadcast_arrays(n_groups, sizes)
    names = [func.__name__.split('_')[0] for func in funcs]
    
    dfs = []
    for func in funcs:
        for n_group, size in zip(n_groups, sizes):
            timings = time_groupby(func, n_group, size, n_iter=n_iter)
            dfs.append(pd.DataFrame({
                'Times (s)': timings,
                'Name': [func.__name__.split('_')[0]]*n_iter,
                'Num Groups': [n_group]*n_iter,
                'Size': [size]*n_iter,
            }))
            
    return pd.concat(dfs)

n_groups = (10 ** np.linspace(0, 4, 10)).astype(int)
sizes = (10 ** np.linspace(2, 6, 10)).astype(int)

funcs = [pandas_groupby, dict_groupby, itertools_groupby,
         masking_groupby, bincount_groupby, sparse_groupby,
         list_groupby, arange_groupby, eye_groupby, ndimage_groupby]
timings_sizes = bench(funcs, 10, sizes, n_iter=100)
timings_groups = bench(funcs, n_groups, 10000, n_iter=100)
```