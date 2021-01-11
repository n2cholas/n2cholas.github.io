---
layout: post
title: Comparing Array Partition Algorithms
comments: True
---

Partitioning an array involves moving all elements that satisfy some predicate
to one side of the array. The most common example is partitioning an array of
numbers based on a *pivot* number, so that all elements on the left side of the
array are less than or equal to the pivot, while all elements on the right are
greater than the pivot. This specific procedure is crucial for *quicksort*, a
popular and efficient sorting algorithm. The more general partition based on an
arbitrary predicate is useful for a wide range of divide-and-conquer style
problems.

Partitioning an array can be done in-place (i.e. rearranging elements within
the given array without making a copy) using constant space and linear time.
That is, partition uses the same amount of memory regardless of the size
of the input, and its running time increases proportionally to the number of
elements in the given array. The two most common algorithms are Lomuto's
Partition and Hoare's Partition (created by Nico Lomuto and Tony Hoare,
respectively). In an [NDC 2016 talk](https://youtu.be/fd1_Miy1Clg), Andrei
Alexandrescu introduces an alternative algorithm which he showed was more
efficient for a variety of data distributions. 

In this post, we'll focus on partition in the context of quicksort.  We'll
implement, visualize, and time these algorithms. Then, we'll try to tease apart
what makes one slower or faster than the other.  All the code to produce the
timings, animations,and plots in this post can be found
[here](https://github.com/n2cholas/array-partition-comparison).

## Lomuto's Partition

Below is a typically `lomuto_partition` as would be used in quicksort. Our
inputs are the array of numbers `arr`, the start index `lo`, and the end index
`hi`. Throughout this article, we can assume `lo=0` and `hi=len(arr)-1`. For
quicksort, we may want to partition a contiguous subsection of the array, which
is why the parameters are required. This functions returns the index at which
the pivot element ends up in our partitioned array.

```python
def lomuto_partition(arr, lo, hi):
    pivot = arr[hi]
    for j in range(lo, hi):
        if arr[j] <= pivot:
            arr[lo], arr[j] = arr[j], arr[lo]
            lo += 1
 
    arr[lo], arr[hi] = arr[hi], arr[lo]
    return lo
```

We first choose the rightmost element as the pivot. In practice, we can choose
any element to be the pivot, but for this post, we'll keep it simple and just
choose either `hi` or `lo` to be the pivot.

On a high level, we have two indices `j` and `lo`. We ensure that every element
to the left of `lo` is less than or equal to the pivot (invariant A). Also,
ensure every element between `lo` and `j` is greater than the pivot (invariant
B).  Elements to the right of `j` are unexplored.

Every time `j` encounters an element smaller or equal to the pivot, it swaps it
with the element at `lo`. We then increment `j` and `lo`. This way, `lo` is
incremented beyond an element less than or equal to the pivot, maintaining
invariant A. We know the element at `j` currently is greater than the pivot.
Why? Since it was previously between `lo` and `j`, this implies that `j` passed
over it before. If it was passed before but does not appear to the left of `lo`
(i.e. was not swapped), that means it is not less than or equal to the pivot,
thus it must be greater, thus maintaining invariant B.

Finally, we swap the pivot from the end into its rightful place between the two
sections, and return the index at which it was placed.

If talking about indices confuses you, here's a visualization of this algorithm
running on a sample array. The heights of the bars represent the value of the
element at that index. Light gray elements are less than or equal to the pivot,
while black elements are greater. In red are the two pointers, `lo` and `j`,
and in blue is the pivot element.

![](/assets/images/posts/array-partition/lomuto_animation.gif#center)

We see our invariants being held here. We can also observe some extra
work happening: some elements are swapped multiple times unnecessarily. Our
next algorithm avoids this extra work by swapping each element at most once.

## Hoare's Partition

Lomuto's partition maintains two pointers that splits the array into three
sections: less than or equal, greater, unexplored. Hoare's partition also
maintains two pointers, but splits the array into less than or equal,
unexplored, and greater. 

```python
def hoare_partition(arr, lo, hi):
    pivot_ind, pivot = hi, arr[hi]
    hi -= 1
    while True:
        while arr[lo] <= pivot and lo < hi: 
            lo += 1 
        while arr[hi] > pivot and lo < hi:
            hi -= 1
        if lo < hi:
            arr[lo], arr[hi] = arr[hi], arr[lo]
            lo += 1
            hi -= 1
        else:
            if arr[lo] <= pivot:
                lo += 1
            arr[pivot_ind], arr[lo] = arr[lo], arr[pivot_ind]
            return lo
```

Our two pointers this time are `lo` and `hi`. `lo` moves from the left to the
center, stopping when an element is greater than the pivot. `hi` moves from the
right towards the center, stopping only when an element is less than or equal
to the pivot. Once both pointers reach an element violating their side's
condition, they swap, then proceed. 

At the end, once again, we do some clean-up, then swap the pivot into the
dividing point then return that index.

Once again, a visualization:

![](/assets/images/posts/array-partition/hoare_animation.gif#center)

The code is slightly more complex, but this time, no work is wasted: each
element is swapped at most once, which is optimal. However, our code has a lot
of bounds checking, which is wasteful. Our next approach solves this issue.

## Alexandrescu's Partition

Hoare's partition was optimal in terms of swaps, but did a lot of array bounds
checking when incrementing the pointers. Alexandrescu's partition uses a
*sentinel* to avoid excessively checking the indices.

Consider this loop from Hoare's partition:

```python
while arr[lo] <= pivot and lo < hi: 
    lo += 1 
```

If we knew for sure that we would encounter an `arr[lo] > pivot` before `lo >=
hi`, we wouldn't need to check `lo < hi`. To ensure this, we simply plant a
sentinel value at the end of the array which is greater than pivot. This way,
we know we'll stop incrementing `lo` once we reach the end.

Now, the code:

```python
def alexandrescu_partition(arr, lo, hi):
    pivot_ind, pivot = lo, arr[lo]
    old_arr_hi, arr[hi] = arr[hi], pivot+1

    while True:
        lo += 1
        while arr[lo] <= pivot: 
            lo += 1
        arr[hi] = arr[lo]

        hi -= 1
        while arr[hi] > pivot: 
            hi -= 1
        if lo >= hi: break
        arr[lo] = arr[hi]

    if (lo == hi + 2):
        arr[lo] = arr[hi + 1]
        lo -= 1

    arr[lo] = old_arr_hi
    if (pivot < old_arr_hi): lo -= 1

    arr[pivot_ind], arr[lo] = arr[lo], arr[pivot_ind]
    return lo
```

We select our pivot as `arr[lo]` this time, but it really doesn't matter.

On the second line, we set the rightmost item to `pivot+1`: our sentinel which
we know is greater than the pivot. We know our left most item is less than or
equal to the pivot, since it is our pivot. Now in our main loop, we can
increment our indices without bounds checking! We do the minimal index check to
ensure our `lo` and `hi` indices don't cross.

You may observe something peculiar in the main loop: there are no swaps! The
sentinel on the right side of the array is an element that did not exist
before. We removed the element that was there before. We call this a *vacancy*.
Now, in our main loop, as we increment our `lo` index, once we reach an element
greater than partition, we can simply move it to the end to fill the vacancy.
The vacany is now at `lo`. We similarly can fill that vacancy after our `hi`
loop. In this partition scheme, a vacancy moves towards the middle of the
array. At the end of the procedure, there is some fix-up code to re-insert the
removed element in the right position.

We illustrate the vacancy with a lighter shade of red, while the other index is
the normal shade:

![](/assets/images/posts/array-partition/alexandrescu_animation.gif#center)

This algorithm seems to do some messy clean-up at the end. This is ok: most of
our time is spent in the main loop, so we want to optimize that. The clean-up
only happens once per partition, so it's negligible for large arrays.

Note that this algorithm was slightly altered from the one presented during the
talk. In the talk, the definition of partition used allowed elements equal to
the pivot to appear on both the left and right sides. Our requirements are more
strict: we want equal elements only on the left side. Using this stricter
definition allows us to generalize our algorithms to the more general partition
which rearranges element based on a predicate as opposed to a pivot element.

## Timings

We time how long it takes to partition arrays of various lengths. These arrays
are NumPy NDArrays filled with 32-bit integers sampled from a discrete uniform
distribution, where the minimum is 0 and maximum is `2*len(arr)`. We run these
algorithms 1000 times on random arrays for each array size. The timings were
measured in Python 3.8.5 on an Intel i5-8265U CPU @ 1.60GHz with 8gb of RAM on
Ubuntu 20.04.

A more thorough investigation would simulate a variety of data distributions to
understand this algorithm's behaviour, but in this post, we're just interested
in understanding the high level effects of some of the design decisions of
these algorithms. 

![](/assets/images/posts/array-partition/timings.png#center)

Nothing surprising here! All these algorithms take time proportional to the
size of the arrays. We see that Alexandrescu's is fastest across the board by a
*bit*, followed by Hoare's, then finally Lomuto's. The bars represent a 95%
confidence interval of the mean time, so our rankings are statistically
significant (but perhaps not practically significant).

## Deep Dive

### Number of Assignments

Recall that Lomuto's algorithm did extra swaps compared to Hoare's. We want to
count and see if this is indeed the case on some random arrays. We will count
the number of assignments into the array instead of swaps, since Alexandrescu's
partition does not do swaps. 

In Python, swapping elements is as easy as `arr[lo], arr[hi] = arr[hi],
arr[lo]`, but under the hood, this is expanded to something like `tmp =
arr[lo]; arr[lo] = arr[hi]; arr[hi] = tmp`, which is a total of 3 assignments,
not 2. For the purposes of our measurements, we'll only count assignments *into
the array*, that is, a swap will only be counted as 2 assignments.

The plot below shows the number of assignments into the array for random arrays
of various sizes.

![](/assets/images/posts/array-partition/num_assigns.png#center)

We can see that the Hoare and Alexandrescu lines overlap, giving evidence to
our previous claim that both do the same number of swaps. Lomuto's, as
expected, does more swaps than the other two algorithms.

## Number of Array Element Comparisons

Comparisons are needed to determine which side of the pivot a particular
element should go. 

![](/assets/images/posts/array-partition/num_elem_cmp.png#center)

All the lines are on top of eachother, indicating that they all do the same
number of comparisons! This makes sense: all the algorithms compare each
element to the pivot exactly once. 

Well, this is a slight lie:

![](/assets/images/posts/array-partition/num_elem_cmp_zoomed.png#center)

Alexandrescu's and Hoare's do a *bit* of extra comparing at the end as cleanup,
while the elegant Lomuto's partition does not.
    
## Number of Array Index Comparisons

We saw the main innovation introduced by Alexandrescu's partition is the
sentinel which eliminated a lot of index checking. Let's take a look at the
number of index comparisons performed by these algorithms:

![](/assets/images/posts/array-partition/num_ind_cmp.png#center)

As expected, Alexandrescu's has the fewest by a landslide.

In the Python code, you can't actually see the index comparisons for Lomuto's
partition. But unde the hood, `range` is doing bounds checks. You can see that
more clearly if you rewrite the algorithm with a `while` loop:

```python
def lomuto_partition_counting(arr, lo, hi):
    pivot = arr[hi]
    j = lo
    while j < hi:
        if arr[j] <= pivot:
            arr[lo], arr[j] = arr[j], arr[lo]
            lo += 1
        j += 1
 
    arr[lo], arr[hi] = arr[hi], arr[lo]
    return lo, pivot
```

You would expect Hoare's and Lomuto's to match, since they do one index check
per index increment/decrement. But actually, Hoare's gets unchecked index
changes after every swap.

## Conclusion

We explored Lomuto's, Hoare's, and Alexandrescu's partition schemes. While
Lomuto's is the simplest and most elegant, it does the most extra work.
Hoare's, for a bit more complexity, removes most of the extra swapping work.
Alexandrescu's uses a *sentinel* to avoid bounds checking. It also uses
interesting half-swaps instead of full swaps, further saving work compared to a
swap (due to not needing a temporary). With increasing complexity, we get
improvements in efficiency, which we explained by counting the swaps, element
comparisons, and index comparisons on random arrays.

Since the [code is
available](https://github.com/n2cholas/array-partition-comparison), try things
you think I missed! A good start would be animating or counting operations on
data distributions other than uniformly random integers. For example, a common
real world array distribution is a partly sorted array. Share what you find!

## Bonus: Cython

We have been doing our investigation in pure Python, which is pretty slow.
Cython will convert your (statically typed) Python code to C, to avoid Python's
overhead. Here's an example of this conversion for Lomuto's in a Jupyter
notebook cell:

```python
%%cython -a
cimport cython
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def lomuto_partition_cython(np.ndarray[int, ndim=1] arr, int lo, int hi):
    cdef int pivot, j
    
    pivot = arr[hi]
    for j in range(lo, hi):
        if arr[j] <= pivot:
            arr[lo], arr[j] = arr[j], arr[lo]
            lo += 1
 
    arr[lo], arr[hi] = arr[hi], arr[lo]
    return lo
```

The output:

![](/assets/images/posts/array-partition/lomuto_cython_out.png#center)

By annotating the types of our parameters and pre-declaring our variables, we
were able to convert this code to C. We can see that our function was
successfully converted with minimal Python interaction (only at the function
call and return). If there was Python interaction, this would slow down our
code as we would have to leave C.

If we had not pre-declared our variables, for example, we would see some Python
interaction, which is suboptimal:

![](/assets/images/posts/array-partition/lomuto_subopt_cython.png#center)

Let's compare the timings for the three algorithms after Cython compilation.
Additionally, we can compare to NumPy's built-in partition, since we're now in
the same order of magnitude of performance. Our experimental set-up is the same
as before, but we do 10,000 runs instead of 1000 to get tighter confidence
intervals.

![](/assets/images/posts/array-partition/cython_timings.png#center)

Woah, our functions run almost 400x faster! Plus, based on our measurements,
our Hoare partition implementation runs faster than the native NumPy
implementation (which is written in C). We only beat NumPy here because our
distribution is a uniformly random. NumPy uses an introspective algorithm,
which essentially monitors the progress of the partition and can switch
partition schemes partway to ensure good average case performance and optimal
worst case performance. This is crucial for real-world data distributions to
ensure consistent performance, but adds some overhead.

I'm puzzled by why Alexandrescu's partition ends up being the slowest of the
bunch after Cython compilation, and would love to hear a Cython expert's
thoughts on this.