---
layout: post
title: Introduction to Data Cleaning with Pandas
---

Through this workshop, you will learn how to use Pandas to explore and “wrangle” datasets. Topics will include an introduction to Jupyter Notebooks/Colab, data cleaning with pandas, feature engineering with pandas, basic visualization and more. This workshop will focus on actual coding.

This article provides a summary of the main workshop, which you can watch [here](https://youtu.be/mesgiVk8G6s). [Here](https://colab.research.google.com/github/n2cholas/dsc-workshops/blob/master/Introduction_to_Data_Cleaning_with_Pandas.ipynb) is a colab link to run all the code.

Indented plain-text blocks in this article contain outputs from the previous code block.


```python
import pandas as pd
import numpy as np

%matplotlib inline
```

## Jupyter Tips

Before starting with pandas, let's look at some useful features Jupyter has that will help us along the way.

Typing a function then pressing tab gives you a list of arguments you can enter. Pressing shift-tab gives you the function signature. Also:


```python
?pd.Series # using one question mark gives you the function/class signature with the description
??pd.Series # two question marks gives you the actual code for that function
```

Timing your pandas code is a very helpful learning tool, so you can figure out the most efficient way to do things. You can time code as follows:


```python
%timeit [i for i in range(500)] # in line mode
```

<div class="output_block">
<pre>
<code class="codeblock">100000 loops, best of 3: 14 µs per loop</code>
</pre>
</div>



```python
%%timeit # time an entire cell
for i in range(10):
    None;
```

<div class="output_block">
<pre>
<code class="codeblock">The slowest run took 5.44 times longer than the fastest. This could mean that an intermediate result is being cached.
1000000 loops, best of 3: 300 ns per loop</code>
</pre>
</div>


Commands prefaced by "%" or "%%" are called magic commands. You can read about more [here](https://ipython.readthedocs.io/en/stable/interactive/magics.html).

## What is Pandas?

Pandas is a Python library for manipulating data and performing analysis. It has too many fefatures to cover in one introductory workshop, but you will find the documentation complete and clear: https://pandas.pydata.org/pandas-docs/stable/. For many tasks, there is likely a Pandas function to make your life easier, so Google away!

The most basic unit in Pandas is called a Series:


```python
s = pd.Series(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
s
```

<div class="output_block">
<pre>
<code class="codeblock">0    a
1    b
2    c
3    d
4    e
5    f
6    g
dtype: object</code>
</pre>
</div>



A series is simply a 1D numpy array with some more functionality built on top. Above, on the left you see an index and on the right are the actual values. The "dtype" is the datatype, which can be anything from objects (usually strings), integers, floats, categorical variables, datetimes, etc. Series are much faster than built in python lists because the numpy backend is written in C.

You can index into a series exactly the same as you would a numpy array:


```python
s[1] # returns the 2nd element (0 indexed)
```

<div class="output_block">
<pre>
<code class="codeblock">'b'</code>
</pre>
</div>




```python
s[1:3] # returns a series from indices 1 to 3 (exclusive)
```

<div class="output_block">
<pre>
<code class="codeblock">1    b
2    c
dtype: object</code>
</pre>
</div>



```python
s[1::2] # returns series from indices 1 to the end, counting by 2s (i.e. 1, 3, 5)
```

<div class="output_block">
<pre>
<code class="codeblock">1    b
3    d
5    f
dtype: object</code>
</pre>
</div>



You also retain the same broadcasting numpy arrays do. For example


```python
s2 = pd.Series([i for i in range(50)])
s2 = s2/50 + 1
```

You can also sample a random element from a series:

```python
s2.sample()
```

<div class="output_block">
<pre>
<code class="codeblock">2    1.04
dtype: float64</code>
</pre>
</div>



Next, let's import some data and jump into Dataframes. Dataframes are tables of data, where each column has a name and is a series of some type. Each column can have a different type.


```python
df = pd.read_csv('https://raw.githubusercontent.com/n2cholas/pokemon-analysis/master/pokemon-data.csv', delimiter=';')
mdf = pd.read_csv('https://raw.githubusercontent.com/n2cholas/pokemon-analysis/master/move-data.csv', delimiter=';')

print('Number of pokemon: ', len(df))
df.sample()
```

<div class="output_block">
<pre>
<code class="codeblock">Number of pokemon:  918</code>
</pre>
</div>



<div style="overflow-x:auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Types</th>
      <th>Abilities</th>
      <th>Tier</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Special Attack</th>
      <th>Special Defense</th>
      <th>Speed</th>
      <th>Next Evolution(s)</th>
      <th>Moves</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>552</th>
      <td>Octillery</td>
      <td>['Water']</td>
      <td>['Moody', 'Sniper', 'Suction Cups']</td>
      <td>PU</td>
      <td>75</td>
      <td>105</td>
      <td>75</td>
      <td>105</td>
      <td>75</td>
      <td>45</td>
      <td>[]</td>
      <td>['Gunk Shot', 'Rock Blast', 'Water Gun', 'Cons...</td>
    </tr>
  </tbody>
</table>
</div>



We can also take samples of different sizes, or look at the top of the dataset, or the bottom:


```python
mdf.head(3)
```


<div style="overflow-x:auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Index</th>
      <th>Name</th>
      <th>Type</th>
      <th>Category</th>
      <th>Contest</th>
      <th>PP</th>
      <th>Power</th>
      <th>Accuracy</th>
      <th>Generation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Pound</td>
      <td>Normal</td>
      <td>Physical</td>
      <td>Tough</td>
      <td>35</td>
      <td>40</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Karate Chop</td>
      <td>Fighting</td>
      <td>Physical</td>
      <td>Tough</td>
      <td>25</td>
      <td>50</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Double Slap</td>
      <td>Normal</td>
      <td>Physical</td>
      <td>Cute</td>
      <td>10</td>
      <td>15</td>
      <td>85</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
mdf.sample(2)
```




<div style="overflow-x:auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Index</th>
      <th>Name</th>
      <th>Type</th>
      <th>Category</th>
      <th>Contest</th>
      <th>PP</th>
      <th>Power</th>
      <th>Accuracy</th>
      <th>Generation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>551</th>
      <td>552</td>
      <td>Fiery Dance</td>
      <td>Fire</td>
      <td>Special</td>
      <td>Beautiful</td>
      <td>10</td>
      <td>80</td>
      <td>100</td>
      <td>5</td>
    </tr>
    <tr>
      <th>84</th>
      <td>85</td>
      <td>Thunderbolt</td>
      <td>Electric</td>
      <td>Special</td>
      <td>Cool</td>
      <td>15</td>
      <td>90</td>
      <td>100</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
mdf.tail()
```




<div style="overflow-x:auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Index</th>
      <th>Name</th>
      <th>Type</th>
      <th>Category</th>
      <th>Contest</th>
      <th>PP</th>
      <th>Power</th>
      <th>Accuracy</th>
      <th>Generation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>723</th>
      <td>724</td>
      <td>Searing Sunraze Smash</td>
      <td>Steel</td>
      <td>Special</td>
      <td>???</td>
      <td>1</td>
      <td>200</td>
      <td>None</td>
      <td>7</td>
    </tr>
    <tr>
      <th>724</th>
      <td>725</td>
      <td>Menacing Moonraze Maelstrom</td>
      <td>Ghost</td>
      <td>Special</td>
      <td>???</td>
      <td>1</td>
      <td>200</td>
      <td>None</td>
      <td>7</td>
    </tr>
    <tr>
      <th>725</th>
      <td>726</td>
      <td>Let's Snuggle Forever</td>
      <td>Fairy</td>
      <td>Physical</td>
      <td>???</td>
      <td>1</td>
      <td>190</td>
      <td>None</td>
      <td>7</td>
    </tr>
    <tr>
      <th>726</th>
      <td>727</td>
      <td>Splintered Stormshards</td>
      <td>Rock</td>
      <td>Physical</td>
      <td>???</td>
      <td>1</td>
      <td>190</td>
      <td>None</td>
      <td>7</td>
    </tr>
    <tr>
      <th>727</th>
      <td>728</td>
      <td>Clangorous Soulblaze</td>
      <td>Dragon</td>
      <td>Special</td>
      <td>???</td>
      <td>1</td>
      <td>185</td>
      <td>None</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



## Initial Processing

We don't need the index column because Pandas gives us a default index, so let's drop that column. 


```python
mdf.drop('Index', inplace=True, axis=1)
# mdf = mdf.drop(columns='Index') # alternative
```

Many pandas functions return a changed version of the dataframe instead of modifying the dataframe itself. We can use inplace=True to do it inplace (which is more efficient). Sometimes, when using multiple commands consecutively, it' easier to chain the commands instead of doing it inplace (as you'll see).


```python
mdf.columns = ['name', 'type', 'category', 'contest', 'pp', 'power', 'accuracy', 'generation'] #set column names

mdf.dtypes
```


<div class="output_block">
<pre>
<code class="codeblock">name          object
type          object
category      object
contest       object
pp             int64
power         object
accuracy      object
generation     int64
dtype: object</code>
</pre>
</div>



Pandas usually does a good job of detecting the datatypes of various columns. We know that power and accuracy should be numbers, but pandas is making them objects (strings). This usually indicates null values. Let's check.


```python
mdf['accuracy'].value_counts()
```


<div class="output_block">
<pre>
<code class="codeblock">100     320
None    280
90       46
95       29
85       26
75       10
80        7
70        4
55        3
50        3
Name: accuracy, dtype: int64</code>
</pre>
</div>



Just as we suspected, there is the string "None" for non-numeric values. Let's fix this.


```python
mdf['accuracy'].replace('None', 0, inplace=True)
# notice mdf.accuracy.replace(..., inplace=True) wouldn't work
mdf['accuracy'] = pd.to_numeric(mdf['accuracy'])
```

Below, we get a boolean series indicating whether the column is 'None' or not. We can use this boolean series to index into the dataframe.


```python
mdf.power == 'None'
```


<div class="output_block">
<pre>
<code class="codeblock">0      False
1      False
2      False
3      False
4      False
5      False
6      False
7      False
8      False
9      False
10     False
11      True
12     False
13      True
14     False
15     False
16     False
17      True
18     False
19     False
20     False
21     False
22     False
23     False
24     False
25     False
26     False
27      True
28     False
29     False
       ...  
698    False
699    False
700    False
701     True
702    False
703    False
704    False
705    False
706    False
707    False
708    False
709    False
710    False
711    False
712    False
713    False
714     True
715    False
716     True
717    False
718    False
719    False
720    False
721    False
722    False
723    False
724    False
725    False
726    False
727    False
Name: power, Length: 728, dtype: bool</code>
</pre>
</div>





```python
mdf[mdf.power == 'None'].head()
```


<div style="overflow-x:auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>type</th>
      <th>category</th>
      <th>contest</th>
      <th>pp</th>
      <th>power</th>
      <th>accuracy</th>
      <th>generation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>Guillotine</td>
      <td>Normal</td>
      <td>Physical</td>
      <td>Cool</td>
      <td>5</td>
      <td>None</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Swords Dance</td>
      <td>Normal</td>
      <td>Status</td>
      <td>Beautiful</td>
      <td>20</td>
      <td>None</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Whirlwind</td>
      <td>Normal</td>
      <td>Status</td>
      <td>Clever</td>
      <td>20</td>
      <td>None</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Sand Attack</td>
      <td>Ground</td>
      <td>Status</td>
      <td>Cute</td>
      <td>15</td>
      <td>None</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Horn Drill</td>
      <td>Normal</td>
      <td>Physical</td>
      <td>Cool</td>
      <td>5</td>
      <td>None</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
mdf.loc[mdf.power == 'None', 'power'].head()
```


<div class="output_block">
<pre>
<code class="codeblock">11    None
13    None
17    None
27    None
31    None
Name: power, dtype: object</code>
</pre>
</div>



.loc is a common way to index into a Dataframe. The first argument is the index (or list of indices), or a boolean array that acts as a mask. iloc can be used similarly, except the first number is the actual numeric index (notice that a Dataframe index can be non-numeric). 


```python
mdf.loc[mdf.power == 'None', 'power'] = 0
mdf['power'] = pd.to_numeric(mdf['power'])

mdf.dtypes
```


<div class="output_block">
<pre>
<code class="codeblock">name          object
type          object
category      object
contest       object
pp             int64
power          int64
accuracy       int64
generation     int64
dtype: object</code>
</pre>
</div>



We were able to convert them with no issues. Notice the two ways to access columns. The only difference between the two is that the dictionary-style access allows you to modify the column, and allows you to create new columns. You can only use the .column method for existing columns, and it returns a copy (so the modifications won't affect the original Dataframe). Also, notice you can't access columns with spaces in their names with the .column notation.

Although the dictionary-style access is more consistent, I like to use the .column access whenever I can because it is faster to type.


```python
df.columns = ['name', 'types', 'abilities', 'tier', 'hp', 'atk', 'def', 'spa', 'spd', 'spe', 'next_evos','moves']
df.dtypes
```


<div class="output_block">
<pre>
<code class="codeblock">name         object
types        object
abilities    object
tier         object
hp            int64
atk           int64
def           int64
spa           int64
spd           int64
spe           int64
next_evos    object
moves        object
dtype: object</code>
</pre>
</div>



We saw above that the next_evos, moves, abilities, and types columns should be lists, so we can do that.


```python
temp_df = df.copy()
```


```python
%%timeit
for ind, row in temp_df.iterrows():
    df.at[ind, 'next_evos'] = eval(row['next_evos'])
```

<div class="output_block">
<pre>
<code class="codeblock">10 loops, best of 3: 108 ms per loop</code>
</pre>
</div>


A few notes. This seems like the most obvious way to achieve what we want. Look through the rows using iterrows, use python's "eval" to turn a string-list into an actual list, then assign it to the dataframe at that index. Notice that we use "at", which is the same as "loc" except it can only access one value at a time. 

This turns out to be the worst way to do this. In pandas, we can almost always avoid explicitly looping through our data.


```python
%%timeit
df['types'] = temp_df.apply(lambda x: eval(x.types), axis=1)
```

<div class="output_block">
<pre>
<code class="codeblock">10 loops, best of 3: 22.4 ms per loop</code>
</pre>
</div>


This is much better. The apply function applies a function you give it to all the rows or columns in the dataframe. The axis argument specifies whether it's rows or columns. We can make this a bit cleaner.


```python
%%timeit
df['abilities'] = temp_df.abilities.map(eval)
```

<div class="output_block">
<pre>
<code class="codeblock">100 loops, best of 3: 6.12 ms per loop</code>
</pre>
</div>


This is very clean. While apply works on a dataframe, map works on a single series. Also, since the value is always just applied to the one column, we can just pass the eval function instead of using a lambda. Our next improvement won't be faster, but it'll be nicer


```python
from tqdm import tqdm
tqdm.pandas()

df['moves'] = temp_df.moves.progress_map(eval)
```

<div class="output_block">
<pre>
<code class="codeblock">100%|██████████| 918/918 [00:00<00:00, 8454.77it/s]</code>
</pre>
</div>


tqdm is a library that provides progress bars for loops, but it can be easily used with pandas to provide a progress bar for your maps and applies. Very useful for doing complex processing on large datasets.

Next, notice that our dataframe has one row per pokemon. It would be nice to index into by the pokemon name rather than a number. If we are going to access rows by pokemon name often, this will give us a speed advantage, since the items in the index are supported in the backend by a hashtable. 


```python
df.set_index('name', inplace=True)

df.loc['Pikachu']
```


<div class="output_block">
<pre>
<code class="codeblock">types                                               [Electric]
abilities                              [Lightning Rod, Static]
tier                                                       NaN
hp                                                          35
atk                                                         55
def                                                         40
spa                                                         50
spd                                                         50
spe                                                         90
next_evos                               [Raichu, Raichu-Alola]
moves        [Tail Whip, Thunder Shock, Growl, Play Nice, T...
Name: Pikachu, dtype: object</code>
</pre>
</div>



We can also reset_index, which can be useful sometimes. Now that we've done some processing, we can produce a summary of the numeric columns:


```python
df.describe()
```


<div style="overflow-x:auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hp</th>
      <th>atk</th>
      <th>def</th>
      <th>spa</th>
      <th>spd</th>
      <th>spe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>918.000000</td>
      <td>918.000000</td>
      <td>918.000000</td>
      <td>918.000000</td>
      <td>918.000000</td>
      <td>918.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>69.558824</td>
      <td>80.143791</td>
      <td>74.535948</td>
      <td>73.297386</td>
      <td>72.384532</td>
      <td>68.544662</td>
    </tr>
    <tr>
      <th>std</th>
      <td>26.066527</td>
      <td>32.697233</td>
      <td>31.225467</td>
      <td>33.298652</td>
      <td>27.889548</td>
      <td>29.472307</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>10.000000</td>
      <td>20.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>50.000000</td>
      <td>55.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>66.500000</td>
      <td>75.000000</td>
      <td>70.000000</td>
      <td>65.000000</td>
      <td>70.000000</td>
      <td>65.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>80.000000</td>
      <td>100.000000</td>
      <td>90.000000</td>
      <td>95.000000</td>
      <td>90.000000</td>
      <td>90.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>255.000000</td>
      <td>190.000000</td>
      <td>230.000000</td>
      <td>194.000000</td>
      <td>230.000000</td>
      <td>180.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Data Correction

Typically, you will find oddities in your data during analysis. Perhaps you visualize a column, and the numbers look off, so you look into the actual data and notice some issues. For the purpose of this workshop, we'll skip the visualization and just correct the data

First, some pokemon have moves duplicated. Let's fix this by making the move-lists into movesets


```python
df['moves'] = df.moves.progress_map(set)
```

<div class="output_block">
<pre>
<code class="codeblock">100%|██████████| 918/918 [00:00<00:00, 68711.23it/s]</code>
</pre>
</div>


Next, I noticed a weird quirk with the strings for the moves. This will cause some trouble if we want to relate the mdf and df tables, so let's fix it.


```python
moves = {move for move_set in df.moves for move in move_set}

weird_moves = {m for m in moves if "'" in m}
weird_moves
```


<div class="output_block">
<pre>
<code class="codeblock">{"Baby'Doll Eyes",
 "Double'Edge",
 "Forest's Curse",
 "Freeze'Dry",
 "King's Shield",
 "Land's Wrath",
 "Lock'On",
 "Mud'Slap",
 "Multi'Attack",
 "Nature's Madness",
 "Power'Up Punch",
 "Self'Destruct",
 "Soft'Boiled",
 "Topsy'Turvy",
 "Trick'or'Treat",
 "U'turn",
 "Wake'Up Slap",
 "Will'O'Wisp",
 "X'Scissor"}</code>
</pre>
</div>



Many of these moves, such as U-turn, should have a dash instead of an apostrophe (according to the moves dataset). Upon closer inspection, it's clear that the only moves that should have an apostrophe are those whose words end with an apostrophe-s. Let's make this correction.


```python
weird_moves.remove("King's Shield")
weird_moves.remove("Forest's Curse")
weird_moves.remove("Land's Wrath")
weird_moves.remove("Nature's Madness")

def clean_moves(x):
  return  {move if move not in weird_moves else 
           move.replace("'", "-")
           for move in x}

df['moves'] = df.moves.progress_map(clean_moves)
```

<div class="output_block">
<pre>
<code class="codeblock">100%|██████████| 918/918 [00:00<00:00, 43018.50it/s]</code>
</pre>
</div>



```python
removal_check = {move for move_set in df.moves 
                      for move in move_set
                      if "'" in move}
removal_check
```


<div class="output_block">
<pre>
<code class="codeblock">{"Forest's Curse", "King's Shield", "Land's Wrath",
 "Nature's Madness"}</code>
</pre>
</div>



The moves dataframe contains moves that are unlearnable by pokemon. These include moves like Struggle (which is a move pokemon use when they have no more pp in their normal moveset) and Z-moves (moves that are activated by a Z-crystal). These moves are characterized by having only 1 PP (which denotes the number of times a pokemon can use the move). Let's remove these.


```python
mdf = mdf[(mdf.pp != 1) | (mdf.name == 'Struggle')]
```

Due to the nature of the site we scraped, some pokemon are missing moves :(. Let's fix part of the problem by adding back some important special moves:


```python
df.loc['Victini', 'moves'].add('V-create')
df.loc['Rayquaza', 'moves'].add('V-create')
df.loc['Celebi', 'moves'].add('Hold Back')

for pok in ['Zygarde', 'Zygarde-10%', 'Zygarde-Complete']:
    df.loc[pok, 'moves'].add('Thousand Arrows')
    df.loc[pok, 'moves'].add('Thousand Waves')
    df.loc[pok, 'moves'].add('Core Enforcer')
```

Let's say for our analysis, we only care about certain tiers. Furthermore, we want to consolidate tiers. Let's do it:


```python
df.loc[df.tier == 'OUBL','tier'] = 'Uber'
df.loc[df.tier == 'UUBL','tier'] = 'OU'
df.loc[df.tier == 'RUBL','tier'] = 'UU'
df.loc[df.tier == 'NUBL','tier'] = 'RU'
df.loc[df.tier == 'PUBL','tier'] = 'NU'
df = df[df['tier'].isin(['Uber', 'OU', 'UU', 'NU', 'RU', 'PU'])]
```

The last line eliminates all pokemon that do not belong to one of those tiers (i.e. LC). 

Since the tiers are a categorical variable, let's covert it to the categorical dtype in pandas. This will come in handy if we decide to use this dataset in a machine learning model, as categorical variables will have a string label but have a corresponding integer code.


```python
df['tier'] = df['tier'].astype('category')
df['tier'].dtype
```

But wait, our tiers do have an order! Let's actually turn them into an ordered categorical variable. This will ensure the codes are in order.


```python
from pandas.api.types import CategoricalDtype

order = ['Uber', 'OU', 'UU', 'NU', 'RU', 'PU']
df['tier'] = df['tier'].astype(CategoricalDtype(categories=order, 
                                                ordered=True))
df['tier'].dtype
```


<div class="output_block">
<pre>
<code class="codeblock">CategoricalDtype(categories=['Uber', 'OU', 'UU', 'NU', 'RU', 'PU'], ordered=True)</code>
</pre>
</div>



We can take a look at the actual codes for the categories:


```python
df['tier'].cat.codes.head(10)
```


<div class="output_block">
<pre>
<code class="codeblock">name
Abomasnow          5
Abomasnow-Mega     4
Absol              5
Absol-Mega         2
Accelgor           3
Aegislash          0
Aegislash-Blade    0
Aerodactyl         4
Aerodactyl-Mega    2
Aggron             5
dtype: int8</code>
</pre>
</div>




```python
list(zip(df['tier'].head(10), df['tier'].cat.codes.head(10)))
```


<div class="output_block">
<pre>
<code class="codeblock">[('PU', 5),
 ('RU', 4),
 ('PU', 5),
 ('UU', 2),
 ('NU', 3),
 ('Uber', 0),
 ('Uber', 0),
 ('RU', 4),
 ('UU', 2),
 ('PU', 5)]</code>
</pre>
</div>



## (very light) Feature Engineering

Let's make a feature counting the number of moves a pokemon can learn.


```python
df['num_moves'] = df.moves.map(len)
```

The base stat total is a common metric players use to assess a Pokemon's overall strength, so let's create a column for this.


```python
df['bst'] = (df['hp'] + df['atk'] + df['def'] + df['spa'] + df['spd']
             + df['spe'])
```

## Anomaly Analysis

This workshop is about data cleaning, but a useful way to look for data issues, gain ideas for feature engineering, and understand your data is to look at anomalies. Plus, we can look at some new pandas techniques.

Let's look at information about the BST by tier:


```python
bstdf = df[['tier', 'bst']].groupby('tier').agg([np.mean, np.std])
bstdf
```




<div style="overflow-x:auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">bst</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>std</th>
    </tr>
    <tr>
      <th>tier</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>NU</th>
      <td>495.132353</td>
      <td>36.655681</td>
    </tr>
    <tr>
      <th>OU</th>
      <td>565.896104</td>
      <td>68.916155</td>
    </tr>
    <tr>
      <th>PU</th>
      <td>464.184685</td>
      <td>59.964976</td>
    </tr>
    <tr>
      <th>RU</th>
      <td>524.486111</td>
      <td>48.101124</td>
    </tr>
    <tr>
      <th>UU</th>
      <td>538.181818</td>
      <td>50.624685</td>
    </tr>
    <tr>
      <th>Uber</th>
      <td>657.042553</td>
      <td>67.435946</td>
    </tr>
  </tbody>
</table>
</div>



First, we get a dataframe containing each pokemon's tier and base stat total. We want the mean and standard deviation of the BST's by tier. So, we group by the tier. In pandas, we can group by multiple columns if you want. Then, we apply aggregate function mean and std. This will calculate mean and std within each tier.

You'll notice that we now have a multiindex for the columns. We will not cover this in this workshop, so we will just simplify the multiindex.


```python
bstdf.columns = ['bst_mean', 'bst_std']
bstdf
```




<div style="overflow-x:auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bst_mean</th>
      <th>bst_std</th>
    </tr>
    <tr>
      <th>tier</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>NU</th>
      <td>495.132353</td>
      <td>36.655681</td>
    </tr>
    <tr>
      <th>OU</th>
      <td>565.896104</td>
      <td>68.916155</td>
    </tr>
    <tr>
      <th>PU</th>
      <td>464.184685</td>
      <td>59.964976</td>
    </tr>
    <tr>
      <th>RU</th>
      <td>524.486111</td>
      <td>48.101124</td>
    </tr>
    <tr>
      <th>UU</th>
      <td>538.181818</td>
      <td>50.624685</td>
    </tr>
    <tr>
      <th>Uber</th>
      <td>657.042553</td>
      <td>67.435946</td>
    </tr>
  </tbody>
</table>
</div>



The main ways to join tables in pandas are join and merge. Join is typically used to join on an index. For example, if you had two tables with the pokemon name as the index, you can do df1.join(df2), and this will horizontally concatenate the tables based on index.

I will show you how to use merge, which is the most general and easiest to understand joining method (though not always the fastest).


```python
df2 = df.reset_index().merge(bstdf, left_on='tier', right_on='tier', 
                             how='left')
# equivalent to bstdf.merge(df, ..., how='right')
df2.sample()
```




<div style="overflow-x:auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>types</th>
      <th>abilities</th>
      <th>tier</th>
      <th>hp</th>
      <th>atk</th>
      <th>def</th>
      <th>spa</th>
      <th>spd</th>
      <th>spe</th>
      <th>next_evos</th>
      <th>moves</th>
      <th>num_moves</th>
      <th>bst</th>
      <th>bst_mean</th>
      <th>bst_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>91</th>
      <td>Crabominable</td>
      <td>[Fighting, Ice]</td>
      <td>[Anger Point, Hyper Cutter, Iron Fist]</td>
      <td>PU</td>
      <td>97</td>
      <td>132</td>
      <td>77</td>
      <td>62</td>
      <td>67</td>
      <td>43</td>
      <td>[]</td>
      <td>{Fling, Bubble Beam, Iron Defense, Hidden Powe...</td>
      <td>54</td>
      <td>478</td>
      <td>464.184685</td>
      <td>59.964976</td>
    </tr>
  </tbody>
</table>
</div>



Basically, pandas looked for where the tier in df equaled tier in bstdf and concatenated those rows. left_on is the column for df, right_on is the column for bstdf (in this case they're the same). You can learn more about how joins work in this article: https://blog.codinghorror.com/a-visual-explanation-of-sql-joins/. The concepts carry over to pandas.

We want to look at anomalous pokemon who's stats seem too low for their tiers. Let's accomplish this:


```python
under = df2[(df2['bst'] < df2['bst_mean'] - 2*df2['bst_std']) 
            & (df2['tier'] != 'PU')]
under
```




<div style="overflow-x:auto; overflow-y:auto; height: 300px;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>types</th>
      <th>abilities</th>
      <th>tier</th>
      <th>hp</th>
      <th>atk</th>
      <th>def</th>
      <th>spa</th>
      <th>spd</th>
      <th>spe</th>
      <th>next_evos</th>
      <th>moves</th>
      <th>num_moves</th>
      <th>bst</th>
      <th>bst_mean</th>
      <th>bst_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Aegislash</td>
      <td>[Steel, Ghost]</td>
      <td>[Stance Change]</td>
      <td>Uber</td>
      <td>60</td>
      <td>50</td>
      <td>150</td>
      <td>50</td>
      <td>150</td>
      <td>60</td>
      <td>[]</td>
      <td>{Hidden Power, Iron Defense, Hyper Beam, Pursu...</td>
      <td>55</td>
      <td>520</td>
      <td>657.042553</td>
      <td>67.435946</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Aegislash-Blade</td>
      <td>[Steel, Ghost]</td>
      <td>[Stance Change]</td>
      <td>Uber</td>
      <td>60</td>
      <td>150</td>
      <td>50</td>
      <td>150</td>
      <td>50</td>
      <td>60</td>
      <td>[]</td>
      <td>{Hidden Power, Iron Defense, Hyper Beam, Pursu...</td>
      <td>55</td>
      <td>520</td>
      <td>657.042553</td>
      <td>67.435946</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Azumarill</td>
      <td>[Water, Fairy]</td>
      <td>[Huge Power, Sap Sipper, Thick Fat]</td>
      <td>OU</td>
      <td>100</td>
      <td>50</td>
      <td>80</td>
      <td>60</td>
      <td>80</td>
      <td>50</td>
      <td>[]</td>
      <td>{Muddy Water, Swagger, Water Pulse, Ice Beam, ...</td>
      <td>96</td>
      <td>420</td>
      <td>565.896104</td>
      <td>68.916155</td>
    </tr>
    <tr>
      <th>115</th>
      <td>Diggersby</td>
      <td>[Normal, Ground]</td>
      <td>[Cheek Pouch, Huge Power, Pickup]</td>
      <td>OU</td>
      <td>85</td>
      <td>56</td>
      <td>77</td>
      <td>50</td>
      <td>77</td>
      <td>78</td>
      <td>[]</td>
      <td>{Rollout, Sandstorm, Fling, Earthquake, Hidden...</td>
      <td>81</td>
      <td>423</td>
      <td>565.896104</td>
      <td>68.916155</td>
    </tr>
    <tr>
      <th>267</th>
      <td>Linoone</td>
      <td>[Normal]</td>
      <td>[Gluttony, Pickup, Quick Feet]</td>
      <td>RU</td>
      <td>78</td>
      <td>70</td>
      <td>61</td>
      <td>50</td>
      <td>61</td>
      <td>100</td>
      <td>[]</td>
      <td>{Thunder Wave, Super Fang, Swagger, Water Puls...</td>
      <td>89</td>
      <td>420</td>
      <td>524.486111</td>
      <td>48.101124</td>
    </tr>
    <tr>
      <th>298</th>
      <td>Marowak-Alola</td>
      <td>[Fire, Ghost]</td>
      <td>[Cursed Body, Lightning Rod, Rock Head]</td>
      <td>UU</td>
      <td>60</td>
      <td>80</td>
      <td>110</td>
      <td>50</td>
      <td>80</td>
      <td>45</td>
      <td>[]</td>
      <td>{Tail Whip, Sandstorm, Fling, Hidden Power, Hy...</td>
      <td>74</td>
      <td>425</td>
      <td>538.181818</td>
      <td>50.624685</td>
    </tr>
    <tr>
      <th>303</th>
      <td>Medicham</td>
      <td>[Fighting, Psychic]</td>
      <td>[Pure Power, Telepathy]</td>
      <td>NU</td>
      <td>60</td>
      <td>60</td>
      <td>75</td>
      <td>60</td>
      <td>75</td>
      <td>80</td>
      <td>[]</td>
      <td>{Rock Slide, Swagger, Meditate, Confusion, Gra...</td>
      <td>96</td>
      <td>410</td>
      <td>495.132353</td>
      <td>36.655681</td>
    </tr>
    <tr>
      <th>521</th>
      <td>Vivillon</td>
      <td>[Bug, Flying]</td>
      <td>[Compound Eyes, Friend Guard, Shield Dust]</td>
      <td>NU</td>
      <td>80</td>
      <td>52</td>
      <td>50</td>
      <td>90</td>
      <td>50</td>
      <td>89</td>
      <td>[]</td>
      <td>{Hidden Power, Iron Defense, Hyper Beam, Rest,...</td>
      <td>59</td>
      <td>411</td>
      <td>495.132353</td>
      <td>36.655681</td>
    </tr>
  </tbody>
</table>
</div>



## Misc.

Pandas also has built in graphing functionalities which behave identically to matplotlib. For example:


```python
df.bst.hist()
```



![png](/assets/images/posts/intro-to-pandas/Introduction_to_Data_Cleaning_with_Pandas_96_1.png#center)



```python
df.plot.scatter('bst', 'atk')
```




![png](/assets/images/posts/intro-to-pandas/Introduction_to_Data_Cleaning_with_Pandas_97_1.png#center)


Finally, we can "pivot" tables as you would in excel. This provides a summary of the data.


```python
df['type_1'] = df['types'].map(lambda x: x[0])

pd.pivot_table(df, index='tier', columns='type_1', values='bst', 
               aggfunc='mean')
```


<div style="overflow-x:auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>type_1</th>
      <th>Bug</th>
      <th>Dark</th>
      <th>Dragon</th>
      <th>Electric</th>
      <th>Fairy</th>
      <th>Fighting</th>
      <th>Fire</th>
      <th>Flying</th>
      <th>Ghost</th>
      <th>Grass</th>
      <th>Ground</th>
      <th>Ice</th>
      <th>Normal</th>
      <th>Poison</th>
      <th>Psychic</th>
      <th>Rock</th>
      <th>Steel</th>
      <th>Water</th>
    </tr>
    <tr>
      <th>tier</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>NU</th>
      <td>476.500000</td>
      <td>494.000000</td>
      <td>487.500000</td>
      <td>460.500000</td>
      <td>473.500000</td>
      <td>469.625000</td>
      <td>534.400000</td>
      <td>479.000000</td>
      <td>483.750000</td>
      <td>506.250000</td>
      <td>486.250</td>
      <td>525.000000</td>
      <td>495.400000</td>
      <td>457.0</td>
      <td>520.000000</td>
      <td>519.50</td>
      <td>520.000000</td>
      <td>520.750000</td>
    </tr>
    <tr>
      <th>OU</th>
      <td>567.500000</td>
      <td>520.000000</td>
      <td>644.444444</td>
      <td>562.142857</td>
      <td>483.000000</td>
      <td>524.250000</td>
      <td>607.600000</td>
      <td>518.333333</td>
      <td>476.000000</td>
      <td>542.166667</td>
      <td>519.000</td>
      <td>505.000000</td>
      <td>497.000000</td>
      <td>495.0</td>
      <td>598.250000</td>
      <td>700.00</td>
      <td>550.000000</td>
      <td>576.428571</td>
    </tr>
    <tr>
      <th>PU</th>
      <td>426.521739</td>
      <td>448.300000</td>
      <td>NaN</td>
      <td>473.800000</td>
      <td>392.666667</td>
      <td>461.333333</td>
      <td>485.454545</td>
      <td>447.090909</td>
      <td>479.400000</td>
      <td>478.476190</td>
      <td>457.875</td>
      <td>511.727273</td>
      <td>457.342857</td>
      <td>472.6</td>
      <td>465.266667</td>
      <td>494.00</td>
      <td>380.000000</td>
      <td>459.000000</td>
    </tr>
    <tr>
      <th>RU</th>
      <td>490.166667</td>
      <td>510.000000</td>
      <td>536.500000</td>
      <td>543.750000</td>
      <td>516.000000</td>
      <td>527.000000</td>
      <td>573.333333</td>
      <td>495.000000</td>
      <td>518.333333</td>
      <td>546.500000</td>
      <td>480.000</td>
      <td>552.500000</td>
      <td>523.571429</td>
      <td>487.0</td>
      <td>545.428571</td>
      <td>505.75</td>
      <td>546.000000</td>
      <td>538.500000</td>
    </tr>
    <tr>
      <th>UU</th>
      <td>485.800000</td>
      <td>531.714286</td>
      <td>598.000000</td>
      <td>540.000000</td>
      <td>525.000000</td>
      <td>531.500000</td>
      <td>517.000000</td>
      <td>536.250000</td>
      <td>500.000000</td>
      <td>586.000000</td>
      <td>512.250</td>
      <td>NaN</td>
      <td>559.500000</td>
      <td>507.5</td>
      <td>547.500000</td>
      <td>585.00</td>
      <td>543.333333</td>
      <td>544.166667</td>
    </tr>
    <tr>
      <th>Uber</th>
      <td>585.000000</td>
      <td>640.000000</td>
      <td>686.800000</td>
      <td>NaN</td>
      <td>680.000000</td>
      <td>612.500000</td>
      <td>613.333333</td>
      <td>626.666667</td>
      <td>600.000000</td>
      <td>NaN</td>
      <td>720.000</td>
      <td>NaN</td>
      <td>655.000000</td>
      <td>540.0</td>
      <td>682.153846</td>
      <td>NaN</td>
      <td>580.000000</td>
      <td>720.000000</td>
    </tr>
  </tbody>
</table>
</div>


## Conclusion

Through this workshop, we've seen an overview of pandas and how it can be useful for data preprocessing. Next, we can use these skills to analyze and model our data using [random forests in scikit-learn](https://nicholasvadivelu.com/2019/09/27/intro-to-random-forests/).