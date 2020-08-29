---
layout: post
title: Introduction to Random Forests
---

Through this workshop you will learn how to quickly model and understand datasets using scikit-learn. Topics will include a basic introduction to using decisions trees and random forests, understanding feature importance, identifying model weaknesses, explaining your model, and more.

If you are not familiar with pandas, check out [this](https://nicholasvadivelu.com/2019/09/27/intro-to-pandas/) post first. This article contains some high level explanations of the code not covered in the live workshop, but skips some examples given in the main presentation. [Here](https://youtu.be/ANdn3CF4bss) is a recording of the main workshop, [here](https://docs.google.com/presentation/d/14e5iw-AswCbli4YUHZI1d9OYu4csM2nyxv84Ytu68QY) are the slides, and [here](https://colab.research.google.com/github/n2cholas/dsc-workshops/blob/master/Random_Forests_Workshop_V2.ipynb) is a colab link to run all the code.

Many of the model interpretation techniques were taken from the [fast.ai ML course](http://course18.fast.ai/ml). Check it out!!

Indented plain-text blocks in this article contain outputs from the previous code block.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```

## Data Wrangling

Data munging/wrangling is the process of transforming your raw data into a form suitable for analysis or modelling. Typically this involves dealing with null values, finding (and possibly correcting) data issues, transforming non-numerical data into numerical data, and more.

In the "real world", you will need to do much more than is shown below. There are numerous Kaggle Kernels demonstrating in depth data cleaning. For the purposes of this workshop, we will do (less than) the bare minimum for two reasons:

1. We are using a relatively clean dataset (with few data oddities).
2. Our model analysis will tell us where to focus our data cleaning and feature engineering efforts.
3. This workshop is only an hour long :P.

[Here](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python) is an example of a great example of some data exploration and correction. 


```python
def show_all(df):
    """Shows our dataframe without cutting off any rows or columns."""
    with pd.option_context("display.max_rows", 1000,
                           "display.max_columns", 1000): 
        display(df)
```

First, we take a look at our data to get a sense of the types of values in the columns. We do this by looking at some of our data and using numerical summaries.


```python
df = pd.read_csv('https://raw.githubusercontent.com/n2cholas/dsc-workshops/master/Random%20Forest%20Workshop/data/train.csv')
show_all(df.head())
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>196.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>706</td>
      <td>Unf</td>
      <td>0</td>
      <td>150</td>
      <td>856</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>1710</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>2003.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>548</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Gd</td>
      <td>ALQ</td>
      <td>978</td>
      <td>Unf</td>
      <td>0</td>
      <td>284</td>
      <td>1262</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>1262</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1976.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>460</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>162.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>486</td>
      <td>Unf</td>
      <td>0</td>
      <td>434</td>
      <td>920</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>1786</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2001.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>608</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>TA</td>
      <td>Gd</td>
      <td>No</td>
      <td>ALQ</td>
      <td>216</td>
      <td>Unf</td>
      <td>0</td>
      <td>540</td>
      <td>756</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>1717</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>1998.0</td>
      <td>Unf</td>
      <td>3</td>
      <td>642</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>350.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>GLQ</td>
      <td>655</td>
      <td>Unf</td>
      <td>0</td>
      <td>490</td>
      <td>1145</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>2198</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>Gd</td>
      <td>9</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2000.0</td>
      <td>RFn</td>
      <td>3</td>
      <td>836</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
</div>



```python
len(df)
```
<div class="output_block">
<pre>
<code class="codeblock">1460</code>
</pre>
</div>


```python
show_all(df.describe())
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>GarageYrBlt</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1201.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1452.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1379.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>730.500000</td>
      <td>56.897260</td>
      <td>70.049958</td>
      <td>10516.828082</td>
      <td>6.099315</td>
      <td>5.575342</td>
      <td>1971.267808</td>
      <td>1984.865753</td>
      <td>103.685262</td>
      <td>443.639726</td>
      <td>46.549315</td>
      <td>567.240411</td>
      <td>1057.429452</td>
      <td>1162.626712</td>
      <td>346.992466</td>
      <td>5.844521</td>
      <td>1515.463699</td>
      <td>0.425342</td>
      <td>0.057534</td>
      <td>1.565068</td>
      <td>0.382877</td>
      <td>2.866438</td>
      <td>1.046575</td>
      <td>6.517808</td>
      <td>0.613014</td>
      <td>1978.506164</td>
      <td>1.767123</td>
      <td>472.980137</td>
      <td>94.244521</td>
      <td>46.660274</td>
      <td>21.954110</td>
      <td>3.409589</td>
      <td>15.060959</td>
      <td>2.758904</td>
      <td>43.489041</td>
      <td>6.321918</td>
      <td>2007.815753</td>
      <td>180921.195890</td>
    </tr>
    <tr>
      <th>std</th>
      <td>421.610009</td>
      <td>42.300571</td>
      <td>24.284752</td>
      <td>9981.264932</td>
      <td>1.382997</td>
      <td>1.112799</td>
      <td>30.202904</td>
      <td>20.645407</td>
      <td>181.066207</td>
      <td>456.098091</td>
      <td>161.319273</td>
      <td>441.866955</td>
      <td>438.705324</td>
      <td>386.587738</td>
      <td>436.528436</td>
      <td>48.623081</td>
      <td>525.480383</td>
      <td>0.518911</td>
      <td>0.238753</td>
      <td>0.550916</td>
      <td>0.502885</td>
      <td>0.815778</td>
      <td>0.220338</td>
      <td>1.625393</td>
      <td>0.644666</td>
      <td>24.689725</td>
      <td>0.747315</td>
      <td>213.804841</td>
      <td>125.338794</td>
      <td>66.256028</td>
      <td>61.119149</td>
      <td>29.317331</td>
      <td>55.757415</td>
      <td>40.177307</td>
      <td>496.123024</td>
      <td>2.703626</td>
      <td>1.328095</td>
      <td>79442.502883</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>334.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>334.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1900.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>34900.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>365.750000</td>
      <td>20.000000</td>
      <td>59.000000</td>
      <td>7553.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1967.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>223.000000</td>
      <td>795.750000</td>
      <td>882.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1129.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>1961.000000</td>
      <td>1.000000</td>
      <td>334.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2007.000000</td>
      <td>129975.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>730.500000</td>
      <td>50.000000</td>
      <td>69.000000</td>
      <td>9478.500000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1994.000000</td>
      <td>0.000000</td>
      <td>383.500000</td>
      <td>0.000000</td>
      <td>477.500000</td>
      <td>991.500000</td>
      <td>1087.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1464.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>1980.000000</td>
      <td>2.000000</td>
      <td>480.000000</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>163000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1095.250000</td>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11601.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2000.000000</td>
      <td>2004.000000</td>
      <td>166.000000</td>
      <td>712.250000</td>
      <td>0.000000</td>
      <td>808.000000</td>
      <td>1298.250000</td>
      <td>1391.250000</td>
      <td>728.000000</td>
      <td>0.000000</td>
      <td>1776.750000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>2002.000000</td>
      <td>2.000000</td>
      <td>576.000000</td>
      <td>168.000000</td>
      <td>68.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>214000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1460.000000</td>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>1474.000000</td>
      <td>2336.000000</td>
      <td>6110.000000</td>
      <td>4692.000000</td>
      <td>2065.000000</td>
      <td>572.000000</td>
      <td>5642.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>3.000000</td>
      <td>14.000000</td>
      <td>3.000000</td>
      <td>2010.000000</td>
      <td>4.000000</td>
      <td>1418.000000</td>
      <td>857.000000</td>
      <td>547.000000</td>
      <td>552.000000</td>
      <td>508.000000</td>
      <td>480.000000</td>
      <td>738.000000</td>
      <td>15500.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
</div>


We want to ensure all our columns (or features) have numerical or categorical values. Let's look at the data types of each column.


```python
show_all(df.dtypes)
```


<div class="output_block">
<pre>
<code class="codeblock">Id                 int64
MSSubClass         int64
MSZoning          object
LotFrontage      float64
LotArea            int64
Street            object
Alley             object
LotShape          object
LandContour       object
Utilities         object
LotConfig         object
LandSlope         object
Neighborhood      object
Condition1        object
Condition2        object
BldgType          object
HouseStyle        object
OverallQual        int64
OverallCond        int64
YearBuilt          int64
YearRemodAdd       int64
RoofStyle         object
RoofMatl          object
Exterior1st       object
Exterior2nd       object
MasVnrType        object
MasVnrArea       float64
ExterQual         object
ExterCond         object
Foundation        object
BsmtQual          object
BsmtCond          object
BsmtExposure      object
BsmtFinType1      object
BsmtFinSF1         int64
BsmtFinType2      object
BsmtFinSF2         int64
BsmtUnfSF          int64
TotalBsmtSF        int64
Heating           object
HeatingQC         object
CentralAir        object
Electrical        object
1stFlrSF           int64
2ndFlrSF           int64
LowQualFinSF       int64
GrLivArea          int64
BsmtFullBath       int64
BsmtHalfBath       int64
FullBath           int64
HalfBath           int64
BedroomAbvGr       int64
KitchenAbvGr       int64
KitchenQual       object
TotRmsAbvGrd       int64
Functional        object
Fireplaces         int64
FireplaceQu       object
GarageType        object
GarageYrBlt      float64
GarageFinish      object
GarageCars         int64
GarageArea         int64
GarageQual        object
GarageCond        object
PavedDrive        object
WoodDeckSF         int64
OpenPorchSF        int64
EnclosedPorch      int64
3SsnPorch          int64
ScreenPorch        int64
PoolArea           int64
PoolQC            object
Fence             object
MiscFeature       object
MiscVal            int64
MoSold             int64
YrSold             int64
SaleType          object
SaleCondition     object
SalePrice          int64
dtype: object</code>
</pre>
</div>


Most of the data is of type `object`. In general, this will be a Python string but could also be other Python objects, such as lists or dicts. Just based on the names, it looks like these are all categorical features. Let's check this by looking at the unique values in each column.


```python
for c in df.columns[df.dtypes=='object']:
    print(df[c].value_counts())
    print()
```

<div class="output_block">
<pre>
<code class="codeblock">RL         1151
RM          218
FV           65
RH           16
C (all)      10
Name: MSZoning, dtype: int64

Pave    1454
Grvl       6
Name: Street, dtype: int64

Grvl    50
Pave    41
Name: Alley, dtype: int64

Reg    925
IR1    484
IR2     41
IR3     10
Name: LotShape, dtype: int64

Lvl    1311
Bnk      63
HLS      50
Low      36
Name: LandContour, dtype: int64

AllPub    1459
NoSeWa       1
Name: Utilities, dtype: int64

Inside     1052
Corner      263
CulDSac      94
FR2          47
FR3           4
Name: LotConfig, dtype: int64

Gtl    1382
Mod      65
Sev      13
Name: LandSlope, dtype: int64

NAmes      225
CollgCr    150
OldTown    113
Edwards    100
Somerst     86
Gilbert     79
NridgHt     77
Sawyer      74
NWAmes      73
SawyerW     59
BrkSide     58
Crawfor     51
Mitchel     49
NoRidge     41
Timber      38
IDOTRR      37
ClearCr     28
StoneBr     25
SWISU       25
MeadowV     17
Blmngtn     17
BrDale      16
Veenker     11
NPkVill      9
Blueste      2
Name: Neighborhood, dtype: int64

Norm      1260
Feedr       81
Artery      48
RRAn        26
PosN        19
RRAe        11
PosA         8
RRNn         5
RRNe         2
Name: Condition1, dtype: int64

Norm      1445
Feedr        6
RRNn         2
PosN         2
Artery       2
RRAe         1
PosA         1
RRAn         1
Name: Condition2, dtype: int64

1Fam      1220
TwnhsE     114
Duplex      52
Twnhs       43
2fmCon      31
Name: BldgType, dtype: int64

1Story    726
2Story    445
1.5Fin    154
SLvl       65
SFoyer     37
1.5Unf     14
2.5Unf     11
2.5Fin      8
Name: HouseStyle, dtype: int64

Gable      1141
Hip         286
Flat         13
Gambrel      11
Mansard       7
Shed          2
Name: RoofStyle, dtype: int64

CompShg    1434
Tar&Grv      11
WdShngl       6
WdShake       5
Membran       1
Roll          1
ClyTile       1
Metal         1
Name: RoofMatl, dtype: int64

VinylSd    515
HdBoard    222
MetalSd    220
Wd Sdng    206
Plywood    108
CemntBd     61
BrkFace     50
WdShing     26
Stucco      25
AsbShng     20
Stone        2
BrkComm      2
AsphShn      1
ImStucc      1
CBlock       1
Name: Exterior1st, dtype: int64

VinylSd    504
MetalSd    214
HdBoard    207
Wd Sdng    197
Plywood    142
CmentBd     60
Wd Shng     38
Stucco      26
BrkFace     25
AsbShng     20
ImStucc     10
Brk Cmn      7
Stone        5
AsphShn      3
CBlock       1
Other        1
Name: Exterior2nd, dtype: int64

None       864
BrkFace    445
Stone      128
BrkCmn      15
Name: MasVnrType, dtype: int64

TA    906
Gd    488
Ex     52
Fa     14
Name: ExterQual, dtype: int64

TA    1282
Gd     146
Fa      28
Ex       3
Po       1
Name: ExterCond, dtype: int64

PConc     647
CBlock    634
BrkTil    146
Slab       24
Stone       6
Wood        3
Name: Foundation, dtype: int64

TA    649
Gd    618
Ex    121
Fa     35
Name: BsmtQual, dtype: int64

TA    1311
Gd      65
Fa      45
Po       2
Name: BsmtCond, dtype: int64

No    953
Av    221
Gd    134
Mn    114
Name: BsmtExposure, dtype: int64

Unf    430
GLQ    418
ALQ    220
BLQ    148
Rec    133
LwQ     74
Name: BsmtFinType1, dtype: int64

Unf    1256
Rec      54
LwQ      46
BLQ      33
ALQ      19
GLQ      14
Name: BsmtFinType2, dtype: int64

GasA     1428
GasW       18
Grav        7
Wall        4
OthW        2
Floor       1
Name: Heating, dtype: int64

Ex    741
TA    428
Gd    241
Fa     49
Po      1
Name: HeatingQC, dtype: int64

Y    1365
N      95
Name: CentralAir, dtype: int64

SBrkr    1334
FuseA      94
FuseF      27
FuseP       3
Mix         1
Name: Electrical, dtype: int64

TA    735
Gd    586
Ex    100
Fa     39
Name: KitchenQual, dtype: int64

Typ     1360
Min2      34
Min1      31
Mod       15
Maj1      14
Maj2       5
Sev        1
Name: Functional, dtype: int64

Gd    380
TA    313
Fa     33
Ex     24
Po     20
Name: FireplaceQu, dtype: int64

Attchd     870
Detchd     387
BuiltIn     88
Basment     19
CarPort      9
2Types       6
Name: GarageType, dtype: int64

Unf    605
RFn    422
Fin    352
Name: GarageFinish, dtype: int64

TA    1311
Fa      48
Gd      14
Ex       3
Po       3
Name: GarageQual, dtype: int64

TA    1326
Fa      35
Gd       9
Po       7
Ex       2
Name: GarageCond, dtype: int64

Y    1340
N      90
P      30
Name: PavedDrive, dtype: int64

Gd    3
Fa    2
Ex    2
Name: PoolQC, dtype: int64

MnPrv    157
GdPrv     59
GdWo      54
MnWw      11
Name: Fence, dtype: int64

Shed    49
Gar2     2
Othr     2
TenC     1
Name: MiscFeature, dtype: int64

WD       1267
New       122
COD        43
ConLD       9
ConLI       5
ConLw       5
CWD         4
Oth         3
Con         2
Name: SaleType, dtype: int64

Normal     1198
Partial     125
Abnorml     101
Family       20
Alloca       12
AdjLand       4
Name: SaleCondition, dtype: int64 </code>
</pre>
</div>
    


Great, now we know all of our object columns contain categories. In reality, this is rarely the case and you would have to do additional cleaning steps.

We should also note that some of these categories are ordered, such as ExterQual and ExterCond.

Also, you may have a dataset with thousands of features, so it is infeasible to look through all the categories like we did. In this case you can look at the number of unique values for each feature. Features with many unique values might be numerical values that weren't properly encoded or free form text, which would have to be dealt with otherwise.

Next thing we want to do is deal with null values. Let's see which columns have null values by showing the proportion of values in each column that are null.


```python
x = (df.isnull().sum()/len(df)).to_frame(name='perc_null')
x['types'] = df.dtypes
x[x.perc_null>0].sort_values('perc_null', ascending=False)
```



<div style="overflow-y:auto; height: 300px;">
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
      <th>perc_null</th>
      <th>types</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PoolQC</th>
      <td>0.995205</td>
      <td>object</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>0.963014</td>
      <td>object</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>0.937671</td>
      <td>object</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>0.807534</td>
      <td>object</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>0.472603</td>
      <td>object</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>0.177397</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>0.055479</td>
      <td>object</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>0.055479</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>0.055479</td>
      <td>object</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>0.055479</td>
      <td>object</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>0.055479</td>
      <td>object</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>0.026027</td>
      <td>object</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>0.026027</td>
      <td>object</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>0.025342</td>
      <td>object</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>0.025342</td>
      <td>object</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>0.025342</td>
      <td>object</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>0.005479</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>0.005479</td>
      <td>object</td>
    </tr>
    <tr>
      <th>Electrical</th>
      <td>0.000685</td>
      <td>object</td>
    </tr>
  </tbody>
</table>
</div>



For the categorical variables, we will simply add a null category, when we turn them into categorical variables, pandas will automatically create a nan category if needed.

Dealing with nans (not a numbers) in numerical columns is more challenging. You typically want to replace the nans with a default value. Is there a reasonable default for this column? Does 0 make sense? What about the min/max/median? There is much discussion about this on the web.

Here, PoolQC and MasVnrArea are null when the house does not have a pool, so it makes sense to fill in 0 for these columns. For LotFrontage and GarageYrBuilt, we use the median. For the latter two, the missing information may have some pattern that helps us predict the price, so we will create an indicator column that tells us whether the column was null or not. This could be useful if, for example, all houses without a LotFrontage had a very small Lot, so they have a lower cost, so the indicator would help our model learn that this is the case.


```python
df.PoolQC.fillna(0, inplace=True)
df.MasVnrArea.fillna(0, inplace=True)

def fill_nulls(col, filler=np.nanmedian):
    df[f'{col}_null'] = 0
    df.loc[df[col].isnull(), f'{col}_null'] = 1
    df[col].fillna(filler(df[col]), inplace=True)

fill_nulls('LotFrontage')
fill_nulls('GarageYrBlt')
```

Now, the "object" dtype columns are ready to be turned into actual categories. You can read more about the "category" dtype on the [pandas documentation page](https://pandas.pydata.org/pandas-docs/version/0.24.2/reference/api/pandas.Categorical.html). Essentially, it changes all the values from strings to numbers, so our model can use them.

For a vast majority of machine learning model types, we would need to do one additional step and "one-hot encode" these categorical variables. Since we are using a tree-based model, we will not need to do this, so we will not cover it.


```python
for col in df.columns[df.dtypes=='object']:
    df[col] = df[col].astype('category')
```

We noted above that some of the categorical features were ordered. We will encode two of them as ordered categorical columns, but leave the rest as an exercise.


```python
from pandas.api.types import CategoricalDtype

order = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
cat_type = CategoricalDtype(categories=order, ordered=True)
df['ExterQual'] = df['ExterQual'].astype(cat_type)
df['ExterCond'] = df['ExterCond'].astype(cat_type)
```

We can take a look at some of the category codes along with the original strings:


```python
print(df['ExterQual'][:10].cat.codes)
print(df['ExterQual'][:10])
```

<div class="output_block">
<pre>
<code class="codeblock">0    3
1    2
2    3
3    2
4    3
5    2
6    3
7    2
8    2
9    2
dtype: int8
0    Gd
1    TA
2    Gd
3    TA
4    Gd
5    TA
6    Gd
7    TA
8    TA
9    TA
Name: ExterQual, dtype: category
Categories (5, object): [Po < Fa < TA < Gd < Ex]</code>
</pre>
</div>


So far, we've done some basic data transformations so the data can be fed into our model. We did no feature engineering or data exploration. Your dataset could have thousands of dimensions, so it is infeasible to extensively explore the data before modelling. Our approach will be to let the model tell us what's important and what's not so we can focus our effots appropriately.

## Modeling


```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
```

Our goal is to predict the price of the houses given the rest of the information. We start by splitting our dataframe into our features (our model inputs) and our target (the price we are trying to predict).




```python
df_y, df_x = df['SalePrice'], df.drop('SalePrice', axis=1)
# same as axis='columns'
```

Scikit-learn expects numerical values for all the columns, but our categorical variables contain text. We will replace the text with their numeric category number:


```python
for col in df_x.columns[df_x.dtypes=='category']:
    df_x[col] = df_x[col].cat.codes
```

In machine learning, you typically split your data into a training, validation, and test set. You train your model on the training set, then evaluate the performance on the validation set. You use this validation set to tune hyperparameters (explained later). After selecting your model, you use the test set as one final check to make sure your model generalizes well (and you didn't just "overfit" to your validation set).

These concepts will be explained in more detail below. We split our data into just training and validation for simplicity, as we won't be doing any rigorous model selection.


```python
X_train, X_valid, y_train, y_valid = train_test_split(
    df_x, df_y, test_size=0.2, random_state=42)
```

Let's create some convenience functions to evaluate our model. Metric selection will not be discussed during this workshop. Here, we use the root mean squared error as our metric. Mean squared error is a common metric used for regression problems, and root mean squared error is nice in that it is in the same units as your original output.


```python
def rmse(x,y): 
    return np.sqrt(np.mean(((x-y)**2)))

def print_score(model, X_t=X_train, y_t=y_train, 
                       X_v=X_valid, y_v=y_valid):
    scores = [rmse(model.predict(X_t), y_t), 
              rmse(model.predict(X_v), y_v),
              model.score(X_t, y_t), 
              model.score(X_v, y_v)]
    
    labels = ['Train RMSE', 'Valid RMSE', 'Train R^2', 'Valid R^2']
    for t, s in zip(labels, scores):
        print(f'{t}: {s}')
```

Time to fit our first model: a DecisionTree! It's as easy as:


```python
tree = DecisionTreeRegressor(min_samples_leaf=50)
tree.fit(X_train, y_train)
```


We used the `min_samples_leaf` argument to limit the size of the tree so we can visualize it below:


```python
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
g = export_graphviz(tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names=X_train.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
```

![png](/assets/images/posts/intro-to-random-forests/Random_Forests_Workshop_V2_36_1.png#center)



Given the features, the decision tree works as follows: we start at the root and look at the feature in question. If the condition at the top is True, we follow the True branch, otherwise we follow the False branch. This process continues until we reach a leaf. The value at the leaf is our prediction. 

The tree is trained as follows. We take the data, and we look at all the features and values of these features to find the best split. The best split is the one such that our evaluation metric (mean squared error) is minimized by the split. In other words, if this split is used, the mean squared error of the model that comes from this split is minimized. The samples represents the number of samples of data that are in the subtree (due to the splits). The value is the mean target value of all the samples in that subtree. Then we repeat until each leaf only contains one sample (usually).


Let's build a full decision tree and evaluate its performance:


```python
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
print_score(model)
```

<div class="output_block">
<pre>
<code class="codeblock">Train RMSE: 0.0
Valid RMSE: 43111.0938316298
Train R^2: 1.0
Valid R^2: 0.7576939544477124</code>
</pre>
</div>


We see the Decision Tree has overfit. So, we introduce Random Forests:


```python
model = RandomForestRegressor(n_estimators=30, 
                              n_jobs=-1)
model.fit(X_train, y_train)
print_score(model)
```

<div class="output_block">
<pre>
<code class="codeblock">Train RMSE: 11972.136923038881
Valid RMSE: 29315.10749027229
Train R^2: 0.9759693433070632
Valid R^2: 0.8879610196550729</code>
</pre>
</div>


There are a few key hyperparameters to tune (shown below): 


```python
model = RandomForestRegressor(n_estimators=50, 
                              n_jobs=-1,
                              min_samples_leaf=1,
                              max_features=0.8) #'sqrt', 'log2'
model.fit(X_train, y_train)
print_score(model)
```

<div class="output_block">
<pre>
<code class="codeblock">Train RMSE: 11464.59450285856
Valid RMSE: 29159.246751559134
Train R^2: 0.9779636487670959
Valid R^2: 0.889149216323862</code>
</pre>
</div>


## Feature Importances

Now that we have a model with reasonable performance, we can use it to understand our dataset (and use that to improve the model). The first is looking at feature importance. 

To calculate feature x's importance, we first shuffle x column. Now, the feature is uncorrelated with the target. Next, we evaluate the model's performance on the training set and see how much it has decreased compared to when the feature was unshuffled. The higher this value, the more important the feature is.


```python  
# properties with an underscore are available after you fit the model
fi = pd.DataFrame(model.feature_importances_,
                  index = X_train.columns,
                  columns=['importance'])
fi = fi.sort_values('importance', ascending=False)
fi[:30].plot.barh(figsize=(12, 7))
```


![png](/assets/images/posts/intro-to-random-forests/Random_Forests_Workshop_V2_45_1.png#center)



```python
fi.loc['YearRemodAdd']
```


<div class="output_block">
<pre>
<code class="codeblock">importance    0.00591
Name: YearRemodAdd, dtype: float64</code>
</pre>
</div>



These importances tell you where to focus your attention on your dataset. Your data may have thousands of features, so having a way to narrow your investigation scope is very helpful. 

Here are a few things to think about:


*   Which features are the most important? This is a good opportunity to do some feature engineering. For example, if a date turned out to be an important feature, you may want to split this up into more granular features, such as splitting up the day/month/year, day of week, adding season, etc.
*   Which features are less important? Can we make them more useful through feature engineering?
*   Do the important features make sense? For example, suppose the data had an ID number that was assigned randomly, but turned out to be important. This is suspicious and you should look into the data and see why this is the case.
*   Are there are any features you expected to be important but aren't? This could indicate some data cleaning work you missed. If not, this is useful to build intuition about your dat.
*   If there are highly correlated features, which of the correlated features are important? How does removing the less important feature affect the model performance? 
*    And many more...

Let's try keeping only the important features and seeing how the model performs:




```python
X_train_new = X_train[fi[fi['importance'] > 0.005].index]
X_valid_new = X_valid[fi[fi['importance'] > 0.005].index]
```


```python
X_train.shape  # see how many features are left
```




<div class="output_block">
<pre>
<code class="codeblock">(1168, 82)</code>
</pre>
</div>




```python
model_new = RandomForestRegressor(n_estimators=30, 
                                  n_jobs=-1,
                                  oob_score=True,
                                  min_samples_leaf=3,
                                  max_features=0.5)
model_new.fit(X_train_new, y_train)
print_score(model_new, X_t=X_train_new, X_v=X_valid_new)
```

<div class="output_block">
<pre>
<code class="codeblock">Train RMSE: 17820.096007344346
Valid RMSE: 29815.94532060853
Train R^2: 0.9467594702883222
Valid R^2: 0.8841000276456914</code>
</pre>
</div>


We can compare the feature importances from the model trained with all the features vs the model with just some of the features.


```python
fi2 = pd.DataFrame(model_new.feature_importances_,
                   index = X_train_new.columns,
                   columns=['importance']).sort_values('importance', ascending=False)

fig, ax = plt.subplots(1,2, figsize=(25,8))
fi[:16].plot.barh(ax=ax[0], title='Nice')
fi2.plot.barh(ax=ax[1])
```




![png](/assets/images/posts/intro-to-random-forests/Random_Forests_Workshop_V2_52_1.png#center)




## Identifying Correlated Features

Identifying and resolving correlated features can sometimes improve your model in a few ways:

*   Decreases the dimensionality of your data
*   Can reduce overfitting
*   Prevents the model from learning unintended biases
*   Makes the model more interpretable (treeinterpeter/partial depencence plots will be more accurate)
*   And more.

For many model types, you would look at normal (linear) correlation, but a random forest is not affected by just linear correlation. Since the trees simply split the data at some point, just the **rank** of the feature matters. 

Consider you sort the data by a feature. The rank is the position of the example in the sorted dataset. Below, we identify rank correlations. 



```python
from scipy.cluster import hierarchy as hc
from scipy import stats
```


```python
df2 = X_train_new[
  X_train_new.columns[X_train_new.dtypes != 'category']
]
corr = np.round(stats.spearmanr(df2).correlation, 4)
fig, ax = plt.subplots(figsize=(10,8))
g = sns.heatmap(corr, ax=ax)
g.set_yticklabels(df2.columns, rotation=0)
g.set_xticklabels(df2.columns, rotation=90)
None
```


![png](/assets/images/posts/intro-to-random-forests/Random_Forests_Workshop_V2_57_0.png#center)



```python
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df2.columns, orientation='left', 
                           leaf_font_size=16)
plt.show()
```


![png](/assets/images/posts/intro-to-random-forests/Random_Forests_Workshop_V2_58_0.png#center)


## Tree Interpreter

Tree interpreter is a great way to understand the individual predictions. Recall for a Decision Tree, at each node, the model makes a "decision" based on a condition on a feature to follow the left child or the right child. Each node contains the average target value for the subset of data that satisfies the condition. The tree interpreter measures how much the average changes for each of these decisions for each feature (averaged across all trees). We call this average change the contribution of the feature.


```python
from treeinterpreter import treeinterpreter as ti

r = X_valid.values[None,0]
_, _, contributions = ti.predict(model, r)

ti_df = pd.DataFrame({
    'feature': X_valid.columns,
    'value': X_valid.iloc[0],
    'contributions': contributions[0],
})
show_all(ti_df.sort_values('contributions'))
```


<div style="overflow-y:auto; height: 300px;">
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
      <th>feature</th>
      <th>value</th>
      <th>contributions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>OverallQual</th>
      <td>OverallQual</td>
      <td>6.0</td>
      <td>-23204.618284</td>
    </tr>
    <tr>
      <th>GrLivArea</th>
      <td>GrLivArea</td>
      <td>1068.0</td>
      <td>-18869.146520</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>GarageArea</td>
      <td>264.0</td>
      <td>-6130.961614</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>GarageCars</td>
      <td>1.0</td>
      <td>-4230.235767</td>
    </tr>
    <tr>
      <th>ExterQual</th>
      <td>ExterQual</td>
      <td>2.0</td>
      <td>-3690.661634</td>
    </tr>
    <tr>
      <th>YearBuilt</th>
      <td>YearBuilt</td>
      <td>1963.0</td>
      <td>-1606.690435</td>
    </tr>
    <tr>
      <th>FullBath</th>
      <td>FullBath</td>
      <td>1.0</td>
      <td>-1549.810925</td>
    </tr>
    <tr>
      <th>MoSold</th>
      <td>MoSold</td>
      <td>2.0</td>
      <td>-1044.108978</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>GarageYrBlt</td>
      <td>1963.0</td>
      <td>-839.353054</td>
    </tr>
    <tr>
      <th>1stFlrSF</th>
      <td>1stFlrSF</td>
      <td>1068.0</td>
      <td>-761.773371</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>FireplaceQu</td>
      <td>-1.0</td>
      <td>-435.448301</td>
    </tr>
    <tr>
      <th>2ndFlrSF</th>
      <td>2ndFlrSF</td>
      <td>0.0</td>
      <td>-353.114027</td>
    </tr>
    <tr>
      <th>KitchenQual</th>
      <td>KitchenQual</td>
      <td>3.0</td>
      <td>-274.866357</td>
    </tr>
    <tr>
      <th>YrSold</th>
      <td>YrSold</td>
      <td>2006.0</td>
      <td>-260.455148</td>
    </tr>
    <tr>
      <th>HouseStyle</th>
      <td>HouseStyle</td>
      <td>2.0</td>
      <td>-241.007907</td>
    </tr>
    <tr>
      <th>HeatingQC</th>
      <td>HeatingQC</td>
      <td>4.0</td>
      <td>-224.041546</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>MasVnrArea</td>
      <td>0.0</td>
      <td>-184.002489</td>
    </tr>
    <tr>
      <th>Id</th>
      <td>Id</td>
      <td>893.0</td>
      <td>-171.481627</td>
    </tr>
    <tr>
      <th>Fireplaces</th>
      <td>Fireplaces</td>
      <td>0.0</td>
      <td>-142.288747</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>BsmtFullBath</td>
      <td>0.0</td>
      <td>-139.435641</td>
    </tr>
    <tr>
      <th>BsmtFinSF2</th>
      <td>BsmtFinSF2</td>
      <td>0.0</td>
      <td>-132.122053</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>BsmtExposure</td>
      <td>3.0</td>
      <td>-116.156015</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>BsmtQual</td>
      <td>3.0</td>
      <td>-111.225296</td>
    </tr>
    <tr>
      <th>Neighborhood</th>
      <td>Neighborhood</td>
      <td>19.0</td>
      <td>-76.460459</td>
    </tr>
    <tr>
      <th>ExterCond</th>
      <td>ExterCond</td>
      <td>2.0</td>
      <td>-58.762708</td>
    </tr>
    <tr>
      <th>HalfBath</th>
      <td>HalfBath</td>
      <td>0.0</td>
      <td>-38.991079</td>
    </tr>
    <tr>
      <th>SaleType</th>
      <td>SaleType</td>
      <td>8.0</td>
      <td>-27.675000</td>
    </tr>
    <tr>
      <th>LotFrontage_null</th>
      <td>LotFrontage_null</td>
      <td>0.0</td>
      <td>-21.285714</td>
    </tr>
    <tr>
      <th>ScreenPorch</th>
      <td>ScreenPorch</td>
      <td>0.0</td>
      <td>-17.295352</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>MiscFeature</td>
      <td>-1.0</td>
      <td>-17.083333</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>GarageFinish</td>
      <td>1.0</td>
      <td>-0.033333</td>
    </tr>
    <tr>
      <th>KitchenAbvGr</th>
      <td>KitchenAbvGr</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Electrical</th>
      <td>Electrical</td>
      <td>4.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LowQualFinSF</th>
      <td>LowQualFinSF</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Utilities</th>
      <td>Utilities</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LotShape</th>
      <td>LotShape</td>
      <td>3.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LandSlope</th>
      <td>LandSlope</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Heating</th>
      <td>Heating</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>GarageYrBlt_null</th>
      <td>GarageYrBlt_null</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Functional</th>
      <td>Functional</td>
      <td>6.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>BldgType</th>
      <td>BldgType</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Street</th>
      <td>Street</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3SsnPorch</th>
      <td>3SsnPorch</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>PoolArea</th>
      <td>PoolArea</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>PoolQC</th>
      <td>PoolQC</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>RoofMatl</th>
      <td>RoofMatl</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MiscVal</th>
      <td>MiscVal</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Condition2</th>
      <td>Condition2</td>
      <td>2.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TotRmsAbvGrd</th>
      <td>TotRmsAbvGrd</td>
      <td>6.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>Alley</td>
      <td>-1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>BsmtHalfBath</th>
      <td>BsmtHalfBath</td>
      <td>1.0</td>
      <td>4.666667</td>
    </tr>
    <tr>
      <th>LandContour</th>
      <td>LandContour</td>
      <td>3.0</td>
      <td>19.390152</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>MasVnrType</td>
      <td>2.0</td>
      <td>20.747510</td>
    </tr>
    <tr>
      <th>Condition1</th>
      <td>Condition1</td>
      <td>2.0</td>
      <td>27.096890</td>
    </tr>
    <tr>
      <th>Foundation</th>
      <td>Foundation</td>
      <td>1.0</td>
      <td>27.947817</td>
    </tr>
    <tr>
      <th>EnclosedPorch</th>
      <td>EnclosedPorch</td>
      <td>0.0</td>
      <td>28.570496</td>
    </tr>
    <tr>
      <th>PavedDrive</th>
      <td>PavedDrive</td>
      <td>2.0</td>
      <td>40.605275</td>
    </tr>
    <tr>
      <th>Exterior1st</th>
      <td>Exterior1st</td>
      <td>6.0</td>
      <td>50.324136</td>
    </tr>
    <tr>
      <th>LotConfig</th>
      <td>LotConfig</td>
      <td>4.0</td>
      <td>56.750000</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>BsmtFinType2</td>
      <td>5.0</td>
      <td>73.333333</td>
    </tr>
    <tr>
      <th>MSSubClass</th>
      <td>MSSubClass</td>
      <td>20.0</td>
      <td>80.537781</td>
    </tr>
    <tr>
      <th>OpenPorchSF</th>
      <td>OpenPorchSF</td>
      <td>0.0</td>
      <td>98.498287</td>
    </tr>
    <tr>
      <th>Exterior2nd</th>
      <td>Exterior2nd</td>
      <td>6.0</td>
      <td>99.801761</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>BsmtFinType1</td>
      <td>2.0</td>
      <td>151.642921</td>
    </tr>
    <tr>
      <th>RoofStyle</th>
      <td>RoofStyle</td>
      <td>3.0</td>
      <td>153.249065</td>
    </tr>
    <tr>
      <th>BedroomAbvGr</th>
      <td>BedroomAbvGr</td>
      <td>3.0</td>
      <td>161.383636</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>BsmtCond</td>
      <td>3.0</td>
      <td>255.793641</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>GarageCond</td>
      <td>4.0</td>
      <td>264.679058</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>Fence</td>
      <td>2.0</td>
      <td>321.183333</td>
    </tr>
    <tr>
      <th>MSZoning</th>
      <td>MSZoning</td>
      <td>3.0</td>
      <td>396.914474</td>
    </tr>
    <tr>
      <th>BsmtFinSF1</th>
      <td>BsmtFinSF1</td>
      <td>663.0</td>
      <td>449.878523</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>GarageQual</td>
      <td>4.0</td>
      <td>462.399980</td>
    </tr>
    <tr>
      <th>WoodDeckSF</th>
      <td>WoodDeckSF</td>
      <td>192.0</td>
      <td>487.761372</td>
    </tr>
    <tr>
      <th>SaleCondition</th>
      <td>SaleCondition</td>
      <td>4.0</td>
      <td>490.223394</td>
    </tr>
    <tr>
      <th>CentralAir</th>
      <td>CentralAir</td>
      <td>1.0</td>
      <td>533.540801</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>GarageType</td>
      <td>1.0</td>
      <td>569.096237</td>
    </tr>
    <tr>
      <th>BsmtUnfSF</th>
      <td>BsmtUnfSF</td>
      <td>396.0</td>
      <td>601.818903</td>
    </tr>
    <tr>
      <th>LotArea</th>
      <td>LotArea</td>
      <td>8414.0</td>
      <td>982.527398</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>LotFrontage</td>
      <td>70.0</td>
      <td>1255.136900</td>
    </tr>
    <tr>
      <th>OverallCond</th>
      <td>OverallCond</td>
      <td>8.0</td>
      <td>2634.189802</td>
    </tr>
    <tr>
      <th>YearRemodAdd</th>
      <td>YearRemodAdd</td>
      <td>2003.0</td>
      <td>2661.275923</td>
    </tr>
    <tr>
      <th>TotalBsmtSF</th>
      <td>TotalBsmtSF</td>
      <td>1059.0</td>
      <td>11313.187661</td>
    </tr>
  </tbody>
</table>
</div>


## Partial Dependence Plots

We will not cover Partial Dependence Plots in this workshop, but here is some code demonstrating how you can use them. 

Partial dependence use the model and dataset as follows. It varies the value of the feature of interest while keeping the rest of the features the same. Then, it uses the model to make a prediction on this augmented data. This way, we can see the effect of that feature alone. This library finds clusters in the dataset predictions for you.


```python
from pdpbox import pdp

def plot_pdp(feat, clusters=None, feat_name=None):
    feat_name = feat_name or feat
    p = pdp.pdp_isolate(model, X_train, X_train.columns, feat)
    return pdp.pdp_plot(p, feat_name, 
                        plot_lines=True,
                        cluster=clusters is not None,
                        n_cluster_centers=clusters)

plot_pdp('GrLivArea')
```



![png](/assets/images/posts/intro-to-random-forests/Random_Forests_Workshop_V2_69_1.png#center)



```python
plot_pdp('OverallQual', clusters=5)
plt.figure()
sns.regplot(data=df, x='OverallQual', y='SalePrice')
```



![png](/assets/images/posts/intro-to-random-forests/Random_Forests_Workshop_V2_70_1.png#center)



![png](/assets/images/posts/intro-to-random-forests/Random_Forests_Workshop_V2_70_2.png#center)


There seems to be an issue in the library related to plotting (or perhaps my code is wrong somehow), which is why I use a try-except block. Due to this issue, the axis are not shown.


```python
feats = ['OverallQual', 'GrLivArea']
p = pdp.pdp_interact(model, X_train, X_train.columns, feats)

try:
  pdp.pdp_interact_plot(p, feats)
except TypeError:
  pass
```


![png](/assets/images/posts/intro-to-random-forests/Random_Forests_Workshop_V2_72_0.png#center)



```python
feats = ['OverallCond', 'OverallQual']
p = pdp.pdp_interact(model, X_train, X_train.columns, feats)

try:
  pdp.pdp_interact_plot(p, feats)
except TypeError:
  pass
```


![png](/assets/images/posts/intro-to-random-forests/Random_Forests_Workshop_V2_73_0.png#center)


## Standard Process for Quick RF Development:

1.	Using scoring metric (provided or decided), create a scoring function (training + validation).
2.	Create a validation set with same properties as test set.
3.	Run a quick random forest on the data with minimal alterations.
4.	Plot feature importance, plot the features, and learn about those features (domain knowledge).
5.	Use a natural breakpoint to remove unimportant features then plot again.
6.	Look at data carefully and encode important features better.
7.	Using heirachical clustering to remove redundant features (scipy)
8.	For interpretation, use partial dependence plots from PDP.
9.	Use Tree Interpreter to explain individual predictions.
