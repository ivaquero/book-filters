# Python

## 1. Basics

### 1.1. Iteration

- `enumerate()` is prefered to `range(len())`

```python
xs = range(3)

# good
for ind, x in enumerate(xs):
  print(f'{ind}: {x}')

# bad
for i in range(len(xs)):
  print(f'{i}: {xs[i]}')
```

## 2. Matplotlib

### 2.1. Subplots

- `Axes` object is prefered to `Figure` object
- use `constrained_layout=True` when draw subplots

```python
# good
_, axes = plt.subplots(1, 2, constrained_layout=True)
axes[0].plot(x1, y1)
axes[1].hist(x2, y2)

# bad
plt.subplot(121)
plt.plot(x1, y1)
plt.subplot(122)
plt.hist(x2, y2)
```

- `axes.flatten()` is prefered to `plt.subplot()` in cases where subplots' data is iterable
- `zip()` or `enumerate()` is prefered to `range()` for iterable objects

```python
# good
_, ax = plt.subplots(2, 2, figsize=[12,8], constrained_layout=True)

for ax, x, y in zip(axes.flatten(), xs, ys):
  ax.plot(x, y)

# bad
for i in range(4):
  ax = plt.subplot(2, 2, i+1)
  ax.plot(x[i], y[i])
```

### 2.2. Decoration

- `set()` method is prefered to `set_*()` method

```python

# good
ax.set(xlabel='x', ylabel='y')

# bad
ax.set_xlabel('x')
ax.set_ylabel('y')
```

- `ax.spines[*].set_visible()` with list is prefered to line-by-line `ax.spines[*].set_visible()`

```python
# good
ax.spines["top", "bottom"].set_visible(False)

# bad
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
```

## 3. pandas

### 3.1. Selection

- `df['col']` is prefered to `df.col`

```python
# good
movies['duration']

# bad
movies.duration
```

- `df.query` is prefered to `df[]` or `df.loc[]` in simple-selection

```python
# good
movies.query('duration >= 200')

# bad
movies[movies['duration'] >= 200]
movies.loc[movies['duration'] >= 200, :]
```

- `df.loc` or `df.iloc` is prefered to `df[]` in multiple-selection

```python
# good
movies.loc[movies['duration'] >= 200, 'genre']
movies.iloc[0:2, :]

# bad
movies[movies['duration'] >= 200].genre
movies[0:2]
```

### 3.2. pandas-vet

Check [pandas-vet](https://github.com/deppen8/pandas-vet)

- **PD001**: pandas should always be imported as `import pandas as pd`
- **PD002**: `inplace = True` should be avoided; it has inconsistent behavior
- **PD003**: `.isna` is prefered to `.isnull`
- **PD004**: `.notna` is preferred to `.notnull`
- **PD005**: arithmetic operator is prefered to method
- **PD006**: comparison operator is prefered to method
- **PD007**: `.loc` or `.iloc` is prefered to `.ix` which is deprecated
- **PD008**: `.loc` is prefered to `.at`. If speed is important, use numpy.
- **PD009**: `.iloc` is prefered to `.iat`. If speed is important, use numpy.
- **PD010**: `.pivot_table` is preferred to `.pivot` or `.unstack`
- **PD011**: `.array` or `.to_array()` is prefered to `.values` which is ambiguous
- **PDO12**: `.read_csv` is preferred to `.read_table`
- **PD013**: `.melt` is preferred to `.stack`
- **PD015**: `.merge` is prefered to `pd.merge`
- **PD901**: `df` is a bad variable name. Be kinder to your future self.

> pay attention to **PD010**, `unstack` is useful for dealing with `MultiIndex` arrays
