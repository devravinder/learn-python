# List vs Tuple vs Set vs Dictionary

## Differences & uses

### 1. List `[]`

* Mutable (can change values)
* Ordered (items keep their position)
* Allows duplicates
* Supports indexing (e.g., `fruits[0]`)
* Good for storing multiple values in a changing sequence

---

### 2. Tuple `()`

* Immutable (cannot change after creation)
* Ordered
* Allows duplicates
* Supports indexing (e.g., `colors[1]`)
* Good for fixed/read-only data and slightly faster than lists

---

### 3. Set `{}`

* Mutable (items can be added/removed)
* Unordered (no index or fixed position)
* Does **not** allow duplicates
* No indexing (e.g., `set[0]` ❌)
* Good when you need unique items and fast membership checks (e.g., `value in my_set`)

---

### 4. Dictionary `{key: value}`

* Mutable
* Ordered (Python 3.7+)
* Keys must be unique (values can repeat)
* Access by key (e.g., `person["age"]`)
* Good for storing mapped/structured data — key → value pairs
