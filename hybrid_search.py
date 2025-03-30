import math

class HybridSearch:
    def __init__(self, data, columns):
        """Initialize hybrid search with Excel data and column names."""
        self.data = data
        self.columns = columns
        self.n = len(data)
        self.bucket_size = max(1, int(math.sqrt(self.n)))
        self.hashmaps = {col: {row[col]: i for i, row in enumerate(data)} for col in columns}
        self.buckets = {}
        self.index_maps = {}
        
        # Create buckets for each column
        for col in columns:
            self.buckets[col], self.index_maps[col] = self._create_buckets(col)

    def _create_buckets(self, column):
        """Sort elements into sqrt(n) partitions and store their original indices."""
        sorted_data = sorted((str(row[column]), i) for i, row in enumerate(self.data))
        buckets = {}
        index_map = {}

        for i in range(self.bucket_size):
            start = i * self.bucket_size
            end = min((i + 1) * self.bucket_size, self.n)
            buckets[i] = [val for val, idx in sorted_data[start:end]]
            for val, idx in sorted_data[start:end]:
                index_map[val] = idx

        return buckets, index_map

    def _binary_search(self, arr, key):
        """Binary search within a bucket."""
        low, high = 0, len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if arr[mid] == key:
                return mid
            elif arr[mid] < key:
                low = mid + 1
            else:
                high = mid - 1
        return -1

    def search(self, key, search_by):
        """Search using HashMap (O(1)) or Bucket Binary Search (O(log√n))."""
        if search_by not in self.columns:
            return -1

        # Try exact match first
        if key in self.hashmaps[search_by]:
            return self.hashmaps[search_by][key]

        # Try fuzzy search in buckets
        approx_index = min(hash(str(key)) % self.bucket_size, self.bucket_size - 1)
        if approx_index in self.buckets[search_by]:
            index_in_bucket = self._binary_search(self.buckets[search_by][approx_index], str(key))
            if index_in_bucket != -1:
                return self.index_maps[search_by][str(key)]

        # Check adjacent buckets
        for adj in [approx_index - 1, approx_index + 1]:
            if adj in self.buckets[search_by]:
                index_in_bucket = self._binary_search(self.buckets[search_by][adj], str(key))
                if index_in_bucket != -1:
                    return self.index_maps[search_by][str(key)]

        return -1

    def get_column_names(self):
        """Return list of available column names."""
        return self.columns 