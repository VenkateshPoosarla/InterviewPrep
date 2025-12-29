## LeetCode 75 (Python)

This folder contains **one file per problem** from LeetCode's "LeetCode 75" study plan,
organized into **category subfolders** (e.g. `array_string/`, `two_pointers/`, etc.).

### How to run

Run any problem file directly; it will execute its built-in tests and then **exit**:

```bash
python leetcode75/array_string/p1768_merge_strings_alternately.py
```

Tip: if you prefer running as a module (no `sys.path` concerns), you can do:

```bash
python -m leetcode75.array_string.p1768_merge_strings_alternately
```

### Conventions

- **Folder structure**: category folders inside `leetcode75/`
- **File naming**: `p<problem_number>_<snake_case_title>.py` inside its category folder
- **Implementation**: `class Solution:` with the LeetCode-style method name.
- **Tests**: small `run_tests()` function using `assert` (no external test runner needed).
- **Visuals**: when helpful, short ASCII walkthroughs are included in comments/docstrings.

### Problem list (75)

#### `array_string/` (9)
- **151. Reverse Words in a String**: `leetcode75/array_string/p0151_reverse_words_in_a_string.py`
- **238. Product of Array Except Self**: `leetcode75/array_string/p0238_product_of_array_except_self.py`
- **334. Increasing Triplet Subsequence**: `leetcode75/array_string/p0334_increasing_triplet_subsequence.py`
- **345. Reverse Vowels of a String**: `leetcode75/array_string/p0345_reverse_vowels_of_a_string.py`
- **443. String Compression**: `leetcode75/array_string/p0443_string_compression.py`
- **605. Can Place Flowers**: `leetcode75/array_string/p0605_can_place_flowers.py`
- **1071. Greatest Common Divisor of Strings**: `leetcode75/array_string/p1071_greatest_common_divisor_of_strings.py`
- **1431. Kids With the Greatest Number of Candies**: `leetcode75/array_string/p1431_kids_with_the_greatest_number_of_candies.py`
- **1768. Merge Strings Alternately**: `leetcode75/array_string/p1768_merge_strings_alternately.py`

#### `two_pointers/` (4)
- **11. Container With Most Water**: `leetcode75/two_pointers/p0011_container_with_most_water.py`
- **283. Move Zeroes**: `leetcode75/two_pointers/p0283_move_zeroes.py`
- **392. Is Subsequence**: `leetcode75/two_pointers/p0392_is_subsequence.py`
- **1679. Max Number of K Sum Pairs**: `leetcode75/two_pointers/p1679_max_number_of_k_sum_pairs.py`

#### `sliding_window/` (4)
- **643. Maximum Average Subarray I**: `leetcode75/sliding_window/p0643_maximum_average_subarray_i.py`
- **1004. Max Consecutive Ones III**: `leetcode75/sliding_window/p1004_max_consecutive_ones_iii.py`
- **1456. Maximum Number of Vowels in a Substring of Given Length**: `leetcode75/sliding_window/p1456_maximum_number_of_vowels_in_a_substring_of_given_length.py`
- **1493. Longest Subarray of 1s After Deleting One Element**: `leetcode75/sliding_window/p1493_longest_subarray_of_1s_after_deleting_one_element.py`

#### `prefix_sum_hashmap/` (6)
- **724. Find Pivot Index**: `leetcode75/prefix_sum_hashmap/p0724_find_pivot_index.py`
- **1207. Unique Number of Occurrences**: `leetcode75/prefix_sum_hashmap/p1207_unique_number_of_occurrences.py`
- **1657. Determine if Two Strings Are Close**: `leetcode75/prefix_sum_hashmap/p1657_determine_if_two_strings_are_close.py`
- **1732. Find the Highest Altitude**: `leetcode75/prefix_sum_hashmap/p1732_find_the_highest_altitude.py`
- **2215. Find the Difference of Two Arrays**: `leetcode75/prefix_sum_hashmap/p2215_find_the_difference_of_two_arrays.py`
- **2352. Equal Row and Column Pairs**: `leetcode75/prefix_sum_hashmap/p2352_equal_row_and_column_pairs.py`

#### `stack/` (4)
- **394. Decode String**: `leetcode75/stack/p0394_decode_string.py`
- **735. Asteroid Collision**: `leetcode75/stack/p0735_asteroid_collision.py`
- **739. Daily Temperatures**: `leetcode75/stack/p0739_daily_temperatures.py`
- **2390. Removing Stars From a String**: `leetcode75/stack/p2390_removing_stars_from_a_string.py`

#### `queue_linked_list/` (6)
- **206. Reverse Linked List**: `leetcode75/queue_linked_list/p0206_reverse_linked_list.py`
- **328. Odd Even Linked List**: `leetcode75/queue_linked_list/p0328_odd_even_linked_list.py`
- **649. Dota2 Senate**: `leetcode75/queue_linked_list/p0649_dota2_senate.py`
- **933. Number of Recent Calls**: `leetcode75/queue_linked_list/p0933_number_of_recent_calls.py`
- **2095. Delete the Middle Node of a Linked List**: `leetcode75/queue_linked_list/p2095_delete_the_middle_node_of_a_linked_list.py`
- **2130. Maximum Twin Sum of a Linked List**: `leetcode75/queue_linked_list/p2130_maximum_twin_sum_of_a_linked_list.py`

#### `binary_tree/` (10)
- **104. Maximum Depth of Binary Tree**: `leetcode75/binary_tree/p0104_maximum_depth_of_binary_tree.py`
- **199. Binary Tree Right Side View**: `leetcode75/binary_tree/p0199_binary_tree_right_side_view.py`
- **236. Lowest Common Ancestor of a Binary Tree**: `leetcode75/binary_tree/p0236_lowest_common_ancestor_of_a_binary_tree.py`
- **437. Path Sum III**: `leetcode75/binary_tree/p0437_path_sum_iii.py`
- **450. Delete Node in a BST**: `leetcode75/binary_tree/p0450_delete_node_in_a_bst.py`
- **700. Search in a Binary Search Tree**: `leetcode75/binary_tree/p0700_search_in_a_binary_search_tree.py`
- **872. Leaf-Similar Trees**: `leetcode75/binary_tree/p0872_leaf_similar_trees.py`
- **1161. Maximum Level Sum of a Binary Tree**: `leetcode75/binary_tree/p1161_maximum_level_sum_of_a_binary_tree.py`
- **1372. Longest ZigZag Path in a Binary Tree**: `leetcode75/binary_tree/p1372_longest_zigzag_path_in_a_binary_tree.py`
- **1448. Count Good Nodes in Binary Tree**: `leetcode75/binary_tree/p1448_count_good_nodes_in_binary_tree.py`

#### `graphs/` (4)
- **547. Number of Provinces**: `leetcode75/graphs/p0547_number_of_provinces.py`
- **841. Keys and Rooms**: `leetcode75/graphs/p0841_keys_and_rooms.py`
- **994. Rotting Oranges**: `leetcode75/graphs/p0994_rotting_oranges.py`
- **1926. Nearest Exit From Entrance in Maze**: `leetcode75/graphs/p1926_nearest_exit_from_entrance_in_maze.py`

#### `heap_priority_queue/` (4)
- **215. Kth Largest Element in an Array**: `leetcode75/heap_priority_queue/p0215_kth_largest_element_in_an_array.py`
- **2336. Smallest Number in Infinite Set**: `leetcode75/heap_priority_queue/p2336_smallest_number_in_infinite_set.py`
- **2462. Total Cost to Hire K Workers**: `leetcode75/heap_priority_queue/p2462_total_cost_to_hire_k_workers.py`
- **2542. Maximum Subsequence Score**: `leetcode75/heap_priority_queue/p2542_maximum_subsequence_score.py`

#### `binary_search/` (4)
- **33. Search in Rotated Sorted Array**: `leetcode75/binary_search/p0033_search_in_rotated_sorted_array.py`
- **74. Search a 2D Matrix**: `leetcode75/binary_search/p0074_search_a_2d_matrix.py`
- **153. Find Minimum in Rotated Sorted Array**: `leetcode75/binary_search/p0153_find_minimum_in_rotated_sorted_array.py`
- **704. Binary Search**: `leetcode75/binary_search/p0704_binary_search.py`

#### `backtracking/` (4)
- **46. Permutations**: `leetcode75/backtracking/p0046_permutations.py`
- **77. Combinations**: `leetcode75/backtracking/p0077_combinations.py`
- **79. Word Search**: `leetcode75/backtracking/p0079_word_search.py`
- **131. Palindrome Partitioning**: `leetcode75/backtracking/p0131_palindrome_partitioning.py`

#### `dynamic_programming/` (8)
- **62. Unique Paths**: `leetcode75/dynamic_programming/p0062_unique_paths.py`
- **72. Edit Distance**: `leetcode75/dynamic_programming/p0072_edit_distance.py`
- **198. House Robber**: `leetcode75/dynamic_programming/p0198_house_robber.py`
- **714. Best Time to Buy and Sell Stock with Transaction Fee**: `leetcode75/dynamic_programming/p0714_best_time_to_buy_and_sell_stock_with_transaction_fee.py`
- **746. Min Cost Climbing Stairs**: `leetcode75/dynamic_programming/p0746_min_cost_climbing_stairs.py`
- **790. Domino and Tromino Tiling**: `leetcode75/dynamic_programming/p0790_domino_and_tromino_tiling.py`
- **1137. N-th Tribonacci Number**: `leetcode75/dynamic_programming/p1137_n_th_tribonacci_number.py`
- **1143. Longest Common Subsequence**: `leetcode75/dynamic_programming/p1143_longest_common_subsequence.py`

#### `bit_manipulation/` (2)
- **338. Counting Bits**: `leetcode75/bit_manipulation/p0338_counting_bits.py`
- **1318. Minimum Flips to Make a OR b Equal to c**: `leetcode75/bit_manipulation/p1318_minimum_flips_to_make_a_or_b_equal_to_c.py`

#### `trie/` (2)
- **208. Implement Trie (Prefix Tree)**: `leetcode75/trie/p0208_implement_trie_prefix_tree.py`
- **1268. Search Suggestions System**: `leetcode75/trie/p1268_search_suggestions_system.py`

#### `intervals/` (2)
- **435. Non-overlapping Intervals**: `leetcode75/intervals/p0435_non_overlapping_intervals.py`
- **452. Minimum Number of Arrows to Burst Balloons**: `leetcode75/intervals/p0452_minimum_number_of_arrows_to_burst_balloons.py`

#### `monotonic_stack/` (2)
- **496. Next Greater Element I**: `leetcode75/monotonic_stack/p0496_next_greater_element_i.py`
- **901. Online Stock Span**: `leetcode75/monotonic_stack/p0901_online_stock_span.py`

