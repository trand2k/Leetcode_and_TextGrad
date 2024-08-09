import textgrad as tg
import os
os.environ['OPENAI_API_KEY'] = "add_key"

tg.set_backward_engine("gpt-3.5-turbo", override=True)

# Step 1: Get an initial response from an LLM.
model = tg.BlackboxLLM("gpt-3.5-turbo")
question_string = """Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

The overall run time complexity should be O(log (m+n)).

Example 1:

Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.
Example 2:

Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.50000
Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.

Constraints:

nums1.length == m
nums2.length == n
0 <= m <= 1000
0 <= n <= 1000
1 <= m + n <= 2000
-106 <= nums1[i], nums2[i] <= 106"""

question = tg.Variable(
    question_string, role_description="question to the LLM", requires_grad=False
)

answer = model(question)
print(answer)

answer.set_role_description("initial model-generated answer")

# Step 2: Define the loss function and the optimizer, just like in PyTorch!
# Here, we don't have SGD, but we have TGD (Textual Gradient Descent)
# that works with "textual gradients".
optimizer = tg.TGD(parameters=[answer])
evaluation_instruction = """Evaluate the provided answer to this coding problem. Be logical and critical in assessing its correctness and efficiency."""

# TextLoss is a natural-language specified loss function that describes
# how we want to evaluate the reasoning.
loss_fn = tg.TextLoss(evaluation_instruction)

# Step 3: Do the loss computation, backward pass, and update the punchline.
loss = loss_fn(answer)
loss.backward()
optimizer.step()

# Print the refined solution
print("Refined solution:")
print(answer.value)

# Result:
# def findMedianSortedArrays(nums1, nums2):
#     if len(nums1) > len(nums2):
#         nums1, nums2 = nums2, nums1

#     m, n = len(nums1), len(nums2)
#     low, high = 0, m

#     while low <= high:
#         partition_nums1 = (low + high) // 2
#         partition_nums2 = (m + n + 1) // 2 - partition_nums1

#         max_left_nums1 = float('-inf') if partition_nums1 == 0 else nums1[partition_nums1 - 1]
#         min_right_nums1 = float('inf') if partition_nums1 == m else nums1[partition_nums1]

#         max_left_nums2 = float('-inf') if partition_nums2 == 0 else nums2[partition_nums2 - 1]
#         min_right_nums2 = float('inf') if partition_nums2 == n else nums2[partition_nums2]

#         if max_left_nums1 <= min_right_nums2 and max_left_nums2 <= min_right_nums1:
#             if (m + n) % 2 == 0:
#                 return (max(max_left_nums1, max_left_nums2) + min(min_right_nums1, min_right_nums2)) / 2
#             else:
#                 return max(max_left_nums1, max_left_nums2)
#         elif max_left_nums1 > min_right_nums2:
#             high = partition_nums1 - 1
#         else:
#             low = partition_nums1 + 1

#     raise ValueError("Input arrays are not sorted.")

# # Test cases
# nums1 = [1, 3]
# nums2 = [2]
# print(findMedianSortedArrays(nums1, nums2))  # Output: 2.0

# nums1 = [1, 2]
# nums2 = [3, 4]
# print(findMedianSortedArrays(nums1, nums2))  # Output: 2.5
