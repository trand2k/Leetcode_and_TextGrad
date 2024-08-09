import textgrad as tg
import os
os.environ['OPENAI_API_KEY'] = 'add_key'
initial_solution = """
def findMedianSortedArrays(nums1, nums2):
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    low, high = 0, m

    while low <= high:
        partition_nums1 = (low + high) // 2
        partition_nums2 = (m + n + 1) // 2 - partition_nums1

        max_left_nums1 = float('-inf') if partition_nums1 == 0 else nums1[partition_nums1 - 1]
        min_right_nums1 = float('inf') if partition_nums1 == m else nums1[partition_nums1]

        max_left_nums2 = float('-inf') if partition_nums2 == 0 else nums2[partition_nums2 - 1]
        min_right_nums2 = float('inf') if partition_nums2 == n else nums2[partition_nums2]

        if max_left_nums1 <= min_right_nums2 and max_left_nums2 <= min_right_nums1:
            if (m + n) % 2 == 0:
                return (max(max_left_nums1, max_left_nums2) + min(min_right_nums1, min_right_nums2)) / 2
            else:
                return max(max_left_nums1, max_left_nums2)
        elif max_left_nums1 > min_right_nums2:
            high = partition_nums1 - 1
        else:
            low = partition_nums1 + 1

    raise ValueError("Input arrays are not sorted.")
"""

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

llm_engine = tg.get_engine("gpt-3.5-turbo")
tg.set_backward_engine(llm_engine)

# Code is the variable of interest we want to optimize -- so requires_grad=True
code = tg.Variable(
    value=initial_solution,
    requires_grad=True,
    role_description="code instance to optimize",
)

# We are not interested in optimizing the problem -- so requires_grad=False
problem = tg.Variable(
    question_string, requires_grad=False, role_description="the coding problem"
)

# Let TGD know to update code!
optimizer = tg.TGD(parameters=[code])

# The system prompt that will guide the behavior of the loss function.
loss_system_prompt = "You are a smart language model that evaluates code snippets. You do not solve problems or propose new code snippets, only evaluate existing solutions critically and give very concise feedback."
loss_system_prompt = tg.Variable(
    loss_system_prompt,
    requires_grad=False,
    role_description="system prompt to the loss function",
)

# The instruction that will be the prefix
instruction = """Think about the problem and the code snippet. Does the code solve the problem? What is the runtime complexity?"""

# The format string and setting up the call
format_string = "{instruction}\nProblem: {{problem}}\nCurrent Code: {{code}}"
format_string = format_string.format(instruction=instruction)

fields = {"problem": None, "code": None}
formatted_llm_call = tg.autograd.FormattedLLMCall(
    engine=llm_engine,
    format_string=format_string,
    fields=fields,
    system_prompt=loss_system_prompt,
)


# Finally, the loss function
def loss_fn(problem: tg.Variable, code: tg.Variable) -> tg.Variable:
    inputs = {"problem": problem, "code": code}

    return formatted_llm_call(
        inputs=inputs,
        response_role_description=f"evaluation of the {code.get_role_description()}",
    )


for iteration in range(5):
    optimizer.zero_grad()
    loss = loss_fn(problem, code)
    loss.backward()
    optimizer.step()

# Print the refined solution
print("Refined solution:")
print(code.value)

# # Result:
# def findMedianSortedArrays(nums1, nums2):
#     # Ensure nums1 is not longer than nums2 for simplicity
#     if len(nums1) > len(nums2):
#         nums1, nums2 = nums2, nums1

#     # Get the lengths of the input arrays
#     m, n = len(nums1), len(nums2)

#     # Initialize pointers for binary search
#     left_pointer, right_pointer = 0, m

#     # Perform binary search to find the median
#     while left_pointer <= right_pointer:
#         # Calculate the partition for nums1
#         partition_nums1 = (left_pointer + right_pointer) // 2
#         # Calculate the partition for nums2 based on partition_nums1
#         partition_nums2 = (m + n + 1) // 2 - partition_nums1

#         # Calculate the elements around the partitions
#         max_left_nums1 = float('-inf') if partition_nums1 == 0 else nums1[partition_nums1 - 1]
#         min_right_nums1 = float('inf') if partition_nums1 == m else nums1[partition_nums1]

#         max_left_nums2 = float('-inf') if partition_nums2 == 0 else nums2[partition_nums2 - 1]
#         min_right_nums2 = float('inf') if partition_nums2 == n else nums2[partition_nums2]

#         # Check if the partitions are at the correct place
#         if max_left_nums1 <= min_right_nums2 and max_left_nums2 <= min_right_nums1:
#             # Calculate the median based on even or odd total elements
#             if (m + n) % 2 == 0:
#                 return (max(max_left_nums1, max_left_nums2) + min(min_right_nums1, min_right_nums2)) / 2
#             else:
#                 return max(max_left_nums1, max_left_nums2)
#         elif max_left_nums1 > min_right_nums2:
#             right_pointer = partition_nums1 - 1
#         else:
#             left_pointer = partition_nums1 + 1

#     raise ValueError("The input arrays do not contain sorted elements. Please ensure both input arrays are sorted.")  # Perform binary search to find the median
