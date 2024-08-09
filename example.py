from leetcode_env.environment import LeetCodeEnv
from leetcode_env.types import LeetCodeSubmission, ProgrammingLanguage
import os 
from leetcode_env.leetcode_requirement import get_problem_details
import langchain
import openai
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI

os.environ['LEETCODE_SESSION'] = '<api_code>'
os.environ['LEETCODE_CSRF_TOKEN'] = '<api_code>'
os.environ['OPENAI_API_KEY'] = '<api_code>'
question_slug='two-sum'

#################### STEP 1 #####################
# define prompt for generate problem

prompt_template = '''
Using python for solve this leetcode problem, in class Solution, sample below:

class Solution:
    def sum(self, a, b):
        return a+b

Given the problem titled {title}

you are required to:
{requirements}

The problem metadata includes:
{meta_data}

Write a function to solve this problem.
'''


llm_basic = OpenAI()

prompt = PromptTemplate(
    input_variables=["title","requirements","meta_data"], template=prompt_template
)

#################### STEP 2 #####################
# get requirements , metadata, title base on question_slug

requirements , metadata, title = get_problem_details(question_slug)


out = prompt.invoke({"title": title, "requirements": requirements,"meta_data" : metadata })
code = llm_basic.invoke(out.text)


################### STEP 3 ######################
# summit code and revice feedback

sub = LeetCodeSubmission(code=code,
                         lang=ProgrammingLanguage.PYTHON3,
                         question_slug=question_slug)

env = LeetCodeEnv()


status, reward, done, submission_result = env.step(sub)

print("Solution::::::::")
print(code)
print("Results:::::::::")
print(submission_result)

#### sample submission_result in fail.json and success.json

