import requests
from bs4 import BeautifulSoup

def get_problem_details(problem_slug):
    url = f"https://leetcode.com/graphql"
    query = """
    query getQuestionDetail($titleSlug: String!) {
        question(titleSlug: $titleSlug) {
            questionId
            title
            content
            difficulty
            codeDefinition
            sampleTestCase
            enableRunCode
            metaData
            translatedContent
            topicTags {
                name
                id
                slug
            }
        }
    }
    """
    variables = {"titleSlug": problem_slug}
    json_data = {
        "query": query,
        "variables": variables,
    }
    
    response = requests.post(url, json=json_data)
    data = response.json()
    data_query = data['data']['question']
    content = data_query["content"]
    soup = BeautifulSoup(content, 'html.parser')

    # Get the text content from the HTML
    cleaned_content = soup.get_text()
    meta_data = data_query["metaData"]
    title = data_query["title"] 
    return cleaned_content, meta_data, title