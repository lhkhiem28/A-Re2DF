import requests
import re
from googlesearch import search
from bs4 import BeautifulSoup

def get_validity_feedback(smiles, code_executor_agent):
    code_output = """
    ```python
from rdkit import Chem

mol = Chem.MolFromSmiles("{}")
if mol is not None:
    print("The modified molecule is chemically valid.")
else:
    print("The modified molecule is not chemically valid.")
    ```
    """.format(smiles)
    obs = code_executor_agent.generate_reply(messages=[{"role": "user", "content": code_output}]).split("Code output: ")[-1].strip()
    obs = re.sub(r'\[\d{2}:\d{2}:\d{2}\] ', '', obs)
    return obs

def get_first_result(query):
    try:
        # Step 1: Search Google for the query
        search_results = search(query, num_results=1)
        first_result_url = next(search_results)
        return first_result_url
    except Exception as e:
        print(f"Error during Google search: {e}")
        return None

def fetch_page_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching page content: {e}")
        return None

def extract_plain_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    plain_text = soup.get_text()
    return plain_text

def doc_retrieve(prop):
    query = f'RDKit compute {prop}'
    url = get_first_result(query)
    if url:
        html_content = fetch_page_content(url)
        if html_content:
            doc = extract_plain_text(html_content)
            return doc.strip()