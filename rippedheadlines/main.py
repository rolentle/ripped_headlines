from langchain import LLMChain
from langchain.chains import APIChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain import OpenAI
import os

from langchain.chains.api import news_docs

llm = OpenAI(temperature=0)
news_chain = APIChain.from_llm_and_api_docs(
    llm, news_docs.NEWS_DOCS, headers={"X-Api-Key": os.getenv('news_api_key')}, output_key="current_event",
    verbose=True
)

# Load the tool configs that are needed.

l_and_o_prompt = """
Create a two paragraph 'Law and Order' plot synopsis based upon the following current event:
{current_event}

Remember to focus the plot around the murder while the current event is used to create characters and settings.
Replace real people's names or companies/brand names with fake names that are very similar names.
Replace locations with locations in New York City.
Do not replace characters names from the show 'Law and Order' or places in New York City.
Remember to set show in New York City
"""
l_and_a_chain = LLMChain(
    llm=OpenAI(temperature=1.0), 
    prompt=PromptTemplate.from_template(l_and_o_prompt),
    verbose=True
)

overall_chain = SimpleSequentialChain(chains=[news_chain, l_and_a_chain], verbose=True)

print(overall_chain.run("Find a few current events that happened in the US written in English"))