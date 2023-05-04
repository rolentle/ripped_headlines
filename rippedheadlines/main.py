from langchain import LLMChain
from langchain.chains import APIChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
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
Create a 'Law and Order' episode plot based upon the following current event:
{current_event}

Guidelines:
Do not draw from Law and Order SVU. Only draw from the original Law and Order.
Remember to focus the plot around the murder while the current event is used to create characters and settings.
The murder must take place in New York City.
Law and Order is a police procedural show set in New York City.
Law and Order show is set in a normal settings and does not incorporate science fiction elements
Replace real peoples, companies, brand, and drug names with fake names that rhyme.
Do not replace names of real cities, states, or countries, such as Paris, Montana, or Russia.
Do not replace characters names from the show 'Law and Order' or places in New York City.
Do not use characters from SVU, only use character from the Original Law and Order.
The plot synopsis begins with a stranger finding the dead body of the murder victim.
The detectives question persons of interest connected to the murder victim.

Here are some common elements you might find in a typical "Law & Order" episode:

1. A crime is committed: The episode usually opens with the discovery of a crime, such as a murder or serious assault, often by ordinary people who come across the scene or the victim.
2. Initial investigation: The detectives, usually a pair, begin their investigation by interviewing witnesses, collecting evidence, and forming an initial hypothesis about the crime.
3. False leads and red herrings: As the detectives follow leads and interview potential suspects, they often encounter dead ends or misdirections that complicate the investigation.
4. Turning point: A key piece of evidence or new witness testimony often emerges, providing a breakthrough in the case and leading the detectives to the true perpetrator(s).
5. Arrest and interrogation: The detectives confront and arrest the main suspect, followed by an interrogation scene where they attempt to extract a confession or gather more information about the crime.
6. Legal wrangling: The case is handed over to the district attorneys, who must navigate legal challenges, such as suppressing evidence or dealing with uncooperative witnesses, as they build their case for trial.
7. The trial: The courtroom drama unfolds, with the prosecution and defense presenting their arguments, examining and cross-examining witnesses, and offering closing statements.
8. Verdict and resolution: The episode typically concludes with the jury's verdict, followed by a brief wrap-up of the case, where the characters may reflect on the outcome or discuss its broader implications.
"""
l_and_o_chain = LLMChain(
    llm=OpenAI(temperature=1.0), 
    prompt=PromptTemplate.from_template(l_and_o_prompt),
    verbose=True
)

filter_principle = ConstitutionalPrinciple(
    name="Filter",
    critique_request="The model should not include the topic of child harm such as school shootings.",
    revision_request="Remove the sentence that includes the topic of child harm as school shootings.",
)

filtered_l_and_o_chain = ConstitutionalChain.from_llm(
    chain= l_and_o_chain,
    constitutional_principles=[filter_principle],
    llm=OpenAI(temperature=0),
    verbose=True,
)

overall_chain = SimpleSequentialChain(chains=[news_chain, filtered_l_and_o_chain], verbose=True)

print(overall_chain.run("Find a few current events that happened in the US written in English"))