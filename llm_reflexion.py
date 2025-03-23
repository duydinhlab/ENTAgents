import os 
from langchain_groq import ChatGroq
from langchain_voyageai import VoyageAIEmbeddings
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from create_agents import create_agent, agent_node
from langgraph.prebuilt import ToolNode
from langchain_community.vectorstores import FAISS
from langchain_core.tools import StructuredTool
import functools
import operator
from typing import Sequence, TypedDict, Annotated, Literal
from dotenv import load_dotenv
load_dotenv()

#Load the API key
google_api_key = os.getenv('GOOGLE_API_KEY')
google_cse_id = os.getenv('GOOGLE_CSE_ID')
os.environ["LANGCHAIN_PROJECT"] = 'llm_reflexion_MCQ'
langchain_api = os.getenv("LANGCHAIN_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
api_key = os.getenv("LANGCHAIN_COHERE_KEY")
voyageai_api_key = os.getenv("VOYAGEAI_API_KEY")

general = 'llama-3.1-70b-versatile'
tool_use = 'llama3-groq-70b-8192-tool-use-preview'

#Function to choose different models based on the needs
def model_select(model):
    groq_chat = ChatGroq(
                groq_api_key=groq_api_key, 
                model_name=model
            )
    return groq_chat

#Reflection Critiques
class Reflection(BaseModel):
    missing: str = Field(description="Critique of how can the answer be elaborated more" )
    superfluous: str = Field(description="Critique of what is redundant in the answer.")

#The format structure of the response from Drafter and Revisor
class AnswerQuestion(BaseModel):
    """Answer the question. Provide an detailed answer, reflection, and then follow up with only 1 query related to the critique of missing to improve the answer."""

    answer: str = Field(description="A detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the latest answer.")
    search_query: list[str] = Field(
        description="Only a list of 1 query for researching improvements to address the critique of missing in reflection if it is necessary"
    )

class ResponderWithRetries:
    def __init__(self, runnable, validator, revise):
        self.runnable = runnable
        self.validator = validator
        self.revise = revise

    def respond(self, state):
        response = []
        for attempt in range(3):
            response = self.runnable.invoke(
                state['messages'], {"tags": [f"attempt:{attempt}"]}
            )
            
            try:
                self.validator.invoke(response)
                return {"messages": [response]}

            except ValidationError as e:
                state = state + [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\n\nPay close attention to the function schema.\n\n"
                        + self.validator.schema_json()
                        + " Respond by fixing all validation errors.",
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]

        return {"messages": [response]}

#General prompt
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a professional otolaryngologist.  

            1. {first_instruction}
            2. Reflect and critique your latest answer to provide new critiques. Be severe to maximize improvement.
            3. Recommend only 1 query that address the critique and research information, thus improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "\n\n<system>Reflect on the user's original question and the"
            " actions taken thus far. Respond using the {function_name} function.</reminder>",
        ),
    ]
)

#Drafter's chain
initial_answer_chain = prompt_template.partial(
    first_instruction="Provide a full detailed answer.",
    function_name=AnswerQuestion.__name__,
) | model_select(general).bind_tools(tools=[AnswerQuestion])

validator = PydanticToolsParser(tools=[AnswerQuestion])

drafter = ResponderWithRetries(
    runnable=initial_answer_chain, validator=validator, revise = False
)

#Revisor prompt
revise_instructions = """Revise your previous answer using the new information provided.
    - You should use the previous critiue of missing to add important information to your answer from the latest message.
    - You should use the previous critique of superflous to remove redundant information from your answer
    - Provide new critiques of reflection if the revised answer is not the best, and finally change the new query to a new one based on reflection to improve the answer.
"""

class ReviseAnswer(AnswerQuestion):
    """Revise your previous answer to your question. """

#Revisor's chain
revision_chain = prompt_template.partial(
    first_instruction=revise_instructions,
    function_name=ReviseAnswer.__name__,
) | model_select(general).bind_tools(tools=[ReviseAnswer])

revision_validator = PydanticToolsParser(tools=[ReviseAnswer])

revisor = ResponderWithRetries(runnable=revision_chain, validator=revision_validator, revise=True)

#Agents used
members = ["Arxiv","Google","Pubmed",]
options = ["FINISH","Arxiv","Google","Pubmed"]

class routeResponse(BaseModel):
    next: Literal["Pubmed","FINISH","Arxiv","Google"]

#Supervisor's prompt
system_prompt = """You are a supervisor tasked with managing a conversation between 
the agents: {members}. Based on the search_queries field in the last message, choose the most appropriate agent to act next: 
- Use Pubmed for medical or scientific queries.
- Use Arxiv for academic or research papers.
- Use Google for general web searches.
- Respond with FINISH when the answer is the best."""

supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),

        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "you should always think about what to do and the action to take based on the last Message, the action should be one of {options}",
        ),
    ]
)

#Supervisor. 
def supervisor_agent(state):
    supervisor_chain = (
        supervisor_prompt.partial(
        options = options,
        members=", ".join(members)
        )
        | model_select(tool_use).with_structured_output(routeResponse)
    )
    state['next'] = None
    return supervisor_chain.invoke(state)

#Building RAG node 
embedding = VoyageAIEmbeddings(
    voyage_api_key=voyageai_api_key, model="voyage-large-2-instruct"
)
vectorstore = FAISS.load_local(
    "FAISS_index", embedding, allow_dangerous_deserialization=True
)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval(search_query: list[str]):
    """Run the search_query by searching similar documents related to the embedded vector of the search_query.""" 
    retrieved_docs = [format_docs(vectorstore.similarity_search_by_vector(
        embedding.embed_query(query), k=6)) for query in search_query
        ]
    return retrieved_docs

rag = StructuredTool.from_function(retrieval, name=AnswerQuestion.__name__, handle_tool_error=True)
rag_node = ToolNode([rag])

#Multiple agents with tools from Google, ArXiv, PubMed
google_search = GoogleSearchAPIWrapper(
      google_api_key = google_api_key,
      google_cse_id = google_cse_id
)

pubmed = PubmedQueryRun(verbose=True)
arxiv = ArxivAPIWrapper(top_k_results=1)
def pubmed_func(search_query: list[str], **kwargs):
    """Run the pubmed tool with search_query as the input.""" 
    return pubmed.batch(search_query,result=1)

def google_func(search_query: list[str], **kwargs):
    """Run the google_search tool with search_query as the input.""" 
    return google_search.run(search_query)

def arxiv_func(search_query: list[str], **kwargs):
    """Run the arxiv tool with search_query as the input.""" 
    return arxiv.run(search_query[0])

arxiv_tool = StructuredTool.from_function(arxiv_func, name='Arxiv')
google_tool = StructuredTool.from_function(google_func, name='google_search')
pubmed_tool = StructuredTool.from_function(pubmed_func, name='PubMed')

google_agent = create_agent(model_select(tool_use), [google_tool], "You are a web researcher. Execute the tool with the search_query field in the latest message as input.")
google_node = functools.partial(agent_node, agent=google_agent, name="Google")

pubmed_agent = create_agent(model_select(tool_use), [pubmed_tool], "You are a literature researcher. Execute the tool with the search_query field in the latest message as input.",)
pubmed_node = functools.partial(agent_node, agent=pubmed_agent, name="Pubmed")

arxiv_agent = create_agent(model_select(tool_use), [arxiv_tool], "You are a literature researcher. Execute the tool with the search_query field in the latest message as input.",)
arxiv_node = functools.partial(agent_node, agent=arxiv_agent, name="Arxiv")

#Building the Agent Graph, all the LLMs and agents used will be connected
from langgraph.graph import END, START,  StateGraph

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

builder = StateGraph(AgentState)
builder.add_node("draft", drafter.respond)
builder.add_node('RAG',rag_node)
builder.add_node("revisor", revisor.respond)
builder.add_node("supervisor", supervisor_agent)
builder.add_node('Pubmed',pubmed_node)
builder.add_node('Google',google_node)
builder.add_node('Arxiv',arxiv_node)

builder.add_edge("draft", "RAG")
builder.add_edge(START, "draft")
builder.add_edge("RAG", "revisor")
builder.add_edge("revisor", "supervisor")
for member in members:
    builder.add_edge(member, "revisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
builder.add_conditional_edges("supervisor", lambda x:  x['next'], conditional_map)
builder.add_edge(START, "draft")
graph = builder.compile()

#Question-Answering
while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chatbot. Goodbye!")
            break
        
        for state in graph.stream({"messages": [HumanMessage(content= user_input)]},
            # {"recursion_limit": 100},
        ): 
            if "__end__" not in state:
                print(state)
                print("----")




