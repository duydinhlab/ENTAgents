from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_agent(llm, tools: list, system_prompt: str):
    # Each worker node will be given a name and a tool.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    messages = state['messages']
    state['messages'] = [messages[0], messages[-1]]
    result = agent.invoke({'messages':state['messages'][-1].tool_calls[0]['args']['search_query']})
    return {"messages": [HumanMessage(content=result['output'], name=name)]}

