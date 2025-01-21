from llm import llm
from graph import graph

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from langchain.tools import Tool

from langchain_community.chat_message_histories import Neo4jChatMessageHistory

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub

from utils import get_session_id
from langchain.callbacks import get_openai_callback

from langchain_core.prompts import PromptTemplate



from tools.cypher import cypher_qa

# Create a powertrain chat chain
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a systems design  expert providing information about vehicle power-trains"),
        ("human", "{input}"),
    ]
)

powertrain_chat = chat_prompt | llm | StrOutputParser()
# Create a set of tools

tools = [
   
    Tool.from_function(
        name="General Chat",
        description="For general chat about power-trains not covered by other tools",
        func=powertrain_chat.invoke,
    ),
    Tool.from_function(
        name="Components Information",
        description="Provide information about components or powertrain questions using Cypher",
        func=cypher_qa

    )
]


# Create chat history callback
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


# Create the agent
agent_prompt = PromptTemplate.from_template("""
You are a vehicle powertrain design expert using a Neo4j knowledge graph.

Your task is to generate valid powertrain configurations in the format:
[Component1]-[Component2]-[Component3]-...-[ComponentN].

Rules:
1. Start with a component that has no input energy (e.g., "FuelTank").
2. Connect components based on matching energy outputs and inputs.
3. Stop when a component with no output energy (e.g., "Vehicle Block") is reached.
4. Use the schema and relationships in the knowledge graph to validate connections.

Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to cars, powertrains  or their components.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response in the form: [Component1]-[Component2]-[Component3]-...-[ComponentN]]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)


# Create a handler to call the agent
def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in UI
    """
    with get_openai_callback() as cb:
        response = chat_agent.invoke(
            {"input": user_input},
            {"configurable": {"session_id": get_session_id()}},)
        print(cb)
    return response['output']


