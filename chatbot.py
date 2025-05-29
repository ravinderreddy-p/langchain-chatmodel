from typing import Sequence
from typing_extensions import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, trim_messages


model = init_chat_model("gpt-4o-mini", model_provider="openai")

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage],  add_messages]
    language: str

trimmer = trim_messages(
    max_tokens=650,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi! I am Ravi"),
    AIMessage(content="hi Ravi!"),
    HumanMessage(content="I like Venila Ice Cream"),
    AIMessage(content="Nice"),
    HumanMessage(content="What's 2 + 2?"),
    AIMessage(content="4"),
    HumanMessage(content="Thanks"),
    AIMessage(content="You're welcome!"),
    HumanMessage(content="Having fun?"),
    AIMessage(content="yes!")
]

trimmer.invoke(messages)


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define a new graph
workflow = StateGraph(state_schema=State)

# Define the function that calls the model
def call_model(state:  State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc345"}}
query = "What math problem did I ask you last?"
language = "English"

input_messages = messages + [HumanMessage(query)]

for chunk, metadata in app.stream(
    {"messages": input_messages, "language": "English"}, config, stream_mode="messages"):
    if isinstance(chunk, AIMessage):
        print(chunk.content, end=" ", flush=True)

# output = app.invoke({"messages": input_messages, "language": "English"}, config)
# output["messages"][-1].pretty_print()
