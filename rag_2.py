import bs4
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.graph import END
from langgraph.checkpoint.memory import MemorySaver

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vectore_store = InMemoryVectorStore(embeddings)

loader = WebBaseLoader(
    web_path = ("https://lilianweng.github.io/posts/2023-06-23-agent/"), # https://lilianweng.github.io/posts/2023-06-23-agent/
    bs_kwargs = dict(
        parse_only=bs4.SoupStrainer(
            class_ = ("post-content", "post-title", "post-header")
        )
    )
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

_ = vectore_store.add_documents(documents=all_splits)

graph_builder = StateGraph(MessagesState)


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to input query"""
    retrieved_docs = vectore_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}") for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Step-1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond"""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

# Step-2: Execute the Retrieval.
tools = ToolNode([retrieve])

# Step-3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get Generated Tool message
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into  prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for questions-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. Use three "
        "sentences maximum and keep the answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message for message in state["messages"] 
        if message.type in ("humnan", "system") 
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

# graph_builder.add_node(query_or_respond)
# graph_builder.add_node(tools)
# graph_builder.add_node(generate)

# graph_builder.set_entry_point("query_or_respond")
# graph_builder.add_conditional_edges(
#     "query_or_respond",
#     tools_condition,
#     {END: END, "tools": "tools"},
# )
# graph_builder.add_edge("tools", "generate")
# graph_builder.add_edge("generate", END)

memory = MemorySaver()

# graph = graph_builder.compile(checkpointer=memory)
agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)

# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}}

# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))

input_message = "What is Task Decomposition?"

for step in agent_executor.stream(
    {"messages": [{"role":  "user",  "content": input_message}]}, 
    stream_mode="values",
    config=config,
):
    step["messages"][-1].pretty_print()


input_message = "Can you look up some common ways of doing it?"

for step in agent_executor.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    step["messages"][-1].pretty_print()