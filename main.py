from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model


model = init_chat_model("gpt-4o-mini", model_provider="openai")


system_template = "Translate the following from english to {language}"

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", "{text}")])

prompt = prompt_template.invoke({"language": "Hindi", "text": "I love programming"})
# print(prompt)

response = model.invoke(prompt)
print(response.content)

