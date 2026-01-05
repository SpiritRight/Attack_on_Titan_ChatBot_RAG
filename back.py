from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from dotenv import load_dotenv

from config import answer_examples

load_dotenv()



store = {}

REPLACEMENTS = {
    "사람을 나타내는 표현": "진격의거인 애니메이션 안의 등장인물이나 사람들",
}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        print(store)
    return store[session_id]


@lru_cache(maxsize=1)
def get_embeddings():
    return OpenAIEmbeddings(model='text-embedding-3-small')


@lru_cache(maxsize=1)
def get_retriever():
    # database = Chroma(collection_name='chroma-inu-new', persist_directory="./chroma_inu-new", embedding_function=get_embeddings()) # 학칙만
    database = Chroma(collection_name='attackTitan', persist_directory="./attackTitan", embedding_function=get_embeddings()) #학칙+장학금
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever

def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def get_llm(model='gpt-5-mini'):
    llm = ChatOpenAI(model=model)
    # llm = Ollama(model=model)
    return llm


def normalize_question(question: str) -> str:
    normalized = question
    for source, target in REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    return normalized


def _sanitize_metadata(metadata: dict) -> dict:
    if not metadata:
        return {}
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def get_answer_chain():
    llm = get_llm()
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )
    system_prompt = (
    """당신은 애니메이션 진격의 거인의 전문가이며, 사용자의 진격의거인 관련 질문에 정확하고 상세한 답변을 제공해야 합니다. 아래의 사항을 철저히 준수하여 응답하세요.

    \n\n{context}"""
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return question_answer_chain


def get_retrieved_context(user_message: str, session_id: str):
    normalized_question = normalize_question(user_message)
    docs = retrieve_docs(normalized_question, session_id)
    return [
        {
            "text": doc.page_content,
            "metadata": _sanitize_metadata(doc.metadata),
        }
        for doc in docs
    ]


def retrieve_docs(question: str, session_id: str):
    history = get_session_history(session_id)
    if len(history.messages) >= 2:
        history_aware_retriever = get_history_retriever()
        return history_aware_retriever.invoke(
            {"input": question, "chat_history": history.messages}
        )
    return get_retriever().invoke(question)


def _stream_answer(chain, inputs: dict, history: BaseChatMessageHistory, user_message: str):
    buffer = []
    for chunk in chain.stream(inputs):
        buffer.append(chunk)
        yield chunk
    answer = "".join(buffer)
    history.add_user_message(user_message)
    history.add_ai_message(answer)


def get_ai_response(user_message, session_id: str = "abc123"):
    normalized_question = normalize_question(user_message)
    docs = retrieve_docs(normalized_question, session_id)
    retrieved_context = [
        {
            "text": doc.page_content,
            "metadata": _sanitize_metadata(doc.metadata),
        }
        for doc in docs
    ]

    history = get_session_history(session_id)
    answer_chain = get_answer_chain()
    inputs = {
        "input": normalized_question,
        "context": docs,
        "chat_history": history.messages,
    }
    ai_response = _stream_answer(answer_chain, inputs, history, user_message)

    return ai_response, retrieved_context
