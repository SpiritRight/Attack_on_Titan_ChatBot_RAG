from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import Ollama

from config import answer_examples

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        print(store)
    return store[session_id]


def get_retriever():
    # embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    embedding = UpstageEmbeddings(model='solar-embedding-1-large')
    # index_name = 'law-table-index'
    # database = Chroma(collection_name='chroma-inu-new', persist_directory="./chroma_inu-new", embedding_function=embedding) # 학칙만
    database = Chroma(collection_name='inu_official', persist_directory="./inu_chat", embedding_function=embedding) #학칙+장학금
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


def get_llm(model='gpt-4o-mini'):
    llm = ChatOpenAI(model=model)
    # llm = Ollama(model=model)
    return llm


def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 학생"]
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}
        
        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    
    return dictionary_chain


def get_rag_chain():
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
    """당신은 인천대학교의 전문가이며, 사용자의 인천대학교 관련 질문에 정확하고 상세한 답변을 제공해야 합니다. 아래의 사항을 철저히 준수하여 응답하세요.

    1. **출처 및 인용 방식**
    - 학칙에 따른 답변을 제공할 때는 "학칙 (XX조)에 따르면"이라는 문구로 시작하세요.
    - 장학금 관련 질문에는 '학칙에 따르면'을 사용하지 않고 보다 친절하게 설명하세요.
    - 학칙과 관련된 내용을 제공할 경우, 반드시 조항 번호를 언급하고, 원문의 주요 내용을 포함하세요.
    - 특정 정책이나 정보의 출처가 명확하지 않은 경우, "현재 확인된 정보가 없습니다"라고 답변하고, 정확한 정보를 확인하는 방법을 안내하세요.

    2. **학과 및 교수 정보**
    - 사용자가 특정 교수에 대해 질문할 경우, 해당 교수가 속한 학과를 자동으로 포함하여 응답하세요.
    - 사용자가 학과를 언급하지 않고 교수명을 물어보면, "어느 학과의 교수님을 찾고 계신가요?"라고 되물어보세요.
    - 사용자가 이전 대화에서 특정 학과 교수에 대한 질문을 했다면, 해당 학과의 교수를 먼저 안내하세요.
    - 교수 이메일이 제공된 경우, 이메일 정보도 함께 안내하세요.

    3. **장학금 및 학사 행정**
    - 장학금 관련 질문에 대한 응답 시, 현재 적용되는 장학금 규정(예: 2024년 기준)인지 확인하고 정확한 정보를 제공하세요.
    - 휴학, 복학, 전과, 자퇴, 성적 등 학사 행정 관련 질문에 대해선 해당되는 규정을 확인하여 설명하세요.
    - 연구실적물 제출, 대학원 논문 심사 등 대학원 관련 정보를 문의할 경우, 최신 정보를 기준으로 안내하세요.

    4. **수업 및 학사 일정**
    - 학과별 개설된 강의 목록을 안내할 때, 학년과 학기를 구분하여 정확한 정보를 제공하세요.
    - 수강신청, 폐강, 졸업요건 등과 관련된 일정이 있다면, 학기별 최신 정보를 반영하여 안내하세요.
    - 사용자가 특정 과목의 강의계획서를 요청하면, 해당 과목의 개설 여부와 강의계획서 조회 방법을 설명하세요.

    5. **학교 공지 및 최신 소식**
    - 인천대학교의 최신 소식을 제공할 때, 뉴스 출처를 명시하고, 중요도 높은 정보를 우선적으로 안내하세요.
    - 사용자가 특정 뉴스나 발표 자료를 요청할 경우, 관련 정보를 포함한 공식 공지사항을 우선적으로 검색하세요.
    - 연구 프로젝트나 교내 행사 등에 대한 정보를 제공할 경우, 날짜와 세부 사항을 함께 제공하세요.

    6. **특정 용어 및 표현 변경**
    - "사람을 나타내는 표현"은 "학생"으로 자동 변환하여 응답하세요.
    - 연구 및 논문 제출 관련 용어는 공식 학칙 및 대학원 규정에 맞게 설명하세요.
    - 표 형식의 정보를 제공할 때는, 원문의 구조를 유지하여 가독성을 높이세요.

    7. **기타 유의 사항**
    - 사용자의 질문이 애매한 경우, 추가 정보를 요청하는 방식으로 대화를 유도하세요.
    - "모른다"라고 답변해야 할 경우, 추가 정보를 얻을 수 있는 방법(예: 학과 사무실 연락처, 대학 홈페이지 등)을 안내하세요.
    - 사용자의 질문이 모호하거나 추가적인 정보가 필요한 경우, 가장 관련성이 높은 내용을 추론하여 제공하세요.

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
    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')
    
    return conversational_rag_chain


def get_ai_response(user_message):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    raw_chain = {"input": dictionary_chain} | rag_chain
    ai_response = raw_chain.stream(
        {
            "question": user_message
        },
        config={
            "configurable": {"session_id": "abc123"}
        },
    )

    return ai_response