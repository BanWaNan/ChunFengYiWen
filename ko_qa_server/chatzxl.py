import json
import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_community.vectorstores import FAISS
import faiss
from pathlib import Path
from build_vector_store import load_documents, file2vector, merge_incomplete_chunks
from langchain_community.docstore.in_memory import InMemoryDocstore
from file2json import extract_text_from_file, extract_text_from_TXT
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, OrGroup
import jieba
from whoosh.analysis import Tokenizer, Token
from langchain.text_splitter import RecursiveCharacterTextSplitter


# 自定义 jieba 分词器，确保每个 token 包含 pos 属性
class JiebaTokenizer(Tokenizer):
    def __call__(self, value, **kwargs):
        if not isinstance(value, str):
            value = value.decode("utf-8")
        t = Token()
        start = 0
        for w in jieba.cut(value, cut_all=False):
            t.original = w
            t.text = w
            t.boost = 1.0
            t.startchar = value.find(w, start)
            t.endchar = t.startchar + len(w)
            t.pos = 0  # 添加默认 pos 属性
            start = t.endchar
            yield t


def jieba_analyzer():
    return JiebaTokenizer()


def prompt_for_RAG_llm(context, question, history=None):
    history_str = ""
    if history and len(history) > 0:
        history_str = '\n\n'.join([f"{message['role']}: {message['content']}" for message in history if message])
    if history_str.strip():
        history_str += "**历史对话**  \n" + history_str
    prompt = f"""
- 目标 -
根据提供的知识库内容，准确回答用户的问题。如果知识库中没有相关信息，请明确告知。

- 角色 -
您扮演的角色是“变压器故障诊断小助手”，专门利用知识库中匹配到的内容为用户提供专业解答。

- 注意事项 -
1. 确保回答与问题完全对应，避免偏离主题。
2. 匹配到的知识存在与问题无关的内容，请注意筛选。充分利用对问题有用知识库内容
3. 如果知识库中没有相关信息，请明确告知用户“当前知识库无法准确回答您的问题，请等待知识库更新...”。
4. 回答应简洁明了，避免冗余信息。

- 知识库内容 -
{"-" * 30}
{context.strip()}
{"-" * 30}

- 问题 -
"{question}"
"""
    return prompt


class Chatzxl:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key="sk-LkwbGvnbbeaRz4vOWeF07xe1mTNBnFWz9EkKKtq5kPwBAKCz",
            # api_key="sk-LGqyU0GSCzetyPXMoTv68FcgGg9Js4g2aHMxQyDJWz65i0zh",
            openai_api_base="https://api.chatanywhere.tech/v1",
            model="gpt-3.5-turbo",
            temperature=0.3,
        )

    def init_database(self, bm25_index_dir='bm25_index_dir', vector_index_dir='vector_stores_bge_m3'):
        # self.bm25_retriever = DocumentRetrievalSystem(index_dir=bm25_index_dir)
        self.embeddings = OpenAIEmbeddings(openai_api_base="https://api.chatanywhere.tech/v1",
                                           openai_api_key="sk-LkwbGvnbbeaRz4vOWeF07xe1mTNBnFWz9EkKKtq5kPwBAKCz")

    def create_new_vector_stroes(self, user_vector_stores_path):
        embeddings = self.embeddings
        dummy_vector = embeddings.embed_query("hello world")
        embedding_dim = len(dummy_vector)
        index = faiss.IndexFlatL2(embedding_dim)
        vector_stores = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        vector_stores.save_local(user_vector_stores_path)

    def update_vector_stores(self, file_path, file_name, userId, filetype):

        if filetype == 'TXT':
            json_data, save_path = extract_text_from_TXT(file_path, file_name)
        else:
            json_data, save_path = extract_text_from_file(file_path, file_name, filetype)
        self.create_keyword_index(save_path, userId)

        docs = load_documents(save_path)
        for doc in docs:
            # 打印文档的唯一标识和标题（假设标题存储在 metadata 中的 "名称" 字段）
            print("Document ID:", doc.metadata.get("id", "N/A"))
            print("Title:", doc.metadata.get("名称", ""))
            print("Content:")
            print(doc.page_content)
            print("=" * 80)
        print("成功加载到用户" + userId + "上传的文档")
        embeddings = self.embeddings

        user_vector_stores_path = Path(f"./user_vector_stores/{userId}").resolve()
        print(11)
        vector_stores = FAISS.load_local(user_vector_stores_path, embeddings, allow_dangerous_deserialization=True)
        print(11)
        user_json_path = Path(f"./user_vector_stores/{userId}/doc_rangs.json").resolve()
        vector_stores = file2vector(docs, vector_stores, file_name, user_json_path)
        vector_stores.save_local(user_vector_stores_path)

        print(f"该文档上传成功！")

    def create_keyword_index(self, json_path, userId):
        with open(json_path, "r", encoding='utf-8') as f:
            json_content = json.load(f)
        # print(json_content['content'])
        # 定义 schema 时为 content 字段指定中文分词器
        schema = Schema(
            title=TEXT(stored=True, analyzer=jieba_analyzer()),
            path=ID(stored=True),
            content=TEXT(stored=True, analyzer=jieba_analyzer())
        )
        index_dir_str = Path(f"./user_vector_stores/{userId}/index").resolve()
        if not os.path.exists(index_dir_str):
            os.makedirs(index_dir_str)
        ix = create_in(index_dir_str, schema)
        print(f"为用户{userId}创建新的关键词索引库！")
        THRESHOLD = 250
        # 优化后的分隔符列表（按优先级排序）
        custom_separators = [
            # 按匹配优先度排序
            # 匹配中文序号（一、 二、 等）
            r'\\n+[一二三四五六七八九十]+、',  # 匹配 "一、" 或 "二."
            # 匹配中文括号序号（如：一、二）包含换行
            r'\\n+[(（]+[一二三四五六七八九十]+[）)]',
            # 保留原有的Markdown标题
            "\n## ", "\n### ",
            # 匹配数字编号（如：1.  2.1）
            r'\\n+[\d]+[.]',
            r'\\n+[\d]+[.]+[\d]'
            # 其他通用分隔符
            "\n\n", "\n", "。", "！", "？"
        ]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=THRESHOLD,
            chunk_overlap=30,
            separators=custom_separators,  # 使用正则表达式
            length_function=len,
            is_separator_regex=True  # 启用正则模式
        )
        segments = text_splitter.split_text(json_content['content'])
        optimized_chunks = merge_incomplete_chunks(segments, THRESHOLD)
        # 将拆分的每个段落分别加入索引中
        writer = ix.writer()
        for idx, segment in enumerate(optimized_chunks):
            # 使用原始 path 加上段落序号，构造唯一标识符
            segment_path = f"{idx}"
            writer.add_document(title=json_content['名称'], path=segment_path, content=segment)
        writer.commit()
        print(f"文档已经存入用户{userId}的关键词索引库中")

    def koquery2kb(self, query, userid):

        rewritten_query = query

        user_vector_stores_path = Path(f"./user_vector_stores/{userid}").resolve()
        user_keyword_index_path = Path(f"./user_vector_stores/{userid}/index").resolve()
        vector_stores_user = FAISS.load_local(str(user_vector_stores_path), embeddings=self.embeddings,
                                              allow_dangerous_deserialization=True)
        print("用户知识库" + str(user_vector_stores_path) + "加载完成")

        ix = open_dir(str(user_keyword_index_path))
        print("用户关键词索引" + str(user_keyword_index_path) + "加载完成")
        # 向量库查询
        vector_docs = vector_stores_user.similarity_search(rewritten_query, k=5)
        # 关键词索引查询
        matched_segments = []  # 用来存储匹配到的文档段
        with ix.searcher() as searcher:
            parser = QueryParser("content", ix.schema, group=OrGroup.factory(0.9))
            query = parser.parse(query)
            results = searcher.search(query, limit=5)
            for hit in results:
                # 收集匹配到的段落内容
                matched_segments.append(hit.get("content"))

        docs = [d.metadata for d in vector_docs]
        for d in docs:
            # if "content" in d.keys():
            #     del d['content']
            if "id" in d.keys():
                del d['id']
        contexts = []
        for d in docs:
            # content = f"\n{d['名称']} \n"
            content = ""
            for k, v in d.items():
                if k == '名称':
                    continue
                content += f"\n{k}: {v} \n"
            contexts.append(content)
        # 将向量库中的上下文拼接起来
        final_context = "---------------".join(contexts)
        # 将关键词匹配的段落拼接起来
        matched_context = "\n---------------\n".join(matched_segments)
        # 将二者合并到一起
        final_context = final_context + "\n---------------\n" + matched_context
        # final_context = "\n---------------\n" + matched_context

        history = None
        prompt = prompt_for_RAG_llm(final_context, rewritten_query, history)
        print("prompt生成完成！")
        print(prompt)
        return prompt

