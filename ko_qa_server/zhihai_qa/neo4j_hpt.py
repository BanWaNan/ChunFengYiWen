import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yaml
from .cypher_tool import query_neo4j
from .kdzn_rule_ent import kdzn_file_rules_kdzn_generate
from pathlib import Path

# 1. 取得当前文件(neo4j_hpt.py)的绝对目录
HERE = Path(__file__).resolve().parent
# 2. 回到 ko_qa_sever 根目录
BASE_DIR = HERE.parent
# 3. 拼出 config.yaml 的完整路径
CFG_FILE = BASE_DIR / "config.yaml"
# 4. 读取
with CFG_FILE.open(encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

reasoning_rules = {
    "父母": [
        """match p=(n:huaputong{name:"{entityname}"})-[r]->(b) where r.name in ["父亲","father","继父"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[{name: "母亲"}]->(b) return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[{name: "妻子"}]->(b) where r1.name in ["父亲","father","继父"] return p""",
    ],
    "父亲": [
        """match p=(n:huaputong{name:"{entityname}"})-[r]->(b) where r.name in ["父亲","father","继父"] return p""",
    ],
    "继父": [
        """match p=(n:huaputong{name:"{entityname}"})-[{name: "继父"}]->(b)  return p""",
    ],
    "母亲": [
        """match p=(n:huaputong{name:"{entityname}"})-[{name: "母亲"}]->(b) return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[{name: "妻子"}]->(b) where r1.name in ["父亲","father","继父"] return p"""
    ],
    "兄弟": [
        """match p=(n:huaputong{name:"{entityname}"})-[r]->(b) where r.name in ["哥哥","兄"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r]->(b) where r.name in ["弟弟","弟"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[{name: "儿子"}]->(b) where r1.name in ["父亲","father","继父"] and b <> n return p"""
    ],
    "哥哥": [
        """match p=(n:huaputong{name:"{entityname}"})-[r]->(b) where r.name in ["哥哥","兄"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[{name: "儿子"}]->(b) where r1.name in ["父亲","father","继父"] and b <> n return p"""
    ],
    "弟弟": [
        """match p=(n:huaputong{name:"{entityname}"})-[r]->(b) where r.name in ["弟弟","弟"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[{name: "儿子"}]->(b) where r1.name in ["父亲","father","继父"] and b <> n return p"""
    ],
    "姐妹": [
        """match p=(n:huaputong{name:"{entityname}"})-[r]->(b) where r.name in ["姐姐","姐"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r]->(b) where r.name in ["妹妹","妹"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[{name: "女儿"}]->(b) where r1.name in ["父亲","father","继父"] and b <> n return p"""
    ],
    "姐姐": [
        """match p=(n:huaputong{name:"{entityname}"})-[r]->(b) where r.name in ["姐姐","姐"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[{name: "女儿"}]->(b) where r1.name in ["父亲","father","继父"] and b <> n return p"""
    ],
    "妹妹": [
        """match p=(n:huaputong{name:"{entityname}"})-[r]->(b) where r.name in ["妹妹","妹"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[{name: "女儿"}]->(b) where r1.name in ["父亲","father","继父"] and b <> n return p"""
    ],
    "丈夫": [
        """match p=(n:huaputong{name:"{entityname}"})-[{name: "丈夫"}]->(b) return p"""
    ],
    "妻子": [
        """match p=(n:huaputong{name:"{entityname}"})-[r]->(b) where r.name in ["妻子","wife"] return p"""
    ],
    "儿子": [
        """match p=(n:huaputong{name:"{entityname}"})-[r]->(b) where r.name in ["儿子","son","继子"] return p"""
    ],
    "女儿": [
        """match p=(n:huaputong{name:"{entityname}"})-[{name: "女儿"}]->(b) return p"""
    ],
    "爷爷": [
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[r2]->(b) where r1.name in ["父亲","father","继父"] and r2.name in ["父亲","father","继父"] return p"""
    ],
    "奶奶": [
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[{name: "母亲"}]->(b) where r1.name in ["父亲","father","继父"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[r2]->()-[r3]->(b) where r1.name in ["父亲","father","继父"] and r2.name in ["父亲","father","继父"] and r3.name in ["妻子","wife"] return p"""
    ],
    "孙女": [
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[{name: "女儿"}]->(b) where r1.name in ["儿子","son","继子"] return p"""
    ],
    "孙子": [
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[r2]->(b) where r1.name in ["儿子","son","继子"] and r2.name in ["儿子","son","继子"] return p"""
    ],
    "姑姑": [
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[r2]->(b) where r1.name in ["父亲","father","继父"] and r2.name in ["姐姐","姐"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[r2]->(b) where r1.name in ["父亲","father","继父"] and r2.name in ["妹妹","妹"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[r2]->()-[r3]->(b) where r1.name in ["父亲","father","继父"] and r2.name in ["父亲","father","继父"] and r3.name in ["女儿"] return p"""
    ],
    "姑父": [
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[r2]->()-[{name: "丈夫"}]->(b) where r1.name in ["父亲","father","继父"] and r2.name in ["姐姐","姐"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[r2]->()-[{name: "丈夫"}]->(b) where r1.name in ["父亲","father","继父"] and r2.name in ["妹妹","妹"] return p"""
    ],
    "伯伯": [
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[r2]->(b) where r1.name in ["父亲","father","继父"] and r2.name in ["哥哥","兄"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[r2]->(b) where r1.name in ["父亲","father","继父"] and r2.name in ["弟弟","弟"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[r2]->()-[r3]->(b) where r1.name in ["父亲","father","继父"] and r2.name in ["父亲","father","继父"] and r3.name in ["儿子","son","继子"] return p"""
    ],
    "婶婶": [
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[r2]->()-[r3]->(b) where r1.name in ["父亲","father","继父"] and r2.name in ["哥哥","兄"] and r3.name in ["妻子","wife"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[r2]->()-[r3]->(b) where r1.name in ["父亲","father","继父"] and r2.name in ["弟弟","弟"] and r3.name in ["妻子","wife"] return p"""
    ],
    "侄子": [
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[r2]->(b) where r1.name in ["哥哥","兄"] and r2.name in ["儿子","son","继子"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[r2]->(b) where r1.name in ["弟弟","弟"] and r2.name in ["儿子","son","继子"] return p"""
    ],
    "侄女": [
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[{name: "女儿"}]->(b) where r1.name in ["哥哥","兄"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[r1]->()-[{name: "女儿"}]->(b) where r1.name in ["弟弟","弟"] return p"""
    ],
    "外公": [
        """match p=(n:huaputong{name:"{entityname}"})-[{name: "母亲"}]->()-[r2]->(b) where r2.name in ["父亲","father","继父"] return p"""
    ],
    "外婆": [
        """match p=(n:huaputong{name:"{entityname}"})-[{name: "母亲"}]->()-[{name: "母亲"}]->(b) return p"""
    ],
    "舅舅": [
        """match p=(n:huaputong{name:"{entityname}"})-[{name: "母亲"}]->()-[r1]->(b) where r1.name in ["哥哥","兄"] return p""",
        """match p=(n:huaputong{name:"{entityname}"})-[{name: "母亲"}]->()-[r1]->(b) where r1.name in ["弟弟","弟"] return p"""
    ],
    "舅妈": [
        """match p=(n:huaputong{name:"{entityname}"})-[{name: "母亲"}]->()-[r1]->()-[r2]->(b) where r1.name in ["哥哥","兄"] and r2.name in ["妻子","wife"] return b.name as name""",
        """match p=(n:huaputong{name:"{entityname}"})-[{name: "母亲"}]->()-[r1]->()-[r2]->(b) where r1.name in ["弟弟","弟"] and r2.name in ["妻子","wife"] return b.name as name"""
    ],
    "姨妈": [
        """match p=(n:huaputong{name:"{entityname}"})-[{name: "母亲"}]->()-[r1]->(b) where r1.name in ["姐姐","姐"] return b.name as name""",
        """match p=(n:huaputong{name:"{entityname}"})-[{name: "母亲"}]->()-[r1]->(b) where r1.name in ["妹妹","妹"] return b.name as name"""
    ],
    "姨父": [
        """match p=(n:huaputong{name:"{entityname}"})-[{name: "母亲"}]->()-[r1]->()-[r2]->(b) where r1.name in ["姐姐","姐"] and r2.name in ["丈夫"] return b.name as name""",
        """match p=(n:huaputong{name:"{entityname}"})-[{name: "母亲"}]->()-[r1]->()-[r2]->(b) where r1.name in ["妹妹","妹"] and r2.name in ["丈夫"] return b.name as name"""
    ],
    "校友": [
        """MATCH p=(n:ownthink{name:"{entityname}"})-[{name:"毕业院校"}]->()<-[{name:"毕业院校"}]-(b) RETURN p LIMIT 15"""],
    "同事": [
        """MATCH p=(n:ownthink{name:"{entityname}"})-[{name:"就职于"}]->()<-[{name:"就职于"}]-(b) RETURN p LIMIT 15"""],
    "同行": [
        """MATCH p=(n:ownthink{name:"{entityname}"})-[{name:"研究领域"}]->()<-[{name:"研究领域"}]-(b) RETURN p LIMIT 15"""],
    "老乡": [
        """MATCH p=(n:ownthink{name:"{entityname}"})-[{name:"出生地"}]->()<-[{name:"出生地"}]-(b) RETURN p LIMIT 15"""],
}


def remove_substrings(str_list):
    """
    移除列表中是其他字符串子串的字符串
    """
    sorted_list = sorted(str_list, key=len, reverse=True)
    result = []
    for i, s1 in enumerate(sorted_list):
        is_substring = False
        # 只需要和比自己长的字符串比较
        for s2 in sorted_list[:i]:
            if s1 in s2:
                is_substring = True
                break
        if not is_substring:
            result.append(s1)
    return result


def get_chinese_properties(node):
    return {k: v for k, v in node._properties.items() if
            isinstance(k, str) and any('\u4e00' <= char <= '\u9fff' for char in k) or k == "requirement"}


def get_text_embedding(text, model, tokenizer):
    # 对文本进行编码
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # 获取BERT模型的输出
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取 [CLS] token 的表示，通常用于句子级别的表示
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # 获取 [CLS] 位置的向量
    return embeddings


def getYiChang(query_text, fuzzy_relation, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    all_query_result = {}
    all_embeddings = {}
    kdzn_file_rules_kdzn = kdzn_file_rules_kdzn_generate()
    print("1111111", kdzn_file_rules_kdzn)
    for key, value in kdzn_file_rules_kdzn.items():
        if value[0]["getIn4simi"] != """""":
            query_result = query_neo4j(value[0]["getIn4simi"], kg='kdzn')
            ents = []
            for record in query_result:
                ent = []
                if ("R1Name" in record[0]):
                    ent.append(record[0]["R1Name"])
                if ("R2Name" in record[0]):
                    ent.append(record[0]["R2Name"])
                if ("RIName" in record[0]):
                    ent.append(record[0]["RIName"])
                ents.append(ent)

            all_query_result[key] = ents
            if ents == []:
                continue
            embeddings = np.array([get_text_embedding("".join(ent_names), model, tokenizer) for ent_names in ents])
            all_embeddings[key] = embeddings
        else:
            all_query_result[key] = []

    # 将文本嵌入向量添加到Faiss索引中
    d = 1024
    index = {}
    for key, value in all_embeddings.items():
        index[key] = faiss.IndexFlatL2(d)  # 创建一个L2距离的索引
        index[key].add(value.astype(np.float32))  # 将嵌入向量添加到Faiss索引中
        # index_filename = f"{key}_index_file.index"
        # faiss.write_index(index[key], index_filename)

    query_embedding = get_text_embedding(query_text, model, tokenizer).reshape(1, -1)  # 将查询文本转换为嵌入向量
    k = 1  # 返回前k个最相似的文本
    ansYiChang = ''  # 异常大类：文件
    ansIndices = 0  # 异常小类下标
    ansDistances = float("inf")  # 最小距离

    # 一个个寻找每个异常大类匹配度最大的异常小类，只保留距离最近的异常小类
    for key, value in index.items():
        distances, indices = index[key].search(query_embedding.astype(np.float32), k)
        if distances.min() < ansDistances:
            ansDistances = distances.min()
            ansYiChang = key
            ansIndices = indices.min()

    ans = all_query_result[ansYiChang][ansIndices][-1]

    # 路径抽取
    choose_kdzn_file_rules = kdzn_file_rules_kdzn
    choose_neo4j = 'kdzn'
    print(f"选中文件及事件：【{ansYiChang}】--【{ans}】")
    rules = choose_kdzn_file_rules.get(ansYiChang)
    if len(rules) == 0:
        return None, None, None
    elif len(rules) == 1:
        rrule = rules[0]
    else:
        file_rules = [rule["rule_name"] for rule in rules]
        max_rule_socre, match_rule = getBest_phrase(fuzzy_relation, file_rules)
        if max_rule_socre > 0:  # 匹配相似规则
            rrule = match_rule
        else:
            rrule = rules[0]

    cyphers = rrule["rule_path"]
    result_names = []
    paths = []
    path2triplet = []
    path_triplets = []
    for cypher in cyphers:
        cypher_result = query_neo4j(cypher.replace("{entityname}", ans), kg=choose_neo4j)
        for record in cypher_result:
            path = record[0]["p"]
            relationships = list(path.relationships)
            path_str = ""
            tail_node = ""
            result = ""
            p2ts = []
            for i in range(len(relationships)):
                nodes = relationships[i].nodes
                start_node = nodes[0]["name"]
                # nodes[0]._properties
                end_node = nodes[1]["name"]
                rel = relationships[i]["name"]
                if rel is None:
                    rel = relationships[i].type
                if (start_node, rel, end_node) not in path_triplets:
                    path_triplets.append((start_node, rel, end_node))
                p2ts.append((start_node, rel, end_node))

                # 增加中文属性三元组到path_triplets

                start_node_chinese_properties = get_chinese_properties(nodes[0])
                end_node_chinese_properties = get_chinese_properties(nodes[1])

                for node, properties in [(start_node, start_node_chinese_properties),
                                         (end_node, end_node_chinese_properties)]:
                    for k, v in properties.items():
                        if k == "requirement":
                            k = "要求"
                        path_triplets.append((node, k, v)) if (node, k, v) not in path_triplets else None
                        p2ts.append((node, k, v)) if (node, k, v) not in path_triplets else None

                # 如果非首段关系，判断关系方向
                if i == 0:
                    if start_node == ans:
                        path_str += f"{start_node}-{rel}->{end_node}"
                        result = end_node
                        tail_node = end_node
                    else:
                        path_str += f"{end_node}<-{rel}-{start_node}"
                        result = start_node
                        tail_node = start_node
                else:
                    if tail_node == start_node:
                        path_str += f"-{rel}->{end_node}"
                        tail_node = end_node
                        result = end_node
                    else:
                        path_str += f"<-{rel}-{start_node}"
                        tail_node = start_node
                        result = start_node

            result_names.append(result) if result not in result_names else None
            paths.append(path_str) if path_str not in paths else None
            path2triplet.append(p2ts) if p2ts not in path2triplet else None

    return result_names, paths, path2triplet, path_triplets


def query_relationship(entityname, relationship):
    cypher = reasoning_rules.get(relationship, [])
    result_names = []
    result_des_n = []
    result_des_b = []
    paths = []
    path_triplets = []
    for query in cypher:
        query_result = query_neo4j(query.replace("{entityname}", entityname))
        for record in query_result:
            path = record[0]["p"]
            relationships = list(path.relationships)

            path_str = ""
            tail_node = ""
            des_n = path.start_node["des"]  # 获取n的des
            des_b = path.end_node["des"]  # 获取b的des

            for i in range(len(relationships)):
                nodes = relationships[i].nodes
                start_node = nodes[0]["name"]
                # nodes[0]._properties
                end_node = nodes[1]["name"]
                rel = relationships[i]["name"]
                if rel is None:
                    rel = relationships[i].type
                if (start_node, rel, end_node) not in path_triplets:
                    path_triplets.append((start_node, rel, end_node))
                # 如果非首段关系，判断关系方向
                if i == 0:
                    if start_node == entityname:
                        path_str += f"{start_node}-{rel}->{end_node}"
                        result = end_node
                        tail_node = end_node
                    else:
                        path_str += f"{end_node}<-{rel}-{start_node}"
                        result = start_node
                        tail_node = start_node
                else:
                    if tail_node == start_node:
                        path_str += f"-{rel}->{end_node}"
                        tail_node = end_node
                        result = end_node
                    else:
                        path_str += f"<-{rel}-{start_node}"
                        tail_node = start_node
                        result = start_node

            # 只添加不重复的des
            if des_n not in result_des_n:
                result_des_n.append(des_n)  # 存储n的des
            if des_b not in result_des_b:
                result_des_b.append(des_b)  # 存储b的des
            if result not in result_names:
                result_names.append(result)
            if path_str not in paths:
                paths.append(path_str)
    return result_names, paths, result_des_n, result_des_b, path_triplets


def query_entity_properties(entityname, label="ownthink"):
    # 查询指定的节点及其属性
    query = """MATCH (n:{__label__}{name: "{entityname}"}) 
                RETURN n LIMIT 1""".replace("{__label__}", label)
    # 执行查询
    query_result = query_neo4j(query.replace("{entityname}", entityname))
    # 打印查询结果以调试
    # 如果查询没有结果，返回空
    if not query_result:
        return None
    # 处理查询结果，提取节点的属性
    for record in query_result:
        # 打印每个记录的内容
        # 正确访问节点信息
        node = record['value']['n']  # 使用 'value' 来访问节点
        # 提取 'des' 字段
        des = node.get('des', 'No description available')  # 使用 .get() 方法
        return des
    # 如果查询没有返回 'des' 字段，返回空
    return None


def getBest_phrases(fuzzy_ent, getEntList):  # 返回相似实体列表
    split_words = ['的', '了', '是', '在', '有', '和', '就', '这', '个', '，', '。', '？', '！', '：', '；', '', ' ', '（',
                   '）', '、', '-', '“', '”']
    q_words = jieba.cut(fuzzy_ent, cut_all=False)
    set_q_words = set(q_words)
    filtered_q_words = set_q_words - set(split_words)
    ent_score = []
    for kg_ent in getEntList:
        p_words = jieba.cut(kg_ent)
        filtered_p_words = set(p_words) - set(split_words)
        # score = len(filtered_q_words.intersection(filtered_p_words)) / len(filtered_p_words)
        score = len(filtered_q_words.intersection(filtered_p_words)) / len(filtered_q_words)
        ent_score.append((kg_ent, score))
    max_score = max(score for _, score in ent_score)
    best_ents = [string for string, score in ent_score if score == max_score]
    return max_score, best_ents


def getBest_phrase(fuzzy_ent, getEntList):
    split_words = ['的', '了', '是', '在', '有', '和', '就', '这', '个', '，', '。', '？', '！', '：', '；', '', ' ', '（',
                   '）', '、', '-', '“', '”']
    q_words = jieba.cut(fuzzy_ent, cut_all=False)
    set_q_words = set(q_words)
    filtered_q_words = set_q_words - set(split_words)
    max_score = 0
    best_ent = str()
    for kg_ent in getEntList:
        p_words = jieba.cut(kg_ent)
        filtered_p_words = set(p_words) - set(split_words)
        # score = len(filtered_q_words.intersection(filtered_p_words)) / len(filtered_p_words)
        score = len(filtered_q_words.intersection(filtered_p_words)) / len(filtered_q_words)
        if score > max_score:
            max_score = score
            best_ent = kg_ent
    return max_score, best_ent


def find_most_similar(query, candidates, model_client):
    """
    输入：query字符串，candidates字符串列表，model_client（有embeddings.create方法）
    输出：最相似的字符串及其相似度
    """

    def cosine_similarity(vec1, vec2):
        """计算两个向量的余弦相似度"""
        vec1, vec2 = np.array(vec1), np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # 拼接输入，先query，后候选
    all_texts = [query] + candidates
    # 批量获取向量（按需适配你的模型调用）
    result = model_client.embeddings.create(
        model="text-embedding-v1",
        input=all_texts,
    )
    embeddings = [emb.embedding for emb in result.data]
    query_vec = embeddings[0]
    candidate_vecs = embeddings[1:]
    # 计算每个候选的相似度
    sims = [cosine_similarity(query_vec, vec) for vec in candidate_vecs]
    max_idx = int(np.argmax(sims))
    return candidates[max_idx], sims[max_idx]


def getBest_rules_vec(query_for, rules_in_file, model_client):
    def cosine_similarity_np(vec1, vec2):
        """计算两个向量的余弦相似度"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)
    print("query_for", query_for)
    print("rules_in_file", rules_in_file)
    embeddings_got = []
    DASHSCOPE_MAX_BATCH_SIZE = 25
    batch_counter = 0
    rules_with_query = [query_for] + rules_in_file
    rules_with_query_simiscore = [1] + [0] * len(rules_in_file)
    for i in range(0, len(rules_with_query), DASHSCOPE_MAX_BATCH_SIZE):
        batch = rules_with_query[i:i + DASHSCOPE_MAX_BATCH_SIZE]
        completion = model_client.embeddings.create(
            model="text-embedding-v1",
            input=batch,
            # dimensions=512,  # v1/v2维度是1536改不了
            # encoding_format="float"
        )
        # print("\033[94m 向量请求 \033[0m")
        # print(f"\033[96mQwen[text-embedding-v3]使用token量：{completion.usage} \033[0m")
        if completion.data and len(completion.data) > 0:
            if not embeddings_got:
                embeddings_got = completion.data
            else:
                embeddings_got.extend(completion.data)
        else:
            print(completion)
        batch_counter += len(batch)
    # 根据向量相似度匹配最相近的规则
    embeddings_list = [emb.embedding for emb in embeddings_got]
    first_tensor = embeddings_list[0]  # 第一个张量：query_for的张量
    max_similarity = -1
    most_similar_index = -1
    for i in range(1, len(embeddings_list)):
        similarity = cosine_similarity_np(first_tensor, embeddings_list[i])
        rules_with_query_simiscore[i] = similarity
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_index = i
    match_best_rule = rules_with_query[most_similar_index]
    return max_similarity, match_best_rule


def find_kdzn(file_ontology, fuzzy_entity, fuzzy_relation, model_client):
    kdzn_file_rules_kdzn = kdzn_file_rules_kdzn_generate()  # 从mysql加载规则
    # print("1111111", kdzn_file_rules_kdzn)
    choose_kdzn_file_rules = kdzn_file_rules_kdzn
    # choose_kdzn_file_rules = kdzn_file_rules
    choose_neo4j = 'kdzn'

    rules = choose_kdzn_file_rules.get(file_ontology)
    if len(rules) == 0:
        return None, None, None, None
    elif len(rules) == 1:
        rrule = rules[0]
    else:
        file_rules = [rule["rule_name"] for rule in rules]
        # 根据qwne向量相似度获取这个文件里面与fuzzy_relation相近的规则头
        max_rule_socre, match_rule_name = getBest_rules_vec(fuzzy_relation, file_rules, model_client)
        print(f"规则匹配：“{fuzzy_relation}”从{len(file_rules)}个规则头中匹配结果:{match_rule_name}，得分：{max_rule_socre:.2f}")
        if max_rule_socre > 0:
            for rule in rules:
                if rule["rule_name"] == match_rule_name:
                    rrule = rule
        else:
            rrule = rules[0]
    query_neo4j_res = query_neo4j(rrule["getIn4simi"], kg=choose_neo4j)
    rule_in_ents = []  # 得到的检索实体（规则的输入实体）
    entstr2RIName = {}
    for record in query_neo4j_res:
        ent = []
        if ("R1Name" in record[0]):
            ent.append(record[0]["R1Name"])
        if ("R2Name" in record[0]):
            ent.append(record[0]["R2Name"])
        if ("RIName" in record[0]):
            ent.append(record[0]["RIName"])
        ent2str = "".join(ent)
        rule_in_ents.append(ent2str)
        entstr2RIName[ent2str] = record[0]["RIName"]
        # rule_in_ents.append(record[0]["RIName"])

    #     匹配相似实体
    # _, match_ent_span = getBest_phrase(fuzzy_entity, rule_in_ents)  # 词频匹配
    match_ent_score, match_ent_span = getBest_rules_vec(fuzzy_entity, rule_in_ents, model_client)  # 向量匹配
    match_ent = entstr2RIName.get(match_ent_span)

    print(
        f"实体段匹配：从{len(rule_in_ents)}个实体段中匹配结果:{match_ent_span}→对应检索实体:{match_ent}，得分：{match_ent_score:.2f}")
    if match_ent == file_ontology:
        match_ent_score = 1.1

    if match_ent_score < 0.5:
        return None, None, None, None
    # 检索结果
    cyphers = rrule["rule_path"]
    result_names = []
    paths = []
    path2triplet = []
    path_triplets = []
    if match_ent is not None:
        for cypher in cyphers:
            cypher_result = query_neo4j(cypher.replace("{entityname}", match_ent), kg=choose_neo4j)
            for record in cypher_result:
                path = record[0]["p"]
                relationships = list(path.relationships)
                path_str = ""
                tail_node = ""
                result = ""
                p2ts = []
                for i in range(len(relationships)):
                    nodes = relationships[i].nodes
                    start_node = nodes[0]["name"]
                    # nodes[0]._properties
                    end_node = nodes[1]["name"]
                    rel = relationships[i]["name"]
                    if rel is None:
                        rel = relationships[i].type
                    if rel == "kedazhineng" or rel == "KedaIntelligent":
                        rel = relationships[i]["edge_name"]
                    if rel is None:
                        rel = relationships[i]["type"]
                    if type(rel) is str:
                        if rel.startswith("包含-"):
                            rel = "包含"
                    if (start_node, rel, end_node) not in path_triplets:
                        path_triplets.append((start_node, rel, end_node))
                    p2ts.append((start_node, rel, end_node))
                    # 增加中文属性三元组到path_triplets
                    start_node_chinese_properties = get_chinese_properties(nodes[0])
                    end_node_chinese_properties = get_chinese_properties(nodes[1])

                    for node, properties in [(start_node, start_node_chinese_properties),
                                             (end_node, end_node_chinese_properties)]:
                        for k, v in properties.items():
                            if k == "requirement":
                                k = "要求"
                            path_triplets.append((node, k, v)) if (node, k, v) not in path_triplets else None
                            p2ts.append((node, k, v)) if (node, k, v) not in p2ts else None
                if len(p2ts) > 1:
                    first_tri, second_tri = p2ts[0], p2ts[1]
                    if first_tri[0] in [second_tri[0], second_tri[2]]:  # 左向路径
                        path_str += f"{first_tri[2]}<-[{first_tri[1]}]-{first_tri[0]}"
                        tail_node = first_tri[0]

                    else:
                        path_str += f"{first_tri[0]}-[{first_tri[1]}]->{first_tri[2]}"
                        tail_node = first_tri[2]
                    for tri in p2ts[1:]:
                        if tail_node == tri[0]:
                            path_str += f"-[{tri[1]}]->{tri[2]}"
                            tail_node = tri[2]
                        else:
                            path_str += f"<-[{tri[1]}]-{tri[0]}"
                            tail_node = tri[0]
                else:
                    tri = p2ts[0]
                    path_str += f"-[{tri[1]}]->{tri[2]}"
                    tail_node = tri[2]
                result = tail_node
                result_names.append(result) if result not in result_names else None
                paths.append(path_str) if path_str not in paths else None
                path2triplet.append(p2ts) if p2ts not in path2triplet else None

    return result_names, remove_substrings(paths), path2triplet, path_triplets


def chinese_text_preprocess(text):
    words = jieba.cut(text)
    return ' '.join([word for word in words if len(word) > 1])


def search_similar_cn(keyword, candidates, top_n):
    """
    中文文本相似度搜索
    参数：
        keyword: 搜索关键词
        candidates: 候选文本列表
        top_n: 返回前N个匹配结果

    返回：
        按相似度降序排列的（文本，分数）元组列表
    """
    # 预处理所有文本
    processed_all = [chinese_text_preprocess(keyword)] + [chinese_text_preprocess(t) for t in candidates]

    # 创建TF-IDF向量器
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tfidf_matrix = vectorizer.fit_transform(processed_all)

    # 分割查询向量和候选向量
    query_vec = tfidf_matrix[0:1]
    candidate_vecs = tfidf_matrix[1:]

    # 计算余弦相似度
    similarities = cosine_similarity(query_vec, candidate_vecs).flatten()

    # 组合结果并排序
    results = sorted(zip(candidates, similarities),
                     key=lambda x: x[1],
                     reverse=True)[:top_n]

    top_keyword = list(result[0] for result in results)
    return top_keyword


def find_topK_kdzn(fuzzy_entity, top_k):
    """
    模糊匹配实体，返回前k个实体
    """
    entities_in_kdzn = cfg["entities_in_kdzn"]
    transformer_oil_gas_analysis_entities = cfg["transformer_oil_gas_analysis_entities"]
    gis_switch_eval_entities = cfg["gis_switch_eval_entities"]
    all_entities = list(set(entities_in_kdzn + transformer_oil_gas_analysis_entities + gis_switch_eval_entities))
    result = search_similar_cn(fuzzy_entity, all_entities, top_k)
    print("全局事件匹配实体：", result)
    paths, path2triplet, path_triplets = find_topK_path(result)

    return paths, path2triplet, path_triplets


def is_sublist(a, b):
    set_b = set(b)
    for item in a:
        if item not in set_b:
            return False
    return True


def find_topK_path(entity_list):
    choose_neo4j = "kdzn"
    path_triplets = []
    all_paths = []
    all_triplets = set()
    path2triplet = []
    path2triplet_pro = []
    filter_entitie_ = ["电力变压器", "检修", "检查", "试验", "故障"]
    for match_ent in entity_list:
        if match_ent is not None:
            paths = []
            cypher = f"""MATCH p=(root:kedazhineng)-[*1..10]->(middle:kedazhineng)-[*0..6]->(leaf)
                    WHERE root.name IN ["电力变压器", "检修", "检查", "试验", "故障"] AND middle.name = "{match_ent}"
                    WITH p, [n IN nodes(p) WHERE n:Entity | n.name] AS pathNames
                    RETURN p, pathNames
                    ORDER BY LENGTH(p) ASC
                    LIMIT 20;"""
            cypher_result = query_neo4j(cypher, kg=choose_neo4j)
            for record in cypher_result:
                path = record[0]["p"]
                relationships = list(path.relationships)
                path_str = ""
                tail_node = ""
                result = ""
                p2ts = []
                has_head = False
                end_node_has_requirement = None
                for i in range(len(relationships)):
                    nodes = relationships[i].nodes
                    start_node = nodes[0]["name"]
                    # nodes[0]._properties
                    end_node = nodes[1]["name"]
                    rel = relationships[i]["name"]
                    if rel is None:
                        rel = relationships[i].type
                    if rel == "kedazhineng":
                        rel = relationships[i]["edge_name"]
                    if rel is None:
                        rel = relationships[i]["type"]
                    if type(rel) is str:
                        if rel.startswith("包含-"):
                            rel = "包含"
                    if (start_node, rel, end_node) not in path_triplets:
                        path_triplets.append((start_node, rel, end_node))
                    p2ts.append((start_node, rel, end_node))
                    # 增加中文属性三元组到path_triplets
                    start_node_chinese_properties = get_chinese_properties(nodes[0])
                    end_node_chinese_properties = get_chinese_properties(nodes[1])
                    # 处理尾结点的属性→三元组
                    for node, properties in [(start_node, start_node_chinese_properties),
                                             (end_node, end_node_chinese_properties)]:
                        for k, v in properties.items():
                            if k == "requirement":
                                k = "要求"
                                end_node_has_requirement = f"-[要求]->{v}"
                            path_triplets.append((node, k, v)) if (node, k, v) not in path_triplets else None
                            p2ts.append((node, k, v)) if (node, k, v) not in path_triplets else None

                    # 拼接路径字符串：如果非首段关系，判断关系方向
                    if not has_head:
                        if start_node not in filter_entitie_:
                            path_str += f"{start_node}-[{rel}]->{end_node}"
                            tail_node = end_node
                            has_head = True
                        # elif end_node not in filter_entities:
                        #     path_str += f"{end_node}"
                        #     tail_node = end_node
                        #     has_head = True
                    else:
                        if tail_node == start_node:
                            path_str += f"-[{rel}]->{end_node}"
                            tail_node = end_node
                        else:
                            path_str += f"<-[{rel}]-{start_node}"
                            tail_node = start_node

                if end_node_has_requirement is not None:
                    path_str += end_node_has_requirement
                paths.append(path_str) if path_str not in paths else None
                path2triplet.append(p2ts) if p2ts not in path2triplet else None
                for pt in p2ts:
                    all_triplets.add(pt)

        # 合并包含路径（保留最长路径）
        paths.sort(key=len, reverse=True)
        unique_paths = []
        for path in paths:
            if not any(p for p in unique_paths if path in p and path != p):
                unique_paths.append(path)
        all_paths += unique_paths

    # 去掉path2triplet子列表
    def remove_sublists(lists):
        non_sublists = []
        for i, lst in enumerate(lists):
            is_sublist = False
            for j, other_lst in enumerate(lists):
                if i != j and set(lst).issubset(set(other_lst)):
                    is_sublist = True
                    break
            if not is_sublist:
                non_sublists.append(lst)
        return non_sublists

    #
    path2triplet_pro_ = remove_sublists(path2triplet)
    for pp2tts in path2triplet_pro_:
        temp_list = []
        for pti in pp2tts:
            if pti[0] not in filter_entitie_:
                temp_list.append(pti)
        path2triplet_pro.append(temp_list)

    return remove_substrings(all_paths), path2triplet_pro, list(all_triplets)


if __name__ == '__main__':
    pass

    # from openai import OpenAI

    # aaaa = OpenAI(
    #     api_key="sk-e4f34f0c88d8467c9cafe7c1d1f7774a",  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope SDK的base_url
    # )
    # a = getBest_rules_vec('应该怎么办', ['检修方法', '检修要求', '检修部件'], aaaa)
    # a, b, c, d = find_kdzn("声音异常检查", "变压器本体发出吱吱声", "问题",aaaa)
    # bb = ['冷却器油路管道内的“哄哄”声', '冷却器油泵的无规则非周期性金属摩擦声',
    #       '冷却器油泵均匀的周期性“咯咯”金属摩擦声', '变压器本体“哇哇”声', '变压器本体“哺咯”的沸腾声',
    #       '变压器本体“嘶嘶”声', '变压器本体“吱吱”或“噼啪”声', '变压器本体异常增大且有明显的杂音',
    #       '变压器本体连续的高频率尖锐声']
    # a = getBest_rules_vec('变压器本体发出吱吱声', bb, aaaa)
    # print(a)
    # for k, v in kdzn_file_rules_kdzn.items():
    #     print('\033[91m' + '-' * 100 + '\033[0m')
    #     print(k)
    #     r_path = v[0]["rule_path"][0]
    #     r_get_e = v[0]["getIn4simi"]
    #     rule_in_ents = []  # 得到的检索实体（规则的输入实体）
    #     query_neo4j_res = query_neo4j(r_get_e, kg='kdzn')
    #     for record in query_neo4j_res:
    #         rule_in_ents.append(record[0]["RIName"])
    #     print(r_get_e)
    #     print(rule_in_ents)
    #     get_p_cy = r_path.replace("{entityname}", rule_in_ents[0])
    #     ppp = query_neo4j(get_p_cy, kg='kdzn')
    #     print(get_p_cy)
    #     for record in ppp:
    #         path = record[0]["p"]
    #         print(path)
    # relationship = "处理建议"
    # entityname = "电抗或阻抗变化明显、频响特性异常、绕组之间或对地电容量变化明显"
    # result_names, paths, result_des_n, result_des_b, path_tris = query_relationship(entityname, relationship)
    # print(result_names)
    # print('-' * 80)
    # print(paths)
    # print('-' * 80)
    # print(path_tris)
    # file_ontology = '器身检修'
    # fuzzy_entity = '绕组的电容量变化比较大'
    # fuzzy_relation = '处理建议'
    # a, b, c, d = find_kdzn(file_ontology, fuzzy_entity, fuzzy_relation)
    # a, b, c, d = getYiChang(fuzzy_entity, fuzzy_relation, model_name="../../model/embeddings/text2vec_large_chinese")
    # print(a)
    # print(b)
    # print(c)
    # q_words = jieba.cut("变压器过度发热，怎么处理", cut_all=False)
    # print(list(q_words))
    # for c in q_words:
    #     print(c)
    # getYiChang("油箱", '处理建议', "../../model/embeddings/text2vec_large_chinese")
