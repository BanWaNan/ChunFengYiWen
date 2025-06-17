import difflib
from .cypher_tool import query_neo4j, query_new_neo4j


def find_most_similar_string(a, B):
    # 初始化相似度和最相近的字符串
    max_similarity = 0
    most_similar_string = ""
    # 遍历列表B中的每个字符串，计算其与字符串a的相似度
    for b in B:
        similarity = difflib.SequenceMatcher(None, a, b).ratio()
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_string = b
    return most_similar_string


def n_hopPath(ent1, ent2, hop_num):
    """
    查询两个实体之间的hop_num跳路径 list[str_path]
    """
    rel_hop_q = "-[]-(:ownthink)" * (hop_num - 1)
    query_ = f"""MATCH p=(a:ownthink){rel_hop_q}-[]-(b:ownthink) WHERE a.name="{ent1}" AND b.name="{ent2}" return p LIMIT 5"""
    result = query_neo4j(query_)
    if not result:
        rel_hop_q = "-[]-(:huaputong)" * (hop_num - 1)
        query_ = f"""MATCH p=(a:huaputong){rel_hop_q}-[]-(b:huaputong) WHERE a.name="{ent1}" AND b.name="{ent2}" return p LIMIT 5"""
        result = query_neo4j(query_)
    formatted_paths = []
    paths_tris = []
    for record in result:
        path = record[0]['p']
        nodes = path.nodes
        relationships = path.relationships
        path_str = f"{nodes[0]['name']}"
        this_path_tris = []
        for i in range(len(relationships)):
            rel = relationships[i]
            direction = ["-", "->"] if rel.start_node == nodes[i] else ["<-", "-"]
            path_str += f"{direction[0]}{rel['name']}{direction[1]}"
            path_str += f"{nodes[i + 1]['name']}"
            this_path_tris.append([rel.start_node['name'], rel['name'], rel.end_node['name']])
        formatted_paths.append(path_str)
        paths_tris.append(this_path_tris)
    return formatted_paths, paths_tris


def mul_hopPath(ent1, ent2):
    """
    查询两个实体之间的1-5跳路径 list[str_path]
    """
    hop_n = 1
    tri_pathss = []
    pathss_tris = []
    while len(tri_pathss) < 11 and hop_n < 6:
        tri_paths, paths_tris = n_hopPath(ent1, ent2, hop_n)
        if tri_paths:
            tri_pathss.extend(tri_paths)
            pathss_tris.extend(paths_tris)
        hop_n += 1
    return tri_pathss, pathss_tris


def find_ent_des(ent):
    """
    获取实体的描述，返回描述三元组[ent,'是',des]
    """
    query_ent = f"""MATCH (a:ownthink{{name:"{ent}"}}) return a.des as des,a.description as desc"""
    result = query_neo4j(query_ent)
    tri = []
    for info in result:
        record = info[0]
        des, description = record["des"], record["desc"]
        if des:
            tri = [ent, '是', des]
        if description:
            tri = [ent, '是', description]
    return tri


def find_shortest_path(head, tail):
    """
    查询最短路径，返回最短路径
    """
    query = f"""match p=shortestpath((a:ownthink)-[rels1*0..]-(b:ownthink)) where a.name="{head}" and b.name="{tail}" return p"""
    result = query_neo4j(query)
    formatted_paths = []
    paths_tris = []
    for record in result:
        path = record[0]['p']
        nodes = path.nodes
        relationships = path.relationships
        path_str = f"{nodes[0]['name']}"
        this_path_tris = []
        for i in range(len(relationships)):
            rel = relationships[i]
            if rel.type == 'SameAs':
                continue
            direction = ["-", "->"] if rel.start_node == nodes[i] else ["<-", "-"]
            rel_name = rel['name'] if rel['name'] else "="
            path_str += f"{direction[0]}{rel_name}{direction[1]}"
            path_str += f"{nodes[i + 1]['name']}"
            this_path_tris.append([rel.start_node['name'], rel['name'], rel.end_node['name']])
        formatted_paths.append(path_str)
        paths_tris.append(this_path_tris)
    return formatted_paths, paths_tris


def extract_triple_rels(relation_path):  # 暂时用不上
    """
    把关系路径转为三元组（方向可能不一致），返回中间的关系 list(str)
    """
    # 用 '->' 分割路径，得到节点和关系的列表
    elements = relation_path.split('-')
    # 初始化三元组列表
    triples = []
    # 每三个元素构成一个三元组
    for i in range(0, len(elements) - 1, 2):
        triple = [elements[i].replace('>', ''), elements[i + 1], elements[i + 2].replace('>', '')]
        triples.append(triple)
    return [tri[1] for tri in triples]


def get_path_by_relP(head, relPath):
    n_hop_qq = ""
    for rel in relPath:
        n_hop_qq += f"""-[:Contain{{name:"{rel[0]}"}}]-(:ownthink)"""
    query_ = f"""MATCH p=(a:ownthink){n_hop_qq} WHERE a.name="{head}" RETURN p ORDER BY rand() LIMIT 15"""
    formatted_paths = []
    paths_tris = []
    result = query_neo4j(query_)
    for record in result:
        path = record[0]['p']
        nodes = path.nodes
        relationships = path.relationships
        path_str = f"{nodes[0]['name']}"
        this_path_tris = []
        for i in range(len(relationships)):
            rel = relationships[i]
            direction = ["-", "->"] if rel.start_node == nodes[i] else ["<-", "-"]
            path_str += f"{direction[0]}{rel['name']}{direction[1]}"
            path_str += f"{nodes[i + 1]['name']}"
            this_path_tris.append([rel.start_node['name'], rel['name'], rel.end_node['name']])
        formatted_paths.append(path_str)
        paths_tris.append(this_path_tris)
    return formatted_paths, paths_tris


def n_hop_ent(head, n_hop):
    """
    查询head的n_hop跳的随机路径，list[str_path]
    """
    n_hop_qq = "-[]-(:ownthink)" * (n_hop - 1)
    query_ = f"""MATCH p=(a:ownthink){n_hop_qq}-[]-(:ownthink) WHERE a.name="{head}" RETURN p ORDER BY rand() LIMIT 100"""
    formatted_paths = []
    paths_tris = []
    result = query_neo4j(query_)
    for record in result:
        path = record[0]['p']
        nodes = path.nodes
        relationships = path.relationships
        path_str = f"{nodes[0]['name']}"
        this_path_tris = []
        for i in range(len(relationships)):
            rel = relationships[i]
            direction = ["-", "->"] if rel.start_node == nodes[i] else ["<-", "-"]
            path_str += f"{direction[0]}{rel['name']}{direction[1]}"
            path_str += f"{nodes[i + 1]['name']}"
            this_path_tris.append([rel.start_node['name'], rel['name'], rel.end_node['name']])
        formatted_paths.append(path_str)
        paths_tris.append(this_path_tris)
    return formatted_paths, paths_tris


def KO_tupu(keywords_list):
    keyword_results = []
    for keyword in keywords_list:
        results, _ = n_hop_ent(keyword, 1)
        keyword_results.append(results)
    return keyword_results


def hop_ent_path(head):
    rel_path_hop, rel_path_hop_tris = [], []
    for n in range(1, 4):
        n_hop_p, n_hop_t = n_hop_ent(head, n)
        rel_path_hop.extend(n_hop_p)
        rel_path_hop_tris.extend(n_hop_t)
    return rel_path_hop, rel_path_hop_tris


def get_rr_byTriPath(tris: list, path1: str):
    """根据三元组列表和路径生成对应关系列表
    e.g
    ###input:
    tris=[['数据整理分析', '领域', '数据挖掘'],['数据整理分析', '学科', '计算机'],['基于点特征', '学科', '计算机'],['基于点特征', '领域', '人工智能']]
    path='数据挖掘<-领域-数据整理分析-学科->计算机<-学科-基于点特征-领域->人工智能'
    ###output:
    [('领域','left'),('学科','right'),('学科','left'),('领域','right')]"""
    rr = []
    path = str(path1)
    for h, r, t in tris:
        if path.startswith(h):
            if path[len(h):].startswith("<-") and path[len(h) + 2:].startswith(r) and path[len(h) + len(r) + 3:].startswith(t):
                rr.append((r, "left"))
            elif path[len(h):].startswith("-") and path[len(h) + 1:].startswith(r) and path[len(h) + len(r) + 3:].startswith(t):
                rr.append((r, "right"))
            path = path[len(h) + len(r) + 3:]
        elif path.startswith(t):
            if path[len(t):].startswith("<-") and path[len(t) + 2:].startswith(r) and path[len(t) + len(r) + 3:].startswith(h):
                rr.append((r, "left"))
            elif path[len(t):].startswith("-") and path[len(t) + 1:].startswith(r) and path[len(t) + len(r) + 3:].startswith(h):
                rr.append((r, "right"))
            path = path[len(t) + len(r) + 3:]
    return rr


def KO_tupu(keywords_list):
    keyword_results = []
    for keyword in keywords_list:
        results, _ = n_hop_ent(keyword, 1)
        keyword_results.append(results)
    return keyword_results


def all_paths_from_entity(head, max_hop=5):
    """
    查询head节点出发，所有长度不超过max_hop的路径
    返回所有路径的字符串和三元组列表
    """
    # Cypher，a为起点，路径长度1到max_hop，路径可以有环
    entityname = head
    query_ = f"""MATCH p=()-[:数据]->(n:kdzn {{name: "{entityname}"}})-[:分析条件]->()-[:输出结果]->()-[:可能原因]->() RETURN p"""
    formatted_paths = []
    paths_tris = []
    result = query_new_neo4j(query_, kg='kdzn')
    print(result)
    for record in result:
        path = record['p']
        nodes = path.nodes
        relationships = path.relationships
        path_str = f"{nodes[0]['name']}"
        this_path_tris = []
        for i in range(len(relationships)):
            rel = relationships[i]
            rel_type = rel.type
            const_dir, const_arrow = ( "-", "->" ) if rel.start_node == nodes[i] else ( "<-", "-" )
            path_str += f"{const_dir}{rel_type}{const_arrow}{nodes[i + 1]['name']}"
            this_path_tris.append([
                rel.start_node['name'],
                rel_type,
                rel.end_node['name']
            ])
        formatted_paths.append(path_str)
        paths_tris.append(this_path_tris)
    return formatted_paths, paths_tris


if __name__ == '__main__':
    pass
    # print(n_hopPath("吴信东", "美国", 3))
    # print('-' * 80)
    # print(n_hopPath("吴信东", "美国", 4))
    # print('-' * 80)
    # print(find_ent_des("吴信东"))
    # print('-' * 80)
    # print((find_shortest_path("吴信东", "美国")))
    # print('-' * 80)
    # print(get_path_by_relP("吴信东", ["毕业院校", "毕业院校"]))
    # print('-' * 80)
    # print(n_hop_ent("吴信东", 2))

    # all_paths_from_entity("35kV整流柜BC线电压")
    # ent = "吴信东"
    # res, res_t = hop_ent_path(ent)
    # print(res)
    # print(res_t)
    # print(len(res))
    # """
    # 1.查询路径（目标实体）√
    # 2.LLM选择与问题相近的路径（筛选）
    # 3.提取中间实体，根据所有路径中实体的度，参考中间实体的度来给目标实体打分。
    # """
    #
    # rel_paths_str = json.dumps(res, ensure_ascii=False)
    # print(rel_paths_str)

#     path_prompt = f"""已知多条关系路径：["张三-民族->汉族","张三-工作单位->A公司-位置->深圳","张三-民族->汉族<-民族-李四-职业->教师","张三-毕业院校->X大学<-毕业院校-王五-出生地->北京市","张三<-负责人-M项目-合作单位->C企业-归属地->北京"]
# 问题：张三的工作地点在哪里？
# 输出:{{"筛选路径三元组":["张三-工作单位->A团队-所属机构->B企业-位于->深圳","张三-毕业院校->X大学<-毕业院校-王五-出生地->北京市","张三<-负责人-M项目-合作单位->C企业-归属地->北京"]}}
#
# 请你参考上面的例子，从下面这几条关系路径中尽可能多地选择与问题接近的多条路径，注意，你只能从给出的路径中选择路径，不可以对路径内容做任何修改。
# 确保输出是紧凑格式的有效 JSON 对象，不包含任何其他解释、转义符、换行符或反斜杠。
#
# 已知多条关系路径：{rel_paths_str}
# 问题：吴信东的校友是谁？
# 输出:"""
# ans1 = gpt_free(path_prompt)
# print(json.loads(ans1))
# choice_paths = json.loads(ans1)['筛选路径三元组']
# for c_p in choice_paths:
#     result = find_most_similar_string(c_p, res)
#     print(result)
# post_prompt = f"""
# 请你根据下面与问题相关的关系路径回答问题。
# 已知关系路径：{choice_paths}
# 问题：吴信东的校友是谁？
# """
# print(gpt_free(post_prompt))
