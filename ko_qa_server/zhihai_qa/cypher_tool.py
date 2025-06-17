from pathlib import Path
import yaml
from neo4j import GraphDatabase

# 1. 取得当前文件(neo4j_hpt.py)的绝对目录
HERE = Path(__file__).resolve().parent
# 2. 回到 ko_qa_sever 根目录
BASE_DIR = HERE.parent
# 3. 拼出 config.yaml 的完整路径
CFG_FILE = BASE_DIR / "config.yaml"
# 4. 读取
with CFG_FILE.open(encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
# 读取默认连接

uri = cfg["neo4j"]["default"]["uri"]
username = cfg["neo4j"]["default"]["username"]
password = cfg["neo4j"]["default"]["password"]

kdzn_uri = cfg["neo4j"]["kdzn"]["uri"]
kdzn_username = cfg["neo4j"]["kdzn"]["username"]
kdzn_password = cfg["neo4j"]["kdzn"]["password"]

new_kdzn_uri = cfg["neo4j"]["new_kdzn"]["uri"]
new_kdzn_username = cfg["neo4j"]["new_kdzn"]["username"]
new_kdzn_password = cfg["neo4j"]["new_kdzn"]["password"]


# 查ko或者kdzn图谱
def query_neo4j(query, timeout=30, kg='ko'):
    """执行neo4j查询
    query:查询语句
    timeout:超时时长/秒
    """
    if kg == 'kdzn':
        driver = GraphDatabase.driver(kdzn_uri, auth=(kdzn_username, kdzn_password))
    else:
        driver = GraphDatabase.driver(uri, auth=(username, password))
    query_ = f"""CALL apoc.cypher.runTimeboxed('{query}',NULL,{timeout*1000})"""
    res = []
    with driver.session() as session:
        result = session.run(query_)
        for record in result:
            res.append(record)
    driver.close()
    return res


# 查ko或者kdzntest图谱
def query_new_neo4j(query, timeout=30, kg='ko'):
    """执行neo4j查询
    query:查询语句
    timeout:超时时长/秒
    """
    if kg == 'kdzn':
        driver = GraphDatabase.driver(new_kdzn_uri, auth=(new_kdzn_username, new_kdzn_password))
    else:
        driver = GraphDatabase.driver(uri, auth=(username, password))
    # query_ = f"""CALL apoc.cypher.runTimeboxed('{query}',NULL,{timeout*1000})"""
    res = []
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            res.append(record)
    driver.close()
    return res


def insert_records_to_graph(json_data, graph_name, kg='kdzn'):
    if not json_data or len(json_data) < 2:
        print("json_data 为空或没有数据行")
        return False
    if kg == 'kdzn':
        driver = GraphDatabase.driver(new_kdzn_uri, auth=(new_kdzn_username, new_kdzn_password))
    else:
        driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session() as session:
            # 提取表头（第一行）
            headers = json_data[0]
            # 数据行（从第二行开始）
            data_rows = json_data[1:]

            inserted_count = 0
            for row in data_rows:
                if len(row) < 2:
                    print(f"跳过无效行: {row}")
                    continue

                root_node = str(row[0])  # 第一列：根节点
                for i in range(len(row) - 1):  # 遍历每一列（除了最后一列）
                    entity_name = str(row[i])  # 实体名是自身
                    type_name = str(headers[i])  # 类型是列名
                    edge_type = str(headers[i + 1]) if i + 1 < len(headers) else None  # 边类型是下一个列名
                    edge_target = str(row[i + 1]) if i + 1 < len(row) else None  # 边目标是下一列的值

                    if edge_type and edge_target:
                        # 构造 Cypher 查询语句，添加 graph_name 属性
                        cypher_query = f"""
                            MERGE (a:kdzn {{name: $entity_name, type: $type_name, root_node: $root_node}})
                            MERGE (b:kdzn {{name: $edge_target, type: $edge_type, root_node: $root_node}})
                            MERGE (a)-[r:{edge_type} {{ graph_name: $graph_name }}]->(b)
                            RETURN a, r, b
                        """

                        # 执行查询
                        session.run(cypher_query, {
                            "root_node": root_node,
                            "entity_name": entity_name,
                            "type_name": type_name,
                            "edge_type": edge_type,
                            "edge_target": edge_target,
                            "graph_name": graph_name
                        })
                        inserted_count += 1

            print(f"成功插入 {inserted_count} 条记录到图谱 {graph_name}。")
            return True

    except Exception as e:
        print(f"插入记录出错: {e}")
        return False
    finally:
        driver.close()


def get_graphnames(kg='kdzn'):
    """
    查询 Neo4j 中所有唯一的 graph_name 值
    返回格式: {'status': bool, 'data': List[str], 'msg': str}
    """
    if kg == 'kdzn':
        driver = GraphDatabase.driver(new_kdzn_uri, auth=(new_kdzn_username, new_kdzn_password))
    else:
        driver = GraphDatabase.driver(uri, auth=(username, password))

    try:
        with driver.session() as session:
            cypher_query = """
                MATCH (a)-[r]->(b)
                WHERE r.graph_name IS NOT NULL
                RETURN DISTINCT r.graph_name AS graph_name
            """
            result = session.run(cypher_query)
            graph_names = [record['graph_name'] for record in result]
            return {
                'status': True,
                'data': graph_names,
                'msg': '查询成功' if graph_names else '图谱为空'
            }
    except Exception as e:
        print(f"查询 graph_name 出错: {e}")
        return {
            'status': False,
            'data': [],
            'msg': f"查询失败: {str(e)}"
        }
    finally:
        driver.close()