import pymysql
import yaml
from pathlib import Path

kdzn_file_rules_kdzn = {
    "典型故障": [
        {"rule_name": "哪些部件可能发生典型故障", "rule_path": [  #
            """MATCH p=(f:kedazhineng {name:"典型故障"})-[:部件]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"典型故障"}) RETURN f.name AS RIName"""},
        {"rule_name": "典型故障", "rule_path": [  #
            """MATCH p=(f:kedazhineng{name:"典型故障"})-[:部件]->()-[:故障类型]->()-[:异常现象]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"典型故障"}) RETURN f.name AS RIName"""},
        {"rule_name": "可能存在什么故障", "rule_path": [  #
            """MATCH p=()-[:部件]->(n:kedazhineng{name:"{entityname}"})-[:故障类型]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"典型故障"})-[:部件]->(RI) RETURN RI.name AS RIName"""},
        {"rule_name": "可能存在什么异常现象", "rule_path": [  #
            """MATCH p=()-[:部件]->(n:kedazhineng{name:"{entityname}"})-[:故障类型]->()-[:异常现象]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"典型故障"})-[:部件]->(RI) RETURN RI.name AS RIName"""},
        {"rule_name": "哪些异常现象", "rule_path": [
            """MATCH p=()-[:部件]->()-[:故障类型]->(n:kedazhineng{name:"{entityname}"})-[:异常现象]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"典型故障"})-[:部件]->(R1)-[:故障类型]->(RI) RETURN R1.name AS R1Name, RI.name AS RIName"""},
        {"rule_name": "部件可能是什么", "rule_path": [
            """MATCH p=()-[:部件]->()-[:故障类型]->(R2)-[:异常现象]->(n:kedazhineng{name:"{entityname}"}) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"典型故障"})-[:部件]->(R1)-[:故障类型]->(R2)-[:异常现象]->(RI) RETURN R1.name AS R1Name, R2.name AS R2Name, RI.name AS RIName"""},
        {"rule_name": "故障可能是什么", "rule_path": [
            """MATCH p=()-[:部件]->()-[:故障类型]->(R2)-[:异常现象]->(n:kedazhineng{name:"{entityname}"}) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"典型故障"})-[:部件]->(R1)-[:故障类型]->(R2)-[:异常现象]->(RI) RETURN R1.name AS R1Name, R2.name AS R2Name, RI.name AS RIName"""},
    ],
    "变压器油中溶解气体分析和判断导则": [
        {"rule_name": "变压器、套管、互感器故障", "rule_path": [
            """MATCH p=()-[:kedazhineng{edge_name: "包含"}]->()-[:kedazhineng{edge_name: "包含"}]->()-[:kedazhineng{edge_name: "故障类型"}]->()-[:kedazhineng{edge_name: "典型故障"}]->(n:kedazhineng{name:"{entityname}"}) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"变压器油中溶解气体分析和判断导则"})-[:kedazhineng{edge_name: "包含"}]->(R1)-[:kedazhineng{edge_name: "故障类型"}]->(R2)-[:kedazhineng{edge_name: "典型故障"}]->(RI) RETURN R1.name AS R1Name, R2.name AS R2Name, RI.name AS RIName"""},
        {"rule_name": "变压器、套管、互感器故障类型", "rule_path": [
            """MATCH p=()-[:kedazhineng{edge_name: "包含"}]->()-[:kedazhineng{edge_name: "包含"}]->()-[:kedazhineng{edge_name: "故障类型"}]->(n:kedazhineng{name:"{entityname}"})-[]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"变压器油中溶解气体分析和判断导则"})-[:kedazhineng{edge_name: "包含"}]->(R1)-[:kedazhineng{edge_name: "故障类型"}]->(RI) RETURN R1.name AS R1Name, RI.name AS RIName"""},
        {"rule_name": "故障原因", "rule_path": [
            """MATCH p=()-[:kedazhineng{edge_name: "包含"}]->()-[:kedazhineng{edge_name: "包含"}]->()-[:kedazhineng{edge_name: "故障原因"}]->(n:kedazhineng{name:"{entityname}"})-[]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"变压器油中溶解气体分析和判断导则"})-[:kedazhineng{edge_name: "包含"}]->(R1)-[:kedazhineng{edge_name: "故障原因"}]->(RI)-[:kedazhineng{edge_name:"故障描述"}]->(R2) RETURN R1.name AS R1Name, R2.name AS R2Name, RI.name AS RIName"""}
    ],
    "干式电力变压器试验": [
        {"rule_name": "干式电力变压器试验有哪些试验类型", "rule_path": [
            """MATCH p=(n:kedazhineng{name:"干式电力变压器"})-[:试验类型]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"干式电力变压器试验"}) RETURN f.name AS RIName"""},
        {"rule_name": "干式电力变压器试验有哪些试验项目", "rule_path": [
            """MATCH p=(n:kedazhineng{name:"干式电力变压器"})-[:试验类型]->()-[:试验项目]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"干式电力变压器试验"}) RETURN f.name AS RIName"""},
        {"rule_name": "干式电力变压器有哪些试验类型", "rule_path": [
            """MATCH p=()-[:试验对象]->(n:kedazhineng{name:"{entityname}"})-[:试验类型]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"干式电力变压器试验"})-[:试验对象]->(RI) RETURN RI.name AS RIName"""},
        {"rule_name": "干式电力变压器有哪些试验项目", "rule_path": [
            """MATCH p=()-[:试验对象]->(n:kedazhineng{name:"{entityname}"})-[:试验类型]->()-[:试验项目]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"干式电力变压器试验"})-[:试验对象]->(RI) RETURN RI.name AS RIName"""},
        {"rule_name": "试验有哪些试验项目", "rule_path": [
            """MATCH p=()-[:试验对象]->()-[:试验类型]->(n:kedazhineng{name:"{entityname}"})-[:试验项目]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"干式电力变压器试验"})-[:试验对象]->(R1)-[:试验类型]->(RI) RETURN R1.name AS R1Name, RI.name AS RIName"""},
        {"rule_name": "项目的试验类型是什么", "rule_path": [
            """MATCH p=()-[:试验类型]->()-[:试验项目]->(n:kedazhineng{name:"{entityname}"}) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"干式电力变压器试验"})-[:试验对象]->(R1)-[:试验类型]->(R2)-[:试验项目]->(RI) RETURN R1.name AS R1Name,R2.name AS R2Name, RI.name AS RIName"""},
    ],
    "油侵式电力变压器试验": [
        {"rule_name": "试验", "rule_path": [
            """MATCH p=()-[:试验对象]->(n:kedazhineng{name:"{entityname}"})-[:试验类型]->()-[:试验项目]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"油侵式电力变压器试验"})-[:试验对象]->(RI) RETURN RI.name AS RIName"""},
        {"rule_name": "有哪些试验类型", "rule_path": [
            """MATCH p=(f:kedazhineng{name:"油侵式电力变压器试验"})-[:试验对象]->()-[:试验类型]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"油侵式电力变压器试验"}) RETURN f.name AS RIName"""},
        # {"rule_name": "试验类型", "rule_path": [
        #     """MATCH p=()-[:试验对象]->(R1)-[:试验类型]->(n:kedazhineng{name:"{entityname}"})-[:试验项目]->(b) RETURN p"""],
        #  "getIn4simi": """MATCH (f:kedazhineng{name:"油侵式电力变压器试验"})-[:试验对象]->(R1)-[:试验类型]->(RI) RETURN R1.name AS R1Name, RI.name AS RIName"""},
        {"rule_name": "有哪些试验项目", "rule_path": [
            """MATCH p=(f:kedazhineng{name:"油侵式电力变压器试验"})-[:试验对象]->()-[:试验类型]->()-[:试验项目]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"油侵式电力变压器试验"}) RETURN f.name AS RIName"""}],
    "绝缘油试验": [
        {"rule_name": "试验要求标准", "rule_path": [
            """MATCH p=()-[:部件]->(n:kedazhineng{name:"{entityname}"})-[:试验项目]->()-[:要求]->()-[:执行标准]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"绝缘油试验"})-[:部件]->(RI) RETURN RI.name AS RIName"""},
        {"rule_name": "试验项目", "rule_path": [
            """MATCH p=()-[:部件]->(R1)-[:试验项目]->(n:kedazhineng{name:"{entityname}"})-[:要求]->()-[:执行标准]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"绝缘油试验"})-[:部件]->(R1)-[:试验项目]->(RI) RETURN R1.name AS R1Name,RI.name AS RIName"""}
    ],
    "不停电检查": [
        {"rule_name": "不停电检查", "rule_path": [
            """MATCH p=(f:kedazhineng{name:"不停电检查"})-[:检查部位]->(part)-[:检查周期]->(cycle)-[:检查项目]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"不停电检查"}) RETURN f.name AS RIName"""},
        {"rule_name": "不停电检查需要检查哪些部位", "rule_path": [
            """MATCH p=(n:kedazhineng{name:""不停电检查""})-[:检查部位]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"不停电检查"}) RETURN f.name AS RIName"""},
        {"rule_name": "有哪些检查项目", "rule_path": [
            """MATCH p=()-[:检查部位]->(n:kedazhineng{name:"{entityname}"})-[:检查周期]->()-[:检查项目]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"不停电检查"})-[:检查部位]->(RI) RETURN RI.name AS RIName"""},
        {"rule_name": "检查周期是什么", "rule_path": [
            """MATCH p=()-[:检查部位]->(n:kedazhineng{name:"{entityname}"})-[:检查周期]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"不停电检查"})-[:检查部位]->(RI) RETURN RI.name AS RIName"""},
        {"rule_name": "项目的检查部位是什么", "rule_path": [
            """MATCH p=(start)-[:检查部位]->(part)-[:检查周期]->(cycle)-[:检查项目]->(n:kedazhineng{name:"{entityname}"}) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"不停电检查"})-[:检查部位]->(R1)-[:检查周期]->(R2)-[:检查项目]->(RI) RETURN R1.name AS R1Name, R2.name AS R2Name, RI.name AS RIName"""},
    ],
    "停电检查": [
        {"rule_name": "检查方式", "rule_path": [  # 检查部位
            """MATCH p=()-[:检查部位]->(n:kedazhineng{name:"{entityname}"})-[:检查周期]->()-[:检查项目]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"停电检查"})-[:检查部位]->(RI) RETURN RI.name AS RIName"""},
        {"rule_name": "检查项目", "rule_path": [
            """MATCH p=()-[:检查部位]->(n:kedazhineng{name:"{entityname}"})-[:检查周期]->()-[:检查项目]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"停电检查"})-[:检查部位]->(RI) RETURN RI.name AS RIName"""},
        {"rule_name": "检查周期", "rule_path": [
            """MATCH p=()-[:检查部位]->(n:kedazhineng{name:"{entityname}"})-[:检查周期]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"停电检查"})-[:检查部位]->(RI) RETURN RI.name AS RIName"""},
        {"rule_name": "检查部位", "rule_path": [
            """MATCH p=(f:kedazhineng{name:"停电检查"})-[:检查部位]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"停电检查"}) RETURN f.name AS RIName"""}
    ],
    "绕组变形检查": [
        {"rule_name": "处理办法", "rule_path": [
            """MATCH p=()-[:异常部位]->()-[:异常现象]->(n:kedazhineng{name:"{entityname}"})-[:可能的异常原因]->()-[:检查方法或部位]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"绕组变形检查"})-[:异常部位]->(R1)-[:异常现象]->(RI) RETURN R1.name AS R1Name, RI.name AS RIName"""},
        {"rule_name": "检查方法或部位", "rule_path": [  # 根据异常原因找检擦方法或部位
            """MATCH p=()-[:异常部位]->()-[:异常现象]->()-[:可能的异常原因]->(n:kedazhineng{name:"{entityname}"})-[:检查方法或部位]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng{name:"绕组变形检查"})-[:异常部位]->(R1)-[:异常现象]->(R2)-[:可能的异常原因]->(RI) RETURN R1.name AS R1Name, R2.name AS R2Name, RI.name AS RIName"""},
        {"rule_name": "异常原因", "rule_path": [
            """MATCH p=()-[:异常部位]->()-[:异常现象]->(n:kedazhineng{name:"{entityname}"})-[:可能的异常原因]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"绕组变形检查"})-[:异常部位]->(R1)-[:异常现象]->(RI) RETURN R1.name AS R1Name, RI.name AS RIName"""},
        {"rule_name": "检查方法", "rule_path": [
            """MATCH p=()-[:异常部位]->()-[:异常现象]->(n:kedazhineng{name:"{entityname}"})-[:可能的异常原因]->()-[:检查方法或部位]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"绕组变形检查"})-[:异常部位]->(R1)-[:异常现象]->(RI) RETURN R1.name AS R1Name, RI.name AS RIName"""},
        {"rule_name": "有哪些异常现象", "rule_path": [
            """MATCH p=(f:kedazhineng{name:"绕组变形检查"})-[:异常部位]->()-[:异常现象]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"绕组变形检查"}) RETURN f.name AS RIName"""}
    ],
    "声音异常检查": [
        {"rule_name": "处理办法", "rule_path": [  # 针对特定异常现象的处理办法
            """MATCH p=()-[:异常部位]->()-[:异常现象]->(n:kedazhineng{name:"{entityname}"})-[:可能的异常原因]->()-[:检查方法或部位]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"声音异常检查"})-[:异常部位]->(R1)-[:异常现象]->(RI) RETURN R1.name AS R1Name, RI.name AS RIName"""},
        # {"rule_name": "检查方法或部位", "rule_path": [  # 针对特定异常原因的检查方法或部位
        #     """MATCH p=()-[:异常部位]->()-[:异常现象]->()-[:可能的异常原因]->(n:kedazhineng{name:"{entityname}"})-[:检查方法或部位]->(b) RETURN p"""],
        #  "getIn4simi": """MATCH (f:kedazhineng{name:"声音异常检查"})-[:异常部位]->(R1)-[:异常现象]->(R2)-[:可能的异常原因]->(RI) RETURN R1.name AS R1Name, R2.name AS R2Name, RI.name AS RIName"""},
        {"rule_name": "异常原因", "rule_path": [  # 针对异常现象的异常原因
            """MATCH p=()-[:异常部位]->()-[:异常现象]->(n:kedazhineng{name:"{entityname}"})-[:可能的异常原因]->()-[:检查方法或部位]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"声音异常检查"})-[:异常部位]->(R1)-[:异常现象]->(RI) RETURN R1.name AS R1Name, RI.name AS RIName"""},
        # {"rule_name": "异常现象", "rule_path": [  # 针对异常原因导致哪些异常现象
        #     """MATCH p=()-[:异常部位]->()-[:异常现象]->()-[:可能的异常原因]->(n:kedazhineng{name:"{entityname}"}) RETURN p"""],
        #  "getIn4simi": """MATCH (f:kedazhineng {name:"声音异常检查"})-[:异常部位]->(R1)-[:异常现象]->(R2)-[:可能的异常原因]->(RI) RETURN R1.name AS R1Name,R2.name AS R2Name, RI.name AS RIName"""},
        {"rule_name": "哪些异常现象", "rule_path": [  # 有哪些异常现象
            """MATCH p=(f:kedazhineng {name:"声音异常检查"})-[:异常部位]->()-[:异常现象]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"声音异常检查"}) RETURN f.name AS RIName"""},
        {"rule_name": "哪些部位会出现异常？", "rule_path": [  # 会出现异常的部位
            """MATCH p=(f:kedazhineng {name:"声音异常检查"})-[:异常部位]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"声音异常检查"}) RETURN f.name AS RIName"""},
    ],
    "放电性异常检查": [
        {"rule_name": "有哪些异常原因", "rule_path": [
            """MATCH p=(f:kedazhineng {name:"电力变压器"})-[:HAS_INSPECTION]->()-[:类型]->()-[:可能的异常原因]->()-[:检查方法或部位]->(b) RETURN p""", ],
         "getIn4simi": """MATCH (f:kedazhineng {name:"放电性异常检查"})  RETURN f.name AS RIName"""},
        {"rule_name": "可能有哪些检查方法", "rule_path": [
            """MATCH p=(f:kedazhineng {name:"电力变压器"})-[:HAS_INSPECTION]->()-[:类型]->()-[:可能的异常原因]->(n:kedazhineng{name:"{entityname}"})-[:检查方法或部位]->(b) RETURN p""", ],
         "getIn4simi": """MATCH (f:kedazhineng {name:"放电性异常检查"})-[:可能的异常原因]->(RI)  RETURN RI.name AS RIName"""},
        {"rule_name": "异常原因导致的现象", "rule_path": [
            """MATCH p=(f:kedazhineng {name:"电力变压器"})-[:HAS_INSPECTION]->()-[:类型]->()-[:可能的异常原因]->(n:kedazhineng{name:"{entityname}"}) RETURN p""", ],
         "getIn4simi": """MATCH (f:kedazhineng {name:"放电性异常检查"})-[:可能的异常原因]->(RI)  RETURN RI.name AS RIName"""}
    ],
    "绝缘受潮异常检查": [
        {"rule_name": "处理办法", "rule_path": [  # 异常现象
            """MATCH p=()-[:异常现象]->(n:kedazhineng{name:"{entityname}"})-[:检查方法或部位]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"绝缘受潮异常检查"})-[:异常现象]->(RI)  RETURN RI.name AS RIName"""},
        {"rule_name": "异常现象", "rule_path": [  # 异常现象
            """MATCH p=()-[:异常现象]->()-[:检查方法或部位]->(n:kedazhineng{name:"{entityname}"}) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"绝缘受潮异常检查"})-[:异常现象]->(R1)-[:检查方法或部位]->(RI)  RETURN RI.name AS RIName"""},
        {"rule_name": "检查的判断措施",  # 检查方法
         "rule_path": [
             """MATCH p=()-[:异常现象]->()-[:检查方法或部位]->(n:kedazhineng{name:"{entityname}"}) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"绝缘受潮异常检查"})-[:异常现象]->()-[:检查方法或部位]->(RI)  RETURN RI.name AS RIName"""},
        {"rule_name": "检查方法和处理措施有哪些",  # 检查方法
         "rule_path": [
             """MATCH p=(f:kedazhineng {name:"绝缘受潮异常检查"})-[:异常现象]->()-[:检查方法或部位]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"绝缘受潮异常检查"}) RETURN f.name AS RIName"""}
    ],
    "过热性异常检查": [
        {"rule_name": "处理措施",  # 异常现象
         "rule_path": [
             """MATCH p=()-[:异常现象]->(n:kedazhineng{name:"{entityname}"})-[:可能的异常原因]->()-[:检查内容或部位]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"过热性异常检查"})-[:异常现象]->(RI)  RETURN RI.name AS RIName"""},
        {"rule_name": "异常原因",  # 异常现象
         "rule_path": [
             """MATCH p=()-[:异常现象]->(n:kedazhineng{name:"{entityname}"})-[:可能的异常原因]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"过热性异常检查"})-[:异常现象]->(RI)  RETURN RI.name AS RIName"""},
        {"rule_name": "检查内容和部位",  # 异常现象
         "rule_path": [
             """MATCH p=()-[:异常现象]->(R1)-[:可能的异常原因]->(n:kedazhineng{name:"{entityname}"})-[:检查内容或部位]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"过热性异常检查"})-[:异常现象]->(R1)-[:可能的异常原因]->(RI)  RETURN R1.name AS R1Name,RI.name AS RIName"""},
        {"rule_name": "原因是什么，怎么处理",  # 异常现象
         "rule_path": [
             """MATCH p=(f:kedazhineng {name:"过热性异常检查"})-[:异常现象]->(R1)-[:可能的异常原因]->()-[:检查内容或部位]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"过热性异常检查"}) RETURN f.name AS RIName"""}
    ],
    "器身检修": [
        {"rule_name": "检修检查",
         "rule_path": [
             """MATCH p=(f:kedazhineng {name:"器身检修"})-[:部位]->()-[:检修内容]->()-[:检修方法]->(n:kedazhineng{name:"{entityname}"}) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"器身检修"})-[:部位]->(R1)-[:检修内容]->(R2)-[:检修方法]->(RI)  RETURN R1.name AS R1Name,R2.name AS R2Name,RI.name AS RIName"""},
        {"rule_name": "检修内容",
         "rule_path": [
             """MATCH p=(f:kedazhineng {name:"器身检修"})-[:部位]->()-[:检修内容]->(n:kedazhineng{name:"{entityname}"})-[:检修方法]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"器身检修"})-[:部位]->(R1)-[:检修内容]->(RI)  RETURN R1.name AS R1Name,RI.name AS RIName"""},
        {"rule_name": "部位检修",
         "rule_path": [
             """MATCH p=(f:kedazhineng {name:"器身检修"})-[:部位]->(n:kedazhineng{name:"{entityname}"})-[:检修内容]->()-[:检修方法]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"器身检修"})-[:部位]->(RI)  RETURN RI.name AS RIName"""},
        {"rule_name": "排查原因的检查措施",  # 异常原因
         "rule_path": [
             """MATCH p=()-[:异常现象]->(R1)-[:可能的异常原因]->(n:kedazhineng{name:"{entityname}"})-[:检查内容或部位]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"过热性异常检查"})-[:异常现象]->()-[:可能的异常原因]->(RI)  RETURN RI.name AS RIName"""},
        {"rule_name": "需要检修部位",
         "rule_path": [
             """MATCH p=(a:kedazhineng {name:"{entityname}"})-[*1..3]->(f:kedazhineng {name:"器身检修"})-[:部位]->(n:kedazhineng) RETURN p"""],
         "getIn4simi": """MATCH (RI)-[]->(f:kedazhineng {name:"器身检修"})-[:部位]->()  RETURN RI.name AS RIName"""}
    ],
    "组件部件检修": [
        {"rule_name": "组件中某部件的检修方法内容要求",  # in：部位
         "rule_path": [
             """MATCH p=()-[:组]->(R1)-[:部位]->(n:kedazhineng{name:"{entityname}"})-[:检修内容]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"组件部件检修"})-[:组]->(R1)-[:部位]->(RI)  RETURN R1.name AS R1Name, RI.name AS RIName"""},
        {"rule_name": "相应部件部位",  # in：检修内容
         "rule_path": [
             """MATCH p=()-[:组]->()-[:部位]->()-[:检修内容]->(b{name:"{entityname}"}) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"组件部件检修"})-[:组]->(R1)-[:部位]->(R2)-[:检修内容]->(RI)  RETURN R1.name AS R1Name,R2.name AS R2Name,RI.name AS RIName"""},
        {"rule_name": "组件装置的检修方法内容要求",  # in：组件
         "rule_path": [
             """MATCH p=()-[:组]->(n:kedazhineng{name:"{entityname}"})-[:部位]->()-[:检修内容]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"组件部件检修"})-[:组]->(RI)  RETURN RI.name AS RIName"""},
        {"rule_name": "组装要求",  # in：部位
         "rule_path": [
             """MATCH p=()-[:组]->(R1)-[:部位]->(n:kedazhineng{name:"{entityname}"})-[:检修内容]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"组件部件检修"})-[:组]->(R1)-[:部位]->(RI)  RETURN R1.name AS R1Name, RI.name AS RIName"""},
        {"rule_name": "组件的部位",  # in：组件
         "rule_path": [
             """MATCH p=()-[:组]->(n:kedazhineng{name:"{entityname}"})-[:部位]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"组件部件检修"})-[:组]->(RI)  RETURN RI.name AS RIName"""},
        {"rule_name": "哪些组/部件",  # in：组件
         "rule_path": [
             """MATCH p=(f:kedazhineng {name:"组件部件检修"})-[:组]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"组件部件检修"})  RETURN f.name AS RIName"""}
    ],
    "气体绝缘金属封闭开关设备状态评价导则": [
        {"rule_name": "状态量影响程度分级",
         "rule_path": [
             """MATCH p=()-[:kedazhineng{edge_name: "包含"}]->()-[:kedazhineng{edge_name: "包含-气体绝缘金属封闭开关设备状态评价导则"}]->()-[:kedazhineng{edge_name: "重要程度"}]->(n:kedazhineng{name:"{entityname}"})-[]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"气体绝缘金属封闭开关设备状态评价导则"})-[:kedazhineng{edge_name: "包含-气体绝缘金属封闭开关设备状态评价导则"}]->(R1)-[:kedazhineng{edge_name: "重要程度"}]->(RI)-[]->(R2)  RETURN R1.name AS R1Name,R2.name AS R2Name, RI.name AS RIName"""},
        {"rule_name": "状态量劣化程度的分级",
         "rule_path": [
             """MATCH p=()-[:kedazhineng{edge_name: "包含"}]->()-[:kedazhineng{edge_name: "包含-气体绝缘金属封闭开关设备状态评价导则"}]->()-[:kedazhineng{edge_name: "劣化程度"}]->(n:kedazhineng{name:"{entityname}"})-[]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"气体绝缘金属封闭开关设备状态评价导则"})-[:kedazhineng{edge_name: "包含-气体绝缘金属封闭开关设备状态评价导则"}]->(R1)-[:kedazhineng{edge_name: "劣化程度"}]->(RI)-[]->(R2)  RETURN R1.name AS R1Name,R2.name AS R2Name, RI.name AS RIName"""},
        {"rule_name": "各类部件不同状态下的扣分值",
         "rule_path": [
             """MATCH p=()-[:kedazhineng{edge_name: "包含"}]->()-[:kedazhineng{edge_name: "包含-气体绝缘金属封闭开关设备状态评价导则"}]->()-[:kedazhineng{edge_name: "部件"}]->(n:kedazhineng{name:"{entityname}"})-[]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"气体绝缘金属封闭开关设备状态评价导则"})-[:kedazhineng{edge_name: "包含-气体绝缘金属封闭开关设备状态评价导则"}]->(R1)-[:kedazhineng{edge_name: "部件"}]->(RI)-[R2]->()  RETURN R1.name AS R1Name, R2.edge_name AS R2Name, RI.name AS RIName"""},
        {"rule_name": "典型缺陷诊断关键和相关状态量",
         "rule_path": [
             """MATCH p=()-[:kedazhineng{edge_name: "包含"}]->()-[:kedazhineng{edge_name: "包含-气体绝缘金属封闭开关设备状态评价导则"}]->()-[:kedazhineng{edge_name: "典型缺陷"}]->(n:kedazhineng{name:"{entityname}"})-[]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"气体绝缘金属封闭开关设备状态评价导则"})-[:kedazhineng{edge_name: "包含-气体绝缘金属封闭开关设备状态评价导则"}]->(R1)-[:kedazhineng{edge_name: "典型缺陷"}]->(RI)-[R2]->()  RETURN R1.name AS R1Name, R2.edge_name AS R2Name, RI.name AS RIName"""},
        {"rule_name": "放电性缺陷状态量分析判断方法",
         "rule_path": [
             """MATCH p=()-[:kedazhineng{edge_name: "包含"}]->()-[:kedazhineng{edge_name: "包含-气体绝缘金属封闭开关设备状态评价导则"}]->()-[:kedazhineng{edge_name: "状态量"}]->(n:kedazhineng{name:"{entityname}"})-[]->() RETURN p"""],
         "getIn4simi": """MATCH (f:kedazhineng {name:"气体绝缘金属封闭开关设备状态评价导则"})-[:kedazhineng{edge_name: "包含-气体绝缘金属封闭开关设备状态评价导则"}]->(R1)-[:kedazhineng{edge_name: "状态量"}]->(RI)-[R2]->()  RETURN R1.name AS R1Name, R2.edge_name AS R2Name, RI.name AS RIName"""},
    ],
    "GIS典型故障": [
        {"rule_name": "GIS典型故障", "rule_path": [  #
            """MATCH p=(f:KedaIntelligent{name:"GIS典型故障"})-[:异常现象]->()-[:诊断方法]->()-[:判定条件]->() RETURN p"""],
         "getIn4simi": """MATCH (f:KedaIntelligent{name:"GIS典型故障"}) RETURN f.name AS RIName"""},
        {"rule_name": "可能存在什么异常现象", "rule_path": [  #
            """MATCH p=(f:KedaIntelligent {name:"GIS典型故障"})-[:异常现象]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:KedaIntelligent{name:"GIS典型故障"}) RETURN f.name AS RIName"""},
        {"rule_name": "异常如何诊断", "rule_path": [
            """MATCH p=()-[:异常现象]->(n:KedaIntelligent{name:"{entityname}"})-[:诊断方法]->() RETURN p"""],
         "getIn4simi": """MATCH (f:KedaIntelligent{name:"GIS典型故障"})-[:异常现象]->(RI) RETURN RI.name AS RIName"""},
        {"rule_name": "异常的判定条件是什么", "rule_path": [
            """MATCH p=()-[:异常现象]->(n:KedaIntelligent{name:"{entityname}"})-[:诊断方法]->()-[:判定条件]->() RETURN p"""],
         "getIn4simi": """MATCH (f:KedaIntelligent{name:"GIS典型故障"})-[:异常现象]->((RI) RETURN RI.name AS RIName"""},
        {"rule_name": "方法的判定条件是什么", "rule_path": [
            """MATCH p=()-[:诊断方法]->(n:KedaIntelligent{name:"{entityname}"})-[:判定条件]->() RETURN p"""],
         "getIn4simi": """MATCH (f:KedaIntelligent{name:"GIS典型故障"})-[:异常现象]->(R1)-[:诊断方法]->(RI) RETURN R1.name AS R1Name, RI.name AS RIName"""},
    ],
    "GIS放电性异常检查": [
        {"rule_name": "有哪些检查项目", "rule_path": [  #
            """MATCH p=(f:KedaIntelligent{name:"GIS"})-[:KedaIntelligent {edge_name:"类型"}]->()-[:KedaIntelligent {edge_name:"检查项目"}]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:KedaIntelligent{name:"GIS"})-[:KedaIntelligent {edge_name:"类型"}]->(RI) RETURN RI.name AS RIName"""},
        {"rule_name": "放电性异常检查", "rule_path": [  #
            """MATCH p=(f:KedaIntelligent{name:"GIS"})-[:KedaIntelligent {edge_name:"类型"}]->()-[:KedaIntelligent {edge_name:"检查项目"}]->()-[:KedaIntelligent {edge_name:"检查内容"}]->()-[:KedaIntelligent {edge_name:"原因与判断"}]->() RETURN p"""],
         "getIn4simi": """MATCH (f:KedaIntelligent{name:"GIS"})-[:KedaIntelligent {edge_name:"类型"}]->(RI) RETURN RI.name AS RIName"""},
        {"rule_name": "需要检查什么内容", "rule_path": [  #
            """MATCH p=(f:KedaIntelligent{name:"GIS"})-[:KedaIntelligent {edge_name:"类型"}]->()-[:KedaIntelligent {edge_name:"检查项目"}]->(n:KedaIntelligent{name:"{entityname}"})-[:KedaIntelligent {edge_name:"检查内容"}]->() RETURN p"""],
         "getIn4simi": """MATCH (f:KedaIntelligent{name:"GIS"})-[:KedaIntelligent {edge_name:"类型"}]->(R1)-[:KedaIntelligent {edge_name:"检查项目"}]->(RI) RETURN R1.name AS R1Name,RI.name AS RIName"""},
        {"rule_name": "如何进行判断", "rule_path": [  #
            """MATCH p=(f:KedaIntelligent{name:"GIS"})-[:KedaIntelligent {edge_name:"类型"}]->()-[:KedaIntelligent {edge_name:"检查项目"}]->(n:KedaIntelligent{name:"{entityname}"})-[:KedaIntelligent {edge_name:"检查内容"}]->()-[:KedaIntelligent {edge_name:"原因与判断"}]->() RETURN p"""],
         "getIn4simi": """MATCH (f:KedaIntelligent{name:"GIS"})-[:KedaIntelligent {edge_name:"类型"}]->(R1)-[:KedaIntelligent {edge_name:"检查项目"}]->(RI) RETURN R1.name AS R1Name,RI.name AS RIName"""},
    ],
    "常见异常": [
        {"rule_name": "有哪些常见异常", "rule_path": [  #
            """MATCH p=(f:KedaIntelligent {name:"常见异常"})-[:KedaIntelligent {edge_name:"异常现象"}]->(b) RETURN p"""],
         "getIn4simi": """MATCH (f:KedaIntelligent{name:"常见异常"}) RETURN f.name AS RIName"""},
        {"rule_name": "常见异常", "rule_path": [  #
            """MATCH p=(f:KedaIntelligent{name:"常见异常"})-[:KedaIntelligent {edge_name:"异常现象"}]->()-[:KedaIntelligent {edge_name:"异常原因"}]->()-[:KedaIntelligent {edge_name:"处理措施"}]->() RETURN p"""],
         "getIn4simi": """MATCH (f:KedaIntelligent{name:"常见异常"}) RETURN f.name AS RIName"""},
        {"rule_name": "异常的原因是什么", "rule_path": [  #
            """MATCH p=()-[:KedaIntelligent {edge_name:"异常现象"}]->(n:KedaIntelligent{name:"{entityname}"})-[:KedaIntelligent {edge_name:"异常原因"}]->() RETURN p"""],
         "getIn4simi": """MATCH (f:KedaIntelligent{name:"常见异常"})-[:KedaIntelligent {edge_name:"异常现象"}]->(RI) RETURN RI.name AS RIName"""},
        {"rule_name": "如何处理异常", "rule_path": [  #
            """MATCH p=()-[:KedaIntelligent {edge_name:"异常现象"}]->(n:KedaIntelligent{name:"{entityname}"})-[:KedaIntelligent {edge_name:"异常原因"}]->()-[:KedaIntelligent {edge_name:"处理措施"}]->() RETURN p"""],
         "getIn4simi": """MATCH (f:KedaIntelligent{name:"常见异常"})-[:KedaIntelligent {edge_name:"异常现象"}]->(RI) RETURN RI.name AS RIName"""},
    ]
}


def connect_to_db():
    try:
        connection = pymysql.connect(
            host='www.zhonghuapu.com',  # 数据库主机
            user='koroot',  # 数据库用户名
            password='DMiC-4092',  # 数据库密码
            database='db_hp',
            charset='utf8'
        )
        return connection
    except pymysql.MySQLError as err:
        print(f"Error: {err}")
        return None


def clean_rule_path(rule_path_list):
    # 去掉三重引号并取第一个路径
    return rule_path_list[0].replace('"""', '').strip()


def insert_data(connection, data_list):
    try:
        with connection.cursor() as cursor:
            insert_query = """
                INSERT INTO ko_kdzn_rule_construct (scene, rulename, rulepath, getin4simi)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_query, tuple(data_list))
        connection.commit()
        print(f"Inserted: {data_list[1]}")
    except pymysql.MySQLError as err:
        print(f"Error while inserting data: {err}")
        connection.rollback()


def import_kdzn_rules(connection, rules_dict):
    for scene, rules in rules_dict.items():
        for rule in rules:
            rulename = rule.get("rule_name", "")
            rulepath = clean_rule_path(rule.get("rule_path", [""]))
            getin4simi = rule.get("getIn4simi", "").replace('"""', '').strip()
            data_list = [scene, rulename, rulepath, getin4simi]
            insert_data(connection, data_list)


def kdzn_file_rules_kdzn_generate():
    connection = connect_to_db()
    if not connection:
        print("无法连接到数据库")
        return {}

    kdzn_file_rules_kdzn = {}  # 初始化空字典
    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            # 查询所有数据
            query = "SELECT * FROM ko_kdzn_rule_construct"
            cursor.execute(query)

            # 获取所有行
            results = cursor.fetchall()

            # 打印数据并构建字典
            if results:
                print("ko_kdzn_rule_construct 表数据：")
                for row in results:
                    scene = row['scene']
                    rule_data = {
                        "rule_name": row['rulename'],
                        "rule_path": [row['rulepath']],  # 保持它是一个列表
                        "getIn4simi": row['getin4simi']
                    }
                    # 如果字典中已经存在该 scene，则追加数据
                    if scene not in kdzn_file_rules_kdzn:
                        kdzn_file_rules_kdzn[scene] = [rule_data]
                    else:
                        kdzn_file_rules_kdzn[scene].append(rule_data)

            else:
                print("表中没有数据")

    except pymysql.MySQLError as err:
        print(f"查询错误: {err}")
        return {}
    finally:
        connection.close()  # 确保关闭连接

    return kdzn_file_rules_kdzn


def update_yaml(key: str, values: list, path: str = "D:\KO_QA_副本\ko_qa_server\config.yaml"):
    """
    把给定列表写入/更新到 YAML 文件的指定键下
    """
    p = Path(path)
    data = {}

    # 1. 先读旧文件（如果存在）
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    # 2. 更新键
    data[key] = values

    # 3. 回写
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def modifi_kdzn_rules(connection, rules_dict):
    for scene, rules in rules_dict.items():
        for rule in rules:
            rid = rule.get("rid", "")
            rulename = rule.get("rule_name", "")
            rulepath = clean_rule_path(rule.get("rule_path", [""]))
            getin4simi = rule.get("getIn4simi", "").replace('"""', '').strip()
            data_list = [rid, scene, rulename, rulepath, getin4simi]
            modifi_data(connection, data_list)


def modifi_data(connection, data_list):
    try:
        with connection.cursor() as cursor:
            update_query = """
                UPDATE ko_kdzn_rule_construct 
                SET scene = %s, rulename = %s, rulepath = %s, getin4simi = %s
                WHERE rid = %s
            """
            # Reorder data_list to match the query: scene, rulename, rulepath, getin4simi, rid
            cursor.execute(update_query, (data_list[1], data_list[2], data_list[3], data_list[4], data_list[0]))
            connection.commit()
            if cursor.rowcount > 0:
                print(f"Updated record with rid: {data_list[0]} for rulename: {data_list[2]}")
            else:
                print(f"No record found with rid: {data_list[0]} for rulename: {data_list[2]}")
    except pymysql.MySQLError as err:
        print(f"Error while updating data: {err}")
        connection.rollback()


if __name__ == '__main__':
    connection = connect_to_db()
    # 输出旧规则id
    # old_file_rules = {
    #     "放电性异常检查": [
    #     {"rule_name": "有哪些异常情况", "rule_path": [
    #         """MATCH p=(f:kedazhineng {name:"电力变压器"})-[:HAS_INSPECTION]->()-[:类型]->()-[:可能的异常原因]->()-[:检查方法或部位]->(b) RETURN p""", ],
    #      "getIn4simi": """MATCH (f:kedazhineng {name:"放电性异常检查"})  RETURN f.name AS RIName"""}
    # ]
    # }
    new_file_rules = {
        "放电性异常检查": [
            {"rid": 40, "rule_name": "有哪些异常情况，对应措施", "rule_path": [
                """MATCH p=(f:kedazhineng {name:"电力变压器"})-[:HAS_INSPECTION]->()-[:类型]->()-[:可能的异常原因]->()-[:检查方法或部位]->(b) RETURN p""", ],
             "getIn4simi": """MATCH (f:kedazhineng {name:"放电性异常检查"})  RETURN f.name AS RIName"""}
        ]
    }
    # 输出旧规则id
    # find_kdzn_rules(connection, old_file_rules)
    modifi_kdzn_rules(connection, new_file_rules)

