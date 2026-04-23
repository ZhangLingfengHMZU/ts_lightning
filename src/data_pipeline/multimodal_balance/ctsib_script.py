"""
CTSIB 改良版：硬编码实验流程（时间轴脚本）
============================================

本文件定义单次平衡/多模态实验的**阶段顺序与名义时长**，供采集仿真或回放使用。
流程为代码内常量，非运行时配置文件；修改后请同步事件命名与落盘逻辑。

阶段概览（顺序即执行顺序）：
  - Test_Trigger:测试开始前的准备/触发窗口
  - Rest EO / Rest EC:静息睁眼、闭眼
  - MVC_Dorsi / MVC_Plantar:sEMG 最大自主收缩标定（各重复 3 次，段间休息由对应 End_* 的 duration 表示）
  - EO / EC / Platform:核心站立任务（与多模态合成中的 task 类型对齐）

数据结构:expriments_sequence
  元素为 dict, 键值对:
    type: str   — 阶段或边界事件标签（需与事件表/流水线约定一致）
    duration: float — 该条目的名义持续时间（秒）；具体语义见各 type 注释
    is_manual: bool — 是否依赖人工确认或外部手动推进

维护：增删阶段或改时长时，请同步 participants/README、BIDS _events.tsv 生成逻辑及依赖该列表的代码。
"""
expriments_sequence = [
    {
        "type": "Test_Trigger", # 开始测试触发器，给受试者10s准备时间
        "duration": 10.0,
        "is_manual": False
    },

    # 开始EEG生理基线阶段
    {"type": "Start_Rest_EO", "duration": 120.0, "is_manual": True},
    {"type": "End_Rest_EO", "duration": 10.0, "is_manual": True},

    {"type": "Start_Rest_EC", "duration": 120.0, "is_manual": True},
    {"type": "End_Rest_EC", "duration": 30.0, "is_manual": True},
    
    # 开始sEMG最大自主收缩, 重复三次，每次休息一分钟
    {"type": "Start_MVC_Dorsi_1", "duration": 5.0, "is_manual": True},
    {"type": "End_MVC_Dorsi_1", "duration": 60.0, "is_manual": True},
    {"type": "Start_MVC_Dorsi_2", "duration": 5.0, "is_manual": True},
    {"type": "End_MVC_Dorsi_2", "duration": 60.0, "is_manual": True},
    {"type": "Start_MVC_Dorsi_3", "duration": 5.0, "is_manual": True},
    {"type": "End_MVC_Dorsi_3", "duration": 60.0, "is_manual": True},

    {"type": "Start_MVC_Plantar_1", "duration": 5.0, "is_manual": True},
    {"type": "End_MVC_Plantar_1", "duration": 60.0, "is_manual": True},
    {"type": "Start_MVC_Plantar_2", "duration": 5.0, "is_manual": True},
    {"type": "End_MVC_Plantar_2", "duration": 60.0, "is_manual": True},
    {"type": "Start_MVC_Plantar_3", "duration": 5.0, "is_manual": True},
    {"type": "End_MVC_Plantar_3", "duration": 60.0, "is_manual": True},

    # 核心测试范式（CTSIB改良版）
    {"type": "Start_EO", "duration": 60.0, "is_manual": True},
    {"type": "End_EO", "duration": 60.0, "is_manual": True},

    {"type": "Start_EC", "duration": 60.0, "is_manual": True},
    {"type": "End_EC", "duration": 120.0, "is_manual": True},

    {"type": "Start_Platform", "duration": 60.0, "is_manual": True},
    {"type": "End_Platform", "duration": 0.0, "is_manual": True}
]