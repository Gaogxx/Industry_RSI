"""快脑小模型（v0.7）：一个真正训练出来的排序模型，拥有快脑的裁决权（通过晋升门之后）。

  features.py   每个候选根因的特征（贝叶斯证据因子 + 贝叶斯没用上的结构信号）
  model.py      Wide & Deep & Cross 排序网络（纯 Python）、训练（含 RSI-Opt 内层控制器）、温度校准
  data.py       训练数据：课程 agent 按配额出题 + 孪生程序给标签；产线确认的收据
  windows.py    训练窗口（A-RSI：事件触发、换池、健康门、晋升门）与保形预测
  optimize.py   训练策略的跨 run 进化（RSI-Opt 外层：存档、提议、验收、事件触发）
  playbook.py   训练知识库（RSI-Opt 的跨 run 知识：事实 / 做法，追责与停用）
"""
