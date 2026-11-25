"""Harness 层（v0.7）：在 RSIAgent 与快脑小模型之上，管理“这一案怎么答”。

  brain.py       快慢双脑（参照 Agnes Harness 的 System 1 / System 2）：快脑 = 训练出来的小模型，慢脑见 slow.py
  slow.py        慢脑 = 双层递归的内层（参照 Harness²）：近邻训练、贝叶斯意见、探测题裁决
  experience.py  改进经验库 = 双层递归的外层（参照 Harness² 的 improvement library）
  policy.py      基础 harness 的组件配置与条件规则
  evolve.py      RRSI：从改进库提议修改基础 harness，正则化验收（事件触发；RRSI 自己的验收与上线门 + 观察期）
  service.py     调度：每晚检查触发条件——训练策略进化 → 快脑训练窗口 → RRSI；控制台总览
"""
