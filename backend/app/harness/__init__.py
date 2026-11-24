"""Harness 层（v0.6）：在 RSIAgent 式递归自我改进之上，管理“怎么用记忆来答题”。

  brain.py       快慢双脑（参照 Agnes Harness 的 System 1 / System 2 路由）
  ttt.py         测试时训练：课程 agent 针对当前案例出近邻练习题，只改一份临时记忆
  recursion.py   双层递归的内层（参照 Harness²：提议 → 探测 → 合成）
  experience.py  双轨任务经验库（参照 TTSE 的 FACT / TIP）与外层递归的经验反思
  policy.py      harness 策略（可进化的旋钮）与存档
  evolve.py      跨 run 策略进化（参照 RRSI 的正则化验收、RSI-Opt / DGM 的存档）
  train.py       可靠度模型的训练窗口（参照 A-RSI 的触发与健康门）与保形预测
"""
