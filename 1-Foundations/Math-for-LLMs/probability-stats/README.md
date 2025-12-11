# 📊 Probability and Statistics for LLMs
## 🎯 概述

本模块系统学习概率统计知识，重点关注在大型语言模型（LLM）中的应用。涵盖从基础概念到高级应用的完整学习路径。

## 📚 目录

### 第一部分：基础概念 
1. [概率基础](./01-basic-concepts/01-probability-fundamentals.md)
   - 概率公理与定义
   - 条件概率与贝叶斯定理
   - 随机变量与分布函数

2. [重要概率分布](./01-basic-concepts/02-probability-distributions.md)
   - 离散分布：伯努利、二项、多项
   - 连续分布：正态、均匀、指数
   - 神经网络中的分布应用

### 第二部分：统计推断 
3. [估计理论](./02-statistical-inference/01-estimation-theory.md)
   - 最大似然估计（MLE）
   - 最大后验估计（MAP）
   - 贝叶斯推断

4. [假设检验](./02-statistical-inference/02-hypothesis-testing.md)
   - 基本概念与p值
   - 置信区间
   - A/B测试原理

### 第三部分：信息论 
5. [信息论基础](./03-information-theory/01-fundamentals.md)
   - 熵、联合熵、条件熵
   - 互信息与KL散度
   - 交叉熵损失函数

### 第四部分：LLM应用 
6. [语言模型的概率视角](./04-llm-applications/01-probabilistic-view.md)
   - n-gram与神经语言模型
   - 生成式模型的概率基础

7. [采样与解码策略](./04-llm-applications/02-sampling-methods.md)
   - 贪心、beam search、温度采样
   - top-p、top-k采样
   - 模型校准

## 🛠️ 代码结构
```
probability-stats/
├── README.md # 本文件
├── 01-basic-concepts/ # 基础概念
│ ├── 01-probability-fundamentals.md
│ ├── 02-probability-distributions.md
│ └── code/
│ ├── basic_probability.py
│ └── distributions_demo.py
├── 02-statistical-inference/ # 统计推断
├── 03-information-theory/ # 信息论
├── 04-llm-applications/ # LLM应用
└── projects/ # 实践项目
├── 01-llm_sampling/
└── 02-model_calibration/
```
