# 医疗智能体ART强化学习训练系统 / Medical AI Agent ART Reinforcement Learning Training System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![ART](https://img.shields.io/badge/ART-Framework-green.svg)](https://github.com/openpipe/art)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

## 📋 项目简介 / Project Overview

本项目是一个基于ART（Automatic Reward Training）框架的医疗智能体强化学习训练系统。该系统能够训练一个专业的医疗问答智能体，具备搜索医疗知识库、回答医疗问题、提供治疗建议等多种能力。

This project is a medical AI agent reinforcement learning training system based on the ART (Automatic Reward Training) framework. The system can train a professional medical Q&A agent with capabilities including medical knowledge base search, medical question answering, and treatment recommendations.

## 🎯 核心功能 / Core Features

### 医疗智能体能力 / Medical Agent Capabilities
- 🔍 **治疗建议搜索** / Treatment advice search
- ⚠️ **不良事件查询** / Adverse events query  
- 💊 **药物概述检索** / Drug overview retrieval
- 🧪 **药物成分分析** / Drug composition analysis
- 🚨 **药物警示与安全性** / Drug warnings and safety
- 💉 **药物依赖与滥用信息** / Drug dependency and abuse information
- 📏 **剂量与用法指导** / Dosage and usage guidance
- 👥 **特定人群用药** / Special population medication
- 🔬 **药理学信息** / Pharmacological information
- 🏥 **临床信息** / Clinical information
- ⚗️ **非临床毒理学** / Non-clinical toxicology
- 👤 **以患者为中心的信息** / Patient-centered information

### 系统特性 / System Features
- 📊 **智能数据管理**: 自动检查HuggingFace数据集，避免重复生成
- 🔧 **配置驱动**: 使用YAML配置文件管理所有参数
- 💾 **双重保存**: 支持HuggingFace Hub和Google Drive模型保存
- 🧪 **独立推理**: 单独的推理测试模块，支持批量测试和结果分析
- 📈 **结果分析**: CSV格式结果保存，支持详细的性能分析
- 🔒 **安全配置**: .gitignore保护敏感配置信息

## 🔒 安全配置 / Security Configuration

⚠️ **重要安全说明 / Important Security Notes:**

- 本项目使用 [config.yaml.template](file:///Users/udpate/Desktop/med_art_rl/config.yaml.template) 作为配置模板
- 请复制模板文件并重命名为 `config.yaml`，然后填入您的API密钥
- **绝对不要**将包含真实API密钥的 `config.yaml` 文件提交到GitHub
- `config.yaml` 文件已在 [.gitignore](file:///Users/udpate/Desktop/med_art_rl/.gitignore) 中被排除，确保不会被意外提交

```bash
# 复制配置文件模板
cp config.yaml.template config.yaml

# 编辑配置文件，填入您的API密钥
vim config.yaml  # 或使用其他编辑器
```

## 📦 安装要求 / Installation Requirements

### Python版本 / Python Version
- Python 3.8+

### 核心依赖 / Core Dependencies
```bash
# ART框架 / ART Framework
openpipe-art[backend]==0.4.8

# 机器学习 / Machine Learning
torch
vllm
triton

# 数据处理 / Data Processing
datasets
pandas
numpy

# 自然语言处理 / NLP
langchain-core
litellm
tenacity

# 配置和工具 / Configuration and Tools
pyyaml
pydantic
tqdm
weave

# 数据库 / Database
sqlite3  # Python内置 / Built-in
```

### 安装步骤 / Installation Steps

1. **克隆仓库 / Clone Repository**
```bash
git clone <repository-url>
cd med_art_rl
```

2. **安装依赖 / Install Dependencies**
```bash
# 使用uv安装（推荐） / Install with uv (recommended)
uv pip install "openpipe-art[backend]==0.4.8" langchain-core tenacity datasets "litellm[proxy]" "gql<4" "protobuf==5.29.5" vllm numpy --prerelease allow --no-cache-dir

# 或使用pip / Or use pip
pip install -r requirements.txt
```

3. **配置API密钥 / Configure API Keys**
```bash
# 复制配置文件模板 / Copy config template
cp config.yaml.template config.yaml

# 编辑配置文件，填入您的API密钥 / Edit config file and fill in your API keys
vim config.yaml
```

4. **HuggingFace认证 / HuggingFace Authentication**
```bash
# 使用您的HuggingFace Token进行认证 / Authenticate with your HuggingFace Token
huggingface-cli login
```

## 🙏 支持声明 / Support Statement

Google AI 开发者计划团队通过提供 Google Cloud Credit 为这项工作提供了支持。
