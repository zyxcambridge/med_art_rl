# -*- coding: utf-8 -*-
"""
医疗智能体ART强化学习训练系统 / Medical AI Agent ART Reinforcement Learning Training System

流程图 / Flowchart:
```mermaid
graph TB
    A[开始 Start] --> B[加载配置文件 Load Config]
    B --> C[检查HF数据集 Check HF Dataset]
    C --> D{数据集存在? Dataset Exists?}
    D -->|是 Yes| E[下载HF数据集 Download HF Dataset]
    D -->|否 No| F[创建医疗数据库 Create Medical DB]
    F --> G[生成训练场景 Generate Training Scenarios]
    G --> H[上传数据集到HF Upload Dataset to HF]
    E --> I[初始化ART后端 Initialize ART Backend]
    H --> I
    I --> J[注册训练模型 Register Training Model]
    J --> K[开始训练循环 Start Training Loop]
    K --> L[执行医疗rollout Execute Medical Rollout]
    L --> M[使用RULER评分 RULER Scoring]
    M --> N[模型训练更新 Model Training Update]
    N --> O{达到最大步数? Max Steps Reached?}
    O -->|否 No| K
    O -->|是 Yes| P[保存模型到HF Save Model to HF]
    P --> Q[保存模型到Drive Save Model to Drive]
    Q --> R[推理测试 Inference Testing]
    R --> S[保存结果到CSV Save Results to CSV]
    S --> T[结束 End]
```

系统架构说明 / System Architecture Description:
1. 配置管理 / Configuration Management: 使用YAML文件管理所有配置参数
2. 数据管理 / Data Management: 智能检查和下载HuggingFace数据集，避免重复生成
3. 模型训练 / Model Training: 基于ART框架的强化学习训练流程
4. 结果评估 / Result Evaluation: 使用RULER框架进行医疗答案质量评估
5. 模型保存 / Model Saving: 支持HuggingFace Hub和Google Drive双重保存
6. 推理测试 / Inference Testing: 独立的推理测试模块，支持批量测试和结果分析

原始文件位置 / Original file location:
https://colab.research.google.com/drive/1bhgi21jrwt47Wc4E4Rtyx0ARmeLkbNjc

作者: Medical AI Team
日期: 2025-08-24
版本: 2.0 - 优化版
"""

# 注释掉Colab的magic命令，保证Python兼容性 / Comment out IPython magic to ensure Python compatibility
# %%capture
# import os
# 
# # 如果不在Colab中，直接使用pip install或uv pip install / If you're not in Colab, just use pip install or uv pip install
# if "COLAB_" not in "".join(os.environ.keys()):
#     !uv pip install "openpipe-art[backend]==0.4.8" langchain-core tenacity datasets "gql<4" --prerelease allow --no-cache-dir

# HuggingFace认证登录 / HuggingFace authentication login
# 注意：请使用您自己的HuggingFace Token / Note: Please use your own HuggingFace Token
# !hf auth login --token YOUR_HF_TOKEN_HERE

# @title 额外的Colab安装 / Extra Colab Installation { display-mode: "form" }
import os  # 操作系统接口模块 / Operating system interface module

# 检查是否在Colab环境中运行 / Check if running in Colab environment
if "COLAB_" in "".join(os.environ.keys()):
    try:
        import numpy  # 数值计算库 / Numerical computing library
        # 获取当前numpy版本号 / Get current numpy version
        get_numpy = f"numpy=={numpy.__version__}"
    except:
        get_numpy = "numpy"  # 如果获取失败则使用默认版本 / Use default version if failed
    try:
        import subprocess  # 子进程管理模块 / Subprocess management module
        # 检查是否为Tesla T4 GPU / Check if it's Tesla T4 GPU
        is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
    except:
        is_tesla_t4 = False  # 如果检查失败则设为False / Set to False if check failed
    # 根据GPU类型选择合适的vllm和triton版本 / Choose appropriate vllm and triton versions based on GPU type
    get_vllm, get_triton = (
        ("vllm==0.9.2", "triton==3.2.0") if is_t4 else ("vllm", "triton")
    )
    # 安装所需的Python包 / Install required Python packages
    !uv pip install --upgrade \
        "openpipe-art[backend]==0.4.8" langchain-core tenacity datasets "litellm[proxy]" "gql<4" "protobuf==5.29.5" {get_vllm} {get_numpy} --prerelease allow --no-cache-dir
    !uv pip install -qqq {get_triton}  # 安装triton库 / Install triton library

# 卸载旧版本litellm并重新安装 / Uninstall old litellm and reinstall
!pip uninstall -y litellm
!pip install litellm --no-cache-dir  # 不使用缓存，确保完整安装 / Don't use cache, ensure complete installation

"""<a name="Environment-Variables"></a>

### 环境变量 / Environment Variables

**OpenAI (用于RULER评判模型 / used for RULER judge model)**

我们的RULER奖励函数查询第三方模型来判断智能体的性能质量。任何LiteLLM支持的模型都可以使用。
在这个示例中我们使用OpenAI的o4-mini模型，因此需要设置`OPENAI_API_KEY`环境变量。

Our RULER reward function queries third-party models to judge the quality of the agent's performance. Any model supported by LiteLLM works. For this example we'll use OpenAI's o4-mini model, so we'll need to set the `OPENAI_API_KEY` environment variable.

**Weights & Biases (可选 / optional)**

在笔记本的后面部分，我们将创建一个能够自动将指标记录到Weights & Biases和将聊天完成情况记录到Weave的模型。
为了做到这一点，您需要将Weights & Biases API密钥作为环境变量提供。

Later on in the notebook, we'll be creating a model that can automatically logs metrics to Weights & Biases and chat completions to Weave. In order to do so, you'll need to provide your Weights & Biases API key as an environment variable.

"""

# 挂载Google Drive到Colab环境 / Mount Google Drive to Colab environment
from google.colab import drive  # 导入Google Drive模块 / Import Google Drive module
drive.mount('/content/drive')  # 挂载Google Drive / Mount Google Drive

# -*- coding: utf-8 -*-
"""
医疗智能体ART强化学习训练系统 / Medical AI Agent ART Reinforcement Learning Training System
使用ART框架训练一个能够搜索和回答医疗问题的智能体 / Use ART framework to train an agent for medical question search and answering

原始文件位置 / Original file location：https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e.ipynb
改造用于医疗数据的强化学习训练 / Modified for medical data reinforcement learning training

医疗智能体功能 / Medical Agent Functions：
- 治疗建议搜索 / Treatment advice search
- 不良事件查询 / Adverse events query
- 药物概述检索 / Drug overview retrieval
- 药物成分分析 / Drug composition analysis
- 药物警示与安全性 / Drug warnings and safety
- 药物依赖与滥用信息 / Drug dependency and abuse information
- 剂量与用法指导 / Dosage and usage guidance
- 特定人群用药 / Special population medication
- 药理学信息 / Pharmacological information
- 临床信息 / Clinical information
- 非临床毒理学 / Non-clinical toxicology
- 以患者为中心的信息 / Patient-centered information
"""

# ==================== 导入必要的库 / Import Necessary Libraries ====================
import os  # 操作系统接口模块 / Operating system interface module
import json  # JSON数据处理模块 / JSON data processing module
import random  # 随机数生成模块 / Random number generation module
import sqlite3  # SQLite数据库接口 / SQLite database interface
import yaml  # YAML文件处理模块 / YAML file processing module
from dataclasses import asdict, dataclass  # 数据类工具 / Data class tools
from datetime import datetime  # 日期时间处理 / Date and time processing
from textwrap import dedent  # 文本格式化工具 / Text formatting tools
from typing import List, Literal, Optional, Dict, Any  # 类型提示 / Type hints

# 第三方库导入 / Third-party library imports
from datasets import Dataset, Features, Sequence, Value, load_dataset  # HuggingFace数据集库 / HuggingFace datasets library
from pydantic import BaseModel, Field  # 数据验证库 / Data validation library
from tqdm import tqdm  # 进度条库 / Progress bar library
import torch  # PyTorch深度学习框架 / PyTorch deep learning framework
import weave  # Weights & Biases Weave跟踪库 / Weights & Biases Weave tracking library
from langchain_core.utils.function_calling import convert_to_openai_tool  # 函数调用工具转换 / Function calling tool conversion
from litellm import acompletion  # LiteLLM异步完成接口 / LiteLLM async completion interface
from tenacity import retry, stop_after_attempt  # 重试机制库 / Retry mechanism library

# ART框架相关导入 / ART framework related imports
import art  # ART强化学习框架 / ART reinforcement learning framework
from art.local import LocalBackend  # ART本地后端 / ART local backend
from art.rewards import ruler_score_group  # RULER评分组功能 / RULER score group function
from art.utils.litellm import convert_litellm_choice_to_openai  # LiteLLM到OpenAI转换工具 / LiteLLM to OpenAI conversion tool
from art.utils import iterate_dataset  # 数据集迭代工具 / Dataset iteration tool

# ==================== 加载配置文件 / Load Configuration File ====================
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    加载YAML配置文件 / Load YAML configuration file
    
    Args:
        config_path: 配置文件路径 / Configuration file path
        
    Returns:
        配置字典 / Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:  # 打开配置文件 / Open config file
            config = yaml.safe_load(file)  # 安全加载YAML / Safely load YAML
        print(f"配置文件加载成功: {config_path}")  # 配置加载成功提示 / Config loaded successfully message
        return config
    except FileNotFoundError:  # 文件未找到异常 / File not found exception
        print(f"警告: 配置文件未找到 {config_path}，使用默认配置")  # 警告信息 / Warning message
        return get_default_config()  # 返回默认配置 / Return default config
    except yaml.YAMLError as e:  # YAML解析错误 / YAML parsing error
        print(f"错误: 配置文件解析失败 {e}，使用默认配置")  # 错误信息 / Error message
        return get_default_config()  # 返回默认配置 / Return default config

def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置（当配置文件不存在时使用） / Get default configuration (used when config file doesn't exist)
    
    Returns:
        默认配置字典 / Default configuration dictionary
    """
    return {
        'api_keys': {
            'openrouter_api_key': "",  # 请在config.yaml中设置您的OpenRouter API密钥 / Please set your OpenRouter API key in config.yaml
            'openai_api_key': "",  # 请在config.yaml中设置您的OpenAI API密钥 / Please set your OpenAI API key in config.yaml
            'wandb_api_key': "",  # 请在config.yaml中设置您的WandB API密钥（可选）/ Please set your WandB API key in config.yaml (optional)
            'hf_token': ""  # 请在config.yaml中设置您的HuggingFace Token / Please set your HuggingFace Token in config.yaml
        },
        'model': {
            'base_model': "Qwen/Qwen2.5-7B-Instruct",
            'name': "medical-agent-001",
            'project': "medical-search-agent",
            'ruler_model': "openrouter/deepseek/deepseek-chat-v3.1",
            'system_prompt_generation_model': "openrouter/deepseek/deepseek-chat-v3.1",
            'input_generation_model': "openrouter/deepseek/deepseek-chat-v3.1"
        },
        'training': {
            'groups_per_step': 2,
            'num_epochs': 10,
            'rollouts_per_group': 4,
            'learning_rate': 0.00001,
            'max_steps': 10,
            'max_turns': 10,
            'temperature': 1.0
        },
        'data': {
            'hf_dataset_repo_name': "update0909/medical-qa-scenarios",
            'training_scenarios_limit': 20,
            'shuffle_data': True,
            'random_seed': 42
        },
        'paths': {
            'drive_save_path': "/content/drive/MyDrive/med_art_rl",
            'art_backend_path': "./.art"
        }
    }

# 加载全局配置 / Load global configuration
CONFIG = load_config()

# ==================== 环境变量设置 / Environment Variables Setup ====================
import os  # 操作系统环境变量管理 / Operating system environment variable management
from dotenv import load_dotenv  # 加载.env文件中的环境变量 / Load environment variables from .env file

# 加载.env文件（如果存在） / Load .env file (if exists)
load_dotenv()

# 从配置文件中获取API密钥 / Get API keys from configuration file
def setup_environment_variables(config: Dict[str, Any]):
    """
    设置环境变量 / Setup environment variables
    
    Args:
        config: 配置字典 / Configuration dictionary
    """
    api_keys = config.get('api_keys', {})  # 获取API密钥配置 / Get API keys configuration
    
    # 设置OpenAI API密钥（用于RULER功能） / Set OpenAI API key (for RULER functionality)
    openai_key = api_keys.get('openai_api_key', '')
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key  # 设置OpenAI API密钥 / Set OpenAI API key
        print("✓ OpenAI API密钥设置成功")  # OpenAI API key set successfully
    else:
        raise ValueError(
            "错误: 未找到OpenAI API密钥，请在config.yaml文件中配置api_keys.openai_api_key\n"
            "Error: OpenAI API key not found, please configure api_keys.openai_api_key in config.yaml"
        )  # Error: OpenAI API key not found
    
    # 设置OpenRouter API密钥（用于数据生成和模型推理） / Set OpenRouter API key (for data generation and model inference)
    openrouter_key = api_keys.get('openrouter_api_key', '')
    if openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = openrouter_key  # 设置OpenRouter API密钥 / Set OpenRouter API key
        print("✓ OpenRouter API密钥设置成功")  # OpenRouter API key set successfully
    else:
        raise ValueError(
            "错误: 未找到OpenRouter API密钥，请在config.yaml文件中配置api_keys.openrouter_api_key\n"
            "Error: OpenRouter API key not found, please configure api_keys.openrouter_api_key in config.yaml"
        )  # Error: OpenRouter API key not found
    
    # 设置WandB API密钥（可选） / Set WandB API key (optional)
    wandb_key = api_keys.get('wandb_api_key', '')
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key  # 设置WandB API密钥 / Set WandB API key
        print("✓ Weights & Biases API密钥设置成功")  # Weights & Biases API key set successfully
    else:
        print("⚠️  Weights & Biases API密钥未设置，将跳过指标记录")  # WandB API key not set, will skip metrics logging

# 设置环境变量 / Setup environment variables
setup_environment_variables(CONFIG)

# 从配置中获取模型参数 / Get model parameters from configuration
RULER_MODEL = CONFIG['model']['ruler_model']  # RULER评判模型 / RULER judge model
SYSTEM_PROMPT_GENERATION_MODEL = CONFIG['model']['system_prompt_generation_model']  # 系统提示生成模型 / System prompt generation model
INPUT_GENERATION_MODEL = CONFIG['model']['input_generation_model']  # 输入生成模型 / Input generation model
HF_DATASET_REPO_NAME = CONFIG['data']['hf_dataset_repo_name']  # HuggingFace数据集仓库名称 / HuggingFace dataset repository name

print(f"✓ 模型配置加载成功:")
print(f"  - RULER模型: {RULER_MODEL}")
print(f"  - 系统提示模型: {SYSTEM_PROMPT_GENERATION_MODEL}")
print(f"  - 输入生成模型: {INPUT_GENERATION_MODEL}")
print(f"  - HF数据集: {HF_DATASET_REPO_NAME}")


# ==================== 数据集管理功能 / Dataset Management Functions ====================

def check_and_download_dataset() -> Optional[List[Dict]]:
    """
    检查HuggingFace上是否已存在数据集，如果存在则下载，避免重复生成
    Check if dataset exists on HuggingFace, download if exists to avoid regeneration
    
    Returns:
        数据集列表或None / Dataset list or None
    """
    try:
        print(f"检查HuggingFace数据集: {HF_DATASET_REPO_NAME}")  # Checking HuggingFace dataset
        
        # 尝试从 HuggingFace 下载数据集 / Try to download dataset from HuggingFace
        dataset = load_dataset(HF_DATASET_REPO_NAME, split='train')  # 加载训练集 / Load training set
        
        print(f"✓ 成功从 HuggingFace 下载数据集，包含 {len(dataset)} 个样本")  # Successfully downloaded dataset from HuggingFace
        
        # 转换为字典列表 / Convert to dictionary list
        scenario_dicts = []
        for item in dataset:  # 遍历数据集中的每个项目 / Iterate through each item in dataset
            try:
                # 反序列化JSON字段 / Deserialize JSON fields
                options = json.loads(item.get('options', '{}'))  # 解析选项 / Parse options
                messages = [json.loads(msg) for msg in item.get('messages', [])]  # 解析消息 / Parse messages
                others = json.loads(item.get('others', '{}'))  # 解析其他信息 / Parse other info
                
                # 重构MedicalScenario对象 / Reconstruct MedicalScenario object
                scenario_dict = {
                    'id': item['id'],
                    'question_type': item['question_type'],
                    'question': item['question'],
                    'correct_answer': item['correct_answer'],
                    'options': options,
                    'reasoning': item.get('reasoning'),
                    'tools': item.get('tools', []),
                    'messages': messages,
                    'others': others,
                    'category': item.get('category', '医疗问答'),  # 默认类别 / Default category
                    'difficulty': item.get('difficulty', 'medium'),
                    'source_ids': item.get('source_ids', []),
                    'split': item.get('split', 'train')
                }
                scenario_dicts.append(scenario_dict)
                
            except json.JSONDecodeError as e:  # JSON解析错误 / JSON parsing error
                print(f"警告: 解析数据项时发生错误: {e}")  # Warning: Error parsing data item
                continue
        
        print(f"✓ 成功解析 {len(scenario_dicts)} 个数据项")  # Successfully parsed data items
        return scenario_dicts
        
    except Exception as e:  # 捕获所有异常 / Catch all exceptions
        print(f"⚠️  无法从 HuggingFace 下载数据集: {e}")  # Cannot download dataset from HuggingFace
        print("将使用本地数据生成新数据集...")  # Will use local data to generate new dataset
        return None

def save_model_to_hf(model, hf_model_repo_name: str):
    """
    保存模型到HuggingFace Hub / Save model to HuggingFace Hub
    
    Args:
        model: 要保存的模型 / Model to save
        hf_model_repo_name: HuggingFace模型仓库名称 / HuggingFace model repository name
    """
    print(f"正在将训练好的模型上传到 HuggingFace: {hf_model_repo_name}")  # Uploading trained model to HuggingFace
    
    try:
        # 尝试使用ART模型的push_to_hub方法 / Try using ART model's push_to_hub method
        if hasattr(model, 'push_to_hub'):  # 检查模型是否有push_to_hub方法 / Check if model has push_to_hub method
            model.push_to_hub(hf_model_repo_name, public=True)  # 上传模型到公开仓库 / Upload model to public repository
            print(f"✓ 训练好的模型已成功上传到 HuggingFace Hub: {hf_model_repo_name}")  # Trained model successfully uploaded
        else:
            print("⚠️  模型对象没有 push_to_hub 方法")  # Model object doesn't have push_to_hub method
            print("请检查 ART 库文档以了解正确的模型上传方式")  # Please check ART library documentation
    except Exception as e:  # 捕获上传异常 / Catch upload exceptions
        print(f"✗ 上传模型到 HuggingFace Hub 时出错: {e}")  # Error uploading model to HuggingFace Hub
        print("请确保已正确安装 huggingface_hub 库且已登录 (`huggingface-cli login`)")  # Please ensure huggingface_hub is installed and logged in

def save_model_to_drive(model, backend, drive_save_path: str):
    """
    保存模型到Google Drive / Save model to Google Drive
    
    Args:
        model: 要保存的模型 / Model to save
        backend: ART后端对象 / ART backend object
        drive_save_path: Google Drive保存路径 / Google Drive save path
    """
    print(f"正在保存训练好的模型到 Google Drive: {drive_save_path}")  # Saving trained model to Google Drive
    
    try:
        import shutil  # 文件操作工具库 / File operation utility library
        
        # 创建目标目录（如果不存在） / Create target directory (if not exists)
        os.makedirs(drive_save_path, exist_ok=True)
        
        # 获取ART保存模型的路径 / Get ART saved model path
        art_model_path = os.path.join(backend.path, model.project, "models", model.name)
        
        # 检查ART模型路径是否存在 / Check if ART model path exists
        if not os.path.exists(art_model_path):
            print(f"✗ 错误: 未找到ART训练好的模型路径: {art_model_path}")  # Error: ART trained model path not found
            return
        
        # 设置目标路径 / Set destination path
        destination_path = os.path.join(drive_save_path, model.name)
        
        # 如果目标目录已存在，先删除 / If destination directory exists, remove first
        if os.path.exists(destination_path):
            print(f"⚠️  警告: 目录 {destination_path} 已存在。正在删除并重新复制...")  # Warning: Directory exists, removing and copying again
            shutil.rmtree(destination_path)  # 删除目录 / Remove directory
        
        # 复制模型文件 / Copy model files
        shutil.copytree(art_model_path, destination_path)  # 复制整个目录树 / Copy entire directory tree
        print(f"✓ 训练好的模型已成功保存到 Google Drive: {destination_path}")  # Trained model successfully saved to Google Drive
        
    except Exception as e:  # 捕获保存异常 / Catch save exceptions
        print(f"✗ 保存模型到 Google Drive 时出错: {e}")  # Error saving model to Google Drive
        print("请确保已正确挂载 Google Drive，并且有写入权限")  # Please ensure Google Drive is mounted and has write permissions

def upload_dataset_to_hf(scenario_dicts: List[Dict], hf_dataset_repo_name: str):
    """
    上传数据集到HuggingFace Hub / Upload dataset to HuggingFace Hub
    
    Args:
        scenario_dicts: 场景字典列表 / List of scenario dictionaries
        hf_dataset_repo_name: HuggingFace数据集仓库名称 / HuggingFace dataset repository name
    """
    print(f"正在将处理后的数据集格式化为 HuggingFace 格式并上传...")  # Formatting processed dataset to HuggingFace format and uploading
    
    try:
        # 定义数据集特征，确保所有字段都被包含 / Define dataset features, ensuring all fields are included
        features = Features({
            'id': Value('string'),                    # 场景唯一标识符 / Scenario unique identifier
            'question_type': Value('string'),         # 问题类型 / Question type
            'question': Value('string'),              # 问题内容 / Question content
            'correct_answer': Value('string'),        # 正确答案 / Correct answer
            'options': Value('string'),               # 选项（存储为字符串） / Options (stored as string)
            'reasoning': Value('string'),             # 推理过程 / Reasoning process
            'tools': Sequence(Value('string')),      # 工具列表 / Tools list
            'messages': Sequence(Value('string')),   # 消息列表（存储为字符串） / Messages list (stored as strings)
            'others': Value('string'),               # 其他信息（存储为字符串） / Other info (stored as string)
            'category': Value('string'),             # 类别 / Category
            'difficulty': Value('string'),           # 难度 / Difficulty
            'source_ids': Sequence(Value('string')), # 源ID列表 / Source IDs list
            'split': Value('string'),                # 数据切分 / Data split
        })
        
        # 将嵌套的字典和列表转换为字符串 / Convert nested dictionaries and lists to strings
        for scenario_dict in scenario_dicts:
            scenario_dict['options'] = json.dumps(scenario_dict['options'], ensure_ascii=False)  # 序列化选项 / Serialize options
            scenario_dict['messages'] = [json.dumps(msg, ensure_ascii=False) for msg in scenario_dict['messages']]  # 序列化消息 / Serialize messages
            scenario_dict['others'] = json.dumps(scenario_dict['others'], ensure_ascii=False)  # 序列化其他信息 / Serialize other info
        
        # 从字典列表创建 Dataset / Create Dataset from dictionary list
        dataset = Dataset.from_list(scenario_dicts, features=features)
        
        # 上传 Dataset 到 HuggingFace Hub / Upload Dataset to HuggingFace Hub
        dataset.push_to_hub(hf_dataset_repo_name)  # 推送到HF Hub / Push to HF Hub
        print(f"✓ 数据集成功上传到 HuggingFace Hub: {hf_dataset_repo_name}")  # Dataset successfully uploaded to HuggingFace Hub
        
    except Exception as e:  # 捕获上传异常 / Catch upload exceptions
        print(f"✗ 上传数据集到 HuggingFace Hub 时出错: {e}")  # Error uploading dataset to HuggingFace Hub
        print("请确保已正确安装 huggingface_hub 库且已登录 (`huggingface-cli login`)")  # Please ensure huggingface_hub is installed and logged in

# ==================== 数据模型定义 ====================

# ==================== 数据模型定义 ====================

class MedicalInfo(BaseModel):
    """医疗信息数据模型"""
    info_id: str  # 信息唯一标识符
    category: str  # 信息分类：治疗建议、不良事件等
    title: Optional[str] = None  # 标题
    content: str  # 内容
    drug_name: Optional[str] = None  # 药物名称
    condition: Optional[str] = None  # 适应症/疾病
    source: Optional[str] = None  # 信息来源
    last_updated: Optional[str] = None  # 最后更新时间
    reliability_score: Optional[float] = None  # 可靠性评分

class MedicalScenario(BaseModel):
    """医疗问答场景数据模型"""
    id: str
    question_type: str  # 问题类型：multi_choice等
    question: str  # 问题
    correct_answer: str  # 正确答案
    options: Optional[Dict[str, str]] = {}  # 选择题选项
    reasoning: Optional[str] = None  # 推理过程
    tools: Optional[List[str]] = []  # 使用的工具
    messages: Optional[List[Dict]] = []  # 消息历史
    others: Optional[Dict[str, Any]] = {}  # 其他信息
    category: str = "general"  # 医疗信息类别
    difficulty: str = "medium"  # 难度级别
    source_ids: List[str] = []  # 参考信息ID列表
    split: Literal["train", "test"] = "train"

@dataclass
class MedicalSearchResult:
    """医疗信息搜索结果"""
    info_id: str
    snippet: str
    category: str
    relevance_score: float = 0.0

class FinalMedicalAnswer(BaseModel):
    """最终医疗答案"""
    answer: str
    source_ids: List[str]
    confidence: float = 0.0
    category: str = "general"

# 数据库配置 / Database configuration
DB_PATH = CONFIG.get('database', {}).get('db_path', './medical_knowledge.db')  # 从配置文件获取数据库路径 / Get database path from config
MEDICAL_DATA_PATH = CONFIG.get('database', {}).get('medical_data_path', './curebench_valset_phase1.jsonl')  # 从配置文件获取医疗数据路径 / Get medical data path from config

# 全局数据库连接
db_conn = None

# ==================== 数据库操作函数 ====================

def create_medical_database():
    """从医疗数据文件创建SQLite数据库"""
    print("正在从医疗数据文件创建数据库...")
    print(f"数据源：{MEDICAL_DATA_PATH}")

    # 数据库表结构
    SQL_CREATE_TABLES = """
    DROP TABLE IF EXISTS medical_info_fts;
    DROP TABLE IF EXISTS medical_info;
    DROP TABLE IF EXISTS drug_interactions;
    DROP TABLE IF EXISTS condition_mappings;

    CREATE TABLE medical_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        info_id TEXT UNIQUE,
        category TEXT,
        title TEXT,
        content TEXT,
        drug_name TEXT,
        condition_name TEXT,
        source TEXT,
        last_updated TEXT,
        reliability_score REAL
    );

    CREATE TABLE drug_interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        drug_name TEXT,
        interaction_type TEXT,
        description TEXT,
        severity TEXT
    );

    CREATE TABLE condition_mappings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        condition_name TEXT,
        category TEXT,
        description TEXT
    );
    """

    SQL_CREATE_INDEXES_TRIGGERS = """
    CREATE INDEX idx_medical_info_category ON medical_info(category);
    CREATE INDEX idx_medical_info_drug ON medical_info(drug_name);
    CREATE INDEX idx_medical_info_condition ON medical_info(condition_name);
    CREATE INDEX idx_medical_info_id ON medical_info(info_id);

    CREATE VIRTUAL TABLE medical_info_fts USING fts5(
        title,
        content,
        drug_name,
        condition_name,
        content='medical_info',
        content_rowid='id'
    );

    CREATE TRIGGER medical_info_ai AFTER INSERT ON medical_info BEGIN
        INSERT INTO medical_info_fts (rowid, title, content, drug_name, condition_name)
        VALUES (new.id, new.title, new.content, new.drug_name, new.condition_name);
    END;

    CREATE TRIGGER medical_info_ad AFTER DELETE ON medical_info BEGIN
        DELETE FROM medical_info_fts WHERE rowid=old.id;
    END;

    CREATE TRIGGER medical_info_au AFTER UPDATE ON medical_info BEGIN
        UPDATE medical_info_fts SET
            title=new.title,
            content=new.content,
            drug_name=new.drug_name,
            condition_name=new.condition_name
        WHERE rowid=old.id;
    END;
    """

    # 创建数据库
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_TABLES)
    conn.commit()

    # 从JSONL文件加载医疗数据
    print("正在加载医疗数据...")
    if not os.path.exists(MEDICAL_DATA_PATH):
        raise FileNotFoundError(f"医疗数据文件不存在：{MEDICAL_DATA_PATH}")

    # 启用事务处理以提高性能
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")
    conn.execute("BEGIN TRANSACTION;")

    record_count = 0

    # 预定义的医疗信息类别和对应的模拟内容
    medical_categories = [
        "治疗建议", "不良事件", "药物概述", "药物成分", "药物警示与安全性",
        "药物依赖与滥用", "剂量与用法", "特定人群用药", "药理学",
        "临床信息", "非临床毒理学", "以患者为中心的信息"
    ]

    try:
        with open(MEDICAL_DATA_PATH, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())

                    # 提取问题信息来生成医疗知识条目
                    question = data.get('question', '')
                    options = data.get('options', {})
                    correct_answer = data.get('correct_answer', '')
                    question_id = data.get('id', f'medical_info_{line_num}')

                    # 为每个医疗类别创建相关的知识条目
                    for category in medical_categories:
                        info_id = f"{question_id}_{category}"

                        # 根据问题和选项生成内容
                        content_parts = [question]
                        if options:
                            content_parts.append("选项：")
                            for key, value in options.items():
                                marker = "✓ " if key == correct_answer else ""
                                content_parts.append(f"{marker}{key}: {value}")

                        content = "\n".join(content_parts)

                        # 从问题中提取可能的药物名称
                        drug_name = None
                        for option_value in options.values():
                            if any(keyword in option_value.lower() for keyword in
                                  ['acid', 'ine', 'ole', 'ide', 'drug', 'medicine']):
                                drug_name = option_value
                                break

                        # 从问题中提取疾病/症状
                        condition_name = None
                        if 'acne' in question.lower():
                            condition_name = "痤疮"
                        elif 'treatment' in question.lower():
                            condition_name = "一般治疗"

                        cursor.execute("""
                            INSERT INTO medical_info
                            (info_id, category, title, content, drug_name, condition_name,
                             source, last_updated, reliability_score)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            info_id,
                            category,
                            f"{category} - {question[:50]}...",
                            content,
                            drug_name,
                            condition_name,
                            "CureBench数据集",
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            0.85
                        ))

                        record_count += 1

                except json.JSONDecodeError as e:
                    print(f"解析第{line_num}行JSON时出错：{e}")
                    continue
                except Exception as e:
                    print(f"处理第{line_num}行数据时出错：{e}")
                    continue

    except FileNotFoundError:
        print(f"找不到文件：{MEDICAL_DATA_PATH}")
        raise

    conn.commit()

    # 创建索引和触发器
    print("正在创建索引和全文搜索...")
    cursor.executescript(SQL_CREATE_INDEXES_TRIGGERS)
    cursor.execute('INSERT INTO medical_info_fts(medical_info_fts) VALUES("rebuild")')
    conn.commit()

    print(f"成功创建医疗知识数据库，包含 {record_count} 条记录")
    return conn

def get_db_connection():
    """获取数据库连接"""
    global db_conn
    if db_conn is None:
        if os.path.exists(DB_PATH):
            print(f"正在加载现有数据库：{DB_PATH}")
            db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        else:
            db_conn = create_medical_database()
    return db_conn

# ==================== 医疗信息搜索函数 ====================

def search_medical_info(
    keywords: List[str],
    category: Optional[str] = None,
    drug_name: Optional[str] = None,
    condition: Optional[str] = None,
    max_results: int = 10
) -> List[MedicalSearchResult]:
    """根据关键词和过滤条件搜索医疗信息"""
    conn = get_db_connection()
    cursor = conn.cursor()

    where_clauses: List[str] = []
    params: List[str | int] = []

    if not keywords:
        raise ValueError("搜索时必须提供关键词")

    if max_results > 10:
        raise ValueError("最大结果数不能超过10")

    # 构建FTS5查询
    fts_query = " ".join(f'"{k.replace('"', '""')}"' for k in keywords)
    where_clauses.append("fts.medical_info_fts MATCH ?")
    params.append(fts_query)

    if category:
        where_clauses.append("m.category = ?")
        params.append(category)

    if drug_name:
        where_clauses.append("m.drug_name LIKE ?")
        params.append(f"%{drug_name}%")

    if condition:
        where_clauses.append("m.condition_name LIKE ?")
        params.append(f"%{condition}%")

    sql = f"""
        SELECT
            m.info_id,
            snippet(medical_info_fts, -1, '<b>', '</b>', ' ... ', 20) as snippet,
            m.category,
            m.reliability_score
        FROM
            medical_info m JOIN medical_info_fts fts ON m.id = fts.rowid
        WHERE
            {" AND ".join(where_clauses)}
        ORDER BY
            m.reliability_score DESC, m.last_updated DESC
        LIMIT ?;
    """
    params.append(max_results)

    cursor.execute(sql, params)
    results = cursor.fetchall()

    return [
        MedicalSearchResult(
            info_id=row[0],
            snippet=row[1],
            category=row[2],
            relevance_score=row[3] or 0.0
        )
        for row in results
    ]

def get_medical_info_detail(info_id: str) -> Optional[MedicalInfo]:
    """根据信息ID获取详细的医疗信息"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT info_id, category, title, content, drug_name, condition_name,
               source, last_updated, reliability_score
        FROM medical_info
        WHERE info_id = ?
    """, (info_id,))

    row = cursor.fetchone()
    if not row:
        return None

    return MedicalInfo(
        info_id=row[0],
        category=row[1],
        title=row[2],
        content=row[3],
        drug_name=row[4],
        condition=row[5],
        source=row[6],
        last_updated=row[7],
        reliability_score=row[8]
    )

def load_medical_scenarios(
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> List[MedicalScenario]:
    """从JSONL文件加载医疗问答场景"""
    print("正在加载医疗问答场景...")

    scenarios = []

    try:
        with open(MEDICAL_DATA_PATH, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if limit and len(scenarios) >= limit:
                    break

                try:
                    data = json.loads(line.strip())

                    scenario = MedicalScenario(
                        id=data.get('id', f'scenario_{line_num}'),
                        question_type=data.get('question_type', 'multi_choice'),
                        question=data.get('question', ''),
                        correct_answer=data.get('correct_answer', ''),
                        options=data.get('options', {}),
                        reasoning=data.get('reasoning', None),
                        tools=data.get('tools', []),
                        messages=data.get('messages', []),
                        others=data.get('others', {}),
                        category="医疗问答",
                        difficulty="medium",
                        source_ids=[],
                        split="train"
                    )

                    scenarios.append(scenario)

                except json.JSONDecodeError as e:
                    print(f"解析第{line_num}行JSON时出错：{e}")
                    continue
                except Exception as e:
                    print(f"处理第{line_num}行数据时出错：{e}")
                    continue

    except FileNotFoundError:
        print(f"找不到医疗数据文件：{MEDICAL_DATA_PATH}")
        return []

    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(scenarios)

    if limit is not None:
        scenarios = scenarios[:limit]

    print(f"加载了 {len(scenarios)} 个医疗问答场景")
    return scenarios

# ==================== 模型配置 / Model Configuration ====================

"""
创建模型 / Creating a Model

现在我们已经定义了环境的规则，可以创建一个学习有效搜索医疗信息的模型。
我们将使用Qwen 2.5 7B模型作为示例。

Now that we've defined the rules of our environment, we can create a model that will learn to search medical information effectively. We'll use a Qwen 2.5 7B model for this example.
"""

# 设置随机种子 / Set random seed
random.seed(CONFIG['data']['random_seed'])  # 使用配置中的随机种子 / Use random seed from config

# 声明训练模型 / Declare training model
model = art.TrainableModel(
    name=CONFIG['model']['name'],  # 模型名称，从配置文件获取 / Model name from config file
    project=CONFIG['model']['project'],  # 项目名称，从配置文件获取 / Project name from config file
    base_model=CONFIG['model']['base_model'],  # 基础模型，从配置文件获取 / Base model from config file
)

print(f"✓ 模型初始化完成:")
print(f"  - 模型名称: {model.name}")
print(f"  - 项目名称: {model.project}")
print(f"  - 基础模型: {CONFIG['model']['base_model']}")

# 针对T4 GPU的配置优化 / Configuration optimization for T4 GPU
# 如果检测到T4 GPU，可以启用以下优化设置 / If T4 GPU detected, enable the following optimization settings
# if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major < 8:
#     model._internal_config = art.dev.InternalModelConfig(
#         init_args=art.dev.InitArgs(
#             max_seq_length=CONFIG['training'].get('max_seq_length', 8192),  # 从配置获取最大序列长度 / Get max seq length from config
#         ),
#         engine_args=art.dev.EngineArgs(
#             enforce_eager=True,  # 强制使用eager模式 / Force eager mode
#             gpu_memory_utilization=CONFIG['training'].get('gpu_memory_utilization', 0.8),  # GPU内存使用率 / GPU memory utilization
#         ),
#     )

# ==================== 评估和判断函数 ====================

class CorrectnessJudgeResponse(BaseModel):
    """正确性判断响应"""
    reasoning: str = Field(description="推理过程的解释")
    accept: bool = Field(description="是否接受AI的答案")

@retry(stop=stop_after_attempt(3))
async def judge_medical_correctness(
    scenario: MedicalScenario, answer: str
) -> CorrectnessJudgeResponse:
    """判断医疗答案的正确性"""
    system_prompt = dedent(
        """
        你是一个医疗知识专家，负责评判AI助手的医疗问答回答质量。

        你会收到一个医疗问题、参考答案（标记为**参考答案**）和AI助手生成的答案（标记为**AI答案**）。

        你的任务是判断AI答案是否正确且应该被接受。如果AI答案包含参考答案的相关信息，你应该接受它。
        如果AI答案缺少问题相关的重要信息，或者与参考答案相矛盾，你不应该接受它。

        在医疗领域，准确性至关重要，因此请严格评判。
        """
    )

    # 构建参考答案
    reference_answer = scenario.correct_answer
    if scenario.options and reference_answer in scenario.options:
        reference_answer = f"{reference_answer}: {scenario.options[reference_answer]}"

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"医疗问题: {scenario.question}\n"
                f"参考答案: {reference_answer}\n"
                f"AI答案: {answer}"
            ),
        },
    ]

    # response = await acompletion(
    #     model="gpt-4.1",
    #     messages=messages,
    #     response_format=CorrectnessJudgeResponse,
    # )
    # RULER_MODEL
    # response = await acompletion(
    #     model=litellm_model_name,
    #     base_url=model.inference_base_url,
    #     api_key=model.inference_api_key,
    #     temperature=0.7,
    #     messages=messages,
    #     response_format=CorrectnessJudgeResponse,
    # )
    print(SYSTEM_PROMPT_GENERATION_MODEL + ": 1 ")
    response = await acompletion(
        model=SYSTEM_PROMPT_GENERATION_MODEL,
        messages=messages,
        response_format=CorrectnessJudgeResponse,
    )
    first_choice = response.choices[0]
    raw_content = first_choice.message.content or "{}"

    try:
        return CorrectnessJudgeResponse.model_validate_json(raw_content)
    except Exception as e:
        return CorrectnessJudgeResponse(
            reasoning=f"解析错误: {e}\n原始内容: {raw_content}",
            accept=False
        )

# ==================== 轨迹和场景类 ====================

class MedicalTrajectory(art.Trajectory):
    """医疗智能体轨迹"""
    final_answer: FinalMedicalAnswer | None = None

class MedicalScenarioWrapper(BaseModel):
    """医疗场景包装器"""
    step: int
    scenario: MedicalScenario

# ==================== Rollout函数 ====================

MAX_TURNS = 10

@weave.op
async def medical_rollout(
    model: art.Model,
    medical_scenario: MedicalScenarioWrapper
) -> MedicalTrajectory:
    """医疗智能体的rollout函数"""
    scenario = medical_scenario.scenario

    traj = MedicalTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={
            "scenario_id": scenario.id,
            "step": medical_scenario.step,
            "category": scenario.category
        },
    )

    system_prompt = dedent(
        f"""
        你是一个专业的医疗信息搜索智能体。你可以使用提供的工具来搜索医疗知识库，
        回答用户的医疗相关问题。你最多可以进行 {MAX_TURNS} 轮搜索，
        如果第一次搜索没有找到答案，可以尝试使用不同的关键词。

        你的专业领域包括：
        - 治疗建议
        - 不良事件
        - 药物概述
        - 药物成分
        - 药物警示与安全性
        - 药物依赖与滥用
        - 剂量与用法
        - 特定人群用药
        - 药理学
        - 临床信息
        - 非临床毒理学
        - 以患者为中心的信息

        请基于搜索到的可靠信息给出准确的答案。
        """
    )

    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario.question},
    ]

    def search_treatment_advice(keywords: list[str]) -> list[dict]:
        """搜索治疗建议相关信息"""
        results = search_medical_info(
            keywords=keywords,
            category="治疗建议",
            max_results=5
        )
        return [asdict(result) for result in results]

    def search_adverse_events(keywords: list[str]) -> list[dict]:
        """搜索不良事件相关信息"""
        results = search_medical_info(
            keywords=keywords,
            category="不良事件",
            max_results=5
        )
        return [asdict(result) for result in results]

    def search_drug_overview(keywords: list[str]) -> list[dict]:
        """搜索药物概述信息"""
        results = search_medical_info(
            keywords=keywords,
            category="药物概述",
            max_results=5
        )
        return [asdict(result) for result in results]

    def search_drug_components(keywords: list[str]) -> list[dict]:
        """搜索药物成分信息"""
        results = search_medical_info(
            keywords=keywords,
            category="药物成分",
            max_results=5
        )
        return [asdict(result) for result in results]

    def search_drug_warnings(keywords: list[str]) -> list[dict]:
        """搜索药物警示与安全性信息"""
        results = search_medical_info(
            keywords=keywords,
            category="药物警示与安全性",
            max_results=5
        )
        return [asdict(result) for result in results]

    def search_drug_dependency(keywords: list[str]) -> list[dict]:
        """搜索药物依赖与滥用信息"""
        results = search_medical_info(
            keywords=keywords,
            category="药物依赖与滥用",
            max_results=5
        )
        return [asdict(result) for result in results]

    def search_dosage_usage(keywords: list[str]) -> list[dict]:
        """搜索剂量与用法信息"""
        results = search_medical_info(
            keywords=keywords,
            category="剂量与用法",
            max_results=5
        )
        return [asdict(result) for result in results]

    def search_special_populations(keywords: list[str]) -> list[dict]:
        """搜索特定人群用药信息"""
        results = search_medical_info(
            keywords=keywords,
            category="特定人群用药",
            max_results=5
        )
        return [asdict(result) for result in results]

    def search_pharmacology(keywords: list[str]) -> list[dict]:
        """搜索药理学信息"""
        results = search_medical_info(
            keywords=keywords,
            category="药理学",
            max_results=5
        )
        return [asdict(result) for result in results]

    def search_clinical_info(keywords: list[str]) -> list[dict]:
        """搜索临床信息"""
        results = search_medical_info(
            keywords=keywords,
            category="临床信息",
            max_results=5
        )
        return [asdict(result) for result in results]

    def search_nonclinical_toxicology(keywords: list[str]) -> list[dict]:
        """搜索非临床毒理学信息"""
        results = search_medical_info(
            keywords=keywords,
            category="非临床毒理学",
            max_results=5
        )
        return [asdict(result) for result in results]

    def search_patient_info(keywords: list[str]) -> list[dict]:
        """搜索以患者为中心的信息"""
        results = search_medical_info(
            keywords=keywords,
            category="以患者为中心的信息",
            max_results=5
        )
        return [asdict(result) for result in results]

    def get_detailed_info(info_id: str) -> Optional[dict]:
        """获取详细的医疗信息"""
        medical_info = get_medical_info_detail(info_id)
        if medical_info:
            return medical_info.dict()
        return None

    def return_final_answer(
        answer: str,
        reference_info_ids: list[str],
        confidence: float = 0.8
    ) -> FinalMedicalAnswer:
        """返回最终的医疗答案"""
        return FinalMedicalAnswer(
            answer=answer,
            source_ids=reference_info_ids,
            confidence=confidence,
            category=scenario.category
        )

    # 定义可用工具
    tools = [
        search_treatment_advice, search_adverse_events, search_drug_overview,
        search_drug_components, search_drug_warnings, search_drug_dependency,
        search_dosage_usage, search_special_populations, search_pharmacology,
        search_clinical_info, search_nonclinical_toxicology, search_patient_info,
        get_detailed_info, return_final_answer
    ]

    tools_by_name = {t.__name__: t for t in tools}
    traj.tools = [convert_to_openai_tool(t) for t in tools]

    # 设置模型推理参数
    if model.trainable:
        litellm_model_name = f"hosted_vllm/{model.name}"
    else:
        litellm_model_name = model.name

    # 开始对话循环
    for turn in range(MAX_TURNS):
        response = await acompletion(
            model=litellm_model_name,
            base_url=model.inference_base_url,
            api_key=model.inference_api_key,
            temperature=1,
            messages=traj.messages(),
            caching=False,
            tools=traj.tools,
        )
      #   response = await acompletion(
      #     model=SYSTEM_PROMPT_GENERATION_MODEL,
      #     messages=messages,
      #     response_format=CorrectnessJudgeResponse,
      #  )

        response_message = response.choices[0].message
        traj.messages_and_choices.append(
            convert_litellm_choice_to_openai(response.choices[0])
        )

        if not response_message.tool_calls:
            return traj

        try:
            for tool_call in response_message.tool_calls:
                tool_name: str = tool_call.function.name
                if tool_name in tools_by_name:
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_to_call = tools_by_name[tool_name]
                    result = tool_to_call(**tool_args)
                    traj.messages_and_choices.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": str(result),
                        }
                    )

                    if tool_name == "return_final_answer":
                        traj.final_answer = result
                        # 评估轨迹质量
                        if traj.final_answer:
                            correctness_judge_response = await judge_medical_correctness(
                                scenario, traj.final_answer.answer
                            )
                            traj.metrics["correct"] = float(
                                correctness_judge_response.accept
                            )
                        return traj
        except Exception as e:
            print(f"解析工具调用时出错：{e}")
            return traj

    return traj



"""### run the train

开始训练


"""

# ==================== 主要训练流程 ====================

async def main():
    """主要训练流程 / Main training workflow"""

    # 初始化后端服务器 / Initialize backend server
    backend = LocalBackend(
        # 在Google Colab中需要设置为True以正确显示输出 / Set to True in Google Colab for correct output display
        in_process=True,
        path=CONFIG['paths']['art_backend_path'],  # 使用配置中的路径 / Use path from config
    )

    # 注册模型到后端（设置日志记录、推理和训练） / Register model to backend (setup logging, inference and training)
    await model.register(backend)

    # 初始化Weave（如果有WANDB API密钥） / Initialize Weave (if WANDB API key exists)
    if os.getenv("WANDB_API_KEY", ""):
        weave.init(model.project, settings={"print_call_link": False})  # 初始化Weave跟踪 / Initialize Weave tracking

    # 检查并下载现有数据集（避免重复生成） / Check and download existing dataset (avoid regeneration)
    existing_scenarios = check_and_download_dataset()
    
    if existing_scenarios:
        # 使用下载的数据集 / Use downloaded dataset
        print(f"✓ 使用从 HuggingFace 下载的数据集，包含 {len(existing_scenarios)} 个场景")  # Using downloaded dataset from HuggingFace
        
        # 转换为MedicalScenario对象 / Convert to MedicalScenario objects
        training_scenarios = []
        for scenario_dict in existing_scenarios:
            try:
                scenario = MedicalScenario(**scenario_dict)  # 创建MedicalScenario对象 / Create MedicalScenario object
                training_scenarios.append(scenario)
            except Exception as e:  # 捕获转换异常 / Catch conversion exceptions
                print(f"警告: 转换数据项时出错: {e}")  # Warning: Error converting data item
                continue
        
        # 限制场景数量（根据配置） / Limit scenario count (based on config)
        limit = CONFIG['data']['training_scenarios_limit']
        if limit and len(training_scenarios) > limit:
            training_scenarios = training_scenarios[:limit]  # 截取指定数量 / Truncate to specified count
            
        print(f"✓ 最终使用 {len(training_scenarios)} 个训练场景")  # Finally using training scenarios
        
    else:
        # 生成新数据集 / Generate new dataset
        print("生成新的训练数据集...")  # Generating new training dataset
        training_scenarios = load_medical_scenarios(
            limit=CONFIG['data']['training_scenarios_limit'],  # 使用配置中的限制 / Use limit from config
            shuffle=CONFIG['data']['shuffle_data'],  # 使用配置中的打乱设置 / Use shuffle setting from config
            seed=CONFIG['data']['random_seed']  # 使用配置中的随机种子 / Use random seed from config
        )
        
        # 上传新生成的数据集到HuggingFace / Upload newly generated dataset to HuggingFace
        scenario_dicts = [scenario.model_dump() for scenario in training_scenarios]  # 转换为字典 / Convert to dictionaries
        upload_dataset_to_hf(scenario_dicts, HF_DATASET_REPO_NAME)  # 上传数据集 / Upload dataset

    print("医疗搜索环境创建完成！")  # Medical search environment creation completed!
    print(f"数据库包含完整的医疗知识库，加载了 {len(training_scenarios)} 个训练圼景")  # Database contains complete medical knowledge base, loaded training scenarios

    # 显示第一个场景示例 / Display first scenario example
    if training_scenarios:
        print("\n示例场景：")  # Example scenario:
        scenario = training_scenarios[0]
        print(f"ID: {scenario.id}")
        print(f"问题: {scenario.question}")  # Question
        print(f"问题类型: {scenario.question_type}")  # Question type
        print(f"正确答案: {scenario.correct_answer}")  # Correct answer
        if scenario.options:
            print("选项:")  # Options:
            for key, value in scenario.options.items():
                print(f"  {key}: {value}")

    # ==================== 训练配置 / Training Configuration ====================
    # 从配置文件加载训练参数 / Load training parameters from config file
    training_config = CONFIG['training']  # 获取训练配置 / Get training configuration
    
    print(f"\n开始训练循环... / Starting training loop...")  # Starting training loop
    print(f"训练配置: {training_config}")  # Training configuration

    # 使用iterate_dataset处理真实训练场景 / Use iterate_dataset to process real training scenarios
    training_iterator = iterate_dataset(
        training_scenarios,  # 训练场景列表 / Training scenarios list
        groups_per_step=training_config["groups_per_step"],  # 每步的组数 / Groups per step
        num_epochs=training_config["num_epochs"],  # 训练轮数 / Training epochs
        initial_step=await model.get_step(),  # 初始步数 / Initial step
    )

    for batch in training_iterator:
        print(f"\nTraining step {batch.step}, Epoch {batch.epoch}, Epoch step {batch.epoch_step}")
        print(f"Batch contains {len(batch.items)} scenarios")

        # Create trajectory groups for this batch
        groups = []
        for scenario in batch.items:
            groups.append(
                art.TrajectoryGroup(
                    (
                        medical_rollout(
                            model,
                            MedicalScenarioWrapper(step=batch.step, scenario=scenario)
                        )
                        for _ in range(training_config["rollouts_per_group"])
                    )
                )
            )

        # Collect all trajectory groups
        finished_groups = await art.gather_trajectory_groups(
            groups,
            pbar_desc="Collecting trajectories",
            max_exceptions=training_config["rollouts_per_group"] * len(batch.items),
        )

        # Score trajectories using RULER
        judged_groups = []
        for group in finished_groups:
            # judged_group = await ruler_score_group(group, "gpt-4o-mini", debug=True)
            judged_group = await ruler_score_group(group,RULER_MODEL, debug=True)
            if judged_group:
                judged_groups.append(judged_group)

        # 删除旧的检查点并训练模型 / Delete old checkpoints and train the model
        await model.delete_checkpoints()  # 删除旧检查点 / Delete old checkpoints
        await model.train(
            judged_groups,  # 已评判的轨迹组 / Judged trajectory groups
            config=art.TrainConfig(learning_rate=training_config["learning_rate"]),  # 使用配置中的学习率 / Use learning rate from config
            # 降低logprob_calculation_chunk_size以节省内存，允许T4上更长的序列（最多8192 tokens） / Lower logprob_calculation_chunk_size to save memory
            _config={"logprob_calculation_chunk_size": training_config.get("logprob_calculation_chunk_size", 8)},  # 使用配置中的块大小 / Use chunk size from config
        )

        print(f"完成训练步骤 {batch.step} / Finished training step {batch.step}")  # Finished training step

        # 为演示目的，达到最大步数后停止 / Stop after max steps for demonstration purposes
        if batch.step >= training_config["max_steps"]:
            break  # 跳出训练循环 / Break training loop

    print("\n训练循环完成 / Training loop finished.")  # Training loop finished

    # ==================== 保存训练好的模型 / Save Trained Model ====================
    print("\n开始保存训练好的模型... / Starting to save trained model...")  # Starting to save trained model
    
    # 保存模型到HuggingFace Hub / Save model to HuggingFace Hub
    hf_model_repo_name = CONFIG['data'].get('hf_model_repo_name', f"update0909/{model.name}")  # 从配置获取模型仓库名称 / Get model repo name from config
    save_model_to_hf(model, hf_model_repo_name)  # 调用保存到HF的函数 / Call save to HF function
    
    # 保存模型到Google Drive / Save model to Google Drive
    drive_save_path = CONFIG['paths'].get('drive_save_path', '/content/drive/MyDrive/med_art_rl')  # 从配置获取Drive路径 / Get Drive path from config
    save_model_to_drive(model, backend, drive_save_path)  # 调用保存到Drive的函数 / Call save to Drive function

    # Use the trained model for testing
    # The model object 'model' should now contain the trained weights
    if training_scenarios:
        test_scenario = training_scenarios[1] if len(training_scenarios) > 1 else training_scenarios[0]

        print(f"测试场景ID: {test_scenario.id}")
        print(f"问题: {test_scenario.question}")
        print(f"预期答案: {test_scenario.correct_answer}")
        if test_scenario.options:
            print("选项:")
            for key, value in test_scenario.options.items():
                print(f"  {key}: {value}")
        print("-" * 50)

        # Run rollout function test the trained model
        test_medical_scenario = MedicalScenarioWrapper(
            step=0,
            scenario=test_scenario
        )
        result_trajectory = await medical_rollout(model, test_medical_scenario)

        print("智能体轨迹:")
        print("-" * 20)

        # Display conversation process
        messages = result_trajectory.messages()
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            if role == "system":
                print(f"[系统]: {content[:100]}..." if len(content) > 100 else f"[系统]: {content}")
            elif role == "user":
                print(f"[用户]: {content}")
            elif role == "assistant":
                if tool_calls:
                    print(f"[助手]: {tool_calls}")
                if content:
                    print(f"[助手]: {content}")
            elif role == "tool":
                tool_name = msg.get("name", "未知工具")
                print(f"[工具 - {tool_name}]: {content[:200]}..." if len(content) > 200 else f"[工具 - {tool_name}]: {content}")
            print()

        print("-" * 50)
        if result_trajectory.final_answer:
            print(f"智能体最终答案: {result_trajectory.final_answer.answer}")
            print(f"使用的信息源ID: {result_trajectory.final_answer.source_ids}")
            print(f"置信度: {result_trajectory.final_answer.confidence}")
        else:
            print("智能体未提供最终答案")

        print(f"\n预期答案: {test_scenario.correct_answer}")
        if test_scenario.options and test_scenario.correct_answer in test_scenario.options:
            print(f"预期答案内容: {test_scenario.options[test_scenario.correct_answer]}")

    print("\n🎉 医疗搜索智能体训练、保存和测试完成！")

# ==================== 运行脚本 ====================
# print("开始运行医疗智能体ART强化学习训练...")
# # asyncio.run(main())
# main()
# print("done.")

if __name__ == "__main__":
    import asyncio

    print("开始运行医疗智能体ART强化学习训练...")

    # 首先测试RULER
    #asyncio.run(test_ruler())

    # 然后运行主训练流程
    asyncio.run(main())
    print("医疗智能体ART强化学习训练完成！")
    print("智能体使用与训练时相同的推理路径，避免了内存问题。")





"""#  run"""