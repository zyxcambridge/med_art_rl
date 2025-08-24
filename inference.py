# -*- coding: utf-8 -*-
"""
医疗智能体推理测试模块
Medical AI Agent Inference Testing Module

功能说明：
- 加载训练好的医疗智能体模型
- 执行多轮推理测试
- 保存结果到CSV文件
- 支持批量测试和性能评估

作者：Medical AI Team
日期：2025-08-24
"""

import os  # 操作系统接口模块 / Operating system interface module
import sys  # 系统特定参数和函数 / System-specific parameters and functions
import csv  # CSV文件读写 / CSV file read/write
import json  # JSON数据处理 / JSON data processing
import yaml  # YAML文件处理 / YAML file processing
import asyncio  # 异步IO支持 / Async IO support
import pandas as pd  # 数据分析库 / Data analysis library
from datetime import datetime  # 日期时间处理 / Date and time processing
from typing import List, Dict, Optional, Any  # 类型提示 / Type hints
from pathlib import Path  # 路径操作 / Path operations
import logging  # 日志记录 / Logging

# 导入ART框架相关模块 / Import ART framework modules
import art  # ART强化学习框架 / ART reinforcement learning framework
from art.local import LocalBackend  # ART本地后端 / ART local backend

# 导入原始训练模块的类和函数 / Import classes and functions from original training module
from med_art_rl import (
    MedicalScenario,  # 医疗场景类 / Medical scenario class
    MedicalScenarioWrapper,  # 医疗场景包装器 / Medical scenario wrapper
    medical_rollout,  # 医疗rollout函数 / Medical rollout function
    load_medical_scenarios,  # 加载医疗场景函数 / Load medical scenarios function
    judge_medical_correctness,  # 医疗正确性判断函数 / Medical correctness judge function
    get_db_connection,  # 获取数据库连接函数 / Get database connection function
    create_medical_database  # 创建医疗数据库函数 / Create medical database function
)

# 配置日志记录 / Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log', encoding='utf-8'),  # 文件日志处理器 / File log handler
        logging.StreamHandler()  # 控制台日志处理器 / Console log handler
    ]
)
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器 / Get logger for current module

class MedicalInferenceEngine:
    """
    医疗智能体推理引擎
    Medical AI Agent Inference Engine
    
    功能：
    - 加载训练好的模型
    - 执行推理测试
    - 记录和分析结果
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化推理引擎 / Initialize inference engine
        
        Args:
            config_path: 配置文件路径 / Configuration file path
        """
        logger.info(f"初始化推理引擎，配置文件：{config_path}")  # 记录初始化信息 / Log initialization info
        
        # 加载配置文件 / Load configuration file
        self.config = self._load_config(config_path)
        
        # 初始化模型相关变量 / Initialize model-related variables
        self.model = None  # 训练好的模型 / Trained model
        self.backend = None  # ART后端 / ART backend
        
        # 创建结果保存目录 / Create results save directory
        self.results_dir = Path(self.config['paths']['inference_results_path'])
        self.results_dir.mkdir(exist_ok=True)  # 如果目录不存在则创建 / Create directory if not exists
        
        # 初始化结果记录列表 / Initialize result recording list
        self.inference_results = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
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
            logger.info("配置文件加载成功")  # 记录成功加载 / Log successful loading
            return config
        except FileNotFoundError:  # 文件未找到异常 / File not found exception
            logger.error(f"配置文件未找到：{config_path}")
            raise
        except yaml.YAMLError as e:  # YAML解析错误 / YAML parsing error
            logger.error(f"配置文件解析错误：{e}")
            raise
            
    async def initialize_model(self, model_name: Optional[str] = None):
        """
        初始化并加载训练好的模型 / Initialize and load trained model
        
        Args:
            model_name: 模型名称（可选） / Model name (optional)
        """
        logger.info("开始初始化模型...")  # 记录模型初始化开始 / Log model initialization start
        
        # 使用配置文件中的模型名称或传入的模型名称 / Use model name from config or passed parameter
        model_name = model_name or self.config['model']['name']
        
        # 初始化ART后端 / Initialize ART backend
        self.backend = LocalBackend(
            in_process=True,  # 在进程内运行 / Run in-process
            path=self.config['paths']['art_backend_path']  # ART后端路径 / ART backend path
        )
        
        # 创建可训练模型实例 / Create trainable model instance
        self.model = art.TrainableModel(
            name=model_name,  # 模型名称 / Model name
            project=self.config['model']['project'],  # 项目名称 / Project name
            base_model=self.config['model']['base_model']  # 基础模型 / Base model
        )
        
        # 注册模型到后端 / Register model to backend
        await self.model.register(self.backend)
        
        logger.info(f"模型 {model_name} 初始化完成")  # 记录模型初始化完成 / Log model initialization completion
        
    def load_test_scenarios(self, limit: Optional[int] = None) -> List[MedicalScenario]:
        """
        加载测试场景 / Load test scenarios
        
        Args:
            limit: 限制场景数量 / Limit number of scenarios
            
        Returns:
            测试场景列表 / List of test scenarios
        """
        logger.info(f"加载测试场景，限制数量：{limit}")  # 记录加载测试场景 / Log loading test scenarios
        
        # 使用配置文件中的参数加载场景 / Load scenarios using config parameters
        scenarios = load_medical_scenarios(
            limit=limit or self.config['inference']['batch_size'],  # 批次大小 / Batch size
            shuffle=self.config['data']['shuffle_data'],  # 是否打乱 / Whether to shuffle
            seed=self.config['data']['random_seed']  # 随机种子 / Random seed
        )
        
        logger.info(f"成功加载 {len(scenarios)} 个测试场景")  # 记录成功加载的场景数量 / Log number of scenarios loaded
        return scenarios
        
    async def run_single_inference(self, scenario: MedicalScenario, test_round: int) -> Dict[str, Any]:
        """
        执行单次推理测试 / Run single inference test
        
        Args:
            scenario: 医疗场景 / Medical scenario
            test_round: 测试轮次 / Test round number
            
        Returns:
            推理结果字典 / Inference result dictionary
        """
        logger.info(f"执行推理测试 - 场景ID: {scenario.id}, 轮次: {test_round}")  # 记录推理测试开始 / Log inference test start
        
        start_time = datetime.now()  # 记录开始时间 / Record start time
        
        try:
            # 创建场景包装器 / Create scenario wrapper
            scenario_wrapper = MedicalScenarioWrapper(
                step=test_round,  # 测试步骤 / Test step
                scenario=scenario  # 医疗场景 / Medical scenario
            )
            
            # 执行医疗rollout / Execute medical rollout
            trajectory = await medical_rollout(self.model, scenario_wrapper)
            
            # 计算推理时间 / Calculate inference time
            inference_time = (datetime.now() - start_time).total_seconds()
            
            # 获取最终答案 / Get final answer
            final_answer = trajectory.final_answer.answer if trajectory.final_answer else "无答案"  # No answer
            source_ids = trajectory.final_answer.source_ids if trajectory.final_answer else []  # 信息源ID列表 / Source IDs list
            confidence = trajectory.final_answer.confidence if trajectory.final_answer else 0.0  # 置信度 / Confidence
            
            # 判断答案正确性 / Judge answer correctness
            is_correct = False
            correctness_reasoning = ""
            
            if trajectory.final_answer:  # 如果有最终答案 / If there's a final answer
                try:
                    # 使用RULER评判正确性 / Use RULER to judge correctness
                    correctness_result = await judge_medical_correctness(scenario, final_answer)
                    is_correct = correctness_result.accept  # 是否接受答案 / Whether to accept answer
                    correctness_reasoning = correctness_result.reasoning  # 推理过程 / Reasoning process
                except Exception as e:  # 评判过程中的异常 / Exception during judging
                    logger.error(f"正确性评判失败：{e}")  # 记录评判失败 / Log judging failure
                    correctness_reasoning = f"评判失败：{str(e)}"  # 评判失败原因 / Reason for judging failure
            
            # 构建结果字典 / Build result dictionary
            result = {
                'test_round': test_round,  # 测试轮次 / Test round
                'scenario_id': scenario.id,  # 场景ID / Scenario ID
                'question': scenario.question,  # 问题 / Question
                'correct_answer': scenario.correct_answer,  # 正确答案 / Correct answer
                'model_answer': final_answer,  # 模型答案 / Model answer
                'is_correct': is_correct,  # 是否正确 / Whether correct
                'confidence': confidence,  # 置信度 / Confidence
                'source_ids': json.dumps(source_ids),  # 信息源ID（JSON格式） / Source IDs (JSON format)
                'inference_time': inference_time,  # 推理时间 / Inference time
                'correctness_reasoning': correctness_reasoning,  # 正确性推理 / Correctness reasoning
                'trajectory_length': len(trajectory.messages_and_choices),  # 轨迹长度 / Trajectory length
                'timestamp': start_time.isoformat()  # 时间戳 / Timestamp
            }
            
            logger.info(f"推理完成 - 场景ID: {scenario.id}, 正确性: {is_correct}")  # 记录推理完成 / Log inference completion
            return result
            
        except Exception as e:  # 推理过程中的异常 / Exception during inference
            logger.error(f"推理失败 - 场景ID: {scenario.id}, 错误: {e}")  # 记录推理失败 / Log inference failure
            
            # 返回错误结果 / Return error result
            return {
                'test_round': test_round,
                'scenario_id': scenario.id,
                'question': scenario.question,
                'correct_answer': scenario.correct_answer,
                'model_answer': f"推理失败：{str(e)}",  # Inference failed
                'is_correct': False,
                'confidence': 0.0,
                'source_ids': "[]",
                'inference_time': (datetime.now() - start_time).total_seconds(),
                'correctness_reasoning': f"推理过程中发生错误：{str(e)}",  # Error during inference
                'trajectory_length': 0,
                'timestamp': start_time.isoformat()
            }
    
    async def run_batch_inference(self, scenarios: List[MedicalScenario], test_rounds: int = 3) -> List[Dict[str, Any]]:
        """
        执行批量推理测试 / Run batch inference tests
        
        Args:
            scenarios: 测试场景列表 / List of test scenarios
            test_rounds: 测试轮次数 / Number of test rounds
            
        Returns:
            所有推理结果列表 / List of all inference results
        """
        logger.info(f"开始批量推理测试 - 场景数: {len(scenarios)}, 轮次数: {test_rounds}")  # 记录批量测试开始 / Log batch test start
        
        all_results = []  # 所有结果列表 / All results list
        
        # 对每个场景执行多轮测试 / Execute multiple rounds for each scenario
        for scenario in scenarios:
            for round_num in range(1, test_rounds + 1):  # 从1开始的轮次编号 / Round numbers starting from 1
                logger.info(f"测试场景 {scenario.id} - 第 {round_num} 轮")  # 记录当前测试 / Log current test
                
                # 执行单次推理 / Execute single inference
                result = await self.run_single_inference(scenario, round_num)
                all_results.append(result)  # 添加到结果列表 / Add to results list
                
                # 保存详细日志（如果配置中启用） / Save detailed logs (if enabled in config)
                if self.config['inference']['save_detailed_logs']:
                    self._log_detailed_result(result)  # 记录详细结果 / Log detailed result
        
        logger.info(f"批量推理测试完成 - 总结果数: {len(all_results)}")  # 记录批量测试完成 / Log batch test completion
        return all_results
    
    def _log_detailed_result(self, result: Dict[str, Any]):
        """
        记录详细的推理结果 / Log detailed inference result
        
        Args:
            result: 推理结果字典 / Inference result dictionary
        """
        logger.info(f"""
详细推理结果 / Detailed Inference Result:
- 场景ID / Scenario ID: {result['scenario_id']}
- 测试轮次 / Test Round: {result['test_round']}
- 问题 / Question: {result['question'][:100]}...
- 正确答案 / Correct Answer: {result['correct_answer']}
- 模型答案 / Model Answer: {result['model_answer'][:100]}...
- 是否正确 / Is Correct: {result['is_correct']}
- 置信度 / Confidence: {result['confidence']:.3f}
- 推理时间 / Inference Time: {result['inference_time']:.2f}s
- 轨迹长度 / Trajectory Length: {result['trajectory_length']}
""")
    
    def save_results_to_csv(self, results: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """
        保存结果到CSV文件 / Save results to CSV file
        
        Args:
            results: 推理结果列表 / List of inference results
            filename: 文件名（可选） / Filename (optional)
            
        Returns:
            保存的文件路径 / Saved file path
        """
        # 生成文件名 / Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 时间戳 / Timestamp
            filename = f"inference_results_{timestamp}.csv"
        
        # 完整文件路径 / Full file path
        filepath = self.results_dir / filename
        
        logger.info(f"保存推理结果到CSV文件：{filepath}")  # 记录保存操作 / Log save operation
        
        # 使用pandas保存到CSV / Use pandas to save to CSV
        df = pd.DataFrame(results)  # 创建DataFrame / Create DataFrame
        df.to_csv(filepath, index=False, encoding='utf-8-sig')  # 保存到CSV文件 / Save to CSV file
        
        logger.info(f"成功保存 {len(results)} 条结果到 {filepath}")  # 记录保存成功 / Log save success
        return str(filepath)
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析推理结果 / Analyze inference results
        
        Args:
            results: 推理结果列表 / List of inference results
            
        Returns:
            分析统计结果 / Analysis statistics
        """
        logger.info("开始分析推理结果...")  # 记录分析开始 / Log analysis start
        
        if not results:  # 如果没有结果 / If no results
            return {}
        
        # 转换为DataFrame进行分析 / Convert to DataFrame for analysis
        df = pd.DataFrame(results)
        
        # 计算基本统计数据 / Calculate basic statistics
        total_tests = len(df)  # 总测试数 / Total tests
        correct_count = df['is_correct'].sum()  # 正确数量 / Correct count
        accuracy = correct_count / total_tests if total_tests > 0 else 0  # 准确率 / Accuracy
        
        avg_confidence = df['confidence'].mean()  # 平均置信度 / Average confidence
        avg_inference_time = df['inference_time'].mean()  # 平均推理时间 / Average inference time
        avg_trajectory_length = df['trajectory_length'].mean()  # 平均轨迹长度 / Average trajectory length
        
        # 按测试轮次分组分析 / Group analysis by test round
        round_stats = df.groupby('test_round').agg({
            'is_correct': ['count', 'sum', 'mean'],  # 正确性统计 / Correctness statistics
            'confidence': 'mean',  # 平均置信度 / Average confidence
            'inference_time': 'mean'  # 平均推理时间 / Average inference time
        }).round(3)
        
        # 构建分析结果 / Build analysis results
        analysis = {
            'total_tests': total_tests,  # 总测试数 / Total tests
            'correct_answers': int(correct_count),  # 正确答案数 / Correct answers
            'accuracy': round(accuracy, 4),  # 准确率 / Accuracy
            'average_confidence': round(avg_confidence, 4),  # 平均置信度 / Average confidence
            'average_inference_time': round(avg_inference_time, 4),  # 平均推理时间 / Average inference time
            'average_trajectory_length': round(avg_trajectory_length, 2),  # 平均轨迹长度 / Average trajectory length
            'round_statistics': round_stats.to_dict(),  # 轮次统计 / Round statistics
            'confidence_threshold_analysis': self._analyze_confidence_threshold(df)  # 置信度阈值分析 / Confidence threshold analysis
        }
        
        logger.info(f"结果分析完成 - 总准确率: {accuracy:.4f}")  # 记录分析完成 / Log analysis completion
        return analysis
    
    def _analyze_confidence_threshold(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        分析置信度阈值对准确率的影响 / Analyze confidence threshold impact on accuracy
        
        Args:
            df: 结果DataFrame / Results DataFrame
            
        Returns:
            置信度阈值分析结果 / Confidence threshold analysis results
        """
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]  # 置信度阈值列表 / Confidence thresholds list
        threshold_analysis = {}  # 阈值分析结果 / Threshold analysis results
        
        for threshold in thresholds:  # 对每个阈值进行分析 / Analyze each threshold
            # 筛选高置信度的预测 / Filter high-confidence predictions
            high_conf_df = df[df['confidence'] >= threshold]
            
            if len(high_conf_df) > 0:  # 如果有高置信度的预测 / If there are high-confidence predictions
                accuracy = high_conf_df['is_correct'].mean()  # 计算准确率 / Calculate accuracy
                coverage = len(high_conf_df) / len(df)  # 计算覆盖率 / Calculate coverage
            else:  # 如果没有高置信度的预测 / If no high-confidence predictions
                accuracy = 0.0
                coverage = 0.0
            
            threshold_analysis[f'threshold_{threshold}'] = {
                'accuracy': round(accuracy, 4),  # 准确率 / Accuracy
                'coverage': round(coverage, 4)  # 覆盖率 / Coverage
            }
        
        return threshold_analysis
    
    def save_analysis_report(self, analysis: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        保存分析报告 / Save analysis report
        
        Args:
            analysis: 分析结果 / Analysis results
            filename: 文件名（可选） / Filename (optional)
            
        Returns:
            保存的文件路径 / Saved file path
        """
        # 生成文件名 / Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 时间戳 / Timestamp
            filename = f"analysis_report_{timestamp}.json"
        
        # 完整文件路径 / Full file path
        filepath = self.results_dir / filename
        
        logger.info(f"保存分析报告到：{filepath}")  # 记录保存操作 / Log save operation
        
        # 保存为JSON文件 / Save as JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)  # 保存分析结果 / Save analysis results
        
        logger.info(f"分析报告保存成功：{filepath}")  # 记录保存成功 / Log save success
        return str(filepath)

async def main():
    """
    主函数：执行完整的推理测试流程 / Main function: Execute complete inference test workflow
    """
    print("=" * 60)
    print("医疗智能体推理测试系统")  # Medical AI Agent Inference Test System
    print("Medical AI Agent Inference Test System")
    print("=" * 60)
    
    try:
        # 1. 初始化推理引擎 / Initialize inference engine
        engine = MedicalInferenceEngine()
        
        # 2. 初始化模型 / Initialize model
        await engine.initialize_model()
        
        # 3. 加载测试场景 / Load test scenarios
        test_scenarios = engine.load_test_scenarios(
            limit=engine.config['inference']['batch_size']  # 使用配置中的批次大小 / Use batch size from config
        )
        
        if not test_scenarios:  # 如果没有测试场景 / If no test scenarios
            logger.error("未找到测试场景")  # No test scenarios found
            return
        
        # 4. 执行批量推理测试 / Execute batch inference tests
        results = await engine.run_batch_inference(
            test_scenarios,
            test_rounds=engine.config['inference']['test_rounds']  # 使用配置中的测试轮次 / Use test rounds from config
        )
        
        # 5. 保存结果到CSV / Save results to CSV
        csv_filepath = engine.save_results_to_csv(results)
        print(f"推理结果已保存到：{csv_filepath}")  # Inference results saved to
        
        # 6. 分析结果 / Analyze results
        analysis = engine.analyze_results(results)
        
        # 7. 保存分析报告 / Save analysis report
        report_filepath = engine.save_analysis_report(analysis)
        print(f"分析报告已保存到：{report_filepath}")  # Analysis report saved to
        
        # 8. 打印摘要 / Print summary
        print("\n" + "=" * 50)
        print("推理测试摘要 / Inference Test Summary")
        print("=" * 50)
        print(f"总测试数 / Total Tests: {analysis['total_tests']}")
        print(f"正确答案数 / Correct Answers: {analysis['correct_answers']}")
        print(f"准确率 / Accuracy: {analysis['accuracy']:.4f} ({analysis['accuracy']*100:.2f}%)")
        print(f"平均置信度 / Average Confidence: {analysis['average_confidence']:.4f}")
        print(f"平均推理时间 / Average Inference Time: {analysis['average_inference_time']:.2f}s")
        print(f"平均轨迹长度 / Average Trajectory Length: {analysis['average_trajectory_length']:.1f}")
        print("=" * 50)
        
        logger.info("推理测试流程完成")  # Inference test workflow completed
        
    except Exception as e:  # 捕获所有异常 / Catch all exceptions
        logger.error(f"推理测试过程中发生错误：{e}")  # Error during inference testing
        raise

if __name__ == "__main__":
    """
    脚本入口点 / Script entry point
    运行方式：python inference.py
    Run with: python inference.py
    """
    print("启动医疗智能体推理测试...")  # Starting medical AI agent inference testing
    asyncio.run(main())  # 运行异步主函数 / Run async main function
    print("推理测试完成！")  # Inference testing completed!