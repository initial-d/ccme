# 信息复杂度与市场效率研究项目

## 项目简介

本项目实现了论文《Computational Complexity and Market Efficiency: A Machine Learning Approach to Information Processing in Financial Markets》中提出的核心算法。该研究探索了信息复杂度对金融市场价格发现效率的影响，并开发了复杂度感知的机器学习模型。

## 🚀 主要特性

- **信息复杂度测量**：Kolmogorov复杂度、计算复杂度和逻辑深度的多维度评估
- **复杂度感知神经网络**：融入计算复杂度约束的深度学习模型
- **可解释AI框架**：SHAP分析和复杂度分解
- **算法交易检测**：基于订单流特征的算法交易识别
- **数据处理管道**：完整的数据清洗和样本构建流程
- **统计分析工具**：回归分析、稳健性检验和可视化

## 📁 项目结构

```
ccme/src/
├── README.md
├── requirements.txt
├── kolmogorov_calculator.py          # Kolmogorov复杂度计算
├── computational_analyzer.py         # 计算复杂度分析
├── complexity_aware_model.py         # 复杂度感知神经网络
├── explainable_ai.py                 # 可解释AI框架
├── algorithmic_trading_detector.py   # 算法交易检测
├── data_processing.py                # 数据处理和质量控制
└── statistical_analysis.py           # 统计分析和可视化
```

## 🛠️ 安装指南

### 环境要求

- Python 3.8+
- PyTorch 1.9+

### 安装步骤

1. **克隆或下载项目文件**

2. **安装依赖**
```bash
pip install -r requirements.txt
```

### 依赖包列表 (requirements.txt)

```txt
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
shap>=0.40.0
networkx>=2.6.0
zstandard>=0.15.0
brotli>=1.0.9
```

## 🚀 快速开始

### 1. 信息复杂度计算

```python
from kolmogorov_calculator import KolmogorovComplexityCalculator
from computational_analyzer import ComputationalComplexityAnalyzer, LogicalDepthEstimator

# 计算Kolmogorov复杂度
calculator = KolmogorovComplexityCalculator()
result = calculator.theoretical_kolmogorov("复杂的金融信息文本...")
print(f"Kolmogorov复杂度: {result.kolmogorov_approx:.4f}")

# 计算计算复杂度
comp_analyzer = ComputationalComplexityAnalyzer()
comp_complexity = comp_analyzer.measure_computational_complexity("财务比率分析数据")
print(f"计算复杂度: {comp_complexity}")

# 估计逻辑深度
depth_estimator = LogicalDepthEstimator()
logical_depth = depth_estimator.estimate_logical_depth("多因子模型分析")
print(f"逻辑深度: {logical_depth}")
```

### 2. 复杂度感知神经网络

```python
import torch
from complexity_aware_model import ComplexityAwareModel

# 创建模型
model = ComplexityAwareModel(vocab_size=10000, d_model=256)

# 准备数据
batch_size = 8
data = {
    'text_ids': torch.randint(0, 1000, (batch_size, 20)),
    'numerical_data': torch.randn(batch_size, 5),
    'kolmogorov': torch.rand(batch_size),
    'computational': torch.randint(1, 100, (batch_size,)).float(),
    'logical_depth': torch.randint(1, 10, (batch_size,)).float()
}

# 模型预测
outputs = model(**data)
print(f"预测的价格发现速度: {outputs['discovery_speed'].mean():.2f} 小时")
print(f"发现效率: {outputs['discovery_efficiency'].mean():.2%}")
```

### 3. 模型解释

```python
from explainable_ai import ModelExplainer

# 创建解释器
explainer = ModelExplainer(model, ['kolmogorov', 'computational', 'logical_depth'])

# 生成解释
explanations = explainer.explain_prediction(data)
report = explainer.generate_explanation_report(data)
print(report)

# 可视化复杂度效应
from explainable_ai import VisualizationTools
VisualizationTools.plot_complexity_effects(
    explanations['complexity_decomposition']['individual_effects'],
    explanations['complexity_decomposition']['contributions']
)
```

### 4. 算法交易检测

```python
from algorithmic_trading_detector import AlgorithmicTradingDetector, SyntheticDataGenerator

# 生成测试数据
data_generator = SyntheticDataGenerator()
training_data = {}
labels = {}

# 生成人工和算法交易数据
for i in range(50):
    training_data[f"human_{i}"] = data_generator.generate_human_trading_session()
    labels[f"human_{i}"] = 0
    
    training_data[f"algo_{i}"] = data_generator.generate_algorithmic_trading_session()
    labels[f"algo_{i}"] = 1

# 训练检测器
detector = AlgorithmicTradingDetector()
results = detector.train(training_data, labels)
print(f"交叉验证AUC: {results['cv_auc_mean']:.4f}")

# 预测新数据
test_data = {
    'test_session': data_generator.generate_human_trading_session()
}
predictions = detector.predict(test_data)
print(f"算法交易概率: {predictions['test_session']:.4f}")
```

### 5. 数据处理和分析

```python
from data_processing import DataQualityController, InformationEventClassifier
from statistical_analysis import StatisticalAnalyzer, VisualizationSuite

# 数据质量控制
quality_controller = DataQualityController()
clean_data = quality_controller.data_quality_control(raw_data)

# 事件分类
classifier = InformationEventClassifier()
classified_events = classifier.classify_information_events(events_list)

# 统计分析
analyzer = StatisticalAnalyzer()
regression_results = analyzer.regression_analysis(
    data, 'discovery_speed', 
    ['complexity_score', 'algo_trading_intensity']
)

# 可视化
visualizer = VisualizationSuite()
visualizer.plot_main_results(regression_results)
```

## 📊 核心算法说明

### 1. 信息复杂度测量

- **Kolmogorov复杂度**：使用多种压缩算法（gzip, bz2, LZMA等）的加权平均
- **计算复杂度**：基于信息类型自动分类，从O(1)到O(2^n)
- **逻辑深度**：通过依赖图分析计算最长处理路径

### 2. 复杂度感知注意力机制

```python
# 核心公式
attention_weights = softmax(QK^T / sqrt(d_k) - λC)
```

其中C是复杂度惩罚矩阵，λ是惩罚强度参数。

### 3. 价格发现指标

- **发现速度**：90%价格调整完成的时间
- **发现效率**：第一天调整占总调整的比例
- **调整质量**：价格调整过程的平滑度

## 🧪 实验结果

### 主要发现

1. **复杂度效应**：信息复杂度每增加1个标准差，价格发现延迟增加23.4%
2. **算法交易缓解**：算法交易将高复杂度信息的处理延迟减少67%
3. **模型性能**：复杂度感知模型的R²达到0.442，显著优于基准模型

### 模型性能对比

| 模型 | R² | MSE | MAE |
|------|----|----|-----|
| 线性回归 | 0.123 | 0.245 | 0.387 |
| 随机森林 | 0.291 | 0.198 | 0.312 |
| XGBoost | 0.324 | 0.189 | 0.298 |
| **复杂度感知模型** | **0.442** | **0.156** | **0.267** |

## 📝 使用注意事项

1. **数据格式**：确保输入数据格式正确，特别是时间戳和数值字段
2. **内存使用**：大规模数据处理时注意内存管理
3. **GPU支持**：神经网络模型支持GPU加速，建议使用CUDA
4. **参数调优**：根据具体数据特征调整模型超参数

## 🔧 配置参数

### 模型配置

```python
model_config = {
    'vocab_size': 10000,     # 词汇表大小
    'd_model': 256,          # 模型维度
    'n_heads': 8,            # 注意力头数
    'num_layers': 6,         # 层数
    'dropout': 0.1           # Dropout率
}
```

### 复杂度计算配置

```python
complexity_config = {
    'missing_threshold': 0.1,        # 缺失值阈值
    'outlier_method': 'modified_tukey',  # 异常值检测方法
    'compression_algorithms': ['gzip', 'bz2', 'lzma']  # 压缩算法
}
```

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

## 📄 许可证

MIT License

## 📞 联系方式

- **作者**：Yimin Du, Guolin Tang
- **邮箱**：sa613403@mail.ustc.edu.cn, guolin_tang@163.com

## 📚 引用

如果使用本项目，请引用：

```bibtex
@article{du2024computational,
  title={Computational Complexity and Market Efficiency: A Machine Learning Approach to Information Processing in Financial Markets},
  author={Du, Yimin and Tang, Guolin},
  journal={IEEE Conference on Computational Intelligence for Financial Engineering and Economics},
  year={2024}
}
```

---

**注意**：本项目仅用于学术研究目的，实际应用前请进行充分的风险评估。
