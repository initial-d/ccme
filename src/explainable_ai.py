import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import torch
from sklearn.base import BaseEstimator
import warnings
warnings.filterwarnings('ignore')

class SHAPAnalyzer:
    """SHAP分析器"""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self, background_data: np.ndarray, explainer_type: str = 'tree'):
        """创建SHAP解释器"""
        if explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif explainer_type == 'kernel':
            self.explainer = shap.KernelExplainer(self.model.predict, background_data)
        elif explainer_type == 'deep':
            self.explainer = shap.DeepExplainer(self.model, background_data)
        else:
            raise ValueError(f"不支持的解释器类型: {explainer_type}")
    
    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """计算SHAP值"""
        if self.explainer is None:
            raise ValueError("请先创建解释器")
        
        self.shap_values = self.explainer.shap_values(X)
        return self.shap_values
    
    def plot_feature_importance(self, max_display: int = 20):
        """绘制特征重要性图"""
        if self.shap_values is None:
            raise ValueError("请先计算SHAP值")
        
        shap.summary_plot(self.shap_values, feature_names=self.feature_names, 
                         max_display=max_display, show=False)
        plt.title('SHAP特征重要性分析')
        plt.tight_layout()
        plt.show()
    
    def plot_waterfall(self, instance_idx: int = 0):
        """绘制瀑布图"""
        if self.shap_values is None:
            raise ValueError("请先计算SHAP值")
        
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[instance_idx],
                base_values=self.explainer.expected_value,
                feature_names=self.feature_names
            )
        )
    
    def get_feature_contributions(self) -> pd.DataFrame:
        """获取特征贡献度统计"""
        if self.shap_values is None:
            raise ValueError("请先计算SHAP值")
        
        # 计算平均绝对SHAP值
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        
        # 创建DataFrame
        contributions = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap,
            'importance_rank': range(1, len(self.feature_names) + 1)
        })
        
        # 按重要性排序
        contributions = contributions.sort_values('mean_abs_shap', ascending=False)
        contributions['importance_rank'] = range(1, len(contributions) + 1)
        
        return contributions

class ComplexityDecomposer:
    """复杂度分解分析器"""
    
    def __init__(self, model):
        self.model = model
    
    def compute_baseline(self, input_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算基线预测"""
        with torch.no_grad():
            outputs = self.model(**input_data)
            return outputs['discovery_speed']
    
    def zero_out_component(self, input_data: Dict[str, torch.Tensor], 
                          component: str) -> Dict[str, torch.Tensor]:
        """将指定复杂度组件置零"""
        modified_data = input_data.copy()
        
        if component == 'K':  # Kolmogorov复杂度
            if 'kolmogorov' in modified_data:
                modified_data['kolmogorov'] = torch.zeros_like(modified_data['kolmogorov'])
        elif component == 'C':  # 计算复杂度
            if 'computational' in modified_data:
                modified_data['computational'] = torch.ones_like(modified_data['computational'])
        elif component == 'D':  # 逻辑深度
            if 'logical_depth' in modified_data:
                modified_data['logical_depth'] = torch.ones_like(modified_data['logical_depth'])
        
        return modified_data
    
    def analyze_interactions(self, input_data: Dict[str, torch.Tensor], 
                           effects: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """分析复杂度组件间的交互作用"""
        interactions = {}
        components = ['K', 'C', 'D']
        
        # 计算两两交互
        for i, comp1 in enumerate(components):
            for comp2 in components[i+1:]:
                # 同时置零两个组件
                modified_data = self.zero_out_component(input_data, comp1)
                modified_data = self.zero_out_component(modified_data, comp2)
                
                with torch.no_grad():
                    joint_effect = self.model(**modified_data)['discovery_speed']
                
                # 交互作用 = 联合效应 - 单独效应之和
                interaction = joint_effect - effects[comp1] - effects[comp2]
                interactions[f'{comp1}_{comp2}'] = interaction
        
        return interactions
    
    def decompose_complexity_effects(self, input_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        算法3: 复杂度效应分解分析
        """
        baseline = self.compute_baseline(input_data)
        effects = {}
        
        # 计算各组件的单独效应
        for component in ['K', 'C', 'D']:
            modified_data = self.zero_out_component(input_data, component)
            with torch.no_grad():
                prediction = self.model(**modified_data)['discovery_speed']
            effects[component] = baseline - prediction
        
        # 分析交互作用
        interactions = self.analyze_interactions(input_data, effects)
        
        # 计算贡献度百分比
        total_effect = sum(effect.abs().mean().item() for effect in effects.values())
        contributions = {}
        for component, effect in effects.items():
            contributions[component] = (effect.abs().mean().item() / total_effect * 100) if total_effect > 0 else 0
        
        return {
            'baseline': baseline,
            'individual_effects': effects,
            'interactions': interactions,
            'contributions': contributions
        }

class ModelExplainer:
    """模型解释器"""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.shap_analyzer = SHAPAnalyzer(model, feature_names)
        self.complexity_decomposer = ComplexityDecomposer(model)
    
    def explain_prediction(self, input_data: Dict[str, torch.Tensor], 
                          background_data: np.ndarray = None) -> Dict[str, Any]:
        """综合解释预测结果"""
        explanations = {}
        
        # 复杂度分解分析
        complexity_analysis = self.complexity_decomposer.decompose_complexity_effects(input_data)
        explanations['complexity_decomposition'] = complexity_analysis
        
        # 如果提供了背景数据，进行SHAP分析
        if background_data is not None:
            try:
                # 将输入数据转换为numpy格式（简化处理）
                X = self._prepare_data_for_shap(input_data)
                
                self.shap_analyzer.create_explainer(background_data, 'kernel')
                shap_values = self.shap_analyzer.compute_shap_values(X)
                
                explanations['shap_analysis'] = {
                    'shap_values': shap_values,
                    'feature_contributions': self.shap_analyzer.get_feature_contributions()
                }
            except Exception as e:
                print(f"SHAP分析失败: {e}")
                explanations['shap_analysis'] = None
        
        return explanations
    
    def _prepare_data_for_shap(self, input_data: Dict[str, torch.Tensor]) -> np.ndarray:
        """为SHAP分析准备数据"""
        # 简化处理：只使用数值特征
        features = []
        
        if 'kolmogorov' in input_data:
            features.append(input_data['kolmogorov'].cpu().numpy())
        if 'computational' in input_data:
            features.append(input_data['computational'].cpu().numpy())
        if 'logical_depth' in input_data:
            features.append(input_data['logical_depth'].cpu().numpy())
        
        if features:
            return np.column_stack(features)
        else:
            return np.random.randn(input_data[list(input_data.keys())[0]].size(0), 3)
    
    def generate_explanation_report(self, input_data: Dict[str, torch.Tensor],
                                  background_data: np.ndarray = None) -> str:
        """生成解释报告"""
        explanations = self.explain_prediction(input_data, background_data)
        
        report = "=== 模型预测解释报告 ===\n\n"
        
        # 复杂度分解部分
        if 'complexity_decomposition' in explanations:
            comp_analysis = explanations['complexity_decomposition']
            report += "1. 复杂度组件分析:\n"
            
            for component, contribution in comp_analysis['contributions'].items():
                component_name = {
                    'K': 'Kolmogorov复杂度',
                    'C': '计算复杂度', 
                    'D': '逻辑深度'
                }.get(component, component)
                report += f"   - {component_name}: {contribution:.1f}%\n"
            
            report += "\n"
        
        # SHAP分析部分
        if explanations.get('shap_analysis'):
            shap_analysis = explanations['shap_analysis']
            if shap_analysis['feature_contributions'] is not None:
                report += "2. 特征重要性分析 (SHAP):\n"
                top_features = shap_analysis['feature_contributions'].head(5)
                for _, row in top_features.iterrows():
                    report += f"   - {row['feature']}: {row['mean_abs_shap']:.4f}\n"
                report += "\n"
        
        # 预测结果
        with torch.no_grad():
            outputs = self.model(**input_data)
            speed = outputs['discovery_speed'].mean().item()
            efficiency = outputs['discovery_efficiency'].mean().item()
            quality = outputs['adjustment_quality'].mean().item()
        
        report += "3. 预测结果:\n"
        report += f"   - 价格发现速度: {speed:.2f} 小时\n"
        report += f"   - 发现效率: {efficiency:.1%}\n"
        report += f"   - 调整质量: {quality:.1%}\n"
        
        return report

class VisualizationTools:
    """可视化工具"""
    
    @staticmethod
    def plot_complexity_effects(effects: Dict[str, torch.Tensor], 
                               contributions: Dict[str, float]):
        """绘制复杂度效应图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 效应大小图
        components = list(effects.keys())
        effect_sizes = [effect.abs().mean().item() for effect in effects.values()]
        
        bars1 = ax1.bar(components, effect_sizes, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('复杂度组件效应大小')
        ax1.set_ylabel('平均绝对效应')
        ax1.set_xlabel('复杂度组件')
        
        # 添加数值标签
        for bar, size in zip(bars1, effect_sizes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{size:.3f}', ha='center', va='bottom')
        
        # 贡献度饼图
        contribution_values = list(contributions.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        wedges, texts, autotexts = ax2.pie(contribution_values, labels=components, 
                                          autopct='%1.1f%%', colors=colors)
        ax2.set_title('复杂度组件贡献度')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_prediction_decomposition(baseline: torch.Tensor, 
                                    effects: Dict[str, torch.Tensor]):
        """绘制预测分解图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 计算累积效应
        baseline_val = baseline.mean().item()
        cumulative = baseline_val
        
        components = ['基线'] + list(effects.keys())
        values = [baseline_val]
        
        for effect in effects.values():
            effect_val = effect.mean().item()
            values.append(effect_val)
            cumulative += effect_val
        
        # 创建瀑布图
        x_pos = range(len(components))
        colors = ['#2E86AB'] + ['#A23B72' if v < 0 else '#F18F01' for v in values[1:]]
        
        bars = ax.bar(x_pos, values, color=colors, alpha=0.7)
        
        # 添加连接线
        running_total = baseline_val
        for i, effect_val in enumerate(values[1:], 1):
            if effect_val >= 0:
                ax.plot([i-0.4, i-0.4, i+0.4], 
                       [running_total, running_total + effect_val, running_total + effect_val],
                       'k--', alpha=0.5)
            else:
                ax.plot([i-0.4, i-0.4, i+0.4], 
                       [running_total + effect_val, running_total, running_total],
                       'k--', alpha=0.5)
            running_total += effect_val
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, 
                   height + (0.01 if height >= 0 else -0.03),
                   f'{val:.3f}', ha='center', 
                   va='bottom' if height >= 0 else 'top')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(components, rotation=45)
        ax.set_ylabel('预测值贡献')
        ax.set_title('价格发现速度预测分解')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 测试代码
if __name__ == "__main__":
    # 创建测试模型和数据
    from complexity_aware_model import ComplexityAwareModel
    
    model = ComplexityAwareModel(vocab_size=1000, d_model=128)
    model.eval()
    
    # 创建测试输入
    test_input = {
        'text_ids': torch.randint(0, 1000, (4, 20)),
        'numerical_data': torch.randn(4, 5),
        'structured_data': torch.randn(4, 10),
        'kolmogorov': torch.rand(4),
        'computational': torch.randint(1, 100, (4,)).float(),
        'logical_depth': torch.randint(1, 10, (4,)).float()
    }
    
    # 创建解释器
    feature_names = ['kolmogorov', 'computational', 'logical_depth']
    explainer = ModelExplainer(model, feature_names)
    
    # 生成解释
    explanations = explainer.explain_prediction(test_input)
    
    # 打印报告
    report = explainer.generate_explanation_report(test_input)
    print(report)
    
    # 可视化
    if 'complexity_decomposition' in explanations:
        comp_analysis = explanations['complexity_decomposition']
        VisualizationTools.plot_complexity_effects(
            comp_analysis['individual_effects'],
            comp_analysis['contributions']
        )
        VisualizationTools.plot_prediction_decomposition(
            comp_analysis['baseline'],
            comp_analysis['individual_effects']
        )
