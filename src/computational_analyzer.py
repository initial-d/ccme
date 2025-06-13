import re
import math
from enum import Enum
from typing import Union, Dict, Any
import ast
import networkx as nx

class InformationType(Enum):
    """信息类型枚举"""
    PRICE_UPDATE = "price_update"
    FINANCIAL_RATIO = "financial_ratio"
    CORRELATION_ANALYSIS = "correlation_analysis"
    FACTOR_INTERACTION = "factor_interaction"
    TEXT_ANALYSIS = "text_analysis"
    EARNINGS_REPORT = "earnings_report"
    NEWS_ARTICLE = "news_article"

class ComputationalComplexityAnalyzer:
    """计算复杂度分析器"""
    
    def __init__(self):
        self.complexity_patterns = {
            InformationType.PRICE_UPDATE: r'price|quote|bid|ask',
            InformationType.FINANCIAL_RATIO: r'ratio|pe|pb|roe|roa',
            InformationType.CORRELATION_ANALYSIS: r'correlation|covariance|beta',
            InformationType.FACTOR_INTERACTION: r'factor|interaction|model',
            InformationType.EARNINGS_REPORT: r'earnings|revenue|profit|loss',
            InformationType.NEWS_ARTICLE: r'news|article|report|announcement'
        }
    
    def classify_information(self, information: Any) -> InformationType:
        """
        分类信息类型
        """
        if isinstance(information, str):
            text = information.lower()
            
            # 使用正则表达式匹配模式
            for info_type, pattern in self.complexity_patterns.items():
                if re.search(pattern, text):
                    return info_type
            
            return InformationType.TEXT_ANALYSIS
        
        elif isinstance(information, dict):
            keys = [str(k).lower() for k in information.keys()]
            key_text = ' '.join(keys)
            
            if any(word in key_text for word in ['price', 'quote', 'bid', 'ask']):
                return InformationType.PRICE_UPDATE
            elif any(word in key_text for word in ['ratio', 'pe', 'pb']):
                return InformationType.FINANCIAL_RATIO
            elif any(word in key_text for word in ['correlation', 'beta']):
                return InformationType.CORRELATION_ANALYSIS
            elif any(word in key_text for word in ['factor', 'interaction']):
                return InformationType.FACTOR_INTERACTION
            else:
                return InformationType.TEXT_ANALYSIS
        
        else:
            return InformationType.TEXT_ANALYSIS
    
    def count_variables(self, information: Any) -> int:
        """计算变量数量"""
        if isinstance(information, dict):
            return len(information)
        elif isinstance(information, (list, tuple)):
            return len(information)
        elif isinstance(information, str):
            # 简单的变量计数（基于数字和关键词）
            numbers = re.findall(r'\d+\.?\d*', information)
            return len(numbers)
        else:
            return 1
    
    def count_factors(self, information: Any) -> int:
        """计算因子数量"""
        if isinstance(information, dict):
            factor_keys = [k for k in information.keys() 
                          if 'factor' in str(k).lower() or 'beta' in str(k).lower()]
            return max(len(factor_keys), 1)
        elif isinstance(information, str):
            factor_mentions = len(re.findall(r'factor|beta|alpha', information.lower()))
            return max(factor_mentions, 1)
        else:
            return 1
    
    def analyze_text_complexity(self, text: str) -> int:
        """分析文本复杂度"""
        # 基于文本长度、句子数量、词汇复杂度等
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        unique_words = len(set(text.lower().split()))
        
        # 简单的复杂度评分
        complexity = math.log(words + 1) * math.log(sentences + 1) * (unique_words / max(words, 1))
        return int(complexity * 10)
    
    def measure_computational_complexity(self, information: Any) -> int:
        """
        算法2: 计算复杂度测量
        """
        info_type = self.classify_information(information)
        
        if info_type == InformationType.PRICE_UPDATE:
            return 1  # O(1)
        
        elif info_type == InformationType.FINANCIAL_RATIO:
            vars_count = self.count_variables(information)
            return vars_count  # O(n)
        
        elif info_type == InformationType.CORRELATION_ANALYSIS:
            vars_count = self.count_variables(information)
            return vars_count ** 2  # O(n²)
        
        elif info_type == InformationType.FACTOR_INTERACTION:
            factors = self.count_factors(information)
            return 2 ** min(factors, 10)  # O(2^n), 限制最大值避免溢出
        
        else:  # TEXT_ANALYSIS, EARNINGS_REPORT, NEWS_ARTICLE
            if isinstance(information, str):
                complexity = self.analyze_text_complexity(information)
                return complexity
            else:
                return 10  # 默认中等复杂度

class LogicalDepthEstimator:
    """逻辑深度估计器"""
    
    def __init__(self):
        self.dependency_patterns = {
            'data_loading': ['raw_data'],
            'data_cleaning': ['data_loading'],
            'basic_calculation': ['data_cleaning'],
            'ratio_calculation': ['basic_calculation'],
            'correlation_analysis': ['ratio_calculation'],
            'factor_analysis': ['correlation_analysis'],
            'model_building': ['factor_analysis'],
            'prediction': ['model_building']
        }
    
    def build_dependency_graph(self, information: Any) -> nx.DiGraph:
        """构建计算依赖图"""
        G = nx.DiGraph()
        
        # 根据信息类型添加节点和边
        info_type = ComputationalComplexityAnalyzer().classify_information(information)
        
        if info_type == InformationType.PRICE_UPDATE:
            G.add_node('price_update')
            return G
        
        elif info_type == InformationType.FINANCIAL_RATIO:
            nodes = ['data_loading', 'basic_calculation', 'ratio_calculation']
            G.add_nodes_from(nodes)
            G.add_edges_from([('data_loading', 'basic_calculation'), 
                             ('basic_calculation', 'ratio_calculation')])
        
        elif info_type == InformationType.CORRELATION_ANALYSIS:
            nodes = ['data_loading', 'data_cleaning', 'basic_calculation', 
                    'ratio_calculation', 'correlation_analysis']
            G.add_nodes_from(nodes)
            edges = [('data_loading', 'data_cleaning'),
                    ('data_cleaning', 'basic_calculation'),
                    ('basic_calculation', 'ratio_calculation'),
                    ('ratio_calculation', 'correlation_analysis')]
            G.add_edges_from(edges)
        
        elif info_type == InformationType.FACTOR_INTERACTION:
            nodes = list(self.dependency_patterns.keys())
            G.add_nodes_from(nodes)
            for node, deps in self.dependency_patterns.items():
                for dep in deps:
                    if dep in nodes:
                        G.add_edge(dep, node)
        
        else:  # 文本分析
            nodes = ['data_loading', 'text_preprocessing', 'feature_extraction', 
                    'analysis', 'interpretation']
            G.add_nodes_from(nodes)
            edges = [('data_loading', 'text_preprocessing'),
                    ('text_preprocessing', 'feature_extraction'),
                    ('feature_extraction', 'analysis'),
                    ('analysis', 'interpretation')]
            G.add_edges_from(edges)
        
        return G
    
    def estimate_logical_depth(self, information: Any) -> int:
        """估计逻辑深度"""
        G = self.build_dependency_graph(information)
        
        if len(G.nodes()) == 0:
            return 1
        
        # 计算最长路径长度
        try:
            longest_path = nx.dag_longest_path_length(G)
            return max(longest_path, 1)
        except:
            # 如果图中有环或其他问题，返回节点数作为近似
            return len(G.nodes())

# 测试代码
if __name__ == "__main__":
    complexity_analyzer = ComputationalComplexityAnalyzer()
    depth_estimator = LogicalDepthEstimator()
    
    test_cases = [
        "AAPL price: 150.00",
        {"pe_ratio": 25.5, "pb_ratio": 3.2, "roe": 0.15},
        "correlation analysis between tech stocks and market volatility",
        "multi-factor model with interaction terms between momentum, value, and quality factors",
        "Apple Inc. reported strong quarterly earnings with revenue growth of 15% year-over-year..."
    ]
    
    for i, test_case in enumerate(test_cases):
        comp_complexity = complexity_analyzer.measure_computational_complexity(test_case)
        logical_depth = depth_estimator.estimate_logical_depth(test_case)
        info_type = complexity_analyzer.classify_information(test_case)
        
        print(f"测试案例 {i+1}: {str(test_case)[:50]}...")
        print(f"  信息类型: {info_type.value}")
        print(f"  计算复杂度: {comp_complexity}")
        print(f"  逻辑深度: {logical_depth}")
        print()
