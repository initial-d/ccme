import gzip
import bz2
import lzma
import zstandard as zstd
import brotli
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import json
import pickle
from dataclasses import dataclass

@dataclass
class ComplexityResult:
    """复杂度计算结果"""
    kolmogorov_approx: float
    error_bound: float
    algorithm_scores: Dict[str, float]
    confidence: float

class KolmogorovComplexityCalculator:
    """Kolmogorov复杂度近似计算器"""
    
    def __init__(self):
        self.algorithms = {
            'gzip': self._compress_gzip,
            'bz2': self._compress_bz2,
            'lzma': self._compress_lzma,
            'zstd': self._compress_zstd,
            'brotli': self._compress_brotli
        }
        
        # 基于理论保证的权重
        self.theoretical_weights = {
            'gzip': 0.25,
            'bz2': 0.20,
            'lzma': 0.30,
            'zstd': 0.15,
            'brotli': 0.10
        }
    
    def serialize_information(self, information: Any) -> str:
        """将不同类型的金融信息序列化为标准字符串"""
        if isinstance(information, str):
            return information
        elif isinstance(information, (int, float)):
            return str(information)
        elif isinstance(information, dict):
            return json.dumps(information, sort_keys=True)
        elif isinstance(information, (list, tuple)):
            return json.dumps(list(information))
        elif isinstance(information, pd.DataFrame):
            return information.to_json(orient='records')
        else:
            return pickle.dumps(information).decode('latin-1')
    
    def _compress_gzip(self, text: str) -> float:
        """使用gzip压缩"""
        original_size = len(text.encode('utf-8'))
        compressed = gzip.compress(text.encode('utf-8'))
        return len(compressed) / original_size
    
    def _compress_bz2(self, text: str) -> float:
        """使用bz2压缩"""
        original_size = len(text.encode('utf-8'))
        compressed = bz2.compress(text.encode('utf-8'))
        return len(compressed) / original_size
    
    def _compress_lzma(self, text: str) -> float:
        """使用LZMA压缩"""
        original_size = len(text.encode('utf-8'))
        compressed = lzma.compress(text.encode('utf-8'))
        return len(compressed) / original_size
    
    def _compress_zstd(self, text: str) -> float:
        """使用Zstandard压缩"""
        original_size = len(text.encode('utf-8'))
        cctx = zstd.ZstdCompressor()
        compressed = cctx.compress(text.encode('utf-8'))
        return len(compressed) / original_size
    
    def _compress_brotli(self, text: str) -> float:
        """使用Brotli压缩"""
        original_size = len(text.encode('utf-8'))
        compressed = brotli.compress(text.encode('utf-8'))
        return len(compressed) / original_size
    
    def compute_theoretical_weight(self, algorithm: str) -> float:
        """计算理论权重"""
        return self.theoretical_weights.get(algorithm, 0.2)
    
    def compute_error_bound(self, ratios: List[float]) -> float:
        """计算误差边界"""
        return np.std(ratios) / np.sqrt(len(ratios))
    
    def theoretical_kolmogorov(self, information: Any) -> ComplexityResult:
        """
        算法1: 理论保证的Kolmogorov复杂度近似
        """
        text = self.serialize_information(information)
        ratios = []
        weights = []
        algorithm_scores = {}
        
        for algorithm_name, compress_func in self.algorithms.items():
            try:
                ratio = compress_func(text)
                weight = self.compute_theoretical_weight(algorithm_name)
                
                ratios.append(ratio)
                weights.append(weight)
                algorithm_scores[algorithm_name] = ratio
            except Exception as e:
                print(f"Error with {algorithm_name}: {e}")
                continue
        
        if not ratios:
            return ComplexityResult(1.0, 1.0, {}, 0.0)
        
        # 加权平均
        k_approx = np.average(ratios, weights=weights)
        error_bound = self.compute_error_bound(ratios)
        confidence = 1.0 - (error_bound / k_approx) if k_approx > 0 else 0.0
        
        return ComplexityResult(
            kolmogorov_approx=k_approx,
            error_bound=error_bound,
            algorithm_scores=algorithm_scores,
            confidence=confidence
        )

# 测试代码
if __name__ == "__main__":
    calculator = KolmogorovComplexityCalculator()
    
    # 测试不同类型的信息
    test_cases = [
        "简单价格更新: AAPL 150.00",
        {"company": "AAPL", "price": 150.00, "volume": 1000000, "pe_ratio": 25.5},
        "复杂的多因子分析报告包含大量技术指标和基本面数据的综合评估...",
        [1, 2, 3, 4, 5] * 100  # 重复模式
    ]
    
    for i, test_case in enumerate(test_cases):
        result = calculator.theoretical_kolmogorov(test_case)
        print(f"测试案例 {i+1}:")
        print(f"  Kolmogorov复杂度: {result.kolmogorov_approx:.4f}")
        print(f"  误差边界: {result.error_bound:.4f}")
        print(f"  置信度: {result.confidence:.4f}")
        print()
