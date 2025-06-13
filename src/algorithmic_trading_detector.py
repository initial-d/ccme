import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import scipy.stats as stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class OrderFlowFeatureExtractor:
    """订单流特征提取器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def extract_order_size_features(self, order_sizes: np.ndarray) -> Dict[str, float]:
        """提取订单大小分布特征"""
        if len(order_sizes) == 0:
            return {}
        
        features = {}
        
        # 基本统计量
        features['order_size_mean'] = np.mean(order_sizes)
        features['order_size_std'] = np.std(order_sizes)
        features['order_size_skew'] = stats.skew(order_sizes)
        features['order_size_kurtosis'] = stats.kurtosis(order_sizes)
        
        # 分位数
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            features[f'order_size_p{p}'] = np.percentile(order_sizes, p)
        
        # 幂律分布检验
        try:
            # 简化的幂律指数估计
            log_sizes = np.log(order_sizes[order_sizes > 0])
            if len(log_sizes) > 10:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    np.log(np.arange(1, len(log_sizes) + 1)),
                    np.sort(log_sizes)[::-1]
                )
                features['power_law_exponent'] = -slope
                features['power_law_r_squared'] = r_value ** 2
            else:
                features['power_law_exponent'] = 0
                features['power_law_r_squared'] = 0
        except:
            features['power_law_exponent'] = 0
            features['power_law_r_squared'] = 0
        
        # 订单大小集中度
        unique_sizes, counts = np.unique(order_sizes, return_counts=True)
        if len(unique_sizes) > 1:
            features['size_concentration'] = np.sum((counts / len(order_sizes)) ** 2)  # HHI指数
            features['size_diversity'] = len(unique_sizes) / len(order_sizes)
        else:
            features['size_concentration'] = 1.0
            features['size_diversity'] = 0.0
        
        return features
    
    def extract_timing_features(self, timestamps: np.ndarray) -> Dict[str, float]:
        """提取时间特征"""
        if len(timestamps) < 2:
            return {}
        
        features = {}
        
        # 计算时间间隔
        intervals = np.diff(timestamps)
        
        # 基本统计量
        features['interval_mean'] = np.mean(intervals)
        features['interval_std'] = np.std(intervals)
        features['interval_cv'] = features['interval_std'] / features['interval_mean'] if features['interval_mean'] > 0 else 0
        
        # 规律性检测
        features['interval_regularity'] = 1.0 / (1.0 + features['interval_cv'])
        
        # 泊松过程检验
        try:
            # 计算到达率
            total_time = timestamps[-1] - timestamps[0]
            arrival_rate = len(timestamps) / total_time if total_time > 0 else 0
            
            # 泊松分布拟合优度
            expected_intervals = np.random.exponential(1/arrival_rate, len(intervals)) if arrival_rate > 0 else intervals
            ks_stat, ks_p_value = stats.kstest(intervals, lambda x: stats.expon.cdf(x, scale=1/arrival_rate))
            
            features['poisson_ks_stat'] = ks_stat
            features['poisson_p_value'] = ks_p_value
            features['arrival_rate'] = arrival_rate
        except:
            features['poisson_ks_stat'] = 1.0
            features['poisson_p_value'] = 0.0
            features['arrival_rate'] = 0.0
        
        # 微秒级精度检测
        microsecond_precision = np.sum(timestamps % 0.001 == 0) / len(timestamps)
        features['microsecond_precision'] = microsecond_precision
        
        return features
    
    def extract_cancellation_features(self, orders_df: pd.DataFrame) -> Dict[str, float]:
        """提取撤单特征"""
        if len(orders_df) == 0:
            return {}
        
        features = {}
        
        # 撤单率
        if 'status' in orders_df.columns:
            total_orders = len(orders_df)
            cancelled_orders = len(orders_df[orders_df['status'] == 'cancelled'])
            features['cancellation_rate'] = cancelled_orders / total_orders if total_orders > 0 else 0
        else:
            features['cancellation_rate'] = 0
        
        # 撤单时间分析
        if 'order_time' in orders_df.columns and 'cancel_time' in orders_df.columns:
            cancelled_df = orders_df[orders_df['status'] == 'cancelled'].copy()
            if len(cancelled_df) > 0:
                cancel_delays = cancelled_df['cancel_time'] - cancelled_df['order_time']
                features['avg_cancel_delay'] = np.mean(cancel_delays)
                features['cancel_delay_std'] = np.std(cancel_delays)
                
                # 快速撤单比例（<1秒）
                fast_cancels = np.sum(cancel_delays < 1.0)
                features['fast_cancel_ratio'] = fast_cancels / len(cancelled_df)
            else:
                features['avg_cancel_delay'] = 0
                features['cancel_delay_std'] = 0
                features['fast_cancel_ratio'] = 0
        else:
            features['avg_cancel_delay'] = 0
            features['cancel_delay_std'] = 0
            features['fast_cancel_ratio'] = 0
        
        return features
    
    def extract_market_impact_features(self, trades_df: pd.DataFrame, 
                                     quotes_df: pd.DataFrame) -> Dict[str, float]:
        """提取市场影响特征"""
        features = {}
        
        if len(trades_df) == 0 or len(quotes_df) == 0:
            return features
        
        try:
            # 价格影响衰减模式
            price_impacts = []
            for _, trade in trades_df.iterrows():
                trade_time = trade['timestamp']
                trade_price = trade['price']
                trade_size = trade['volume']
                
                # 找到交易前后的报价
                pre_quotes = quotes_df[quotes_df['timestamp'] < trade_time].tail(5)
                post_quotes = quotes_df[quotes_df['timestamp'] > trade_time].head(10)
                
                if len(pre_quotes) > 0 and len(post_quotes) > 0:
                    pre_mid = (pre_quotes['bid'].iloc[-1] + pre_quotes['ask'].iloc[-1]) / 2
                    
                    # 计算不同时间窗口的价格影响
                    for window in [1, 3, 5, 10]:
                        if len(post_quotes) >= window:
                            post_mid = (post_quotes['bid'].iloc[window-1] + post_quotes['ask'].iloc[window-1]) / 2
                            impact = abs(post_mid - pre_mid) / pre_mid if pre_mid > 0 else 0
                            price_impacts.append(impact / trade_size if trade_size > 0 else 0)
            
            if price_impacts:
                features['avg_price_impact'] = np.mean(price_impacts)
                features['price_impact_std'] = np.std(price_impacts)
            else:
                features['avg_price_impact'] = 0
                features['price_impact_std'] = 0
            
            # VWAP偏差
            if 'vwap' in trades_df.columns:
                vwap_deviations = abs(trades_df['price'] - trades_df['vwap']) / trades_df['vwap']
                features['avg_vwap_deviation'] = np.mean(vwap_deviations)
            else:
                features['avg_vwap_deviation'] = 0
            
            # 买卖价差影响
            if len(quotes_df) > 0:
                spreads = quotes_df['ask'] - quotes_df['bid']
                features['avg_spread'] = np.mean(spreads)
                features['spread_std'] = np.std(spreads)
            else:
                features['avg_spread'] = 0
                features['spread_std'] = 0
                
        except Exception as e:
            print(f"市场影响特征提取错误: {e}")
            features['avg_price_impact'] = 0
            features['price_impact_std'] = 0
            features['avg_vwap_deviation'] = 0
            features['avg_spread'] = 0
            features['spread_std'] = 0
        
        return features

class AlgorithmicTradingDetector:
    """算法交易检测器"""
    
    def __init__(self):
        self.feature_extractor = OrderFlowFeatureExtractor()
        self.classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
    
    def extract_features(self, order_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """提取所有特征"""
        all_features = []
        
        for session_id, session_data in order_data.items():
            features = {'session_id': session_id}
            
            # 订单大小特征
            if 'order_size' in session_data.columns:
                order_size_features = self.feature_extractor.extract_order_size_features(
                    session_data['order_size'].values
                )
                features.update(order_size_features)
            
            # 时间特征
            if 'timestamp' in session_data.columns:
                timing_features = self.feature_extractor.extract_timing_features(
                    session_data['timestamp'].values
                )
                features.update(timing_features)
            
            # 撤单特征
            cancellation_features = self.feature_extractor.extract_cancellation_features(session_data)
            features.update(cancellation_features)
            
            # 市场影响特征（需要额外的报价数据）
            if 'quotes' in order_data:
                market_impact_features = self.feature_extractor.extract_market_impact_features(
                    session_data, order_data['quotes']
                )
                features.update(market_impact_features)
            
            all_features.append(features)
        
        features_df = pd.DataFrame(all_features)
        
        # 填充缺失值
        features_df = features_df.fillna(0)
        
        return features_df
    
    def train(self, training_data: Dict[str, pd.DataFrame], 
              labels: Dict[str, int]) -> Dict[str, float]:
        """训练分类器"""
        # 提取特征
        features_df = self.extract_features(training_data)
        
        # 准备标签
        y = np.array([labels.get(session_id, 0) for session_id in features_df['session_id']])
        
        # 准备特征矩阵
        X = features_df.drop(['session_id'], axis=1)
        self.feature_names = X.columns.tolist()
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练模型
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True
        
        # 交叉验证评估
        cv_scores = cross_val_score(self.classifier, X_scaled, y, 
                                   cv=TimeSeriesSplit(n_splits=5), 
                                   scoring='roc_auc')
        
        # 特征重要性
        feature_importance = dict(zip(self.feature_names, self.classifier.feature_importances_))
        
        return {
            'cv_auc_mean': np.mean(cv_scores),
            'cv_auc_std': np.std(cv_scores),
            'feature_importance': feature_importance
        }
    
    def predict(self, test_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """预测算法交易概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        # 提取特征
        features_df = self.extract_features(test_data)
        
        # 准备特征矩阵
        X = features_df.drop(['session_id'], axis=1)
        
        # 确保特征顺序一致
        X = X.reindex(columns=self.feature_names, fill_value=0)
        
        # 标准化
        X_scaled = self.scaler.transform(X)
        
        # 预测
        probabilities = self.classifier.predict_proba(X_scaled)[:, 1]
        
        # 返回结果
        results = {}
        for session_id, prob in zip(features_df['session_id'], probabilities):
            results[session_id] = prob
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

class AlgorithmicTradingIntensityCalculator:
    """算法交易强度计算器"""
    
    def __init__(self, detector: AlgorithmicTradingDetector):
        self.detector = detector
    
    def calculate_intensity(self, trading_sessions: Dict[str, pd.DataFrame],
                          time_window: str = '1H') -> pd.DataFrame:
        """计算算法交易强度"""
        # 预测各交易会话的算法交易概率
        algo_probs = self.detector.predict(trading_sessions)
        
        # 构建时间序列数据
        intensity_data = []
        
        for session_id, session_data in trading_sessions.items():
            if 'timestamp' in session_data.columns and 'volume' in session_data.columns:
                algo_prob = algo_probs.get(session_id, 0)
                
                for _, row in session_data.iterrows():
                    intensity_data.append({
                        'timestamp': row['timestamp'],
                        'volume': row['volume'],
                        'algo_prob': algo_prob,
                        'algo_volume': row['volume'] * algo_prob
                    })
        
        if not intensity_data:
            return pd.DataFrame()
        
        intensity_df = pd.DataFrame(intensity_data)
        intensity_df['timestamp'] = pd.to_datetime(intensity_df['timestamp'])
        intensity_df = intensity_df.set_index('timestamp')
        
        # 按时间窗口聚合
        aggregated = intensity_df.resample(time_window).agg({
            'volume': 'sum',
            'algo_volume': 'sum'
        })
        
        # 计算算法交易强度
        aggregated['algo_intensity'] = aggregated['algo_volume'] / aggregated['volume']
        aggregated['algo_intensity'] = aggregated['algo_intensity'].fillna(0)
        
        return aggregated

# 数据生成器（用于测试）
class SyntheticDataGenerator:
    """合成数据生成器"""
    
    @staticmethod
    def generate_human_trading_session(n_orders: int = 100) -> pd.DataFrame:
        """生成人工交易会话数据"""
        np.random.seed(42)
        
        # 人工交易特征：不规律的时间间隔，多样化的订单大小
        timestamps = np.cumsum(np.random.exponential(30, n_orders))  # 平均30秒间隔
        order_sizes = np.random.lognormal(mean=6, sigma=1.5, size=n_orders)  # 对数正态分布
        
        # 添加一些噪声
        timestamps += np.random.normal(0, 5, n_orders)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'order_size': order_sizes,
            'status': np.random.choice(['filled', 'cancelled'], n_orders, p=[0.8, 0.2])
        })
    
    @staticmethod
    def generate_algorithmic_trading_session(n_orders: int = 100) -> pd.DataFrame:
        """生成算法交易会话数据"""
        np.random.seed(123)
        
        # 算法交易特征：规律的时间间隔，标准化的订单大小
        timestamps = np.arange(0, n_orders * 5, 5)  # 每5秒一个订单
        order_sizes = np.random.choice([100, 200, 500, 1000], n_orders, p=[0.4, 0.3, 0.2, 0.1])
        
        # 添加微秒级精度
        timestamps = timestamps + np.random.choice([0, 0.001, 0.002], n_orders, p=[0.1, 0.45, 0.45])
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'order_size': order_sizes,
            'status': np.random.choice(['filled', 'cancelled'], n_orders, p=[0.9, 0.1])
        })

# 测试代码
if __name__ == "__main__":
    # 生成测试数据
    data_generator = SyntheticDataGenerator()
    
    # 训练数据
    training_data = {}
    labels = {}
    
    # 生成人工交易数据
    for i in range(50):
        session_id = f"human_{i}"
        training_data[session_id] = data_generator.generate_human_trading_session()
        labels[session_id] = 0  # 人工交易
    
    # 生成算法交易数据
    for i in range(50):
        session_id = f"algo_{i}"
        training_data[session_id] = data_generator.generate_algorithmic_trading_session()
        labels[session_id] = 1  # 算法交易
    
    # 创建检测器并训练
    detector = AlgorithmicTradingDetector()
    training_results = detector.train(training_data, labels)
    
    print("训练结果:")
    print(f"交叉验证AUC: {training_results['cv_auc_mean']:.4f} ± {training_results['cv_auc_std']:.4f}")
    
    # 显示特征重要性
    importance_df = detector.get_feature_importance()
    print("\n前10个重要特征:")
    print(importance_df.head(10))
    
    # 测试数据
    test_data = {
        'test_human': data_generator.generate_human_trading_session(),
        'test_algo': data_generator.generate_algorithmic_trading_session()
    }
    
    # 预测
    predictions = detector.predict(test_data)
    print("\n预测结果:")
    for session_id, prob in predictions.items():
        print(f"{session_id}: {prob:.4f}")
    
    # 计算算法交易强度
    intensity_calculator = AlgorithmicTradingIntensityCalculator(detector)
    
    # 为测试数据添加volume列
    for session_data in test_data.values():
        session_data['volume'] = session_data['order_size'] * np.random.uniform(0.5, 1.5, len(session_data))
    
    intensity_df = intensity_calculator.calculate_intensity(test_data, '10T')  # 10分钟窗口
    print("\n算法交易强度:")
    print(intensity_df.head())
