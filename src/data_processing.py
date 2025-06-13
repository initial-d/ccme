import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime, timedelta
from scipy import stats
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityController:
    """数据质量控制器"""
    
    def __init__(self, missing_threshold: float = 0.1, outlier_method: str = 'modified_tukey'):
        self.missing_threshold = missing_threshold
        self.outlier_method = outlier_method
        self.outlier_bounds = {}
        
    def missing_value_check(self, record: pd.Series) -> float:
        """检查缺失值比例"""
        return record.isnull().sum() / len(record)
    
    def outlier_detection(self, record: pd.Series, column: str = None) -> bool:
        """异常值检测"""
        if self.outlier_method == 'modified_tukey':
            return self._modified_tukey_outlier(record, column)
        elif self.outlier_method == 'iqr':
            return self._iqr_outlier(record)
        elif self.outlier_method == 'zscore':
            return self._zscore_outlier(record)
        else:
            return False
    
    def _modified_tukey_outlier(self, record: pd.Series, column: str = None) -> bool:
        """修正的Tukey方法检测异常值"""
        numeric_cols = record.select_dtypes(include=[np.number]).index
        
        for col in numeric_cols:
            if pd.isna(record[col]):
                continue
                
            # 计算MAD (Median Absolute Deviation)
            if column and col == column:
                # 使用预计算的边界
                if col in self.outlier_bounds:
                    lower, upper = self.outlier_bounds[col]
                    if record[col] < lower or record[col] > upper:
                        return True
            else:
                # 实时计算（用于单个记录）
                median_val = record[col] if pd.notna(record[col]) else 0
                mad = abs(record[col] - median_val) if pd.notna(record[col]) else 0
                
                # 简化的异常值检测
                if mad > 3.5 * np.std([record[col]]) if pd.notna(record[col]) else False:
                    return True
        
        return False
    
    def _iqr_outlier(self, record: pd.Series) -> bool:
        """IQR方法检测异常值"""
        numeric_cols = record.select_dtypes(include=[np.number]).index
        
        for col in numeric_cols:
            if pd.isna(record[col]):
                continue
            
            # 简化的IQR检测（需要更多数据点才有意义）
            # 这里只是示例实现
            if abs(record[col]) > 1e6:  # 简单的阈值检测
                return True
        
        return False
    
    def _zscore_outlier(self, record: pd.Series, threshold: float = 3.0) -> bool:
        """Z-score方法检测异常值"""
        numeric_cols = record.select_dtypes(include=[np.number]).index
        
        for col in numeric_cols:
            if pd.isna(record[col]):
                continue
            
            # 简化的Z-score检测
            if abs(record[col]) > threshold * np.std([record[col]]) if pd.notna(record[col]) else False:
                return True
        
        return False
    
    def winsorize_outliers(self, record: pd.Series, lower_percentile: float = 0.01, 
                          upper_percentile: float = 0.99) -> pd.Series:
        """Winsorize异常值"""
        numeric_cols = record.select_dtypes(include=[np.number]).index
        winsorized_record = record.copy()
        
        for col in numeric_cols:
            if pd.notna(record[col]):
                # 简化的winsorize（实际应用中需要基于历史数据计算分位数）
                if col in self.outlier_bounds:
                    lower, upper = self.outlier_bounds[col]
                    winsorized_record[col] = np.clip(record[col], lower, upper)
        
        return winsorized_record
    
    def consistency_check(self, record: pd.Series) -> bool:
        """一致性检查"""
        # 检查价格相关字段的一致性
        if 'bid' in record.index and 'ask' in record.index:
            if pd.notna(record['bid']) and pd.notna(record['ask']):
                if record['bid'] > record['ask']:
                    return False
        
        # 检查时间戳的合理性
        if 'timestamp' in record.index:
            if pd.notna(record['timestamp']):
                try:
                    ts = pd.to_datetime(record['timestamp'])
                    # 检查时间戳是否在合理范围内
                    if ts.year < 2000 or ts.year > 2030:
                        return False
                except:
                    return False
        
        # 检查交易量的合理性
        if 'volume' in record.index:
            if pd.notna(record['volume']) and record['volume'] < 0:
                return False
        
        return True
    
    def data_quality_control(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        算法4: 数据质量控制过程
        """
        clean_data = []
        total_records = len(raw_data)
        
        logger.info(f"开始处理 {total_records} 条记录")
        
        for idx, record in raw_data.iterrows():
            # 检查缺失值
            missing_ratio = self.missing_value_check(record)
            if missing_ratio > self.missing_threshold:
                continue
            
            # 异常值检测和处理
            if self.outlier_detection(record):
                record = self.winsorize_outliers(record)
            
            # 一致性检查
            if self.consistency_check(record):
                clean_data.append(record)
        
        clean_df = pd.DataFrame(clean_data)
        logger.info(f"清洗后保留 {len(clean_df)} 条记录 ({len(clean_df)/total_records:.2%})")
        
        return clean_df

class InformationEventClassifier:
    """信息事件分类器"""
    
    def __init__(self):
        self.classification_patterns = {
            'earnings': ['earnings', 'profit', 'revenue', 'income', 'quarterly'],
            'announcement': ['announce', 'declare', 'disclose', 'release'],
            'merger': ['merger', 'acquisition', 'takeover', 'buyout'],
            'dividend': ['dividend', 'payout', 'distribution'],
            'management': ['ceo', 'cfo', 'management', 'executive', 'director'],
            'regulatory': ['regulatory', 'compliance', 'sec', 'investigation'],
            'guidance': ['guidance', 'forecast', 'outlook', 'projection']
        }
        
        self.confidence_threshold = 0.8
    
    def extract_features(self, event: Dict[str, Any]) -> Dict[str, float]:
        """提取事件特征"""
        features = {}
        
        # 文本特征
        text_content = ""
        if 'title' in event:
            text_content += str(event['title']).lower() + " "
        if 'content' in event:
            text_content += str(event['content']).lower() + " "
        if 'summary' in event:
            text_content += str(event['summary']).lower() + " "
        
        # 关键词匹配特征
        for category, keywords in self.classification_patterns.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text_content)
            features[f'{category}_keywords'] = keyword_count
            features[f'{category}_presence'] = 1 if keyword_count > 0 else 0
        
        # 数值特征
        features['text_length'] = len(text_content)
        features['word_count'] = len(text_content.split())
        
        # 时间特征
        if 'timestamp' in event:
            try:
                ts = pd.to_datetime(event['timestamp'])
                features['hour'] = ts.hour
                features['day_of_week'] = ts.dayofweek
                features['is_trading_hours'] = 1 if 9 <= ts.hour <= 15 else 0
            except:
                features['hour'] = 0
                features['day_of_week'] = 0
                features['is_trading_hours'] = 0
        
        # 来源特征
        if 'source' in event:
            source = str(event['source']).lower()
            features['is_official'] = 1 if any(word in source for word in ['official', 'company', 'sec']) else 0
            features['is_news'] = 1 if any(word in source for word in ['news', 'media', 'press']) else 0
        
        return features
    
    def classify_information_events(self, events: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str]]:
        """
        算法5: 自动化信息事件分类
        """
        classified_events = []
        
        for event in events:
            features = self.extract_features(event)
            
            # 简单的规则基分类
            category_scores = {}
            for category in self.classification_patterns.keys():
                score = features.get(f'{category}_keywords', 0) * 0.7 + features.get(f'{category}_presence', 0) * 0.3
                category_scores[category] = score
            
            # 选择最高分的类别
            if category_scores:
                best_category = max(category_scores, key=category_scores.get)
                confidence = category_scores[best_category] / sum(category_scores.values()) if sum(category_scores.values()) > 0 else 0
            else:
                best_category = 'other'
                confidence = 0
            
            # 如果置信度低，标记为需要人工审核
            if confidence < self.confidence_threshold:
                best_category = 'manual_review'
            
            classified_events.append((event, best_category))
        
        return classified_events

class SampleConstructor:
    """样本构建器"""
    
    def __init__(self):
        self.data_quality_controller = DataQualityController()
        self.event_classifier = InformationEventClassifier()
    
    def filter_stocks(self, stock_universe: pd.DataFrame, 
                     criteria: Dict[str, Any]) -> pd.DataFrame:
        """股票筛选"""
        filtered_stocks = stock_universe.copy()
        
        # 市值筛选
        if 'market_cap_min' in criteria:
            filtered_stocks = filtered_stocks[filtered_stocks['market_cap'] >= criteria['market_cap_min']]
        if 'market_cap_max' in criteria:
            filtered_stocks = filtered_stocks[filtered_stocks['market_cap'] <= criteria['market_cap_max']]
        
        # 交易量筛选
        if 'volume_min' in criteria:
            filtered_stocks = filtered_stocks[filtered_stocks['avg_volume'] >= criteria['volume_min']]
        
        # 上市时间筛选
        if 'listing_days_min' in criteria:
            filtered_stocks = filtered_stocks[filtered_stocks['listing_days'] >= criteria['listing_days_min']]
        
        # 排除ST股票
        if criteria.get('exclude_st', True):
            filtered_stocks = filtered_stocks[~filtered_stocks['stock_code'].str.contains('ST')]
        
        logger.info(f"筛选后保留 {len(filtered_stocks)} 只股票")
        return filtered_stocks
    
    def construct_event_study_sample(self, events_df: pd.DataFrame, 
                                   stock_prices_df: pd.DataFrame,
                                   event_window: Tuple[int, int] = (-5, 20)) -> pd.DataFrame:
        """构建事件研究样本"""
        event_study_data = []
        
        for _, event in events_df.iterrows():
            stock_code = event['stock_code']
            event_date = pd.to_datetime(event['date'])
            
            # 获取股票价格数据
            stock_data = stock_prices_df[stock_prices_df['stock_code'] == stock_code].copy()
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            stock_data = stock_data.sort_values('date')
            
            # 定义事件窗口
            start_date = event_date + timedelta(days=event_window[0])
            end_date = event_date + timedelta(days=event_window[1])
            
            # 提取事件窗口内的数据
            window_data = stock_data[
                (stock_data['date'] >= start_date) & 
                (stock_data['date'] <= end_date)
            ].copy()
            
            if len(window_data) < abs(event_window[0]) + event_window[1] - 5:  # 至少要有足够的数据点
                continue
            
            # 计算累积异常收益
            window_data['return'] = window_data['close'].pct_change()
            window_data['days_from_event'] = (window_data['date'] - event_date).dt.days
            
            # 估计正常收益（使用事件前的数据）
            pre_event_data = window_data[window_data['days_from_event'] < 0]
            if len(pre_event_data) > 0:
                normal_return = pre_event_data['return'].mean()
                window_data['abnormal_return'] = window_data['return'] - normal_return
                window_data['cumulative_abnormal_return'] = window_data['abnormal_return'].cumsum()
            else:
                continue
            
            # 添加事件信息
            for col in event.index:
                if col not in window_data.columns:
                    window_data[col] = event[col]
            
            event_study_data.append(window_data)
        
        if event_study_data:
            return pd.concat(event_study_data, ignore_index=True)
        else:
            return pd.DataFrame()

class PriceDiscoveryMetrics:
    """价格发现指标计算器"""
    
    @staticmethod
    def calculate_discovery_speed(car_series: pd.Series, threshold: float = 0.9) -> float:
        """计算价格发现速度（90%调整完成的时间）"""
        if len(car_series) == 0:
            return np.nan
        
        final_car = car_series.iloc[-1]
        if abs(final_car) < 1e-6:  # 避免除零
            return 0
        
        target_car = threshold * abs(final_car)
        
        for i, car in enumerate(car_series):
            if abs(car) >= target_car:
                return i  # 返回天数
        
        return len(car_series)  # 如果没有达到阈值，返回整个窗口长度
    
    @staticmethod
    def calculate_discovery_efficiency(car_series: pd.Series) -> float:
        """计算价格发现效率（第一天调整比例）"""
        if len(car_series) < 2:
            return np.nan
        
        first_day_car = abs(car_series.iloc[1])  # 第一天的CAR
        final_car = abs(car_series.iloc[-1])
        
        if final_car < 1e-6:
            return 1.0 if first_day_car < 1e-6 else 0.0
        
        return min(first_day_car / final_car, 1.0)
    
    @staticmethod
    def calculate_adjustment_quality(car_series: pd.Series) -> float:
        """计算调整质量（调整过程的平滑度）"""
        if len(car_series) < 3:
            return np.nan
        
        # 计算CAR变化的方差
        car_changes = car_series.diff().dropna()
        final_car = abs(car_series.iloc[-1])
        
        if final_car < 1e-6:
            return 1.0
        
        variance_ratio = car_changes.var() / (final_car ** 2)
        quality = 1.0 / (1.0 + variance_ratio)  # 方差越小，质量越高
        
        return quality

class ExperimentalDesign:
    """实验设计框架"""
    
    def __init__(self):
        self.sample_constructor = SampleConstructor()
        self.metrics_calculator = PriceDiscoveryMetrics()
    
    def prepare_cross_sectional_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备横截面分析数据"""
        # 按股票和时间分组，计算价格发现指标
        analysis_data = []
        
        grouped = data.groupby(['stock_code', 'event_id'])
        
        for (stock_code, event_id), group in grouped:
            if len(group) < 10:  # 需要足够的观测值
                continue
            
            # 计算价格发现指标
            car_series = group.sort_values('days_from_event')['cumulative_abnormal_return']
            
            metrics = {
                'stock_code': stock_code,
                'event_id': event_id,
                'discovery_speed': self.metrics_calculator.calculate_discovery_speed(car_series),
                'discovery_efficiency': self.metrics_calculator.calculate_discovery_efficiency(car_series),
                'adjustment_quality': self.metrics_calculator.calculate_adjustment_quality(car_series)
            }
            
            # 添加复杂度和控制变量
            event_data = group.iloc[0]
            metrics.update({
                'complexity_score': event_data.get('complexity_score', 0),
                'kolmogorov_complexity': event_data.get('kolmogorov_complexity', 0),
                'computational_complexity': event_data.get('computational_complexity', 0),
                'logical_depth': event_data.get('logical_depth', 0),
                'algo_trading_intensity': event_data.get('algo_trading_intensity', 0),
                'market_cap': event_data.get('market_cap', 0),
                'volatility': event_data.get('volatility', 0),
                'liquidity': event_data.get('liquidity', 0)
            })
            
            analysis_data.append(metrics)
        
        return pd.DataFrame(analysis_data)
    
    def prepare_time_series_analysis(self, data: pd.DataFrame, 
                                   time_freq: str = 'M') -> pd.DataFrame:
        """准备时间序列分析数据"""
        data['date'] = pd.to_datetime(data['date'])
        data['period'] = data['date'].dt.to_period(time_freq)
        
        # 按时间期间聚合
        time_series_data = data.groupby('period').agg({
            'complexity_score': 'mean',
            'discovery_speed': 'mean',
            'discovery_efficiency': 'mean',
            'adjustment_quality': 'mean',
            'algo_trading_intensity': 'mean',
            'market_cap': 'mean',
            'volatility': 'mean'
        }).reset_index()
        
        # 计算复杂度效应（滚动回归系数）
        time_series_data['complexity_effect'] = np.nan
        
        window_size = 12  # 12个月滚动窗口
        for i in range(window_size, len(time_series_data)):
            window_data = time_series_data.iloc[i-window_size:i]
            
            try:
                # 简单线性回归
                X = window_data['complexity_score'].values
                y = window_data['discovery_speed'].values
                
                if len(X) > 5 and np.std(X) > 0:
                    slope, _, _, _, _ = stats.linregress(X, y)
                    time_series_data.loc[i, 'complexity_effect'] = slope
            except:
                continue
        
        return time_series_data

# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    
    # 模拟股票数据
    stock_universe = pd.DataFrame({
        'stock_code': [f'00{i:04d}' for i in range(1, 1001)],
        'market_cap': np.random.lognormal(10, 2, 1000),
        'avg_volume': np.random.lognormal(15, 1, 1000),
        'listing_days': np.random.randint(100, 3000, 1000)
    })
    
    # 模拟事件数据
    events_data = []
    for i in range(100):
        event = {
            'event_id': i,
            'stock_code': f'00{np.random.randint(1, 1001):04d}',
            'date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365)),
            'title': f'Company announcement {i}',
            'content': 'Important business update with financial implications',
            'source': 'official'
        }
        events_data.append(event)
    
    events_df = pd.DataFrame(events_data)
    
    # 测试数据质量控制
    quality_controller = DataQualityController()
    
    # 添加一些异常数据进行测试
    test_data = pd.DataFrame({
        'price': [100, 101, 999999, 102, np.nan, 103],  # 包含异常值和缺失值
        'volume': [1000, 1100, 1200, -500, 1300, 1400],  # 包含负值
        'bid': [99, 100, 101, 102, 103, 104],
        'ask': [100, 101, 102, 101, 104, 105]  # 包含bid > ask的情况
    })
    
    print("原始数据:")
    print(test_data)
    
    clean_data = quality_controller.data_quality_control(test_data)
    print("\n清洗后数据:")
    print(clean_data)
    
    # 测试事件分类
    event_classifier = InformationEventClassifier()
    classified_events = event_classifier.classify_information_events(events_data[:5])
    
    print("\n事件分类结果:")
    for event, category in classified_events:
        print(f"事件 {event['event_id']}: {category}")
    
    # 测试样本构建
    sample_constructor = SampleConstructor()
    
    # 股票筛选
    filter_criteria = {
        'market_cap_min': 1e9,  # 10亿市值以上
        'volume_min': 1e6,      # 100万交易量以上
        'listing_days_min': 365, # 上市一年以上
        'exclude_st': True
    }
    
    filtered_stocks = sample_constructor.filter_stocks(stock_universe, filter_criteria)
    print(f"\n筛选后股票数量: {len(filtered_stocks)}")
    
    # 测试价格发现指标计算
    metrics_calc = PriceDiscoveryMetrics()
    
    # 模拟CAR序列
    car_series = pd.Series([0, 0.01, 0.015, 0.018, 0.019, 0.02, 0.02, 0.02])
    
    speed = metrics_calc.calculate_discovery_speed(car_series)
    efficiency = metrics_calc.calculate_discovery_efficiency(car_series)
    quality = metrics_calc.calculate_adjustment_quality(car_series)
    
    print(f"\n价格发现指标:")
    print(f"发现速度: {speed}")
    print(f"发现效率: {efficiency:.4f}")
    print(f"调整质量: {quality:.4f}")
