import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class StatisticalAnalyzer:
    """统计分析器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def descriptive_statistics(self, data: pd.DataFrame, 
                             variables: List[str]) -> pd.DataFrame:
        """计算描述性统计"""
        stats_data = []
        
        for var in variables:
            if var in data.columns:
                series = data[var].dropna()
                
                stats_dict = {
                    'Variable': var,
                    'N': len(series),
                    'Mean': series.mean(),
                    'Std': series.std(),
                    'Min': series.min(),
                    'P25': series.quantile(0.25),
                    'P50': series.median(),
                    'P75': series.quantile(0.75),
                    'Max': series.max(),
                    'Skewness': stats.skew(series),
                    'Kurtosis': stats.kurtosis(series)
                }
                stats_data.append(stats_dict)
        
        return pd.DataFrame(stats_data)
    
    def correlation_analysis(self, data: pd.DataFrame, 
                           variables: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """相关性分析"""
        corr_data = data[variables].corr()
        
        # 计算p值
        n = len(data)
        p_values = pd.DataFrame(index=variables, columns=variables)
        
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    corr_coef = corr_data.loc[var1, var2]
                    t_stat = corr_coef * np.sqrt((n - 2) / (1 - corr_coef**2))
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                    p_values.loc[var1, var2] = p_val
                else:
                    p_values.loc[var1, var2] = 0
        
        return corr_data, p_values
    
    def regression_analysis(self, data: pd.DataFrame, 
                          dependent_var: str, 
                          independent_vars: List[str],
                          include_interactions: bool = False) -> Dict[str, Any]:
        """回归分析"""
        # 准备数据
        analysis_data = data[independent_vars + [dependent_var]].dropna()
        
        X = analysis_data[independent_vars]
        y = analysis_data[dependent_var]
        
        # 添加交互项
        if include_interactions and len(independent_vars) >= 2:
            for i, var1 in enumerate(independent_vars):
                for var2 in independent_vars[i+1:]:
                    interaction_name = f'{var1}_x_{var2}'
                    X[interaction_name] = X[var1] * X[var2]
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # 回归
        model = LinearRegression()
        model.fit(X_scaled_df, y)
        
        # 预测
        y_pred = model.predict(X_scaled_df)
        
        # 计算统计量
        n = len(y)
        k = X_scaled_df.shape[1]
        
        # R²和调整R²
        r2 = r2_score(y, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        
        # 残差分析
        residuals = y - y_pred
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        
        # 系数和t统计量
        coefficients = pd.DataFrame({
            'Variable': X_scaled_df.columns,
            'Coefficient': model.coef_,
            'Std_Error': np.sqrt(np.diag(np.linalg.inv(X_scaled_df.T @ X_scaled_df) * mse)),
        })
        
        coefficients['t_statistic'] = coefficients['Coefficient'] / coefficients['Std_Error']
        coefficients['p_value'] = 2 * (1 - stats.t.cdf(np.abs(coefficients['t_statistic']), n - k - 1))
        
        # 显著性标记
        coefficients['Significance'] = coefficients['p_value'].apply(
            lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        )
        
        return {
            'model': model,
            'coefficients': coefficients,
            'r_squared': r2,
            'adj_r_squared': adj_r2,
            'rmse': rmse,
            'residuals': residuals,
            'fitted_values': y_pred,
            'n_obs': n
        }

class VisualizationSuite:
    """可视化套件"""
    
    def __init__(self):
        self.color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    def plot_complexity_distribution(self, data: pd.DataFrame, 
                                   complexity_col: str = 'complexity_score',
                                   category_col: str = 'information_type') -> None:
        """绘制复杂度分布图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 整体分布
        ax1.hist(data[complexity_col].dropna(), bins=50, alpha=0.7, color=self.color_palette[0])
        ax1.set_xlabel('信息复杂度分数')
        ax1.set_ylabel('频数')
        ax1.set_title('信息复杂度整体分布')
        ax1.grid(True, alpha=0.3)
        
        # 按类别分布
        categories = data[category_col].unique()
        for i, category in enumerate(categories):
            category_data = data[data[category_col] == category][complexity_col].dropna()
            ax2.hist(category_data, bins=30, alpha=0.6, 
                    label=category, color=self.color_palette[i % len(self.color_palette)])
        
        ax2.set_xlabel('信息复杂度分数')
        ax2.set_ylabel('频数')
        ax2.set_title('不同信息类型的复杂度分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_main_results(self, regression_results: Dict[str, Any]) -> None:
        """绘制主要回归结果"""
        coeffs = regression_results['coefficients']
        
        # 筛选显著的系数
        significant_coeffs = coeffs[coeffs['p_value'] < 0.1].copy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 系数图
        y_pos = range(len(significant_coeffs))
        bars = ax1.barh(y_pos, significant_coeffs['Coefficient'], 
                       color=[self.color_palette[0] if x > 0 else self.color_palette[1] 
                             for x in significant_coeffs['Coefficient']])
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(significant_coeffs['Variable'])
        ax1.set_xlabel('回归系数')
        ax1.set_title('显著变量的回归系数')
        ax1.grid(True, alpha=0.3)
        
        # 添加显著性标记
        for i, (coef, sig) in enumerate(zip(significant_coeffs['Coefficient'], 
                                          significant_coeffs['Significance'])):
            ax1.text(coef + 0.01 if coef > 0 else coef - 0.01, i, sig, 
                    ha='left' if coef > 0 else 'right', va='center', fontweight='bold')
        
        # 拟合优度图
        fitted = regression_results['fitted_values']
        residuals = regression_results['residuals']
        
        ax2.scatter(fitted, residuals, alpha=0.6, color=self.color_palette[2])
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax2.set_xlabel('拟合值')
        ax2.set_ylabel('残差')
        ax2.set_title(f'残差图 (R² = {regression_results["r_squared"]:.4f})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_heterogeneous_effects(self, data: pd.DataFrame,
                                 complexity_col: str = 'complexity_score',
                                 algo_col: str = 'algo_trading_intensity',
                                 outcome_col: str = 'discovery_speed') -> None:
        """绘制异质性效应图"""
        # 创建复杂度和算法交易的分组
        data['complexity_quintile'] = pd.qcut(data[complexity_col], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        data['algo_high'] = (data[algo_col] > data[algo_col].median()).astype(int)
        
        # 计算各组的平均效应
        grouped_effects = data.groupby(['complexity_quintile', 'algo_high'])[outcome_col].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制线图
        for algo_level in [0, 1]:
            subset = grouped_effects[grouped_effects['algo_high'] == algo_level]
            label = '高算法交易' if algo_level == 1 else '低算法交易'
            color = self.color_palette[algo_level]
            
            ax.plot(subset['complexity_quintile'], subset[outcome_col], 
                   marker='o', linewidth=3, markersize=8, label=label, color=color)
        
        ax.set_xlabel('信息复杂度分位数')
        ax.set_ylabel('价格发现速度 (小时)')
        ax.set_title('算法交易对不同复杂度信息的异质性效应')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for algo_level in [0, 1]:
            subset = grouped_effects[grouped_effects['algo_high'] == algo_level]
            for _, row in subset.iterrows():
                ax.annotate(f'{row[outcome_col]:.2f}', 
                          (row['complexity_quintile'], row[outcome_col]),
                          textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.show()
    
    def plot_time_series_effects(self, data: pd.DataFrame,
                               time_col: str = 'period',
                               effect_col: str = 'complexity_effect') -> None:
        """绘制时间序列效应图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 复杂度效应时间序列
        ax1.plot(data[time_col], data[effect_col], 
                linewidth=2, color=self.color_palette[0], marker='o')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax1.set_ylabel('复杂度效应系数')
        ax1.set_title('复杂度效应的时间变化')
        ax1.grid(True, alpha=0.3)
        
        # 滚动相关性
        if 'algo_trading_intensity' in data.columns:
            rolling_corr = data['complexity_effect'].rolling(window=6).corr(data['algo_trading_intensity'])
            ax2.plot(data[time_col], rolling_corr, 
                    linewidth=2, color=self.color_palette[1], marker='s')
            ax2.set_ylabel('滚动相关系数')
            ax2.set_xlabel('时间')
            ax2.set_title('复杂度效应与算法交易强度的滚动相关性')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard(self, data: pd.DataFrame) -> None:
        """创建交互式仪表板"""
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('复杂度分布', '价格发现效率', '算法交易影响', '时间趋势'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # 复杂度分布直方图
        fig.add_trace(
            go.Histogram(x=data['complexity_score'], name='复杂度分布', 
                        marker_color=self.color_palette[0]),
            row=1, col=1
        )
        
        # 价格发现效率散点图
        fig.add_trace(
            go.Scatter(x=data['complexity_score'], y=data['discovery_efficiency'],
                      mode='markers', name='发现效率',
                      marker=dict(color=self.color_palette[1], opacity=0.6)),
            row=1, col=2
        )
        
        # 算法交易影响（双轴图）
        fig.add_trace(
            go.Scatter(x=data['algo_trading_intensity'], y=data['discovery_speed'],
                      mode='markers', name='发现速度',
                      marker=dict(color=self.color_palette[2])),
            row=2, col=1
        )
        
        # 时间趋势
        if 'date' in data.columns:
            monthly_data = data.groupby(data['date'].dt.to_period('M')).agg({
                'complexity_score': 'mean',
                'discovery_speed': 'mean'
            }).reset_index()
            
            fig.add_trace(
                go.Scatter(x=monthly_data['date'].astype(str), 
                          y=monthly_data['complexity_score'],
                          mode='lines+markers', name='平均复杂度',
                          line=dict(color=self.color_palette[3])),
                row=2, col=2
            )
        
        # 更新布局
        fig.update_layout(
            title_text="信息复杂度与价格发现效率分析仪表板",
            showlegend=True,
            height=800
        )
        
        fig.show()

class RobustnessTests:
    """稳健性检验"""
    
    def __init__(self):
        self.statistical_analyzer = StatisticalAnalyzer()
    
    def placebo_test(self, data: pd.DataFrame, 
                    dependent_var: str,
                    true_treatment: str,
                    n_placebo: int = 100) -> Dict[str, Any]:
        """安慰剂检验"""
        # 真实效应
        true_result = self.statistical_analyzer.regression_analysis(
            data, dependent_var, [true_treatment]
        )
        true_coef = true_result['coefficients'][
            true_result['coefficients']['Variable'] == true_treatment
        ]['Coefficient'].iloc[0]
        
        # 生成随机安慰剂变量
        placebo_coefs = []
        for i in range(n_placebo):
            placebo_var = f'placebo_{i}'
            data[placebo_var] = np.random.permutation(data[true_treatment])
            
            placebo_result = self.statistical_analyzer.regression_analysis(
                data, dependent_var, [placebo_var]
            )
            placebo_coef = placebo_result['coefficients'][
                placebo_result['coefficients']['Variable'] == placebo_var
            ]['Coefficient'].iloc[0]
            placebo_coefs.append(placebo_coef)
            
            # 清理临时变量
            data.drop(placebo_var, axis=1, inplace=True)
        
        # 计算p值
        placebo_coefs = np.array(placebo_coefs)
        p_value = np.mean(np.abs(placebo_coefs) >= np.abs(true_coef))
        
        return {
            'true_coefficient': true_coef,
            'placebo_coefficients': placebo_coefs,
            'placebo_p_value': p_value,
            'placebo_mean': np.mean(placebo_coefs),
            'placebo_std': np.std(placebo_coefs)
        }
    
    def subsample_stability(self, data: pd.DataFrame,
                          dependent_var: str,
                          independent_vars: List[str],
                          n_subsamples: int = 50,
                          subsample_ratio: float = 0.8) -> Dict[str, Any]:
        """子样本稳定性检验"""
        subsample_results = []
        n_obs = len(data)
        subsample_size = int(n_obs * subsample_ratio)
        
        for i in range(n_subsamples):
            # 随机抽样
            subsample_idx = np.random.choice(n_obs, subsample_size, replace=False)
            subsample_data = data.iloc[subsample_idx]
            
            # 回归分析
            result = self.statistical_analyzer.regression_analysis(
                subsample_data, dependent_var, independent_vars
            )
            
            subsample_results.append(result['coefficients'])
        
        # 汇总结果
        stability_summary = {}
        for var in independent_vars:
            var_coefs = []
            var_pvals = []
            
            for result in subsample_results:
                var_result = result[result['Variable'] == var]
                if len(var_result) > 0:
                    var_coefs.append(var_result['Coefficient'].iloc[0])
                    var_pvals.append(var_result['p_value'].iloc[0])
            
            stability_summary[var] = {
                'mean_coefficient': np.mean(var_coefs),
                'std_coefficient': np.std(var_coefs),
                'significant_ratio': np.mean(np.array(var_pvals) < 0.05),
                'coefficient_range': (np.min(var_coefs), np.max(var_coefs))
            }
        
        return stability_summary

# 测试和演示代码
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    n_obs = 1000
    
    # 生成模拟的分析数据
    simulation_data = pd.DataFrame({
        'complexity_score': np.random.beta(2, 5, n_obs),
        'kolmogorov_complexity': np.random.beta(3, 4, n_obs),
        'computational_complexity': np.random.lognormal(2, 1, n_obs),
        'logical_depth': np.random.poisson(5, n_obs),
        'algo_trading_intensity': np.random.beta(2, 3, n_obs),
        'market_cap': np.random.lognormal(10, 2, n_obs),
        'volatility': np.random.gamma(2, 0.01, n_obs),
        'liquidity': np.random.lognormal(15, 1, n_obs),
        'information_type': np.random.choice(['earnings', 'announcement', 'merger', 'guidance'], n_obs),
        'date': pd.date_range('2020-01-01', periods=n_obs, freq='D')
    })
    
    # 生成因变量（价格发现指标）
    simulation_data['discovery_speed'] = (
        2 + 
        0.3 * simulation_data['complexity_score'] +
        -0.2 * simulation_data['algo_trading_intensity'] +
        -0.1 * simulation_data['complexity_score'] * simulation_data['algo_trading_intensity'] +
        0.1 * np.log(simulation_data['market_cap']) +
        0.05 * simulation_data['volatility'] +
        np.random.normal(0, 0.2, n_obs)
    )
    
    simulation_data['discovery_efficiency'] = np.clip(
        0.5 + 
        -0.2 * simulation_data['complexity_score'] +
        0.15 * simulation_data['algo_trading_intensity'] +
        np.random.normal(0, 0.1, n_obs),
        0, 1
    )
    
    # 创建分析器
    analyzer = StatisticalAnalyzer()
    visualizer = VisualizationSuite()
    robustness = RobustnessTests()
    
    # 描述性统计
    variables = ['complexity_score', 'discovery_speed', 'discovery_efficiency', 
                'algo_trading_intensity', 'market_cap', 'volatility']
    
    desc_stats = analyzer.descriptive_statistics(simulation_data, variables)
    print("描述性统计:")
    print(desc_stats.round(4))
    
    # 相关性分析
    corr_matrix, p_values = analyzer.correlation_analysis(simulation_data, variables)
    print("\n相关系数矩阵:")
    print(corr_matrix.round(4))
    
    # 主回归分析
    main_regression = analyzer.regression_analysis(
        simulation_data, 
        'discovery_speed',
        ['complexity_score', 'algo_trading_intensity', 'market_cap', 'volatility'],
        include_interactions=True
    )
    
    print("\n主回归结果:")
    print(main_regression['coefficients'].round(4))
    print(f"R²: {main_regression['r_squared']:.4f}")
    print(f"调整R²: {main_regression['adj_r_squared']:.4f}")
    
    # 可视化
    print("\n生成可视化图表...")
    
    # 复杂度分布图
    visualizer.plot_complexity_distribution(simulation_data)
    
    # 主要结果图
    visualizer.plot_main_results(main_regression)
    
    # 异质性效应图
    visualizer.plot_heterogeneous_effects(simulation_data)
    
    # 稳健性检验
    print("\n进行稳健性检验...")
    
    # 安慰剂检验
    placebo_result = robustness.placebo_test(
        simulation_data, 'discovery_speed', 'complexity_score', n_placebo=50
    )
    
    print(f"安慰剂检验结果:")
    print(f"真实系数: {placebo_result['true_coefficient']:.4f}")
    print(f"安慰剂系数均值: {placebo_result['placebo_mean']:.4f}")
    print(f"安慰剂p值: {placebo_result['placebo_p_value']:.4f}")
    
    # 子样本稳定性检验
    stability_result = robustness.subsample_stability(
        simulation_data, 'discovery_speed', 
        ['complexity_score', 'algo_trading_intensity'], 
        n_subsamples=30
    )
    
    print(f"\n子样本稳定性检验:")
    for var, stats in stability_result.items():
        print(f"{var}:")
        print(f"  平均系数: {stats['mean_coefficient']:.4f}")
        print(f"  系数标准差: {stats['std_coefficient']:.4f}")
        print(f"  显著比例: {stats['significant_ratio']:.2%}")
    
    # 创建交互式仪表板
    print("\n创建交互式仪表板...")
    visualizer.create_interactive_dashboard(simulation_data)
    
    print("\n分析完成！")
