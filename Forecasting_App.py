# Set page config FIRST - before any other Streamlit commands
import streamlit as st

st.set_page_config(
    page_title="Advanced Demand Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# Now import other libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import io
import base64

# Core ML libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Time series libraries with better error handling
TENSORFLOW_AVAILABLE = False
STATSMODELS_AVAILABLE = False
PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Smoothing libraries
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

# Show library status
st.sidebar.info(f"""
📦 **Library Status:**
- TensorFlow (LSTM): {'✅' if TENSORFLOW_AVAILABLE else '❌'}
- Statsmodels (ARIMA): {'✅' if STATSMODELS_AVAILABLE else '❌'}
- Prophet: {'✅' if PROPHET_AVAILABLE else '❌'}
""")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .smoothing-info {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .forecast-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎯 Advanced Demand Forecasting Dashboard</h1>', unsafe_allow_html=True)

# Installation instructions
if not all([TENSORFLOW_AVAILABLE, STATSMODELS_AVAILABLE, PROPHET_AVAILABLE]):
    with st.expander("📋 Installation Instructions for Missing Libraries"):
        st.markdown("""
        To enable all features, install missing libraries:
        
        ```bash
        # For TensorFlow (LSTM models)
        pip install tensorflow
        
        # For Statsmodels (ARIMA/SARIMA)
        pip install statsmodels
        
        # For Prophet
        pip install prophet
        
        # Install all at once
        pip install tensorflow statsmodels prophet
        ```
        
        After installation, restart your Streamlit app.
        """)

# Sidebar
st.sidebar.header("📊 Configuration")

@st.cache_data
def load_sample_data():
    """Create sample retail data with realistic patterns"""
    np.random.seed(42)
    
    # Generate more realistic time series data
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 6, 30)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Create base demand with multiple patterns
    n_days = len(dates)
    
    # Trend component (gradual increase over time)
    trend = np.linspace(100, 150, n_days)
    
    # Seasonal component (yearly pattern)
    seasonal_yearly = 20 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    
    # Weekly pattern (higher demand on weekends)
    weekly_pattern = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    
    # Random noise
    noise = np.random.normal(0, 15, n_days)
    
    # Special events (holiday spikes)
    special_events = np.zeros(n_days)
    for i, date in enumerate(dates):
        if date.month == 12 and date.day in [24, 25]:  # Christmas
            special_events[i] = 50
        elif date.month == 11 and date.day in [24, 25]:  # Black Friday
            special_events[i] = 40
        elif date.month == 1 and date.day == 1:  # New Year
            special_events[i] = 30
    
    # Combine all components
    demand = trend + seasonal_yearly + weekly_pattern + special_events + noise
    demand = np.maximum(demand, 10)  # Ensure positive values
    
    # Create DataFrame
    data = {
        'Date': dates,
        'Demand': demand,
        'Revenue': demand * np.random.uniform(15, 25, n_days),
        'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books'], n_days),
        'Year': [d.year for d in dates],
        'Month': [d.month for d in dates],
        'DayOfWeek': [d.weekday() for d in dates],
        'Quarter': [(d.month - 1) // 3 + 1 for d in dates]
    }
    
    return pd.DataFrame(data)

# Data loading
uploaded_file = st.sidebar.file_uploader("Upload your retail dataset", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("✅ File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")
        df = load_sample_data()
else:
    st.sidebar.info("📋 Using sample retail data")
    df = load_sample_data()

# Data preprocessing
def preprocess_data(df):
    """Preprocess data with error handling"""
    try:
        df_processed = df.copy()
        
        # Handle date column
        date_columns = ['Date', 'date', 'DATE', 'ds']
        date_col = None
        for col in date_columns:
            if col in df_processed.columns:
                date_col = col
                break
        
        if date_col:
            df_processed['Date'] = pd.to_datetime(df_processed[date_col], errors='coerce')
            df_processed = df_processed.dropna(subset=['Date'])
        else:
            # Create date range if no date column
            df_processed['Date'] = pd.date_range(start='2020-01-01', periods=len(df_processed), freq='D')
        
        # Handle demand column
        demand_columns = ['Demand', 'demand', 'Sales', 'Quantity', 'Volume', 'y']
        demand_col = None
        for col in demand_columns:
            if col in df_processed.columns:
                demand_col = col
                break
        
        if demand_col and demand_col != 'Demand':
            df_processed['Demand'] = pd.to_numeric(df_processed[demand_col], errors='coerce')
        elif 'Demand' not in df_processed.columns:
            # Create demand from revenue or other metrics
            if 'Revenue' in df_processed.columns:
                df_processed['Demand'] = pd.to_numeric(df_processed['Revenue'], errors='coerce')
            else:
                df_processed['Demand'] = np.random.normal(100, 20, len(df_processed))
        
        # Ensure demand is numeric and positive
        df_processed['Demand'] = pd.to_numeric(df_processed['Demand'], errors='coerce')
        df_processed = df_processed.dropna(subset=['Demand'])
        df_processed['Demand'] = np.maximum(df_processed['Demand'], 0.1)
        
        # Create time features
        df_processed['Year'] = df_processed['Date'].dt.year
        df_processed['Month'] = df_processed['Date'].dt.month
        df_processed['DayOfWeek'] = df_processed['Date'].dt.dayofweek
        df_processed['Quarter'] = df_processed['Date'].dt.quarter
        df_processed['DayOfYear'] = df_processed['Date'].dt.dayofyear
        
        # Sort by date
        df_processed = df_processed.sort_values('Date').reset_index(drop=True)
        
        return df_processed
    
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return load_sample_data()

df = preprocess_data(df)

# Sidebar configuration
st.sidebar.subheader("🔍 Data Smoothing Options")

# Smoothing techniques selection
smoothing_methods = st.sidebar.multiselect(
    "Select Smoothing Methods",
    ["Moving Average", "Exponential Smoothing", "Savitzky-Golay Filter", "Gaussian Filter"],
    default=["Moving Average"]
)

# Smoothing parameters
ma_window = 7
exp_alpha = 0.3
savgol_window = 11
savgol_poly = 2
gaussian_sigma = 1.0

if "Moving Average" in smoothing_methods:
    ma_window = st.sidebar.slider("Moving Average Window", 3, 30, 7)

if "Exponential Smoothing" in smoothing_methods:
    exp_alpha = st.sidebar.slider("Exponential Smoothing Alpha", 0.1, 0.9, 0.3)

if "Savitzky-Golay Filter" in smoothing_methods:
    savgol_window = st.sidebar.slider("Savitzky-Golay Window", 5, 51, 11)
    savgol_poly = st.sidebar.slider("Polynomial Order", 1, 5, 2)

if "Gaussian Filter" in smoothing_methods:
    gaussian_sigma = st.sidebar.slider("Gaussian Sigma", 0.5, 5.0, 1.0)

# Model selection
st.sidebar.subheader("🤖 Model Selection")
available_models = ["Random Forest", "Linear Regression"]

if TENSORFLOW_AVAILABLE:
    available_models.append("LSTM")
if STATSMODELS_AVAILABLE:
    available_models.extend(["ARIMA", "SARIMA", "Exponential Smoothing"])
if PROPHET_AVAILABLE:
    available_models.append("Prophet")

model_type = st.sidebar.selectbox("Select Forecasting Model", available_models)

# Forecast parameters
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)

# Data filtering
if 'Product_Category' in df.columns:
    categories = st.sidebar.multiselect(
        "Select Product Categories",
        df['Product_Category'].unique(),
        default=df['Product_Category'].unique()[:3] if len(df['Product_Category'].unique()) > 3 else df['Product_Category'].unique()
    )
    if categories:
        df_filtered = df[df['Product_Category'].isin(categories)]
    else:
        df_filtered = df
else:
    df_filtered = df

# Aggregate data by date
@st.cache_data
def aggregate_data(df):
    """Aggregate data by date for time series analysis"""
    try:
        df_agg = df.groupby('Date').agg({
            'Demand': 'sum',
            'Revenue': 'sum' if 'Revenue' in df.columns else 'mean'
        }).reset_index()
        df_agg = df_agg.sort_values('Date').reset_index(drop=True)
        return df_agg
    except Exception as e:
        st.error(f"Error in aggregation: {str(e)}")
        return df

df_ts = aggregate_data(df_filtered)

# Apply smoothing techniques
def apply_smoothing(data, methods, params):
    """Apply various smoothing techniques to the data"""
    smoothed_data = data.copy()
    smoothing_info = []
    
    if "Moving Average" in methods:
        window = params.get('ma_window', 7)
        smoothed_data['MA_Smoothed'] = data['Demand'].rolling(window=window, center=True).mean()
        smoothing_info.append(f"Moving Average (window={window}): Reduces noise by averaging {window} neighboring points")
    
    if "Exponential Smoothing" in methods:
        alpha = params.get('exp_alpha', 0.3)
        smoothed_data['EXP_Smoothed'] = data['Demand'].ewm(alpha=alpha).mean()
        smoothing_info.append(f"Exponential Smoothing (α={alpha}): Weighted average with {alpha} decay factor")
    
    if "Savitzky-Golay Filter" in methods:
        window = params.get('savgol_window', 11)
        poly_order = params.get('savgol_poly', 2)
        if len(data) >= window:
            smoothed_data['SAVGOL_Smoothed'] = savgol_filter(data['Demand'], window, poly_order)
            smoothing_info.append(f"Savitzky-Golay (window={window}, poly={poly_order}): Polynomial smoothing preserving peaks/valleys")
    
    if "Gaussian Filter" in methods:
        sigma = params.get('gaussian_sigma', 1.0)
        smoothed_data['GAUSSIAN_Smoothed'] = gaussian_filter1d(data['Demand'], sigma=sigma)
        smoothing_info.append(f"Gaussian Filter (σ={sigma}): Gaussian kernel smoothing with standard deviation {sigma}")
    
    return smoothed_data, smoothing_info

# Apply smoothing
smoothing_params = {
    'ma_window': ma_window,
    'exp_alpha': exp_alpha,
    'savgol_window': savgol_window,
    'savgol_poly': savgol_poly,
    'gaussian_sigma': gaussian_sigma
}

df_smoothed, smoothing_info = apply_smoothing(df_ts, smoothing_methods, smoothing_params)

# Display smoothing information
if smoothing_info:
    st.markdown('<div class="smoothing-info">', unsafe_allow_html=True)
    st.markdown("### 🔧 Applied Smoothing Techniques:")
    for info in smoothing_info:
        st.markdown(f"• {info}")
    st.markdown('</div>', unsafe_allow_html=True)

# Choose which smoothed series to use for modeling
selected_smooth = 'Original'
if smoothing_methods:
    smoothed_columns = [col for col in df_smoothed.columns if 'Smoothed' in col]
    if smoothed_columns:
        selected_smooth = st.selectbox("Select smoothed series for modeling", 
                                     ['Original'] + smoothed_columns, 
                                     index=0)
        if selected_smooth != 'Original':
            df_ts['Demand'] = df_smoothed[selected_smooth]

# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>{len(df_filtered):,}</h3>
        <p>Total Records</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_demand = df_ts['Demand'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <h3>{avg_demand:.1f}</h3>
        <p>Avg Daily Demand</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    total_demand = df_ts['Demand'].sum()
    st.markdown(f"""
    <div class="metric-card">
        <h3>{total_demand:,.0f}</h3>
        <p>Total Demand</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    date_range = (df_ts['Date'].max() - df_ts['Date'].min()).days
    st.markdown(f"""
    <div class="metric-card">
        <h3>{date_range}</h3>
        <p>Days of Data</p>
    </div>
    """, unsafe_allow_html=True)

# Model training functions
def train_lstm_model(data, forecast_days=30, lookback=60):
    """Train LSTM model for time series forecasting"""
    if not TENSORFLOW_AVAILABLE:
        return None, None, None, None
    
    try:
        # Prepare data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data['Demand'].values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        with st.spinner("Training LSTM model..."):
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Future predictions
        last_sequence = scaled_data[-lookback:]
        future_pred = []
        
        for _ in range(forecast_days):
            next_pred = model.predict(last_sequence.reshape(1, lookback, 1), verbose=0)
            future_pred.append(next_pred[0, 0])
            last_sequence = np.append(last_sequence[1:], next_pred[0, 0])
        
        future_pred = scaler.inverse_transform(np.array(future_pred).reshape(-1, 1))
        
        return y_test_actual.flatten(), y_pred.flatten(), future_pred.flatten(), scaler
    
    except Exception as e:
        st.error(f"LSTM model error: {str(e)}")
        return None, None, None, None

def train_arima_model(data, forecast_days=30):
    """Train ARIMA model"""
    if not STATSMODELS_AVAILABLE:
        return None, None, None
    
    try:
        # Prepare data
        ts_data = data['Demand'].values
        split_idx = int(len(ts_data) * 0.8)
        train_data = ts_data[:split_idx]
        test_data = ts_data[split_idx:]
        
        # Fit ARIMA model
        with st.spinner("Training ARIMA model..."):
            model = ARIMA(train_data, order=(5, 1, 2))
            fitted_model = model.fit()
        
        # Predictions
        y_pred = fitted_model.forecast(steps=len(test_data))
        future_pred = fitted_model.forecast(steps=forecast_days)
        
        return test_data, y_pred, future_pred
    
    except Exception as e:
        st.error(f"ARIMA model error: {str(e)}")
        return None, None, None

def train_sarima_model(data, forecast_days=30):
    """Train SARIMA model"""
    if not STATSMODELS_AVAILABLE:
        return None, None, None
    
    try:
        ts_data = data['Demand'].values
        split_idx = int(len(ts_data) * 0.8)
        train_data = ts_data[:split_idx]
        test_data = ts_data[split_idx:]
        
        with st.spinner("Training SARIMA model..."):
            model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            fitted_model = model.fit(disp=False)
        
        y_pred = fitted_model.forecast(steps=len(test_data))
        future_pred = fitted_model.forecast(steps=forecast_days)
        
        return test_data, y_pred, future_pred
    
    except Exception as e:
        st.error(f"SARIMA model error: {str(e)}")
        return None, None, None

def train_prophet_model(data, forecast_days=30):
    """Train Prophet model and return aligned y_test, y_pred, and future forecast."""
    if not PROPHET_AVAILABLE:
        return None, None, None

    try:
        # Prepare Prophet-friendly data
        prophet_data = data[['Date', 'Demand']].copy()
        prophet_data.columns = ['ds', 'y']

        split_idx = int(len(prophet_data) * 0.8)
        train_data = prophet_data.iloc[:split_idx]
        test_data = prophet_data.iloc[split_idx:]

        # Fit the model
        with st.spinner("Training Prophet model..."):
            model = Prophet(daily_seasonality=True, yearly_seasonality=True)
            model.fit(train_data)

        # Predict future
        future_dates = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future_dates)

        # --- ALIGN predicted and actual values on 'ds' ---
        test_merged = test_data.merge(forecast[['ds', 'yhat']], on='ds', how='inner')

        # Now y_test and y_pred are aligned by date
        y_test = test_merged['y'].values
        y_pred = test_merged['yhat'].values

        # Get future predictions
        future_pred = forecast.iloc[-forecast_days:]['yhat'].values

        return y_test, y_pred, future_pred

    except Exception as e:
        st.error(f"Prophet model error: {str(e)}")
        return None, None, None


def train_sklearn_model(data, model_type, forecast_days=30):
    """Train sklearn models"""
    try:
        # Create features
        df_features = data.copy()
        df_features['Year'] = df_features['Date'].dt.year
        df_features['Month'] = df_features['Date'].dt.month
        df_features['DayOfYear'] = df_features['Date'].dt.dayofyear
        df_features['DayOfWeek'] = df_features['Date'].dt.dayofweek
        
        # Lag features
        for lag in [1, 7, 30]:
            df_features[f'Demand_lag_{lag}'] = df_features['Demand'].shift(lag)
        
        # Rolling features
        df_features['Demand_rolling_7'] = df_features['Demand'].rolling(7).mean()
        df_features['Demand_rolling_30'] = df_features['Demand'].rolling(30).mean()
        
        df_features = df_features.dropna()
        
        feature_cols = ['Year', 'Month', 'DayOfYear', 'DayOfWeek',
                       'Demand_lag_1', 'Demand_lag_7', 'Demand_lag_30',
                       'Demand_rolling_7', 'Demand_rolling_30']
        
        X = df_features[feature_cols]
        y = df_features['Demand']
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Future predictions (simplified)
        last_values = X.iloc[-1:].copy()
        future_pred = []
        
        for i in range(forecast_days):
            pred = model.predict(last_values)[0]
            future_pred.append(pred)
            
            # Update lag features for next prediction
            last_values.iloc[0, last_values.columns.get_loc('Demand_lag_1')] = pred
            if i >= 6:
                last_values.iloc[0, last_values.columns.get_loc('Demand_lag_7')] = future_pred[i-6]
            if i >= 29:
                last_values.iloc[0, last_values.columns.get_loc('Demand_lag_30')] = future_pred[i-29]
        
        return y_test.values, y_pred, np.array(future_pred)
    
    except Exception as e:
        st.error(f"Sklearn model error: {str(e)}")
        return None, None, None

# Train selected model
if len(df_ts) > 10:
    y_test, y_pred, future_pred = None, None, None
    
    if model_type == "LSTM":
        y_test, y_pred, future_pred, scaler = train_lstm_model(df_ts, forecast_days)
    elif model_type == "ARIMA":
        y_test, y_pred, future_pred = train_arima_model(df_ts, forecast_days)
    elif model_type == "SARIMA":
        y_test, y_pred, future_pred = train_sarima_model(df_ts, forecast_days)
    elif model_type == "Prophet":
        y_test, y_pred, future_pred = train_prophet_model(df_ts, forecast_days)
    else:
        y_test, y_pred, future_pred = train_sklearn_model(df_ts, model_type, forecast_days)
    
    if y_test is not None and y_pred is not None:
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Handle potential division by zero
        if np.mean(y_test) != 0:
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            accuracy = max(0, 100 - mape)
        else:
            mape = 0
            accuracy = 0
        
        # Display metrics
        st.subheader("📊 Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RMSE", f"{rmse:.2f}")
        with col2:
            st.metric("MAE", f"{mae:.2f}")
        with col3:
            st.metric("MAPE", f"{mape:.2f}%")
        with col4:
            st.metric("Accuracy", f"{accuracy:.1f}%")
        
        # Visualizations
        st.subheader("📈 Forecasting Visualizations")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Data & Smoothing", "Actual vs Predicted", "Future Forecast", "Model Analysis"])
        
        with tab1:
            # Show original and smoothed data
            fig = go.Figure()
            
            # Original data
            original_demand = df_ts['Demand'] if selected_smooth == 'Original' else df_smoothed['Demand']
            fig.add_trace(go.Scatter(
                x=df_ts['Date'],
                y=original_demand,
                name='Original Data',
                line=dict(color='blue', width=1)
            ))
            
            # Add smoothed series
            if smoothing_methods:
                colors = ['red', 'green', 'orange', 'purple']
                for i, method in enumerate(smoothing_methods):
                    if method == "Moving Average" and 'MA_Smoothed' in df_smoothed.columns:
                        fig.add_trace(go.Scatter(
                            x=df_smoothed['Date'],
                            y=df_smoothed['MA_Smoothed'],
                            name=f'Moving Average ({ma_window})',
                            line=dict(color=colors[i % len(colors)], width=2)
                        ))
                    elif method == "Exponential Smoothing" and 'EXP_Smoothed' in df_smoothed.columns:
                        fig.add_trace(go.Scatter(
                            x=df_smoothed['Date'],
                            y=df_smoothed['EXP_Smoothed'],
                            name=f'Exponential Smoothing (α={exp_alpha})',
                            line=dict(color=colors[i % len(colors)], width=2)
                        ))
                    elif method == "Savitzky-Golay Filter" and 'SAVGOL_Smoothed' in df_smoothed.columns:
                        fig.add_trace(go.Scatter(
                            x=df_smoothed['Date'],
                            # The code continues from the cut-off section for visualizing Savitzky-Golay Filter and beyond

y = df_smoothed['SAVGOL_Smoothed'],
name='Savitzky-Golay Filter',
line=dict(color=colors[i % len(colors)], width=2)
))
elif method == "Gaussian Filter" and 'GAUSSIAN_Smoothed' in df_smoothed.columns:
    fig.add_trace(go.Scatter(
        x=df_smoothed['Date'],
        y=df_smoothed['GAUSSIAN_Smoothed'],
        name='Gaussian Filter',
        line=dict(color=colors[i % len(colors)], width=2)
    ))

fig.update_layout(title="Demand Smoothing Visualization",
                  xaxis_title="Date",
                  yaxis_title="Demand",
                  template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=y_test,
        name='Actual',
        mode='lines'))
    fig2.add_trace(go.Scatter(
        y=y_pred,
        name='Predicted',
        mode='lines'))
    fig2.update_layout(title="Actual vs Predicted Demand",
                       xaxis_title="Time",
                       yaxis_title="Demand",
                       template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    forecast_dates = pd.date_range(df_ts['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=forecast_dates, y=future_pred, name="Forecast", line=dict(color='orange')))
    fig3.update_layout(title="Future Forecast",
                       xaxis_title="Date",
                       yaxis_title="Forecasted Demand",
                       template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.markdown('<div class="forecast-summary">', unsafe_allow_html=True)
    st.markdown(f"### 🔍 Model: {model_type}")
    st.markdown(f"- **RMSE**: {rmse:.2f}")
    st.markdown(f"- **MAE**: {mae:.2f}")
    st.markdown(f"- **MAPE**: {mape:.2f}%")
    st.markdown(f"- **Accuracy**: {accuracy:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

st.warning("📉 Not enough data to forecast. Please upload more complete dataset.")
