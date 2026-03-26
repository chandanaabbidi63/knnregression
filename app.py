import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Prediction Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    :root {
        --primary-color: #6366f1;
        --secondary-color: #ec4899;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.95) !important;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 10px;
    }
    
    h1, h2, h3 {
        color: #fff;
        font-weight: 700;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(99, 102, 241, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv('task1_dataset.csv')
    return df

# Preprocess data
@st.cache_resource
def preprocess_data(df):
    df_processed = df.copy()
    
    # Handle missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Encode categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in categorical_cols:
        if col != 'date':
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            le_dict[col] = le
    
    return df_processed, le_dict

# Train Random Forest model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    return model, scaler, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test

# Header with animations
st.markdown("""
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%); 
    border-radius: 15px; margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
        <h1 style='font-size: 2.5em; margin: 0; color: white;'>🤖 Advanced ML Prediction Dashboard</h1>
        <p style='color: rgba(255,255,255,0.9); font-size: 1.1em; margin: 10px 0 0 0;'>
            Random Forest Model with Real-time Analytics
        </p>
    </div>
""", unsafe_allow_html=True)

# Main content
try:
    # Load data
    df = load_data()
    
    # Sidebar configuration
    st.sidebar.markdown("## 📊 Configuration")
    show_raw_data = st.sidebar.checkbox("📈 Show Raw Data", value=False)
    
    if show_raw_data:
        st.sidebar.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        with st.expander("🔍 View Raw Data"):
            st.dataframe(df.head(10), width='stretch')
    
    # Preprocess data
    df_processed, le_dict = preprocess_data(df)
    
    # Separate features and target
    X = df_processed.drop(['target', 'date'], axis=1, errors='ignore')
    y = df_processed['target']
    
    # Train model
    model, scaler, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = train_model(X, y)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Performance", "🎯 Predictions", "📈 Feature Importance", "🔮 Make Predictions"])
    
    # Tab 1: Model Performance
    with tab1:
        st.markdown("### Model Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📌 R² Score (Test)", f"{test_r2:.4f}", f"{(test_r2-train_r2)*100:.2f}%")
        with col2:
            st.metric("📍 RMSE (Test)", f"${test_rmse:.2f}", f"-${test_rmse*100/np.mean(y_test):.2f}%")
        with col3:
            st.metric("🎯 MAE (Test)", f"${test_mae:.2f}", f"-${test_mae*100/np.mean(y_test):.2f}%")
        
        # Animated prediction scatter plot
        st.markdown("### Actual vs Predicted Values")
        
        # Create animated scatter plot with Plotly
        fig = go.Figure()
        
        # Add test data points with animation
        fig.add_trace(go.Scatter(
            x=y_test.values,
            y=y_pred_test,
            mode='markers',
            marker=dict(
                size=8,
                color=np.abs(y_test.values - y_pred_test),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Error"),
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=[f"Actual: ${v:.2f}<br>Predicted: ${p:.2f}<br>Error: ${abs(v-p):.2f}" 
                  for v, p in zip(y_test.values, y_pred_test)],
            hovertemplate='%{text}<extra></extra>',
            name='Test Predictions'
        ))
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', width=2, dash='dash'),
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title='Actual vs Predicted Test Values (with Animation)',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            hovermode='closest',
            height=500,
            template='plotly_dark',
            font=dict(color='white'),
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Residuals analysis with animation
        st.markdown("### Residuals Distribution")
        residuals = y_test.values - y_pred_test
        
        fig_residuals = go.Figure()
        
        fig_residuals.add_trace(go.Histogram(
            x=residuals,
            nbinsx=30,
            marker=dict(
                color='#6366f1',
                opacity=0.7,
                line=dict(color='white', width=1)
            ),
            name='Residuals',
            hovertemplate='Residuals Range: %{x}<br>Frequency: %{y}<extra></extra>'
        ))
        
        fig_residuals.update_layout(
            title='Distribution of Prediction Residuals',
            xaxis_title='Residual Value',
            yaxis_title='Frequency',
            height=400,
            template='plotly_dark',
            font=dict(color='white'),
            showlegend=False
        )
        
        st.plotly_chart(fig_residuals, width='stretch')
    
    # Tab 2: Predictions Visualization
    with tab2:
        st.markdown("### Predictions Timeline")
        
        # Create a dataset with predictions
        predictions_df = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': y_pred_test,
            'Index': range(len(y_test))
        })
        
        # Animated line plot
        fig_timeline = go.Figure()
        
        fig_timeline.add_trace(go.Scatter(
            x=predictions_df['Index'],
            y=predictions_df['Actual'],
            mode='lines',
            name='Actual',
            line=dict(color='#10b981', width=2),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.2)',
            hovertemplate='Index: %{x}<br>Actual: $%{y:.2f}<extra></extra>'
        ))
        
        fig_timeline.add_trace(go.Scatter(
            x=predictions_df['Index'],
            y=predictions_df['Predicted'],
            mode='lines',
            name='Predicted',
            line=dict(color='#ec4899', width=2, dash='dash'),
            fill='tozeroy',
            fillcolor='rgba(236, 72, 153, 0.2)',
            hovertemplate='Index: %{x}<br>Predicted: $%{y:.2f}<extra></extra>'
        ))
        
        fig_timeline.update_layout(
            title='Actual vs Predicted Values Over Time',
            xaxis_title='Sample Index',
            yaxis_title='Value',
            height=500,
            template='plotly_dark',
            font=dict(color='white'),
            hovermode='x unified',
        )
        
        st.plotly_chart(fig_timeline, width='stretch')
    
    # Tab 3: Feature Importance
    with tab3:
        st.markdown("### Feature Importance Analysis")
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig_importance = go.Figure(data=[
            go.Bar(
                x=feature_importance['Importance'],
                y=feature_importance['Feature'],
                orientation='h',
                marker=dict(
                    color=feature_importance['Importance'],
                    colorscale='Plasma',
                    showscale=False,
                    line=dict(color='white', width=1)
                ),
                text=[f"{v:.4f}" for v in feature_importance['Importance']],
                textposition='auto',
                hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
            )
        ])
        
        fig_importance.update_layout(
            title='Top 10 Most Important Features',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=500,
            template='plotly_dark',
            font=dict(color='white'),
            showlegend=False
        )
        
        st.plotly_chart(fig_importance, width='stretch')
    
    # Tab 4: Make Predictions
    with tab4:
        st.markdown("### Enter Data to Make Predictions")
        
        # Create input form
        with st.form("prediction_form"):
            cols = st.columns(3)
            input_data = {}
            
            for idx, col_name in enumerate(X.columns):
                col = cols[idx % 3]
                
                if X[col_name].dtype in [np.int64, np.float64]:
                    min_val = X[col_name].min()
                    max_val = X[col_name].max()
                    mean_val = X[col_name].mean()
                    
                    input_data[col_name] = col.slider(
                        f"📊 {col_name}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(mean_val),
                        step=(max_val - min_val) / 100
                    )
            
            submit_button = st.form_submit_button("🔮 Make Prediction", width='stretch')
        
        if submit_button:
            # Prepare input
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display prediction with animation
            st.success(f"### 🎯 Predicted Value: **${prediction:,.2f}**")
            
            # Create gauge chart for prediction
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Prediction Value"},
                delta={'reference': y_test.mean()},
                gauge={
                    'axis': {'range': [y.min(), y.max()]},
                    'bar': {'color': "#6366f1"},
                    'steps': [
                        {'range': [y.min(), y.quantile(0.33)], 'color': "#ef4444"},
                        {'range': [y.quantile(0.33), y.quantile(0.66)], 'color': "#f59e0b"},
                        {'range': [y.quantile(0.66), y.max()], 'color': "#10b981"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': y_test.mean()
                    }
                }
            ))
            
            fig_gauge.update_layout(
                height=400,
                template='plotly_dark',
                font=dict(color='white'),
            )
            
            st.plotly_chart(fig_gauge, width='stretch')
    
    # Footer
    st.markdown("""
        ---
        <div style='text-align: center; padding: 20px; color: rgba(255,255,255,0.7);'>
            <p>🚀 Built with Streamlit | 🤖 Powered by Random Forest | 📊 Data Analytics Dashboard</p>
            <p style='font-size: 0.9em;'>© 2024 ML Prediction System</p>
        </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"❌ Error loading or processing data: {str(e)}")
    st.info("Make sure task1_dataset.csv is in the same directory as this app.")
