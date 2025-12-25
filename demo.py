import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import urllib.request
import json

# Azure ML Endpoint Config
AZURE_ML_URL = 'https://churn-prediction-dwwwt.centralus.inference.ml.azure.com/score'
AZURE_API_KEY = 'EzNClNF4E9Wu30gN5fe7kP5CdYdxmrFivALxuP5OkKy2Yk7ygdnmJQQJ99BLAAAAAAAAAAAAINFRAZML3ANj'

def get_predictions(df):
    """Send data to Azure ML endpoint and get predictions"""
    try:
        # Send as JSON records directly (no wrapper)
        data = df.to_dict(orient='records')
        
        body = str.encode(json.dumps(data))
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {AZURE_API_KEY}'
        }
        
        req = urllib.request.Request(AZURE_ML_URL, body, headers)
        response = urllib.request.urlopen(req)
        result = response.read().decode('utf-8')
        
        # Handle double JSON encoding - keep parsing until we get a dict/list
        while isinstance(result, str):
            result = json.loads(result)
        
        return result, None
    except urllib.error.HTTPError as error:
        error_msg = f"Error {error.code}: {error.read().decode('utf8', 'ignore')}"
        return None, error_msg
    except Exception as e:
        return None, str(e)

def assign_intervention(probability):
    """Assign intervention level based on probability"""
    if probability >= 0.6:
        return 'Immediate Offer'
    elif probability >= 0.4:
        return 'Email Campaign'
    else:
        return 'Normal Customer Service'

# Page config
st.set_page_config(
    page_title="Customer Risk Dashboard",
    page_icon="📊",
    layout="wide"
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #1a1a2e;
    }
    .main-header {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #16213e;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .card-title {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .metric-value {
        color: #4ecca3;
        font-size: 2rem;
        font-weight: bold;
    }
    div[data-testid="stDataFrame"] {
        background-color: #16213e;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">📊 Customer Risk Dashboard</p>', unsafe_allow_html=True)

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    st.markdown("---")
    st.markdown("### 🔧 How it works:")
    st.markdown("1. Upload your customer CSV")
    st.markdown("2. Click 'Get Predictions'")
    st.markdown("3. View churn probabilities")
    
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.session_state['input_data'] = df_input
        st.success(f"✅ Loaded {len(df_input)} rows")
        
        if st.button("🚀 Get Predictions", type="primary", use_container_width=True):
            with st.spinner("Calling Azure ML endpoint..."):
                result, error = get_predictions(df_input)
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.success("✅ Predictions received!")
                    
                    # Parse the response - your API returns {"predictions": [...], "probabilities": [...]}
                    probabilities = None
                    
                    if isinstance(result, dict):
                        # Your exact format: {"predictions": [...], "probabilities": [...]}
                        if "probabilities" in result:
                            probabilities = result["probabilities"]
                        elif "predictions" in result:
                            probabilities = result["predictions"]
                        else:
                            # Try other common keys
                            for key in ['scores', 'result', 'output']:
                                if key in result:
                                    probabilities = result[key]
                                    break
                    elif isinstance(result, list):
                        probabilities = result
                    
                    # Last resort
                    if probabilities is None:
                        st.error("❌ Could not parse probabilities from response")
                        st.stop()
                    
                    # Validate length
                    if len(probabilities) != len(df_input):
                        st.error(f"❌ Length mismatch: Got {len(probabilities)} predictions for {len(df_input)} rows")
                        st.warning("Check the debug output above to see the raw response format")
                        st.stop()
                    
                    # Build results dataframe
                    results_df = pd.DataFrame({
                        'customer_id': df_input.iloc[:, 0] if len(df_input.columns) > 0 else range(len(probabilities)),
                        'probability': probabilities
                    })
                    results_df['intervention'] = results_df['probability'].apply(assign_intervention)
                    
                    st.session_state['data'] = results_df

# Main content
if 'data' in st.session_state:
    df = st.session_state['data']
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Total Customers", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        avg_prob = df['probability'].mean() if 'probability' in df.columns else 0
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Avg Probability", f"{avg_prob:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        high_risk = len(df[df['probability'] >= 0.6]) if 'probability' in df.columns else 0
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Immediate Offer", high_risk)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        email_campaign = len(df[(df['probability'] >= 0.4) & (df['probability'] < 0.6)]) if 'probability' in df.columns else 0
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Email Campaign", email_campaign)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Two columns: Table and Pie Chart
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        st.markdown("### 📋 Customer Predictions")
        
        # Style the dataframe
        display_df = df.copy()
        if 'probability' in display_df.columns:
            display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name="customer_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with right_col:
        st.markdown("### 🎯 Intervention Distribution")
        
        if 'intervention' in df.columns:
            intervention_counts = df['intervention'].value_counts()
            
            colors = ['#4ecca3', '#ff6b6b', '#ffd93d', '#6bcbff', '#c44dff']
            
            fig = go.Figure(data=[go.Pie(
                labels=intervention_counts.index,
                values=intervention_counts.values,
                hole=0.5,
                marker_colors=colors[:len(intervention_counts)],
                textinfo='percent',
                textfont_size=14,
                textfont_color='black',
                textposition='inside'
            )])
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                showlegend=True,
                legend=dict(
                    font=dict(color='white'),
                    bgcolor='rgba(0,0,0,0)'
                ),
                height=350,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No 'intervention' column found")

else:
    # Empty state
    st.markdown("""
    <div style="text-align: center; padding: 100px; color: #666;">
        <h2> Upload a CSV file to get started</h2>
     
    </div>
    """, unsafe_allow_html=True)
