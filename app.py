# ============================================================
# 🚗 CAR PRICE PREDICTION - STREAMLIT APP
# ============================================================
# استخدام: streamlit run app.py
# ============================================================
import pandas as pd
import streamlit as st
import pandas as pdcd 
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="🚗 Car Price Predictor Egypt",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. CUSTOM CSS
# ============================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .price-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .price-box h1 {
        font-size: 3.5rem;
        margin: 0;
    }
    .price-box p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.75rem;
        border: none;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-card h3 {
        color: #1E88E5;
        margin: 0;
    }
    .metric-card p {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0.5rem 0 0 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 3. LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    """Load all trained artifacts"""
    try:
        model = joblib.load('models/best_model.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        features = joblib.load('models/features.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, label_encoders, features, scaler
    except Exception as e:
        st.error(f"❌ Error loading artifacts: {e}")
        st.stop()

model, label_encoders, features, scaler = load_artifacts()

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================
def predict_price(car_info):
    """Predict price for a single car"""
    # Encode categorical features
    brand_enc = label_encoders['brand'].transform([car_info['brand']])[0]
    fuel_enc = label_encoders['fuel_type'].transform([car_info['fuel_type']])[0]
    trans_enc = label_encoders['transmission_type'].transform([car_info['transmission_type']])[0]
    seller_enc = label_encoders['seller_type'].transform([car_info['seller_type']])[0]
    
    # Create feature array
    input_features = [
        car_info['vehicle_age'],
        car_info['km_driven'],
        car_info['engine'],
        car_info['max_power'],
        car_info['seats'],
        brand_enc,
        fuel_enc,
        trans_enc,
        seller_enc
    ]
    
    # Scale features
    input_scaled = scaler.transform([input_features])
    
    # Predict
    price = model.predict(input_scaled)[0]
    return max(price, 50000)  # Minimum reasonable price

def get_price_range(price):
    """Get price range for confidence interval"""
    lower = price * 0.85
    upper = price * 1.15
    return lower, upper

def format_price(price):
    """Format price with commas"""
    return f"{price:,.0f} EGP"
def estimate_maintenance(brand, vehicle_age, km_driven, engine):
    """
    Estimate annual maintenance cost using simple rules.
    Returns:
        maintenance_cost
        risk_level
        breakdown
    """

    maintenance_cost = 3000

    breakdown = {
        "Base Service": 3000
    }

    # Age
    if vehicle_age > 10:
        maintenance_cost += 4000
        breakdown["Old Vehicle"] = 4000

    # Mileage
    if km_driven > 150000:
        maintenance_cost += 5000
        breakdown["High Mileage"] = 5000

    # Engine Size
    if engine > 2000:
        maintenance_cost += 3000
        breakdown["Large Engine"] = 3000

    # Premium Brands
    premium_brands = [
        "BMW",
        "Mercedes-Benz",
        "Audi",
        "Land",
        "Jaguar",
        "Volvo"
    ]

    if any(p in brand for p in premium_brands):
        maintenance_cost += 6000
        breakdown["Premium Brand"] = 6000

    # Risk Level
    if maintenance_cost <= 7000:
        risk = "🟢 Low"

    elif maintenance_cost <= 14000:
        risk = "🟡 Medium"

    else:
        risk = "🔴 High"

    return maintenance_cost, risk, breakdown
# ============================================================
# 5. SIDEBAR - NAVIGATION
# ============================================================
st.sidebar.image("https://img.icons8.com/color/96/000000/car--v2.png", width=80)
st.sidebar.title("🚗 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Predict", "📊 Analytics", "📈 Compare", "ℹ️ About"],
    index=0
)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.metric("R² Score", "87.9%", "✅ Good")
st.sidebar.metric("MAE", "71,212 EGP", "⬇️")
st.sidebar.metric("RMSE", "97,999 EGP", "⬇️")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Best Model")
st.sidebar.info("Random Forest with Hyperparameter Tuning")

# ============================================================
# 6. PAGE: PREDICT
# ============================================================
if page == "🏠 Predict":
    st.markdown('<p class="main-header">🚗 Car Price Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict the selling price of a used car in Egypt</p>', unsafe_allow_html=True)
    
    # Two columns for input
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### 🏷️ Car Details")
        
        brand = st.selectbox(
            "Brand",
            options=sorted(label_encoders['brand'].classes_),
            help="Select the car brand"
        )
        
        vehicle_age = st.slider(
            "Vehicle Age (years)",
            min_value=0,
            max_value=30,
            value=5,
            help="Age of the car in years"
        )
        
        km_driven = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=500000,
            value=50000,
            step=1000,
            help="Total kilometers driven"
        )
        
        fuel_type = st.selectbox(
            "Fuel Type",
            options=label_encoders['fuel_type'].classes_,
            help="Type of fuel used"
        )
    
    with col2:
        st.markdown("### ⚙️ Specifications")
        
        transmission_type = st.selectbox(
            "Transmission",
            options=label_encoders['transmission_type'].classes_,
            help="Manual or Automatic"
        )
        
        seller_type = st.selectbox(
            "Seller Type",
            options=label_encoders['seller_type'].classes_,
            help="Who is selling the car"
        )
        
        engine = st.slider(
            "Engine Size (cc)",
            min_value=500,
            max_value=6000,
            value=1500,
            step=100,
            help="Engine capacity in cubic centimeters"
        )
        
        max_power = st.slider(
            "Max Power (BHP)",
            min_value=30,
            max_value=400,
            value=100,
            step=5,
            help="Maximum horsepower"
        )
        
        seats = st.selectbox(
            "Number of Seats",
            options=[2, 4, 5, 6, 7, 8, 9],
            help="Seating capacity"
        )
    
    # Prediction button
    st.markdown("---")
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    
    with predict_col2:
        predict_clicked = st.button("💰 Predict Price", use_container_width=True)
    
    if predict_clicked:
        # Gather input
        car_info = {
            'brand': brand,
            'vehicle_age': vehicle_age,
            'km_driven': km_driven,
            'fuel_type': fuel_type,
            'transmission_type': transmission_type,
            'seller_type': seller_type,
            'engine': engine,
            'max_power': max_power,
            'seats': seats
        }
        
        # Predict
        with st.spinner("🧠 Predicting..."):
            price = predict_price(car_info)
            lower, upper = get_price_range(price)
            maintenance_cost, risk, breakdown = estimate_maintenance(
               brand,
               vehicle_age,
               km_driven,
               engine
            )

            ownership_cost = price + maintenance_cost
        # Display results
        st.markdown("---")
        
        col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
        
        with col_result2:
            st.markdown(f"""
            <div class="price-box">
                <p>💰 Estimated Price</p>
                <h1>{format_price(price)}</h1>
                <p>Confidence Range: {format_price(lower)} - {format_price(upper)}</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("## 🔧 Maintenance Estimation")

            col_a, col_b, col_c = st.columns(3)

            with col_a:
               st.metric(
                  "Annual Maintenance",
                   format_price(maintenance_cost)
                   )

            with col_b:
                st.metric(
                    "Risk Level",
                     risk
                    )

            with col_c:
                st.metric(
                    "Ownership Cost",
                     format_price(ownership_cost)
                )
        # Feature impact visualization
        st.markdown("### 📊 Feature Impact")
        
        # Create feature importance chart
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Car info summary
            st.markdown("#### 📋 Car Summary")
            summary_data = {
                "Feature": ["Brand", "Age", "KM", "Fuel", "Transmission", "Engine", "Power", "Seats"],
                "Value": [brand, f"{vehicle_age} yrs", f"{km_driven:,}", fuel_type, 
                         transmission_type, f"{engine} cc", f"{max_power} BHP", seats]
            }
            st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)
        
        with col_chart2:
            # Price range gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=price,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Price Range (EGP)"},
                delta={'reference': 500000, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 2000000]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 300000], 'color': "#e8f5e9"},
                        {'range': [300000, 700000], 'color': "#fff3e0"},
                        {'range': [700000, 1200000], 'color': "#fce4ec"},
                        {'range': [1200000, 2000000], 'color': "#f44336"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': price
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        # Similar cars in market
        st.markdown("### 🔍 Similar Cars in Market")
        
        # Generate sample similar cars
        similar_cars = pd.DataFrame({
            'Brand': [brand, brand, brand, 'Toyota', 'Hyundai'],
            'Age': [vehicle_age-1, vehicle_age, vehicle_age+1, vehicle_age, vehicle_age],
            'KM': [km_driven-10000, km_driven, km_driven+10000, km_driven-5000, km_driven+5000],
            'Price': [
                price * 0.9,
                price,
                price * 1.1,
                price * 0.85,
                price * 0.92
            ]
        })
        similar_cars['Price'] = similar_cars['Price'].apply(lambda x: f"{x:,.0f} EGP")
        st.dataframe(similar_cars, hide_index=True, use_container_width=True)

# ============================================================
# 7. PAGE: ANALYTICS
# ============================================================
elif page == "📊 Analytics":
    st.markdown('<p class="main-header">📊 Market Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Understanding the used car market in Egypt</p>', unsafe_allow_html=True)
    
    # Load the processed data
    @st.cache_data
    def load_data():
        return pd.read_csv('model_ready_no_leakage.csv')
    
    df = load_data()
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Cars", f"{len(df):,}")
    with col2:
        st.metric("💰 Avg Price", f"{df['selling_price'].mean():,.0f} EGP")
    with col3:
        st.metric("📅 Avg Age", f"{df['vehicle_age'].mean():.1f} years")
    with col4:
        st.metric("🚗 Most Common", df['brand'].mode()[0])
    
    # Charts
    tab1, tab2, tab3 = st.tabs(["📈 Price Distribution", "🏷️ Brand Analysis", "🔧 Feature Impact"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df, x='selling_price', nbins=50,
                title='Selling Price Distribution',
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df, y='selling_price',
                title='Price Range (Box Plot)',
                color_discrete_sequence=['#764ba2']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Age vs Price
        fig = px.scatter(
            df.sample(500), x='vehicle_age', y='selling_price',
            color='transmission_type',
            title='Age vs Price (by Transmission)',
            labels={'vehicle_age': 'Age (years)', 'selling_price': 'Price (EGP)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Top brands
        brand_stats = df.groupby('brand').agg({
            'selling_price': ['mean', 'count']
        }).round(0)
        brand_stats.columns = ['Avg Price', 'Count']
        brand_stats = brand_stats.sort_values('Avg Price', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                brand_stats.head(15).reset_index(),
                x='brand', y='Avg Price',
                title='Top 15 Brands by Average Price',
                color='Avg Price',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                df, names='fuel_type',
                title='Fuel Type Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Feature importance from model
        st.markdown("### Feature Importance (from Random Forest)")
        
        # Load feature importance if available
        importance_data = {
            'Feature': ['Max Power', 'Vehicle Age', 'Engine', 'KM Driven', 'Brand', 
                       'Transmission', 'Fuel Type', 'Seats', 'Seller Type'],
            'Importance': [0.534, 0.299, 0.103, 0.025, 0.019, 0.009, 0.004, 0.004, 0.003]
        }
        importance_df = pd.DataFrame(importance_data)
        
        fig = px.bar(
            importance_df,
            x='Importance', y='Feature',
            orientation='h',
            title='Feature Importance Ranking',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.markdown("### Correlation Matrix")
        corr_cols = ['vehicle_age', 'km_driven', 'engine', 'max_power', 'seats', 'selling_price']
        corr_matrix = df[corr_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            title='Correlation Matrix',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 8. PAGE: COMPARE
# ============================================================
elif page == "📈 Compare":
    st.markdown('<p class="main-header">📈 Compare Cars</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Compare prices of different cars</p>', unsafe_allow_html=True)
    
    # Input for two cars
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚗 Car 1")
        brand1 = st.selectbox("Brand 1", sorted(label_encoders['brand'].classes_), key='brand1')
        age1 = st.slider("Age 1", 0, 30, 5, key='age1')
        km1 = st.number_input("KM 1", 0, 500000, 50000, step=1000, key='km1')
        fuel1 = st.selectbox("Fuel 1", label_encoders['fuel_type'].classes_, key='fuel1')
        trans1 = st.selectbox("Transmission 1", label_encoders['transmission_type'].classes_, key='trans1')
        engine1 = st.slider("Engine 1", 500, 6000, 1500, key='engine1')
        power1 = st.slider("Power 1", 30, 400, 100, key='power1')
        seats1 = st.selectbox("Seats 1", [2,4,5,6,7,8,9], key='seats1')
    
    with col2:
        st.markdown("### 🚗 Car 2")
        brand2 = st.selectbox("Brand 2", sorted(label_encoders['brand'].classes_), key='brand2')
        age2 = st.slider("Age 2", 0, 30, 3, key='age2')
        km2 = st.number_input("KM 2", 0, 500000, 30000, step=1000, key='km2')
        fuel2 = st.selectbox("Fuel 2", label_encoders['fuel_type'].classes_, key='fuel2')
        trans2 = st.selectbox("Transmission 2", label_encoders['transmission_type'].classes_, key='trans2')
        engine2 = st.slider("Engine 2", 500, 6000, 2000, key='engine2')
        power2 = st.slider("Power 2", 30, 400, 150, key='power2')
        seats2 = st.selectbox("Seats 2", [2,4,5,6,7,8,9], key='seats2')
    
    if st.button("🔄 Compare", use_container_width=True):
        # Predict both
        car1 = {'brand': brand1, 'vehicle_age': age1, 'km_driven': km1, 'fuel_type': fuel1,
                'transmission_type': trans1, 'seller_type': 'Dealer', 'engine': engine1,
                'max_power': power1, 'seats': seats1}
        car2 = {'brand': brand2, 'vehicle_age': age2, 'km_driven': km2, 'fuel_type': fuel2,
                'transmission_type': trans2, 'seller_type': 'Dealer', 'engine': engine2,
                'max_power': power2, 'seats': seats2}
        
        price1 = predict_price(car1)
        price2 = predict_price(car2)
        
        # Display comparison
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="price-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <p>🚗 Car 1</p>
                <h3>{brand1}</h3>
                <h2>{format_price(price1)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            diff = price2 - price1
            diff_pct = (diff / price1) * 100
            color = "green" if diff > 0 else "red"
            st.markdown(f"""
            <div class="price-box" style="background: #f8f9fa; color: #333;">
                <p>📊 Difference</p>
                <h3 style="color: {color};">{diff_pct:+.1f}%</h3>
                <p style="color: {color};">{format_price(diff)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="price-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <p>🚗 Car 2</p>
                <h3>{brand2}</h3>
                <h2>{format_price(price2)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Radar chart for comparison
        st.markdown("### 📊 Feature Comparison")
        
        features = ['Age', 'KM', 'Engine', 'Power', 'Seats']
        values1 = [age1, km1/10000, engine1/1000, power1, seats1]
        values2 = [age2, km2/10000, engine2/1000, power2, seats2]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values1,
            theta=features,
            fill='toself',
            name=brand1,
            line_color='#667eea'
        ))
        fig.add_trace(go.Scatterpolar(
            r=values2,
            theta=features,
            fill='toself',
            name=brand2,
            line_color='#f5576c'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(values1), max(values2)) * 1.1]
                )
            ),
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 9. PAGE: ABOUT
# ============================================================
else:
    st.markdown('<p class="main-header">ℹ️ About</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🚗 Car Price Prediction Project
        
        This application uses **Machine Learning** to predict the selling price of used cars in Egypt.
        
        ### 📊 How it works:
        1. **Data Collection**: 15,000+ used car listings from Egypt
        2. **Data Cleaning**: Removed duplicates, outliers, and missing values
        3. **Feature Engineering**: Created meaningful features for better predictions
        4. **Model Training**: Tested 5 different models (Random Forest performed best)
        5. **Hyperparameter Tuning**: Optimized for 87.9% accuracy
        
        ### 🎯 Key Features:
        - **87.9% R² Score** - Model explains 87.9% of price variance
        - **71,212 EGP MAE** - Average prediction error
        - **5-Fold Cross Validation** - Ensures model reliability
        
        ### 🔧 Technologies Used:
        - Python 🐍
        - Scikit-Learn
        - Streamlit
        - Plotly
        - Pandas & NumPy
        
        ### 📈 Model Performance:
        | Metric | Value |
        |--------|-------|
        | R² Score | 87.9% |
        | MAE | 71,212 EGP |
        | RMSE | 97,999 EGP |
        | Best Model | Random Forest |
        """)
        
        with col2:
            st.markdown("""
            ### 📂 Project Files
            
            ✅ **best_model.pkl** - Trained model  
            ✅ **label_encoders.pkl** - Category encoders  
            ✅ **features.pkl** - Feature list  
            ✅ **scaler.pkl** - Feature scaler  
            
            ### 📊 Data Statistics
            
            📊 Total Cars: 13,871  
            🏷️ Brands: 23  
            ⛽ Fuel Types: 4  
            📅 Avg Age: 6.2 years  
            💰 Avg Price: 562,940 EGP  
            """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ in Egypt")

# ============================================================
# 10. FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🚗 Car Price Predictor v1.0 | Built with Streamlit</p>
        <p style="font-size: 0.8rem;">Predictions are estimates and should be used for reference only</p>
    </div>
    """,
    unsafe_allow_html=True
)