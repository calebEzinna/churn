import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv('customer_churn_dataset-training-master.csv')
    return data

# Load data
data = load_data()

# Remove null values
data.dropna(inplace=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Predictive Model", "Revenue Impact of Churn"])

# Download button in sidebar
st.sidebar.download_button(
    label="Download Full Dataset",
    data=data.to_csv(index=False).encode('utf-8'),
    file_name='customer_churn_dataset-training-master.csv',
    mime='text/csv'
)

# Define categorization functions
def categorize_age(age):
    if age < 25:
        return "<25"
    elif 25 <= age < 35:
        return "25-34"
    elif 35 <= age < 45:
        return "35-44"
    elif 45 <= age < 55:
        return "45-54"
    elif 55 <= age < 65:
        return "55-64"
    else:
        return "65+"

def categorize_tenure(tenure):
    if tenure < 12:
        return "<1 yr"
    elif 12 <= tenure < 24:
        return "1-2 yrs"
    elif 24 <= tenure < 36:
        return "2-3 yrs"
    elif 36 <= tenure < 48:
        return "3-4 yrs"
    elif 48 <= tenure < 60:
        return "4-5 yrs"
    else:
        return "5+ yrs"

def categorize_spend(spend):
    if spend <= 500:
        return "0-500"
    elif 500 < spend <= 1000:
        return "500-1000"
    elif 1000 < spend <= 1500:
        return "1000-1500"
    elif 1500 < spend <= 2000:
        return "1500-2000"
    else:
        return "2000+"

# Apply categorization functions to data
data['Age Category'] = data['Age'].apply(categorize_age)
data['Tenure Category'] = data['Tenure'].apply(categorize_tenure)
data['Spend Category'] = data['Total Spend'].apply(categorize_spend)

# Page 1: Dashboard
if page == "Dashboard":
    st.title("Customer Churn Analysis")
    
    # Calculate KPIs
    total_customers = len(data)
    churn_rate = data['Churn'].mean() * 100
    high_value_churn_rate = data[data['Total Spend'] > data['Total Spend'].quantile(0.75)]['Churn'].mean() * 100

    # Display KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{total_customers}")
    col2.metric("Churn Rate", f"{churn_rate:.2f}%")
    col3.metric("High-Value Customer Churn Rate", f"{high_value_churn_rate:.2f}%")

    # Gender Distribution and Churn by Gender 
    
    col4, col5 = st.columns(2)
    fig_gender_dist = px.pie(data, names="Gender", title="Gender Distribution", hole=0.4)
    fig_gender_dist.update_traces(textinfo='percent+label')
    col4.plotly_chart(fig_gender_dist, use_container_width=True)

    x = data[data['Churn'] == 1.0]
    churn_by_gender = x['Gender'].value_counts().reset_index()
    fig_churn_gender = px.pie(churn_by_gender, names="Gender", values="count", title="Churn by Gender", hole=0.5)
    fig_churn_gender.update_traces(textinfo='percent+label')
    col5.plotly_chart(fig_churn_gender, use_container_width=True)

    # Churn by Subscription Type 
    churn_by_subscription = x.groupby('Subscription Type').agg(
        churn_rate=('Churn', 'size'),
        total_customers=('Churn', 'count')
    ).reset_index()
    fig_churn_subscription = px.scatter(
        churn_by_subscription,
        x="Subscription Type",
        y="churn_rate",
        size="total_customers",
        color="Subscription Type",
        title="Churn Rate by Subscription Type",
        size_max=60
    )
    fig_churn_subscription.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
    fig_churn_subscription.update_traces(text=churn_by_subscription["total_customers"], textposition='top center')
    st.plotly_chart(fig_churn_subscription, use_container_width=True)

    # Churn by Age Category 
    churn_by_age_category = x['Age Category'].value_counts().reindex(['<25', '25-34', '35-44', '45-54', '55-64', '65+']).reset_index()
    fig_churn_age = px.bar(
        churn_by_age_category,
        x="Age Category",
        y="count",
        title="Churn by Age Category",
        color="Age Category"
    )
    st.plotly_chart(fig_churn_age, use_container_width=True)

    # Churn by Spend Category
    churn_by_spend_category = x.groupby('Spend Category')['Churn'].size().reset_index()
    fig_churn_spend = px.pie(
        churn_by_spend_category,
        names="Spend Category",
        values="Churn",
        title="Churn by Total Spend Category",
        hole=0.4
    )
    st.plotly_chart(fig_churn_spend, use_container_width=True)

    # Churn by Tenure Category
    churn_by_tenure = x.groupby('Tenure Category')['Churn'].size().reindex(['<1 yr', '1-2 yrs', '2-3 yrs', '3-4 yrs', '4-5 yrs', '5+ yrs']).reset_index()
    fig_churn_tenure = go.Figure(data=[go.Bar(
        x=churn_by_tenure['Tenure Category'],
        y=churn_by_tenure['Churn'],
        marker_color='indianred',
        text=churn_by_tenure['Churn'],
        textposition='auto'
    )])
    fig_churn_tenure.update_layout(
        title="Churn by Tenure Category",
        xaxis_title="Tenure Category",
        yaxis_title="Churn Rate"
    )
    st.plotly_chart(fig_churn_tenure, use_container_width=True)

# Page 2: Predictive Model
elif page == "Predictive Model":
    st.title("🔮 Predictive Model for Customer Churn")

    # Load the saved model
    model = joblib.load("GradientBoostingClassifier_Churn_model.joblib")

    # Input Form
    st.header("📝 Enter Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("👤 Age", min_value=18, max_value=100, value=30)
        tenure = st.number_input("📅 Tenure (in months)", min_value=0, value=12)
        usage_frequency = st.number_input("📊 Usage Frequency", min_value=0.0, step=0.1, value=2.5)

    with col2:
        support_calls = st.number_input("📞 Support Calls", min_value=0, value=1)
        payment_delay = st.number_input("⏳ Payment Delay (in days)", min_value=0, value=5)
        total_spend = st.number_input("💵 Total Spend", min_value=0.0, step=0.01, value=500.0)

    last_interaction = st.number_input("📅 Last Interaction (in days)", min_value=0, value=15)

    # Show Summary of Inputs
    with st.expander("🔍 Review Input Summary"):
        st.write(
            f"""
            - **Age:** {age}  
            - **Tenure:** {tenure} months  
            - **Usage Frequency:** {usage_frequency}  
            - **Support Calls:** {support_calls}  
            - **Payment Delay:** {payment_delay} days  
            - **Total Spend:** ${total_spend:,.2f}  
            - **Last Interaction:** {last_interaction} days ago  
            """
        )

    # Predict Button
    if st.button("💡 Predict Churn"):
        # Prepare data for prediction
        input_data = pd.DataFrame({
            'Age': [age],
            'Tenure': [tenure],
            'Usage Frequency': [usage_frequency],
            'Support Calls': [support_calls],
            'Payment Delay': [payment_delay],
            'Total Spend': [total_spend],
            'Last Interaction': [last_interaction]
        })

        # Make prediction
        prediction = model.predict(input_data)

        # Display Result with Visual Feedback
        if prediction[0] == 1:
            st.error("⚠️ This customer is **likely to churn**.")
        else:
            st.success("✅ This customer is **not likely to churn**.")

    # Model Information
    with st.expander("📊 Model Details"):
        st.write("This model uses a Gradient Boosting Classifier trained on customer churn data.")
        st.markdown("#### Key Features:")
        st.write("- Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, Last Interaction")
        st.image("f_importance.png",caption="Feature importance", use_column_width=True)


# Page 3: Revenue Impact of Churn
elif page == "Revenue Impact of Churn":
    st.title("Revenue Impact of Churn")

    # Display an image
    st.image("revenue loss.jpeg", caption="Churn Revenue Impact", use_column_width=True)

    # Calculate Revenue Impact Metrics
    total_revenue = data['Total Spend'].sum()
    avg_revenue_loss = data[data['Churn'] == 1.0]['Total Spend'].mean()
    total_revenue_loss = data[data['Churn'] == 1.0]['Total Spend'].sum()

    # Display revenue impact metrics
    st.write("### Key Revenue Impact Metrics")
   
    st.markdown(
    f"<p style='font-size:24px; font-weight:bold;'>Total Revenue: <span style='color:green;'>${total_revenue:,.2f}</span></p>", 
    unsafe_allow_html=True
    )

    st.markdown(
    f"<p style='font-size:24px; font-weight:bold;'>Average Revenue Loss per Churned Customer: <span style='color:red;'>${avg_revenue_loss:,.2f}</span></p>", 
    unsafe_allow_html=True
    )

    st.markdown(
    f"<p style='font-size:24px; font-weight:bold;'>Total Revenue Loss Due to Churn: <span style='color:red;'>${total_revenue_loss:,.2f}</span></p>", 
    unsafe_allow_html=True
    )
