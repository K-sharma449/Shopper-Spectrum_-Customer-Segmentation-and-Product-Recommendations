# Technical Tags: #Pandas #Numpy #DataCleaning #FeatureEngineering #EDA #RFMAnalysis #CustomerSegmentation #KMeansClustering #CollaborativeFiltering #CosineSimilarity #ProductRecommendation #ScikitLearn #StandardScaler #StreamlitApp #MachineLearning #DataVisualization #PivotTables #DataTransformation #RealTimePrediction

# Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce
# This script performs data preprocessing, EDA, RFM analysis, clustering, model evaluation,
# and builds the recommendation system. It saves models and artifacts for the Streamlit app.
# The dataset is fetched directly from the Google Drive link into memory without local download.
# Handles confirmation for large files. Updated to read as CSV since the file is CSV.
# The second part of the script runs a Streamlit dashboard for interaction.
# Note: 3D plot removed to avoid potential Matplotlib version issues; 2D scatter plot retained.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from datetime import datetime
import sys
import streamlit as st
import requests
import io

# Function to fetch file from Google Drive, handling large file confirmation
def download_from_google_drive(file_id):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    params = {'id': file_id}
    response = session.get(URL, params=params, stream=True)

    # Get confirmation token from cookies
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    # If token found, make second request with confirmation
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # Collect content in memory
    content = io.BytesIO()
    for chunk in response.iter_content(32768):
        if chunk:
            content.write(chunk)
    content.seek(0)
    return content

# Check if running as a Streamlit app or data processing script
if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
    # Streamlit app code will run here
    pass
else:
    # Data processing and model building code
    # Step 1: Dataset Collection and Understanding
    # Fetch the dataset directly from Google Drive into memory
    file_id = '1rzRwxm_CJxcRzfoo9Ix37A2JTlMummY-'
    content = download_from_google_drive(file_id)
    df = pd.read_csv(content, encoding='ISO-8859-1')  # Read as CSV with encoding for special chars
    # Alternative for local file (uncomment if using Option 2): df = pd.read_csv('online_retail.csv', encoding='ISO-8859-1')

    # Ensure InvoiceNo is string for filtering
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    # Drop rows with missing Description to avoid issues in pivot
    df = df.dropna(subset=['Description'])

    # Explore the dataset
    print(df.head())
    print(df.info())
    print(df.describe())

    # Identify missing values, duplicates, and unusual records
    print(df.isnull().sum())
    print(f"Duplicates: {df.duplicated().sum()}")

    # Step 2: Data Preprocessing
    # Remove rows with missing CustomerID
    df = df.dropna(subset=['CustomerID'])

    # Exclude cancelled invoices (InvoiceNo starting with 'C')
    df = df[~df['InvoiceNo'].str.startswith('C', na=False)]

    # Remove negative or zero quantities and prices
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]

    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Add TotalPrice column
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Step 3: Exploratory Data Analysis (EDA)
    # Analyze transaction volume by country
    plt.figure(figsize=(12, 6))
    country_volume = df['Country'].value_counts()
    sns.barplot(x=country_volume.index, y=country_volume.values, palette='viridis')
    plt.title('Transaction Volume by Country')
    plt.xlabel('Country')
    plt.ylabel('Number of Transactions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('country_volume.png')
    plt.close()

    # Identify top-selling products (limited to top 20)
    plt.figure(figsize=(12, 6))
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(500)  # Limit to top 500 for filtering
    top_20_products = top_products.head(20)
    sns.barplot(x=top_20_products.index, y=top_20_products.values, palette='magma')
    plt.title('Top 20 Selling Products')
    plt.xlabel('Product')
    plt.ylabel('Quantity Sold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('top_products.png')
    plt.close()

    # Add category wise (using country as proxy for category) product sold chart
    plt.figure(figsize=(12, 6))
    country_quantity = df.groupby('Country')['Quantity'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=country_quantity.index, y=country_quantity.values, palette='coolwarm')
    plt.title('Top 10 Countries by Quantity Sold')
    plt.xlabel('Country')
    plt.ylabel('Total Quantity Sold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('country_quantity.png')
    plt.close()

    # Filter dataset to top products
    df = df[df['Description'].isin(top_products.index)]

    # Visualize purchase trends over time
    plt.figure(figsize=(12, 6))
    df['Month'] = df['InvoiceDate'].dt.to_period('M')
    monthly_sales = df.groupby('Month')['TotalPrice'].sum()
    sns.lineplot(x=monthly_sales.index.astype(str), y=monthly_sales.values, color='purple')
    plt.title('Monthly Sales Trends')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('monthly_sales.png')
    plt.close()

    # Inspect monetary distribution per transaction and customer
    plt.figure(figsize=(12, 6))
    sns.histplot(x=df['TotalPrice'], bins=50, kde=True, color='teal')
    plt.title('Monetary Distribution per Transaction')
    plt.xlabel('Transaction Amount (£)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('monetary_transaction.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    customer_monetary = df.groupby('CustomerID')['TotalPrice'].sum()
    sns.histplot(x=customer_monetary, bins=50, kde=True, color='coral')
    plt.title('Monetary Distribution per Customer')
    plt.xlabel('Total Spend (£)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('monetary_customer.png')
    plt.close()

    # Step 4: Clustering Methodology
    # 1. Feature Engineering: Calculate RFM
    latest_date = df['InvoiceDate'].max()
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',  # Frequency
        'TotalPrice': 'sum'  # Monetary
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm.reset_index()

    # RFM distributions
    plt.figure(figsize=(12, 4))
    sns.histplot(x=rfm['Recency'], bins=50, kde=True, color='green')
    plt.title('Recency Distribution')
    plt.xlabel('Days Since Last Purchase')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('recency_dist.png')
    plt.close()

    plt.figure(figsize=(12, 4))
    sns.histplot(x=rfm['Frequency'], bins=50, kde=True, color='blue')
    plt.title('Frequency Distribution')
    plt.xlabel('Number of Purchases')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('frequency_dist.png')
    plt.close()

    plt.figure(figsize=(12, 4))
    sns.histplot(x=rfm['Monetary'], bins=50, kde=True, color='red')
    plt.title('Monetary Distribution')
    plt.xlabel('Total Spend (£)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('monetary_dist.png')
    plt.close()

    # 2. Standardize/Normalize the RFM values
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # 3. Choose Clustering Algorithm: KMeans
    # 4. Use Elbow Method and Silhouette Score to decide the number of clusters
    inertias = []
    sil_scores = []
    k_range = range(2, 11)  # Start from 2 for silhouette
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        inertias.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))

    # Elbow curve
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=k_range, y=inertias, marker='o', color='purple')
    plt.title('Elbow Curve for Cluster Selection')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.tight_layout()
    plt.savefig('elbow_curve.png')
    plt.close()

    # Silhouette scores
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=k_range, y=sil_scores, marker='o', color='orange')
    plt.title('Silhouette Scores for Cluster Selection')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.tight_layout()
    plt.savefig('silhouette_scores.png')
    plt.close()

    # Assuming optimal is 4 based on typical RFM (adjust if needed after plots)
    n_clusters = 4

    # 5. Run Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Label the clusters by interpreting their RFM averages
    cluster_means = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()

    # Assign labels: Higher score (low Recency, high Freq/Mon) -> better segment
    cluster_means['Score'] = -cluster_means['Recency'] + cluster_means['Frequency'] + cluster_means['Monetary']
    sorted_clusters = cluster_means.sort_values('Score', ascending=False)
    labels = ['High-Value', 'Regular', 'Occasional', 'At-Risk']
    cluster_to_label = {sorted_clusters.index[i]: labels[i] for i in range(n_clusters)}

    # Customer cluster profiles
    plt.figure(figsize=(12, 6))
    cluster_means[['Recency', 'Frequency', 'Monetary']].plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Customer Cluster Profiles')
    plt.xlabel('Cluster')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig('cluster_profiles.png')
    plt.close()

    # 6. Visualize the clusters using a scatter plot (2D: Frequency vs Monetary, colored by Cluster)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Frequency', y='Monetary', hue='Cluster', data=rfm, palette='viridis', s=100)
    plt.title('Clusters: Frequency vs Monetary')
    plt.xlabel('Number of Purchases')
    plt.ylabel('Total Spend (£)')
    plt.tight_layout()
    plt.savefig('cluster_scatter.png')
    plt.close()

    # 3D plot removed to avoid Matplotlib compatibility issues; 2D visualization retained

    # 7. Save the best performing model for Streamlit usage
    joblib.dump(kmeans, 'kmeans_model.pkl', compress=3)
    joblib.dump(scaler, 'scaler.pkl', compress=3)
    joblib.dump(cluster_to_label, 'cluster_labels.pkl', compress=3)

    # Recommendation System Approach
    # Use Item-based Collaborative Filtering
    # Create CustomerID-Description matrix (using Quantity)
    user_item = df.pivot_table(index='CustomerID', columns='Description', values='Quantity', aggfunc='sum', fill_value=0)

    # Compute cosine similarity between products (item-item)
    item_similarity = cosine_similarity(user_item.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item.columns, columns=user_item.columns)

    # Precompute recommendations
    recommendations = {}
    for product in item_similarity_df.index:
        similar_scores = item_similarity_df[product].sort_values(ascending=False)
        recommendations[product] = similar_scores.index[1:6].tolist()

    # Product recommendation heatmap / similarity matrix (subset for visualization)
    plt.figure(figsize=(12, 10))
    sns.heatmap(item_similarity_df.iloc[:20, :20], cmap='coolwarm', annot=False)
    plt.title('Product Similarity Heatmap (Subset)')
    plt.tight_layout()
    plt.savefig('product_heatmap.png')
    plt.close()

    # Save for Streamlit
    joblib.dump(item_similarity_df, 'item_similarity.pkl', compress=3)
    joblib.dump(recommendations, 'recommendations.pkl', compress=3)
    joblib.dump(list(user_item.columns), 'product_list.pkl', compress=3)

    print("Analysis complete. Models and artifacts saved for Streamlit app.")

# Streamlit App Code (runs if "streamlit" argument is provided or when run with streamlit command)
if len(sys.argv) > 1 and sys.argv[1] == "streamlit" or __name__ == "__main__":
    @st.cache_data
    def load_data():
        kmeans = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        cluster_labels = joblib.load('cluster_labels.pkl')
        item_similarity_df = joblib.load('item_similarity.pkl')
        recommendations = joblib.load('recommendations.pkl')
        product_list = joblib.load('product_list.pkl')
        return kmeans, scaler, cluster_labels, item_similarity_df, recommendations, product_list

    kmeans, scaler, cluster_labels, item_similarity_df, recommendations, product_list = load_data()

    # Custom CSS for larger radio buttons and visual enhancements
    st.markdown("""
        <style>
        div.stRadio > div > label > div > p {
            font-size: 22px;
            font-weight: bold;
            padding: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 60px;
            white-space: pre-wrap;
            width: fit-content;
            min-width: 160px;
            font-size: 20px;
            padding: 10px;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 5px;
        }
        .stSelectbox > div > div > div {
            font-size: 16px;
        }
        .stNumberInput > div > div > input {
            font-size: 16px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title('🛒 Shopper Spectrum: Customer Segmentation and Product Recommendations')

    # Sidebar for module selection using radio buttons with emojis
    module = st.sidebar.radio('Navigation 🚀', ['🏠 Home', '🔍 Clustering', '💡 Recommendation'])

    if module == '🏠 Home':
        st.header('Welcome to Shopper Spectrum 🌟')
        st.write('This app provides insightful customer segmentation and product recommendations based on e-commerce data. Explore the visualizations below to understand the dataset and analysis.')

        st.subheader('📊 Exploratory Data Analysis')

        st.subheader('1. Transaction Volume by Country 📍')
        st.image('country_volume.png', caption='Transaction Volume by Country')
        st.write("""
        Description: This bar chart shows how many transactions occur in each country, highlighting where sales are concentrated.
        - It identifies key markets, like if the UK dominates, showing where most customers are located.
        - Useful for planning where to focus marketing or expand operations based on transaction volume.
        """)

        st.subheader('2. Top 20 Selling Products 🛍️')
        st.image('top_products.png', caption='Top 20 Selling Products')
        st.write("""
        Description: This bar chart lists the 20 products with the highest quantity sold, showing what customers buy most.
        - Each bar represents a product, with height indicating units sold, helping identify popular items.
        - Guides inventory stocking and promotional strategies to boost sales of top products.
        """)

        st.subheader('3. Top 10 Countries by Quantity Sold 🌍')
        st.image('country_quantity.png', caption='Top 10 Countries by Quantity Sold')
        st.write("""
        Description: This bar chart displays the total quantity of products sold in the top 10 countries, showing sales volume by region.
        - It highlights which countries contribute most to product sales, like a leading market driving revenue.
        - Helps prioritize markets for logistics, advertising, or customer engagement efforts.
        """)

        st.subheader('4. Monthly Sales Trends 📈')
        st.image('monthly_sales.png', caption='Monthly Sales Trends')
        st.write("""
        Description: This line chart tracks total sales amounts over months, revealing patterns in sales activity.
        - Peaks might indicate holiday seasons or promotions, while dips show quieter periods.
        - Useful for planning inventory or marketing campaigns based on seasonal trends.
        """)

        st.subheader('5. Monetary Distribution per Transaction 💰')
        st.image('monetary_transaction.png', caption='Monetary Distribution per Transaction')
        st.write("""
        Description: This histogram shows the amount spent per transaction, with most purchases being smaller amounts.
        - The curve smooths the data to show the overall spending pattern, often clustered at lower values.
        - Helps set pricing strategies or discounts to encourage higher transaction values.
        """)

        st.subheader('6. Monetary Distribution per Customer 👥')
        st.image('monetary_customer.png', caption='Monetary Distribution per Customer')
        st.write("""
        Description: This histogram shows total spending by each customer, often with many spending less and few spending more.
        - It reveals the range of customer spending, identifying high-value customers.
        - Guides efforts to retain big spenders or encourage others to spend more.
        """)

        st.subheader('🔍 Clustering Insights')

        st.subheader('7. Elbow Curve for Cluster Selection 📉')
        st.image('elbow_curve.png', caption='Elbow Curve for Cluster Selection')
        st.write("""
        Description: This line plot helps choose the best number of customer groups by showing how compact the groups are.
        - The 'elbow' point, like around 4, suggests a good balance for grouping customers effectively.
        - Ensures the number of groups is practical for marketing or analysis purposes.
        """)

        st.subheader('8. Silhouette Scores for Cluster Selection 📏')
        st.image('silhouette_scores.png', caption='Silhouette Scores for Cluster Selection')
        st.write("""
        Description: This plot shows how well customers fit into their groups for different numbers of clusters.
        - Higher scores mean clearer group separation, confirming the best number of groups.
        - Helps ensure groups are distinct for targeted customer strategies.
        """)

        st.subheader('9. Customer Cluster Profiles 👥')
        st.image('cluster_profiles.png', caption='Customer Cluster Profiles')
        st.write("""
        Description: This bar chart compares average recency, frequency, and monetary values for each customer group.
        - Shows differences, like groups with frequent purchases or high spending, for easy comparison.
        - Helps tailor marketing, like special offers for high-value or at-risk customers.
        """)

        st.subheader('10. Clusters: Frequency vs Monetary 🔵')
        st.image('cluster_scatter.png', caption='Clusters: Frequency vs Monetary')
        st.write("""
        Description: This scatter plot groups customers by how often they buy and how much they spend, with colors for each group.
        - Each dot is a customer, showing patterns like frequent high-spenders in one group.
        - Useful for visualizing how customer behaviors differ across segments.
        """)

        st.subheader('11. Product Similarity Heatmap (Subset) 🔥')
        st.image('product_heatmap.png', caption='Product Similarity Heatmap (Subset)')
        st.write("""
        Description: This heatmap shows how similar products are based on customer purchases, with darker colors indicating stronger similarity.
        - Based on a subset of products, it reveals which items are often bought together.
        - Supports recommendations by suggesting products likely to appeal to similar customers.
        """)

    elif module == '🔍 Clustering':
        st.header('🔍 Customer Segmentation Module')
        st.write('Explore RFM metrics and predict customer segments. 📊')

        tab4, tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "⏳ Recency", "🔄 Frequency", "💵 Monetary"])

        with tab4:
            st.subheader('🔮 Predict Customer Segment')
            st.write('Enter RFM values to classify the customer. Results include a remark based on the segment.')

            recency = st.number_input('⏳ Recency (days since last purchase)', min_value=0.0, step=1.0)
            frequency = st.number_input('🔄 Frequency (number of purchases)', min_value=0.0, step=1.0)
            monetary = st.number_input('💵 Monetary (total spend)', min_value=0.0, step=0.01)

            if st.button('Predict Segment 🚀'):
                input_data = np.array([[recency, frequency, monetary]])
                scaled_input = scaler.transform(input_data)
                cluster = kmeans.predict(scaled_input)[0]
                label = cluster_labels.get(cluster, 'Unknown')
                st.subheader('📋 Predicted Segment:')
                st.info(f"With Recency: {recency} days, Frequency: {frequency} purchases, Monetary: £{monetary}, this customer belongs to **{label}** shoppers.")

        with tab1:
            st.subheader('⏳ Recency: Days Since Last Purchase')
            st.image('recency_dist.png', caption='Recency Distribution 📅')
            st.write("""
            Description: This histogram shows how many days have passed since each customer's last purchase, with most having recent activity.
            - Many customers shop recently (low days), while some haven't in a long time, forming a skewed shape.
            - Helps identify active customers versus those who might need reminders or offers to return.
            """)

        with tab2:
            st.subheader('🔄 Frequency: Number of Purchases')
            st.image('frequency_dist.png', caption='Frequency Distribution 🔢')
            st.write("""
            Description: This histogram displays how many times customers have made purchases, with most having fewer transactions.
            - A few customers buy frequently, creating a long tail in the distribution.
            - Useful for spotting loyal customers to reward or encouraging others to buy more often.
            """)

        with tab3:
            st.subheader('💵 Monetary: Total Spend')
            st.image('monetary_dist.png', caption='Monetary Distribution 💲')
            st.write("""
            Description: This histogram shows the total amount spent by each customer, typically with many spending smaller amounts.
            - A small group of high-spenders stands out, contributing significantly to overall revenue.
            - Guides strategies to retain big spenders or boost spending among others.
            """)

    elif module == '💡 Recommendation':
        st.header('💡 Product Recommendation Module')
        st.write('Discover similar products based on collaborative filtering. 🔗')

        # Text input for product name
        product_name = st.selectbox('Select or Type Product Name 🛒', options=[''] + product_list, index=0)
        if not product_name:
            product_name = st.text_input('Or Enter Product Name Manually 📝')

        if st.button('Get Recommendations 🚀'):
            if product_name in recommendations:
                similar_products = recommendations[product_name]
                st.subheader('Recommended Products 🌟')
                cols = st.columns(5)
                for i, prod in enumerate(similar_products):
                    with cols[i]:
                        st.markdown(f"**{prod}** 📦")
            else:
                st.error('Product not found in the dataset. Please try another name. ❌')
                
