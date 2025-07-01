import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def engineer_proxy_target(df, 
                          customer_id_col='CustomerId',
                          date_col='TransactionStartTime',
                          freq_col='TransactionId',
                          monetary_col='Amount',
                          n_clusters=3,
                          random_state=42):
  
    # Ensure datetime format
    df[date_col] = pd.to_datetime(df[date_col])

    # Set snapshot date (1 day after last transaction)
    snapshot_date = df[date_col].max() + timedelta(days=1)

    # Calculate RFM
    rfm = df.groupby(customer_id_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,
        freq_col: 'count',
        monetary_col: 'sum'
    }).rename(columns={
        date_col: 'Recency',
        freq_col: 'Frequency',
        monetary_col: 'Monetary'
    }).reset_index()

    # Scale RFM
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Determine high-risk cluster (typically lowest frequency & monetary)
    cluster_stats = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_stats['Frequency'].idxmin()  # Or use more logic if needed

    # Assign binary label
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)

    # Merge back to original dataset
    df = df.merge(rfm[[customer_id_col, 'is_high_risk']], on=customer_id_col, how='left')
    df['is_high_risk'] = df['is_high_risk'].fillna(0).astype(int)

    return df
