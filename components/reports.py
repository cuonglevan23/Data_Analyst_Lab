import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def show():
    st.title('Trang Báo Cáo')
    st.write('Dưới đây là một số báo cáo và biểu đồ dữ liệu.')

    # Tải dữ liệu từ file CSV
    df = pd.read_csv("dataset/Car Sales.xlsx - car_data.csv")

    # Thêm lựa chọn cho người dùng để chọn báo cáo
    report_type = st.selectbox(
        'Chọn một báo cáo để hiển thị',
        [
            'Doanh số theo từng tháng',
            'Top 10 Xe Đắt Nhất',
            'Top 10 Xe Rẻ Nhất',
            'Phân Phối Giá',
            'Giá Xe Theo Đại Lý',
            'Top Khu Vực Bán Xe Nhiều Nhất',
            'Top 10 Khách Hàng Mua Nhiều Xe Nhất',

            'Doanh Số Bán Xe Theo Đại Lý',
            'PCA và Phân Cụm K-Means',
            'Phân Tích Chi Tiết',
        ]
    )

    if report_type == 'Doanh số theo từng tháng':
        st.write("### Doanh số bán xe theo từng tháng")

        # Chuyển đổi cột 'Date' thành định dạng ngày tháng và tạo cột 'Month'
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.to_period('M').astype('str')

        # Chọn khoảng thời gian
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        start_date, end_date = st.slider(
            'Chọn Khoảng Thời Gian:',
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM"
        )

        # Lọc dữ liệu theo khoảng thời gian
        filtered_df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]

        # Nhóm dữ liệu theo tháng và tính doanh số
        monthly_sales = filtered_df.groupby('Month').size().reset_index(name='Number of Cars Sold')
        monthly_sales['Month'] = pd.to_datetime(monthly_sales['Month'], format='%Y-%m')

        # Tính doanh thu cho từng tháng
        monthly_revenue = filtered_df.groupby('Month')['Price ($)'].sum().reset_index()
        monthly_revenue['Month'] = pd.to_datetime(monthly_revenue['Month'], format='%Y-%m')

        # Vẽ biểu đồ doanh số bán xe theo thời gian
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.lineplot(x='Month', y='Number of Cars Sold', data=monthly_sales, marker='o', label='Doanh số', ax=ax)
        ax.set_xlabel('Tháng')
        ax.set_ylabel('Số Lượng Xe Bán Ra')
        ax.set_title('Doanh Số Bán Xe Theo Thời Gian')
        plt.xticks(rotation=45)
        ax.legend(loc='upper left')
        st.pyplot(fig)

        # Vẽ biểu đồ doanh thu theo thời gian
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.lineplot(x='Month', y='Price ($)', data=monthly_revenue, marker='o', color='orange', label='Doanh thu',
                     ax=ax)
        ax.set_xlabel('Tháng')
        ax.set_ylabel('Doanh Thu ($)')
        ax.set_title('Doanh Thu Theo Thời Gian')
        plt.xticks(rotation=45)
        ax.legend(loc='upper left')
        st.pyplot(fig)

        # Tính doanh số bán cho từng dòng xe trong khoảng thời gian đã chọn
        car_sales = filtered_df.groupby('Model').size().reset_index(name='Number of Cars Sold')

        # Chọn 10 dòng xe bán chạy nhất
        top_10_cars = car_sales.nlargest(10, 'Number of Cars Sold')

        # Vẽ biểu đồ doanh số của 10 dòng xe bán chạy nhất
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=top_10_cars, x='Model', y='Number of Cars Sold', ax=ax, palette='viridis')
        ax.set_title('Top 10 Dòng Xe Bán Chạy Nhất Trong Khoảng Thời Gian')
        ax.set_xlabel('Dòng Xe')
        ax.set_ylabel('Số Lượng Xe Bán Ra')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)
    elif report_type == 'Top 10 Xe Đắt Nhất':
        st.write("### Top 10 Xe Đắt Nhất")
        top_10_expensive = df.nlargest(10, 'Price ($)')

        # Chọn cột mẫu xe (model) và giá
        st.dataframe(top_10_expensive[['Company', 'Price ($)']])

        # Vẽ biểu đồ
        fig, ax = plt.subplots()
        sns.barplot(data=top_10_expensive, x='Company', y='Price ($)', ax=ax, palette='viridis')
        ax.set_title('Top 10 Xe Đắt Nhất')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

    elif report_type == 'Top 10 Xe Rẻ Nhất':
        st.write("### Top 10 Xe Rẻ Nhất")
        top_10_cheapest = df.nsmallest(10, 'Price ($)')
        st.dataframe(top_10_cheapest[['Company', 'Price ($)']])

        fig, ax = plt.subplots()
        sns.barplot(data=top_10_cheapest, x='Company', y='Price ($)', ax=ax, palette='magma')
        ax.set_title('Top 10 Xe Rẻ Nhất')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

    elif report_type == 'Phân Phối Giá':
        st.write("### Phân Phối Giá")
        fig, ax = plt.subplots()
        sns.histplot(df['Price ($)'], bins=30, kde=True, ax=ax, color='blue')
        ax.set_title('Phân Phối Giá Xe')
        st.pyplot(fig)

    elif report_type == 'Giá Xe Theo Đại Lý':
        st.write("### Giá Xe Theo Đại Lý")
        dealer_price_summary = df.groupby('Dealer_Name')['Price ($)'].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(data=dealer_price_summary, x='Dealer_Name', y='Price ($)', ax=ax, palette='coolwarm')
        ax.set_title('Giá Trung Bình Của Xe Theo Đại Lý')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
        st.pyplot(fig)
        # Biểu đồ Top Khu Vực Bán Xe Nhiều Nhất
    elif report_type == 'Top Khu Vực Bán Xe Nhiều Nhất':
        st.write("### Top Khu Vực Bán Xe Nhiều Nhất")
        region_sales = df.groupby('Dealer_Region').size().reset_index(name='Number of Cars Sold')
        region_sales = region_sales.sort_values(by='Number of Cars Sold', ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Number of Cars Sold', y='Dealer_Region', data=region_sales, palette='viridis', ax=ax)
        ax.set_title('Top Khu Vực Bán Xe Nhiều Nhất')
        ax.set_xlabel('Số Lượng Xe Bán Ra')
        ax.set_ylabel('Khu Vực')
        st.pyplot(fig)

    elif report_type == 'Top 10 Khách Hàng Mua Nhiều Xe Nhất':
        st.write("#### Top 10 Khách Hàng Mua Nhiều Xe Nhất")
        purchases_by_customer = df['Customer Name'].value_counts().reset_index()
        purchases_by_customer.columns = ['Customer Name', 'Number of Cars Purchased']
        top_customers = purchases_by_customer.head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(top_customers['Customer Name'], top_customers['Number of Cars Purchased'], color='skyblue')
        ax.set_title('Top 10 Khách Hàng Mua Nhiều Xe Nhất')
        ax.set_xlabel('Tên Khách Hàng')
        ax.set_ylabel('Số Lượng Xe Đã Mua')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)
    elif report_type == 'Doanh Số Bán Xe Theo Đại Lý':
        # Tính toán tổng doanh số bán xe cho mỗi đại lý
        sales_by_dealer = df.groupby('Dealer_Name')['Price ($)'].sum().reset_index()

        # Tạo một hình ảnh và trục
        fig, ax = plt.subplots(figsize=(10, 6))

        # Vẽ biểu đồ thanh hiển thị doanh số bán xe của mỗi đại lý
        ax.bar(sales_by_dealer['Dealer_Name'], sales_by_dealer['Price ($)'], color='skyblue')

        # Thêm tiêu đề và nhãn cho các trục
        ax.set_title('Doanh Số Bán Xe Theo Đại Lý')
        ax.set_xlabel('Tên Đại Lý')
        ax.set_ylabel('Tổng Doanh Số Bán Xe ($)')

        # Xoay các nhãn trục x để dễ đọc hơn
        ax.set_xticklabels(sales_by_dealer['Dealer_Name'], rotation=45, ha='right')

        # Hiển thị biểu đồ trên Streamlit
        st.pyplot(fig)

    elif report_type == 'Phân Tích Chi Tiết':
        st.write("### Phân Tích Chi Tiết")

        st.write("#### Số Lượng Xe Theo Loại Thân Xe và Giới Tính")
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.subplot(121)
        sns.histplot(data=df, x='Body Style', color='skyblue')
        plt.title('Số Lượng Xe Theo Loại Thân Xe')

        plt.subplot(122)
        sns.histplot(data=df, x='Body Style', hue='Gender')
        plt.title('Số Lượng Xe Theo Loại Thân Xe và Giới Tính')

        st.pyplot(fig)

        st.write("#### Số Lượng Xe Theo Khu Vực Đại Lý")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.countplot(data=df, x='Dealer_Region', palette='Blues', ax=ax)
        ax.set_title('Số Lượng Xe Theo Khu Vực Đại Lý')
        st.pyplot(fig)

        st.write("#### Phân Phối Giá Theo Loại Thân Xe")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=df, x="Price ($)", y="Body Style", hue='Body Style', palette='crest', ax=ax)
        ax.set_title('Phân Phối Giá Theo Loại Thân Xe')
        st.pyplot(fig)

        st.write("#### Phân Phối Giá Theo Động Cơ")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=df, x="Price ($)", y="Engine", hue='Engine', palette='viridis', ax=ax)
        ax.set_title('Phân Phối Giá Theo Động Cơ')
        st.pyplot(fig)

        st.write("#### Số Lượng Xe Theo Loại Thân Xe")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(y='Body Style', data=df, ax=ax)
        ax.set_title('Số Lượng Xe Theo Loại Thân Xe')
        ax.set_xlabel('Số Lượng')
        ax.set_ylabel('Loại Thân Xe')
        st.pyplot(fig)

        st.write("#### Xu Hướng Giá Xe Qua Thời Gian (Trung Bình Tháng)")
        # Chuyển đổi cột 'Date' sang định dạng datetime và tạo cột 'Month'
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.to_period('M')
        monthly_price = df.groupby('Month')['Price ($)'].mean().reset_index()
        monthly_price['Month'] = monthly_price['Month'].dt.to_timestamp()

        # Vẽ xu hướng giá trung bình theo tháng
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='Month', y='Price ($)', data=monthly_price, ax=ax)
        ax.set_title('Xu Hướng Giá Xe Qua Thời Gian (Trung Bình Tháng)')
        ax.set_xlabel('Tháng')
        ax.set_ylabel('Giá Trung Bình ($)')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif report_type == 'PCA và Phân Cụm K-Means':
        st.write("### PCA và Phân Cụm K-Means")

        # Chuẩn bị dữ liệu cho PCA
        df_pca = df[['Price ($)', 'Body Style', 'Engine', 'Transmission', 'Company', 'Color', 'Dealer_Region']]
        df_pca = pd.get_dummies(df_pca)  # Chuyển đổi các biến phân loại thành các biến giả/chỉ báo

        # Áp dụng PCA
        pca = PCA(n_components=3)
        reduced_data = pca.fit_transform(df_pca)

        # Tạo một DataFrame với kết quả PCA
        df_pca_reduced = pd.DataFrame(reduced_data, columns=['PC1', 'PC2', 'PC3'])

        # Áp dụng K-Means để tạo cụm
        kmeans = KMeans(n_clusters=10, random_state=42)
        clusters = kmeans.fit_predict(df_pca_reduced)

        # Thêm thông tin cụm vào DataFrame
        df_pca_reduced['Cluster'] = clusters

        # Tạo biểu đồ tán xạ 3D cho phân cụm
        fig = px.scatter_3d(df_pca_reduced, x='PC1', y='PC2', z='PC3', color='Cluster', opacity=0.7, size_max=5,
                            title='Phân Cụm 3D')
        st.plotly_chart(fig)
