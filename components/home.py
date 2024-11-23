import streamlit as st
import pandas as pd
from datetime import datetime

def show():
    st.title('Trang Chủ')
    st.write('Chào mừng bạn đến với trang chủ của dự án.')

    # Upload and load CSV file
    uploaded_file = st.file_uploader("Chọn một file CSV", type=["csv"])
    if uploaded_file is not None:
        # Load data from the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Display data overview
        st.write("### Tổng quan về dữ liệu")
        st.write("Đây là bản xem trước của dữ liệu:")
        st.dataframe(df.head())

        # Display descriptive statistics of the dataset
        st.write("### Mô tả dữ liệu")
        st.write(df.describe())
    else:
        st.write("Vui lòng chọn một file CSV để tải dữ liệu.")

    # Tải dữ liệu từ file CSV
    df = pd.read_csv("dataset/Car Sales.xlsx - car_data.csv")

    # Tính toán các chỉ số để hiển thị
    total_sales = df['Price ($)'].sum()
    num_dealers = df['Dealer_Name'].nunique()
    num_cars = df.shape[0]

    # Thêm CSS để tạo khung với màu sắc và khoảng cách
    st.markdown("""
    <style>
    .metrics-container {
        display: flex;
        justify-content: space-between;
        gap: 10px;
    }
    .metric-box {
        flex: 1;
        padding: 20px;
        color: white;
        border-radius: 10px;
        text-align: center;
    }
    .blue { background-color: #1f77b4; }
    .red { background-color: #d62728; }
    .yellow { background-color: #ffbb78; }
    </style>
    """, unsafe_allow_html=True)

    # Hiển thị các ô vuông với màu sắc và khoảng cách
    st.markdown("""
    <div class="metrics-container">
        <div class="metric-box blue">
            <h3>Total Sales ($)</h3>
            <p>${:,.2f}</p>
        </div>
        <div class="metric-box red">
            <h3>Number of Dealers</h3>
            <p>{}</p>
        </div>
        <div class="metric-box yellow">
            <h3>Number of Cars</h3>
            <p>{}</p>
        </div>
    </div>
    """.format(total_sales, num_dealers, num_cars), unsafe_allow_html=True)

    # Hiển thị thông tin tổng quan về dữ liệu
    st.write("### Tổng quan về dữ liệu")
    st.write("Đây là bản xem trước của dữ liệu:")
    st.dataframe(df.head())

    # Hiển thị thông tin mô tả của dữ liệu
    st.write("### Mô tả dữ liệu")
    st.write(df.describe())
