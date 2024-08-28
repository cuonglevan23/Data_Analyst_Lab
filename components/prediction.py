import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from components.price_trend_prediction import plot_price_trend_prediction

def show():
    # Tải các mô hình đã lưu
    models = {
        'Linear Regression': joblib.load(
            '/Users/lvc/PycharmProjects/DataAnalyst_vietnamese-car-price/Decision_Tree_.joblib'),
        'Ridge Regression': joblib.load(
            '/Users/lvc/PycharmProjects/DataAnalyst_vietnamese-car-price/Ridge_Regression.joblib'),
        'Lasso Regression': joblib.load(
            '/Users/lvc/PycharmProjects/DataAnalyst_vietnamese-car-price/Lasso_Regression.joblib'),
        'Decision Tree': joblib.load(
            '/Users/lvc/PycharmProjects/DataAnalyst_vietnamese-car-price/Decision_Tree_.joblib'),
        'Random Forest': joblib.load(
            '/Users/lvc/PycharmProjects/DataAnalyst_vietnamese-car-price/Random_Forest_.joblib'),
        'Gradient Boosting': joblib.load(
            '/Users/lvc/PycharmProjects/DataAnalyst_vietnamese-car-price/Gradient_Boosting_.joblib'),
        'SVR': joblib.load('/Users/lvc/PycharmProjects/DataAnalyst_vietnamese-car-price/SVR.joblib')
    }

    # Tải dữ liệu để mã hóa các cột danh mục
    df = pd.read_csv(
        '/Users/lvc/PycharmProjects/DataAnalyst_vietnamese-car-price/dataset/Car Sales.xlsx - car_data.csv')

    # Tạo ánh xạ từ số thành giá trị gốc
    label_mappings = {}
    categorical_cols = ['Gender', 'Company', 'Model', 'Engine', 'Transmission', 'Color', 'Body Style', 'Dealer_Region']

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_mappings[col] = dict(enumerate(le.classes_))

    # Chọn các đặc trưng cho dự đoán
    X = df[['Gender', 'Model', 'Engine', 'Transmission', 'Company', 'Color', 'Body Style', 'Dealer_Region']]
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    option = st.sidebar.radio('Chọn trang', ['Dự đoán giá xe', 'Dự đoán xu hướng giá'])

    if option == 'Dự đoán giá xe':
        st.title('Dự đoán giá xe')

        # Chọn mô hình
        model_name = st.selectbox('Chọn mô hình', list(models.keys()))

        # Nhập dữ liệu đặc trưng của xe để dự đoán
        gender = st.selectbox('Giới tính', options=list(label_mappings['Gender'].values()))
        model = st.selectbox('Mẫu xe', options=list(label_mappings['Model'].values()))
        engine = st.selectbox('Động cơ', options=list(label_mappings['Engine'].values()))
        transmission = st.selectbox('Hộp số', options=list(label_mappings['Transmission'].values()))
        company = st.selectbox('Hãng xe', options=list(label_mappings['Company'].values()))
        color = st.selectbox('Màu sắc', options=list(label_mappings['Color'].values()))
        body_style = st.selectbox('Kiểu dáng thân xe', options=list(label_mappings['Body Style'].values()))
        dealer_region = st.selectbox('Khu vực đại lý', options=list(label_mappings['Dealer_Region'].values()))

        # Dự đoán giá xe
        if st.button('Dự đoán'):
            # Chuyển đổi dữ liệu đầu vào thành DataFrame
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Model': [model],
                'Engine': [engine],
                'Transmission': [transmission],
                'Company': [company],
                'Color': [color],
                'Body Style': [body_style],
                'Dealer_Region': [dealer_region]
            })

            # Chuyển đổi dữ liệu đầu vào thành mã số
            for col in categorical_cols:
                input_data[col] = [list(label_mappings[col].keys())[list(label_mappings[col].values()).index(val)] for
                                   val in input_data[col]]

            # Áp dụng PCA
            input_data_pca = pca.transform(input_data)

            # Dự đoán giá
            model = models[model_name]
            price_prediction = model.predict(input_data_pca)

            # Hiển thị dự đoán
            st.write(f'Giá dự đoán của xe là ${price_prediction[0]:,.2f}')

    elif option == 'Dự đoán xu hướng giá':
        st.title('Dự đoán xu hướng giá')

        st.write("### Dự đoán xu hướng giá xe")
        # Gọi hàm từ file bên ngoài để vẽ dự đoán xu hướng giá
        fig = plot_price_trend_prediction(df)
        st.pyplot(fig)
