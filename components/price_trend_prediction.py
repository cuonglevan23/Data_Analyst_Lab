import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def plot_price_trend_prediction(df):
    # Chuẩn bị dữ liệu cho mô hình
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M').astype('str')
    df['Price ($)'] = df['Price ($)'].astype(float)

    # Nhóm dữ liệu theo tháng và tạo các đặc trưng và biến mục tiêu
    monthly_data = df.groupby('Month')['Price ($)'].mean().reset_index()
    monthly_data['Month'] = pd.to_datetime(monthly_data['Month'])

    # Chuẩn bị dữ liệu cho mô hình
    X = monthly_data.index.values.reshape(-1, 1)  # Sử dụng chỉ số làm đặc trưng
    y = monthly_data['Price ($)'].values

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tiền xử lý dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Khởi tạo các mô hình
    models = {
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
    }

    # Tạo một hình ảnh để vẽ các dự đoán của các mô hình khác nhau
    fig, ax = plt.subplots(figsize=(12, 8))

    # Dự đoán giá bằng các mô hình đã chọn
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predicted_prices = model.predict(scaler.transform(X.reshape(-1, 1)))
        monthly_data[f'{model_name} Predicted Price'] = predicted_prices

        # Vẽ đường dự đoán của mô hình
        sns.lineplot(x='Month', y=f'{model_name} Predicted Price', data=monthly_data, label=model_name, ax=ax, linestyle='--')

    # Vẽ đường giá thực tế
    sns.lineplot(x='Month', y='Price ($)', data=monthly_data, label='Actual Price', ax=ax)

    ax.set_title('Giá xe thực tế và dự đoán')
    ax.set_xlabel('Tháng')
    ax.set_ylabel('Giá ($)')
    plt.xticks(rotation=45)
    ax.legend()
    return fig
