import streamlit as st
from components import home, reports, prediction
from components.sidebar import show_sidebar


def main():
    st.title('Dashboard')

    # Hiển thị thanh điều hướng (sidebar)
    show_sidebar()

    # Hiển thị nội dung của các trang
    page = st.session_state.page

    if page == 'Trang Chủ':
        home.show()
    elif page == 'Báo Cáo':
        reports.show()

    elif page == 'Dự Đoán':
          prediction.show()





if __name__ == '__main__':
    main()


