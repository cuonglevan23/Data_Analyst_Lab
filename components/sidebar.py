import streamlit as st

def show_sidebar():
    st.sidebar.title('Điều Hướng')
    pages = ['Trang Chủ', 'Báo Cáo', 'Dự Đoán']
    selection = st.sidebar.radio('Đi đến', pages)
    st.session_state.page = selection
