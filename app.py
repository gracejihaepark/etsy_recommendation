import functions as func
import streamlit as st
import pandas as pd


# df = pd.read_csv('dfclusterid.csv')
# df = df.drop(columns=['Unnamed: 0'])
#
#
#
# termdf = pd.read_csv('clustertermdf.csv')
# termdf = termdf.drop(columns=['Unnamed: 0'])
#
#

st.markdown("""
<style>
body {
    background-color: #ffffff;
}

</style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #eb6d20; font: Times New Roman; font-size: 72px;''>ETSY RECOMMENDER</h1>", unsafe_allow_html=True)


st.markdown("<h6 style='text-align: left; color: black; font: Times New Roman; font-size: 18px;''>What would you like to search for?</h6>", unsafe_allow_html=True)
tokens = st.text_input('')


if tokens != '':
    for i in range(10):
        st.write(func.top10(tokens)[i])
