import streamlit as st

st.title("My first streamlit App created by Ravi teja M")

st.write("Welcome! This app caculates the squar of a number")

st.header("Select a Number")
number = st.slider("Pick a number",0,100,5)

st.subheader("Result")

square_number = number * number

st.write(f"The square of **{number}** is **{square_number}**.")