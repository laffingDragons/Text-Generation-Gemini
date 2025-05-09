import streamlit as st
import google.generativeai as genai

# Streamlit app title and description
st.title("Story Generator")
st.write("Enter your Gemini API key and a story title to generate a hooking story!")

# Input field for Gemini API key
api_key = st.text_input("Gemini API Key", type="password", placeholder="Enter your API key")

# Input field for the story title
title = st.text_input("Story Title", placeholder="e.g., A Magical Adventure")

# Button to generate the story
if st.button("Generate Story"):
    if not api_key.strip():
        st.error("Please enter a Gemini API key.")
    elif not title.strip():
        st.error("Please enter a story title.")
    else:
        with st.spinner("Configuring API and generating story..."):
            try:
                # Configure the Google Generative AI model with the provided API key
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")

                # Generate the story
                prompt = f"Generate a hooking story about {title}"
                response = model.generate_content(prompt)
                story = response.text

                # Display the story
                st.subheader(f"Story: {title}")
                st.write(story)
            except Exception as e:
                st.error(f"An error occurred: {str(e)} (Check if your API key is valid)")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Google Generative AI")