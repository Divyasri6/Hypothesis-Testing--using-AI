import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

import os

llm=OpenAI(openai_api_key=os.environ["OPEN_API_KEY"],temperature=0.6)

def perform_z_test(sample_mean, population_mean, sample_stddev, sample_size, alpha):
    z_score = (sample_mean - population_mean) / (sample_stddev / np.sqrt(sample_size))
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
    #z_statistic, p_value = ztest_1samp((sample_mean,), population_mean, sigma=sample_stddev/np.sqrt(sample_size))
    if p_value < alpha:
      result = "Reject Null Hypothesis"
      interpretation = f"At a significance level of {alpha}, we can reject the null hypothesis. This suggests a statistically significant difference between the sample mean ({sample_mean:.2f}) and the population mean ({population_mean:.2f})."
    else:
      result = "Fail to Reject Null Hypothesis"
      interpretation = f"At a significance level of {alpha}, we fail to reject the null hypothesis. This means we cannot conclude a statistically significant difference between the sample mean ({sample_mean:.2f}) and the population mean ({population_mean:.2f}) based on this data."
    return result, interpretation

# Function to perform t-test
def perform_t_test(sample1, sample2, alpha):
    _, p_value = stats.ttest_ind(sample1, sample2, equal_var=False)
    if p_value < alpha:
        return "Reject Null Hypothesis"
    else:
        return "Fail to Reject Null Hypothesis"

def provide_test_guidance(data_description):
    prompt_template = PromptTemplate(
        input_variables=["data_description", "research_question"],
        template=f"Based on the provided data description ('{{data_description}}') , which hypothesis test is most suitable?"
        #template=f"Based on the provided data description ('{{data_description}}') and research question ('{{research_question}}'), which hypothesis test is most suitable?"
    )
    formatted_prompt = prompt_template.format(data_description=data_description)
    response = llm.generate([formatted_prompt])
    return response.generations[0][0].text.strip()
# Main function
def main():
    st.title("Hypothesis Testing Tool")
    tab1, tab2, = st.tabs(["perform hypothesis testing","Guidance"])

    with tab2:
        data_description = st.text_area("Describe your data")
        if st.button("Get Test Guidance"):
            if data_description :
                guidance = provide_test_guidance(data_description)
                st.write(guidance)
            else:
                st.write("Please provide data description")

      # Upload CSV data
    with tab1:
        data_source = st.radio("Data Source", ("Enter data manually","Upload CSV file"))
        if data_source == "Upload CSV file":
            uploaded_file = st.file_uploader("Upload CSV Data:", type="csv")
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.success("Data uploaded successfully!")
                except pd.errors.ParserError:
                    st.error("Error parsing CSV file. Please ensure it's valid.")

                st.header("Data Preview")
                st.write(data.head())

                test_type = st.radio("Select Test", ("Z-Test", "T-Test"))
                if test_type == "Z-Test":
                    st.subheader("Z-Test")
                    selected_column = st.selectbox("Select column", data.columns,help="Choose the column containing your sample data.")
                    sample_mean = data[selected_column].mean()
                    population_mean = st.number_input("Population Mean", help="Enter the theoretical average value of the entire population.")
                    sample_stddev = data[selected_column].std()
                    sample_size = len(data)
                    alpha = st.number_input("Significance Level (alpha)", value=0.05, help="Set the probability of rejecting the null hypothesis when it's true.")
                    if st.button("Perform Z-Test", key="ztest_button"):
                        result, interpretation = perform_z_test(sample_mean, population_mean, sample_stddev, sample_size, alpha)
                        st.write("Result:", result)
                        st.write(interpretation)

                elif test_type == "T-Test":
                    st.subheader("T-Test")
                    all_columns = list(data.columns)
                    sample1_choices = st.selectbox("Sample 1 ",all_columns, help="Select the column containing your first sample data.")
                    sample2_choices = st.selectbox("Sample 2 ",all_columns, help="Select the column containing your second sample data.")
                    alpha = st.number_input("Significance Level (alpha)", value=0.05, help="Set the probability of rejecting the null hypothesis when it's true.")
                    if st.button("Perform T-Test"):
                        sample1 = data[sample1_choices].to_numpy()
                        sample2 = data[sample2_choices].to_numpy()
                        result = perform_t_test(sample1, sample2, alpha)
                        st.write("Result:", result)
        elif data_source == "Enter data manually":
            st.header("Enter Data Manually")
            test_type = st.radio("Select Test", ("Z-Test", "T-Test"))
            if test_type == "Z-Test":
                sample_mean = st.number_input("Sample Mean", help="Enter the mean value of your sample data.")
                population_mean = st.number_input("Population Mean")
                sample_stddev = st.number_input("Sample Standard Deviation", help="Enter the standard deviation of your sample data.")
                sample_size = st.number_input("Sample Size", help="Enter the number of data points in your sample.")
                alpha = st.number_input("Significance Level (alpha)", value=0.05, help="Set the significance level for the test.")
                if st.button("Perform Z-Test"):
                    result , interpretation = perform_z_test(sample_mean, population_mean, sample_stddev, sample_size, alpha)
                    st.write("Result:", result)
                    st.write(interpretation)

            elif test_type == "T-Test":
                sample1 = st.text_area("Enter Sample 1 (comma-separated)", help="Enter the data points for Sample 1, separated by commas. for ex:1,2,3")
                sample2 = st.text_area("Enter Sample 2 (comma-separated)", help="Enter the data points for Sample 2, separated by commas.")
                alpha = st.number_input("Significance Level (alpha)", value=0.05, help="Set the significance level for the test.")
                try:
                # Convert user input to arrays
                    sample1 = np.array([float(x.strip()) for x in sample1.split(",")if x.strip()])
                    sample2 = np.array([float(x.strip()) for x in sample2.split(",")if x.strip()])
                except ValueError:
                    st.error("Error: Data contains non-numeric values.")
                else:
                    result = perform_t_test(sample1, sample2, alpha)
                    st.write("Result:", result)
            
if __name__ == "__main__":
    main()

