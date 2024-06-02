import pandas as pd
import numpy as np
import json
import os
def templates():
    # Reading json data
    data_folder = "C:\Personal\Personal documents\Github\FolioLLM\data"
    json_data = data_folder +"\\"+ "etf_data_v3_clean.json"
    etf_db = pd.read_json(json_data)

    return etf_db, data_folder



def gen_q1_prompt_response(etf_db, number):
    # Question #1 prompt generation
    # Sort the DataFrame by YTD_Return in descending order
    etf_db_sorted = etf_db.sort_values(by='ytd_return', ascending=False)

    # Sample 'number' ETFs from the sorted DataFrame
    sampled_etfs = etf_db_sorted.head(number)

    # Create a list of ETF names
    etf_ticker_list = sampled_etfs['ticker'].tolist()
    etf_names_list = sampled_etfs['etf_name'].tolist()


    # Calculate equal weights
    weight = 100 / number

    # Generate the question
    question = f"Create a portfolio of {number} ETFs with equal weights?"

    # Generate the response
    response = "Here is the list of suitable ETFs and their weightings:\n"
    response_data = []

    for etf in etf_names_list:
        response += f"{etf}: {weight:.2f}%, "
        response_data.append({"ETF": etf, "Weight (%)": weight})

    return question, response, response_data

def store_q1(etf_db, data_folder):
    # Iterate through the number of ETFs from 1 to 30 and store the results
    questions_responses = []
    response_dataframes = []

    for num_etfs in range(1, 31):
        question, response, _ = gen_q1_prompt_response(etf_db, num_etfs)
        questions_responses.append({
            "question": question,
            "response": response
        })

    # Store the questions and responses in a JSON file
    json_file_path = os.path.join(data_folder, 'q1_responses.json')
    with open(json_file_path, 'w') as file:
        json.dump(questions_responses, file, indent=4)


    # Example of accessing the stored questions, responses, and DataFrames
    for i, (question, response) in enumerate(questions_responses):
        print(f"Question {i + 1}:")
        print(question)
        print(response)
        print(response_dataframes[i])
        print("\n")

    # Optionally, you can concatenate all response dataframes if needed
    all_responses_df = pd.concat(response_dataframes, keys=range(1, 31), names=['Number of ETFs'])
    print(all_responses_df)