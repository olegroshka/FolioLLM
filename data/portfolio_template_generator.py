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

def gen_q2_prompt_response2(etf_db, number, portfolio_share):
    df_sorted = etf_db.sort_values(by='ytd_return', ascending=False)
    sampled_etfs = df_sorted.head(number)
    etf_list = sampled_etfs['etf_name'].tolist()
    weight = portfolio_share / number
    question = f"Create a portfolio of {number} ETFs with equal weights, investing only {portfolio_share}% of the total portfolio: {', '.join(etf_list)}."
    response = "Here is the list of suitable ETFs and their weightings: "
    response_data = []
    for etf in etf_list:
        response += f"{etf}: {weight:.2f}%, "
        response_data.append({"ETF": etf, "Weight (%)": weight})
    response = response.rstrip(', ')
    return question, response, response_data


# Function to generate portfolio question and response for Q3 with long and short positions
def generate_portfolio_question_and_response_q3(etf_db, number_long, number_short):
    df_sorted = etf_db.sort_values(by='ytd_return', ascending=False)
    long_etfs = df_sorted.head(number_long)['etf_name'].tolist()
    short_etfs = df_sorted.tail(number_short)['etf_name'].tolist()
    weight_long = 100 / (2 * number_long)
    weight_short = 100 / (2 * number_short)
    question = f"Create a portfolio with {number_long} long ETF positions and {number_short} short ETF positions so that net position will be zero."
    response = "Here is the list of suitable ETFs and their weightings:\nLong positions: "
    response_data = []
    for etf in long_etfs:
        response += f"{etf}: {weight_long:.2f}% long, "
        response_data.append({"ETF": etf, "Weight (%)": weight_long, "Position": "Long"})
    response = response.rstrip(', ') + "\nShort positions: "
    for etf in short_etfs:
        response += f"{etf}: {weight_short:.2f}% short, "
        response_data.append({"ETF": etf, "Weight (%)": -weight_short, "Position": "Short"})
    response = response.rstrip(', ')
    return question, response, response_data





def store_questions_responses(questions_responses, filename, data_folder):
    os.makedirs(data_folder, exist_ok=True)
    json_file_path = os.path.join(data_folder, filename)
    with open(json_file_path, 'w') as file:
        json.dump(questions_responses, file, indent=4)

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
    store_questions_responses(questions_responses, 'q1_responses.json')

def store_q2(etf_db, data_folder):
    # Iterate through the number of ETFs from 1 to 30 and portfolio shares from 10% to 90% for Q2
    for portfolio_share in range(10, 100, 10):
        questions_responses_q2 = []
        for num_etfs in range(1, 31):
            question, response, _ = gen_q2_prompt_response2(etf_db, num_etfs, portfolio_share)
            questions_responses_q2.append({
                "question": question,
                "response": response
            })
    filename = f'q2_responses.json'
    store_questions_responses(questions_responses_q2, filename, data_folder)