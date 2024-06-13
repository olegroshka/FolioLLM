# FolioLLM - ETF Portfolio Construction Using LLM

## Description
**FolioLLM** is a project for Stanford University's CS224n course, aimed at developing a domain-specific large language model (LLM) to assist investors and financial professionals in constructing optimal portfolios of ETFs. Leveraging advancements in LLM technology, FolioLLM is designed to interpret user preferences and market conditions, providing personalized and actionable investment advice.

## Objectives
- **Primary Goal:** Investigate the capability of a fine-tuned LLM to effectively understand user preferences and market conditions to offer personalized ETF portfolio suggestions.
- **Secondary Goals:**
  - Evaluate FolioLLM's performance across various metrics including financial knowledge, portfolio optimization, and personalized recommendations.
  - Compare the efficacy of FolioLLM with existing baselines, such as traditional portfolio optimization methods and other financial LLMs like FinGPT.

## Methodology
- **Data Sources:** Utilizes a blend of macroeconomic data, descriptive data of investment funds and ETFs, and securities information from financial databases like Bloomberg.
- **Approach:**
  - **Pre-training:** Begins with a pre-trained model, enhancing its financial understanding through domain-specific data and texts.
  - **Fine-tuning:** Adapts the model specifically for ETF and portfolio management using curated datasets.
  - **Retrieval-Augmented Generation:** Enhances the model's ability to provide relevant responses based on user queries.
  - **Portfolio Optimization:** Integrates both traditional and modern optimization techniques to formulate optimal ETF allocations.

## Evaluation
- **Financial Metrics:** Uses Sharpe Ratio and Information Ratio to assess the risk-adjusted performance and benchmark comparisons.
- **NLP Metrics:** Evaluates the coherence and relevance of the model's responses to ensure quality and accuracy.

## Ethical Considerations
- **Bias and Fairness:** Ensures a balanced training dataset to minimize biases in recommendations.
- **Transparency and Explainability:** Aims to enhance model transparency by implementing methods that clarify the decision-making process, supported by visualizations and detailed explanations.

## Additional Information
The project also explores innovative approaches like Low-Rank Adaptation (LoRA) and Kolmogorov-Arnold Network (KAN) techniques to further enhance the model's performance. The integration of these methods aims to improve the model's understanding of complex interactions in ETF data and deliver more accurate and relevant financial advice.

## Links to Project Outputs
- [Final Report](https://drive.google.com/file/d/1TWmDpB4n24w1lkFbqEMh2ynUEldjEQPR/view?usp=drive_link): This document provides a comprehensive overview of the FolioLLM project, including the methodology, experiments, results, and analysis.
- [Project Poster](https://drive.google.com/file/d/1D4SFW0DBvX38lqY1XVp2wXAGlf4SJucQ/view?usp=drive_link): A visual summary of the project, highlighting key objectives, methods, and findings. 

These documents offer detailed insights into the development and evaluation of FolioLLM, showcasing its potential in the financial domain.
