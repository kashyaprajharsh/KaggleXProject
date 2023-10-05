# KaggleXProject
# Finpro- Your Financial assistant
# AI for Invetors(creating Large Language model application for Nifty 50 Indian company using earning Calls from them )

A Financial Research assistant for Nifty-50 Indian Companies who can read earning Calls and annual reports for you and give You your answer.
It is using data upto August 2023.
We are currently developing a application that uses  Large Language Model that focuses on all the Nifty-50 Indian companies using their financial reports and earning calls as our primary data source. Our goal is to create a user-friendly web application that can provide investors with accurate and in-depth information about a company's performance, goals, and financial reports for any given financial year. With this tool, investors can easily access data related to a particular company and make informed investment decisions. I am utilizing the open-source Langchain framework and Hugging Face's Text-Generation bigscience/bloomz-1b7, which is an open-source Large Language Model. I am using the auto tokenizer and embedding to create tokens for the Bloom model, and I plan to fine-tune my large language model to provide accurate answers. I am planning to develop a web application using Streamlit that will allow users to upload their financial documents and ask questions related to them. I believe that using Generative AI can significantly improve financial research for any organization. It can provide smart insights to help them make better decisions.



## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`VertexAi_api`

`Huggingface_hub_api`


## Installation

Install my-project with Pip

```bash
  Pip install -r requirements.txt 
```
    
## Tech Stack

 **Library:** Langchain, HuggingFaceEmbedding, Python,StreamLit

**LLMs:** VertexAi API,Replicate,LLalma2 ,Palm2


## Screenshots

![App Screenshot](https://i.ibb.co/F3Hkhd5/image.png)

![App Screenshot](https://i.ibb.co/CJLVGNg/image.png)

## Acknowledgements

 - [Langchain](https://python.langchain.com/docs/get_started/introduction)
 - [bigscience/bloomz](https://huggingface.co/bigscience/bloomz)
 - [Paml2 LLMs](https://developers.generativeai.google/)

