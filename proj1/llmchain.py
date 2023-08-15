from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Step 1: Import the necessary modules
from dotenv import load_dotenv
import os

# Step 2: Set the OpenAI API Key
load_dotenv()

# Step 3: Get user input
user_input = input("Enter a concept: ")

# Step 4: Define the prompt template
prompt = PromptTemplate(
    input_variables=["concept"],
    template="Define {concept} with a real-world example?",
)

# Step 5: Print the Prompt Template
print(prompt.format(concept=user_input))

# Step 6: Instantiate the LLMChain
llm = OpenAI(temperature=0.9)
chain = LLMChain(llm=llm, prompt=prompt)

# Step 7: Run the LLMChain
output = chain.run(user_input)
print(output)