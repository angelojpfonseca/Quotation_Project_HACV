from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.7)

def analyze_product(product_description):
    prompt = PromptTemplate(
        input_variables=["product"],
        template="Analyze the following product and extract key features:\n{product}\n\nKey features:"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(product_description)

def compare_products(product1, product2):
    prompt = PromptTemplate(
        input_variables=["product1", "product2"],
        template="Compare the following two products:\n\nProduct 1: {product1}\n\nProduct 2: {product2}\n\nComparison:"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(product1=product1, product2=product2)

# Usage example:
# features = analyze_product(daikin_product)
# comparison = compare_products(daikin_product, melco_product)