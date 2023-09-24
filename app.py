"""Chainlit UI implementation"""

import chainlit as cl

from main import chain, vector

template = """Question: {question}

Answer: Let's think step by step."""


@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    llm_chain = chain

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(question: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  ## type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall({'question':question, 'context':vector._run(question)}, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Do any post processing here

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.
    await cl.Message(content=res["text"]).send()
