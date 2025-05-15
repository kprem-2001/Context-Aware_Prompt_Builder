from logger_config import logging
from langchain_pinecone import PineconeVectorStore
from embedding import embedding

logging.info("Starting the prompt ")
docsearch = PineconeVectorStore.from_existing_index(
    index_name="hii",
    embedding=embedding

)

retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": 1})


user_prompt = input("Enter your story continuation instruction: ")  # Prompt input from user
print(f"User Instruction: {user_prompt}\n")


retrieved_docs = retriever.invoke(user_prompt)

retrieved_docs

if not retrieved_docs:
    print("No relevant slide found.")
else:
   
    most_relevant_slide_doc = retrieved_docs[0] if isinstance(retrieved_docs, list) else retrieved_docs
    relevant_slide_content = most_relevant_slide_doc.page_content

    slide_label = ""
    if relevant_slide_content == "Meena met Mathur in Lodhi Garden, under the rain.":
        slide_label = "Slide 1: "
    elif relevant_slide_content == "They shared chai and laughter in Khan Market.":
        slide_label = "Slide 2: "
    elif relevant_slide_content == "Mathur reveals he’s leaving for London in a week.": 
        slide_label = "Slide 3: "

    print("Most Relevant Slide:") 
    print(f'{slide_label}"{relevant_slide_content}"\n') 

    # 3. Print the "Final Prompt"
    final_prompt = (
        "You are continuing a love story between Meena and Mathur.\n"
        f"Previously, {relevant_slide_content.rstrip('.')}.\n" 
        f"Now, write the next slide with more emotional tension and vulnerability."
    )
    print("Final Prompt:") 
    print(final_prompt)
