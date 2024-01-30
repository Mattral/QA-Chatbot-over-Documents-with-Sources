# QA-Chatbot-over-Documents-with-Sources


## Introduction
Let’s explore a more advanced application of Artificial Intelligence - building a Question Answering (QA) Chatbot that works over documents and provides sources of information for its answers. Our QA Chatbot uses a chain (specifically, the RetrievalQAWithSourcesChain), and leverages it to sift through a collection of documents, extracting relevant information to answer queries.

The chain sends structured prompts to the underlying language model to generate responses. These prompts are crafted to guide the language model's generation, thereby improving the quality and relevance of the responses. Additionally, the retrieval chain is designed to keep track of the sources of information it retrieves to provide answers, offering the ability to back up its responses with credible references.

As we proceed, we'll learn how to:

1. Scrape online articles and store each article's text content and URL.
2. Use an embedding model to compute embeddings of these documents and store them in Deep Lake, a vector database.
3. Split the article texts into smaller chunks, keeping track of each chunk's source.
4. Utilize RetrievalQAWithSourcesChain to create a chatbot that retrieves answers and tracks their sources.
5. Generate a response to a query using the chain and display the answer along with its sources.
This knowledge can be transformative, allowing you to create intelligent chatbots capable of answering questions with sourced information, increasing the trustworthiness and utility of the chatbot.

Let's dive in!

## Setup
Remember to install the required packages with the following command: pip install langchain deeplake openai==0.27.8 tiktoken. Additionally, install the newspaper3k package with version 0.2.8.

```
!pip install -q newspaper3k==0.2.8 python-dotenv
```

Then, you need to add your OpenAI and Deep Lake API keys to the environment variables. The LangChain library will read the tokens and use them in the integrations.

```

import os

os.environ["OPENAI_API_KEY"] = "<YOUR-OPENAI-API-KEY>"
os.environ["ACTIVELOOP_TOKEN"] = "<YOUR-ACTIVELOOP-API-KEY>"
```

## Scrapping for the News
Now, let's begin by fetching some articles related to AI news. We're particularly interested in the text content of each article and the URL where it was published.

In the code, you’ll see the following:

- Imports: We begin by importing necessary Python libraries. requests are used to send HTTP requests, the newspaper is a fantastic tool for extracting and curating articles from a webpage, and time will help us introduce pauses during our web scraping task.
- Headers: Some websites may block requests without a proper User-Agent header as they may consider it as a bot's action. Here we define a User-Agent string to mimic a real browser's request.
- Article URLs: We have a list of URLs for online articles related to artificial intelligence news that we wish to scrape.
- Web Scraping: We create an HTTP session using requests.Session() allows us to make multiple requests within the same session. We also define an empty list of pages_content to store our scraped articles.


```

import requests
from newspaper import Article # https://github.com/codelucas/newspaper
import time

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

article_urls = [
    "https://www.artificialintelligence-news.com/2023/05/16/openai-ceo-ai-regulation-is-essential/",
    "https://www.artificialintelligence-news.com/2023/05/15/jay-migliaccio-ibm-watson-on-leveraging-ai-to-improve-productivity/",
    "https://www.artificialintelligence-news.com/2023/05/15/iurii-milovanov-softserve-how-ai-ml-is-helping-boost-innovation-and-personalisation/",
    "https://www.artificialintelligence-news.com/2023/05/11/ai-and-big-data-expo-north-america-begins-in-less-than-one-week/",
    "https://www.artificialintelligence-news.com/2023/05/02/ai-godfather-warns-dangers-and-quits-google/",
    "https://www.artificialintelligence-news.com/2023/04/28/palantir-demos-how-ai-can-used-military/"
]

session = requests.Session()
pages_content = [] # where we save the scraped articles

for url in article_urls:
    try:
        time.sleep(2) # sleep two seconds for gentle scraping
        response = session.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            article = Article(url)
            article.download() # download HTML of webpage
            article.parse() # parse HTML to extract the article text
            pages_content.append({ "url": url, "text": article.text })
        else:
            print(f"Failed to fetch article at {url}")
    except Exception as e:
        print(f"Error occurred while fetching article at {url}: {e}")

#If an error occurs while fetching an article, we catch the exception and print
#an error message. This ensures that even if one article fails to download,
#the rest of the articles can still be processed.
```

Next, we'll compute the embeddings of our documents using an embedding model and store them in Deep Lake, a multimodal vector database. OpenAIEmbeddings will be used to generate vector representations of our documents. These embeddings are high-dimensional vectors that capture the semantic content of the documents. When we create an instance of the Deep Lake class, we provide a path that starts with hub://... that specifies the database name, which will be stored on the cloud.


```
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "<YOUR_ORGANIZATION_ID>"
my_activeloop_dataset_name = "langchain_course_qabot_with_source"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
```

This is a crucial part of the setup because it prepares the system for storing and retrieving the documents based on their semantic content. This functionality is key for the following steps, where we’d find the most relevant documents to answer a user's question.

Then, we'll break down these articles into smaller chunks, and for each chunk, we'll save its corresponding URL as a source. This division helps in efficiently processing the data, making the retrieval task more manageable, and focusing on the most relevant pieces of text when answering a question.

RecursiveCharacterTextSplitter is created with a chunk size of 1000, and 100 characters overlap between chunks. The chunk_size parameter defines the length of each text chunk, while chunk_overlap sets the number of characters that adjacent chunks will share. For each document in pages_content, the text will be split into chunks using the .split_text() method.

```
# We split the article texts into small chunks. While doing so, we keep track of each
# chunk metadata (i.e. the URL where it comes from). Each metadata is a dictionary and
# we need to use the "source" key for the document source so that we can then use the
# RetrievalQAWithSourcesChain class which will automatically retrieve the "source" item
# from the metadata dictionary.

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

all_texts, all_metadatas = [], []
for d in pages_content:
    chunks = text_splitter.split_text(d["text"])
    for chunk in chunks:
        all_texts.append(chunk)
        all_metadatas.append({ "source": d["url"] })
```

The "source" key is used in the metadata dictionary to align with the RetrievalQAWithSourcesChain class's expectations, which will automatically retrieve this "source" item from the metadata. We then add these chunks to our Deep Lake database along with their respective metadata.

```
# we add all the chunks to the deep lake, along with their metadata
db.add_texts(all_texts, all_metadatas)
```

Now comes the fun part - building the QA Chatbot. We'll create a RetrievalQAWithSourcesChain chain that not only retrieves relevant document snippets to answer the questions but also keeps track of the sources of these documents.

## Setting up the Chain 
We then create an instance of RetrievalQAWithSourcesChain using the from_chain_type method. This method takes the following parameters:

- LLM: This argument expects to receive an instance of a model (GPT-3, in this case) with a temperature of 0. The temperature controls the randomness of the model's outputs - a higher temperature results in more randomness, while a lower temperature makes the outputs more deterministic.
- chain_type="stuff": This defines the type of chain being used, which influences how the model processes the retrieved documents and generates responses. 
- retriever=db.as_retriever(): This sets up the retriever that will fetch the relevant documents from the Deep Lake database. Here, the Deep Lake database instance db is converted into a retriever using its as_retriever method.

```
# we create a RetrievalQAWithSourcesChain chain, which is very similar to a
# standard retrieval QA chain but it also keeps track of the sources of the
# retrieved documents

from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
                                                    chain_type="stuff",
                                                    retriever=db.as_retriever())
```

Lastly, we'll generate a response to a question using the chain. The response includes the answer and its corresponding sources.


```

# We generate a response to a query using the chain. The response object is a dictionary containing
# an "answer" field with the textual answer to the query, and a "sources" field containing a string made
# of the concatenation of the metadata["source"] strings of the retrieved documents.
d_response = chain({"question": "What does Geoffrey Hinton think about recent trends in AI?"})

print("Response:")
print(d_response["answer"])
print("Sources:")
for source in d_response["sources"].split(", "):
    print("- " + source)
```

```
Response:
 Geoffrey Hinton has expressed concerns about the potential dangers of AI, such as false text, images, and videos created by AI, and the impact of AI on the job market. He believes that AI has the potential to replace humans as the dominant species on Earth.

Sources:
- https://www.artificialintelligence-news.com/2023/05/02/ai-godfather-warns-dangers-and-quits-google/
- https://www.artificialintelligence-news.com/2023/05/15/iurii-milovanov-softserve-how-ai-ml-is-helping-boost-innovation-and-personalisation/
```

That's it! You've now built a question-answering chatbot that can provide answers from a collection of documents and indicate where it got its information.

# Conclusion
The chatbot was able to provide an answer to the question, giving a brief overview of Geoffrey Hinton's views on recent trends in AI. The sources provided and the answer traces back to the original articles expressing these views. This process adds a layer of credibility and traceability to the chatbot's responses. The presence of multiple sources also suggests that the chatbot was able to draw information from various documents to provide a comprehensive answer, demonstrating the effectiveness of the RetrievalQAWithSourcesChain in retrieving information.
