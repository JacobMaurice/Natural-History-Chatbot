This is a local chat bot trained entirely on Pliny the Elder's Natural History.

The training data was downloaded from Project Gutenberg (https://www.gutenberg.org/ebooks/author/50041).

The Data folder contains the manually cleaned versions of each volume, with the footnotes being separated into their own files for easier use.

data_prep.ipynb is used to prep the volumes to be chunked with the appropriate metadata for referencing. all_chunks.csv is the resulting chunks prepared for embedding and indexing.

build_index.py embeds every chunk (pliny_meta.pkl) and builds a FAISS flat inner-product index (pliny.index)

chatbot.py embeds your question and retrieves the top 5 chunks. It labels them [1]–[5] with Book and Chapter, and sends them to a local model through Ollama (llama3.1:8b in this case).

The chatbot.py SYSTEM_PROMPT and config.py TOP_K can be modified to adjust how the chat bot replies.
