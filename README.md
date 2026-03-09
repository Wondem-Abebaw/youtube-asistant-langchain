# YouTube Assistant

Ask questions about any YouTube video to this LLM powered assistant.

## Running it locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the streamlit app:

```bash
streamlit run main.py
```

![YouTube Assistant App](/YouTube-Assistant.png)

**The overall flow in summary:**

YouTube URL → fetch transcript → split into chunks →
embed chunks → store in FAISS → user asks question →
find relevant chunks → send to LLM with prompt → display answer

## Hosted On

The web-app uses streamlit and is hosted on [Azure Container Apps.](https://azure.microsoft.com/en-ca/products/container-apps)

## Author

- LinkedIn: [wondem-Abebaw](https://www.linkedin.com/in/wondem-abebaw-185612209/)
