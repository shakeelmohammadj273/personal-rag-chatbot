# Projects

## Projects by company

Projects Mohammad has done in each company:

- EXL Solutions (Dubai, 2026 – present): PPE Detection System and Truck Speed Monitoring, both real-time computer vision systems for industrial safety using YOLO and OpenCV.
- IGO Solutions (Chennai, 2025): Container Stacking Prediction, a machine learning solution for terminal port operations using XGBoost, AWS SageMaker and Amazon S3.
- InstaVC (Hyderabad, 2024 – 2025): real-time video conferencing solutions with React.js and WebRTC, plus serverless Python APIs on AWS Lambda.
- Peoplelink Unified Communication (Hyderabad, 2023 – 2024): Frame Xtract human-detection video frames, a Flask and OpenCV face detection web app, and OCR-based frame extraction.

## Personal RAG Chatbot (October 2025)

Mohammad built a Retrieval-Augmented Generation assistant that answers questions about his background using a local LLM. It uses Python, Ollama with Llama 3.2, LangChain, ChromaDB, Sentence Transformers (BAAI/bge-small-en-v1.5), FastAPI and React. It loads his documents, splits them into chunks, creates embeddings and stores them in ChromaDB. At question time it retrieves the most relevant chunks and gives them to Llama 3.2, which runs locally through Ollama, so it does not depend on external hosted model APIs. The application is split into document ingestion, retrieval, context building and response generation.

## PPE Detection System (EXL Solutions)

An AI system that detects workers who are not wearing required PPE (helmet and safety vest) from CCTV camera streams and raises an alert on each violation. Built with Python, YOLO, OpenCV and Docker.

## Truck Speed Monitoring (EXL Solutions)

An AI system that detects and tracks trucks across frames in terminal yards, estimates their speed, and alerts when a speed threshold is exceeded. Built with Python, OpenCV and YOLO.

## Container Stacking Prediction (IGO Solutions, 2025)

A machine learning solution for container placement in terminal port operations, built with Python, XGBoost and Scikit-learn. The trained models were integrated with AWS SageMaker and Amazon S3, and business rules were applied on top of the predictions.

## Speech To Text Translation Using NLP (May 2024)

A multilingual translation project built in Google Colab. It records audio in real time, transcribes it to text with automatic speech recognition, tokenizes the text and translates it into other languages, using deep learning and a Speech-to-Text API.

## Other Projects

Mohammad has also worked on Invoice OCR (using PaddleOCR), Container Number Recognition, an STS Crane Camera project and an AI Camera System.