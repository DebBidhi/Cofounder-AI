import asyncio
import datetime
import logging
import os
import time
from functools import wraps
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def timeit(func):
    @wraps(func)
    async def inner(*args, **kwargs):
        t_start = time.time()
        res = await func(*args, **kwargs)
        t_exec = time.time() - t_start
        return res, t_exec * 1000
    return inner

class Vectorstore:
    def __init__(self, embedder: SentenceTransformer, index_name: str, namespace: str = "default"):
        self.embedder = embedder
        self.index_name = index_name
        self.namespace = namespace
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(self.index_name)

    async def add_conversation(self, speaker: str, message: str, timestamp: float):
        text = f"{speaker}: {message}"
        chunks = self._chunk_text(text)
        embeddings = await asyncio.to_thread(self.embedder.encode, chunks)
        
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{timestamp}_{i}"
            vectors.append((vector_id, embedding.tolist(), {
                "text": chunk,
                "timestamp": timestamp,
                "speaker": speaker,
                "chunk_index": i
            }))
        
        await asyncio.to_thread(self.index.upsert, vectors=vectors, namespace=self.namespace)

    def _chunk_text(self, text: str, chunk_size: int = 200) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 for space
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    async def get_conversation_context(self, query: str, k: int = 5, window_size: int = 3) -> List[Dict[str, Any]]:
        query_embedding = await asyncio.to_thread(self.embedder.encode, [query])
        query_embedding = query_embedding[0].tolist()
        results = await asyncio.to_thread(
            self.index.query,
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            namespace=self.namespace
        )

        context = []
        for result in results.matches:
            timestamp = result.metadata.get('timestamp')
            if timestamp is None:
                logging.warning(f"Timestamp missing for result: {result.metadata}")
                continue
            
            try:
                window_before = await asyncio.to_thread(
                    self.index.query,
                    filter={"timestamp": {"$lt": timestamp}},
                    top_k=window_size,
                    include_metadata=True,
                    namespace=self.namespace
                )
                window_after = await asyncio.to_thread(
                    self.index.query,
                    filter={"timestamp": {"$gt": timestamp}},
                    top_k=window_size,
                    include_metadata=True,
                    namespace=self.namespace
                )

                context.extend([m.metadata for m in window_before.matches])
                context.append(result.metadata)
                context.extend([m.metadata for m in window_after.matches])
            except Exception as e:
                logging.error(f"Error querying window for timestamp {timestamp}: {e}")

        # Remove duplicates and sort by timestamp
        seen = set()
        unique_context = []
        for item in context:
            timestamp = item.get('timestamp')
            if timestamp is not None and timestamp not in seen:
                seen.add(timestamp)
                unique_context.append(item)

        return sorted(unique_context, key=lambda x: x.get('timestamp', 0))

    @classmethod
    def create(cls, embedder: SentenceTransformer, index_name: str, namespace: str = "default") -> "Vectorstore":
        load_dotenv()
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        try:
            if index_name not in pc.list_indexes():
                pc.create_index(
                    name=index_name,
                    dimension=embedder.get_sentence_embedding_dimension(),
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                logging.info(f"Created new index: {index_name}")
        except Exception as e:
            if "ALREADY_EXISTS" in str(e):
                logging.warning(f"Index '{index_name}' already exists. Using the existing index.")
            else:
                logging.error(f"Error creating index: {e}")
                raise
        
        return cls(embedder=embedder, index_name=index_name, namespace=namespace)

    async def load_recent_conversations(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            query_embedding = await asyncio.to_thread(self.embedder.encode, [query])
            query_embedding = query_embedding[0].tolist()
            results = await asyncio.to_thread(
                self.index.query,
                vector=query_embedding,
                top_k=limit,
                include_metadata=True,
                namespace=self.namespace,
            )

            conversations = sorted(
                [result.metadata for result in results.matches if 'timestamp' in result.metadata],
                key=lambda x: x['timestamp'],
                reverse=True
            )

            return conversations

        except Exception as e:
            logging.error(f"Error loading recent conversations: {e}")
            return []

    async def search(self, query: str, k: int = 5) -> Tuple[List[Dict[str, Any]], List[float]]:
        query_embedding = await asyncio.to_thread(self.embedder.encode, [query])
        query_embedding = query_embedding[0].tolist()
        results = await asyncio.to_thread(
            self.index.query,
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            namespace=self.namespace
        )
        docs = [result.metadata for result in results.matches]
        scores = [result.score for result in results.matches]
        return docs, scores

    async def add_context_from_file(self, file_path: str):
        with open(file_path, 'r') as f:
            content = f.read()
        chunks = self._chunk_text(content)
        timestamp = time.time()
        for chunk in chunks:
            embedding = await asyncio.to_thread(self.embedder.encode, [chunk])
            embedding = embedding[0].tolist()
            await asyncio.to_thread(
                self.index.upsert,
                vectors=[(str(timestamp), embedding, {"text": f"Context: {chunk}"})],
                namespace=self.namespace
            )
            timestamp += 0.001  # Ensure different timestamps for each chunk

    async def consolidate_memory(self, older_than_days: int = 30):
        cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
        
        # Retrieve old conversations
        old_conversations = await self.load_recent_conversations(
            query="",
            limit=1000,
            filter={"timestamp": {"$lt": cutoff_time}}
        )

        if not old_conversations:
            return

        # Group conversations by day
        grouped_conversations = {}
        for conv in old_conversations:
            date = datetime.datetime.fromtimestamp(conv['timestamp']).date()
            if date not in grouped_conversations:
                grouped_conversations[date] = []
            grouped_conversations[date].append(conv['text'])

        # Summarize each day's conversations
        for date, conversations in grouped_conversations.items():
            summary = await self.summarize_conversations(conversations)
            
            # Store the summary as a new vector
            summary_text = f"Summary for {date}: {summary}"
            summary_embedding = await asyncio.to_thread(self.embedder.encode, [summary_text])
            summary_embedding = summary_embedding[0].tolist()
            await asyncio.to_thread(
                self.index.upsert,
                vectors=[(str(date), summary_embedding, {"text": summary_text, "timestamp": date.timestamp(), "type": "summary"})],
                namespace=self.namespace
            )

        # Remove old individual conversation vectors
        await asyncio.to_thread(
            self.index.delete,
            filter={"timestamp": {"$lt": cutoff_time}, "type": {"$ne": "summary"}},
            namespace=self.namespace
        )

    async def summarize_conversations(self, conversations: List[str]) -> str:
        # Implement your summarization logic here
        # This could be a call to an LLM or a custom summarization algorithm
        pass

class CofounderAI:
    def __init__(self, vectorstore: Vectorstore):
        self.vectorstore = vectorstore
        load_dotenv()
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.conversation_history = []
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self):
        with open('config.txt', 'r') as f:
            return f.read().strip()

    async def chat(self, user_input: str):
        try:
            relevant_context = await self.vectorstore.get_conversation_context(user_input, k=5)
            recent_context = await self.vectorstore.load_recent_conversations(user_input, limit=3)
            
            context = self._format_context(relevant_context + recent_context)
            system_prompt = self._create_system_prompt(context)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]

            chat_completion = await self.groq_client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",
            )
            ai_response = chat_completion.choices[0].message.content.strip()

            timestamp = time.time()
            await asyncio.gather(
                self.vectorstore.add_conversation("User", user_input, timestamp),
                self.vectorstore.add_conversation("AI", ai_response, timestamp + 0.001)
            )

            self.conversation_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": ai_response}
            ])

            return ai_response
        except Exception as e:
            logging.error(f"Error in chat method: {e}")
            return "I apologize, but I encountered an error while processing your request. Could you please try again?"

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        formatted_context = []
        for item in context:
            timestamp = item.get('timestamp', 'Unknown time')
            text = item.get('text', 'No text available')
            formatted_context.append(f"[{timestamp}] {text}")
        return "\n".join(formatted_context)

    def _create_system_prompt(self, context: str) -> str:
        return f"""{self.system_prompt}

Use the following context from previous conversations if relevant:

{context}

Please provide a contextually relevant response based on the given information and the user's input. If the context doesn't seem relevant, feel free to generate a response based on your general knowledge."""

    async def add_context_from_file(self, file_path: str):
        await self.vectorstore.add_context_from_file(file_path)

# Initialize the embedding model and vector store
print("Loading the embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
vs = Vectorstore.create(embedder=model, index_name="cofounder-ai", namespace="default")

# Initialize the co-founder AI
cofounder = CofounderAI(vs)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('send_message')
async def handle_message(data):
    user_input = data['message']
    try:
        ai_response = await cofounder.chat(user_input)
        await socketio.emit('receive_message', {'sender': 'AI Co-founder', 'message': ai_response})
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        await socketio.emit('receive_message', {'sender': 'System', 'message': 'An error occurred. Please try again.'})

@socketio.on('add_context')
async def handle_add_context(data):
    file_path = data['file_path']
    try:
        await cofounder.add_context_from_file(file_path)
        await socketio.emit('receive_message', {'sender': 'System', 'message': f"Context added from {file_path}"})
    except Exception as e:
        logging.error(f"Error adding context: {e}")
        await socketio.emit('receive_message', {'sender': 'System', 'message': f"Error adding context: {str(e)}"})

if __name__ == '__main__':
    socketio.run(app, debug=True, use_reloader=False)