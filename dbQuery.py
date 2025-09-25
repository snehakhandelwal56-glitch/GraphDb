from sentence_transformers import SentenceTransformer, util
from neo4j import GraphDatabase
import numpy as np
import torch

# Neo4j connection details
URI = "bolt://localhost:7687"  # Replace with your Neo4j URI
USER = "neo4j"                 # Replace with your Neo4j username
PASSWORD = "xyz"             # Replace with your Neo4j password

# Connect to Neo4j
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# Load the pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and fast model

# Function to load precomputed embeddings
def load_precomputed_embeddings():
    return np.load("questions_embedding.npy")  # Replace with your file path

# Function to find the most similar question using precomputed embeddings 
def find_similar_question(user_question, precomputed_embeddings):
    # Step 1: Compute embedding for the user's question
    user_embedding = model.encode(user_question, convert_to_tensor=True)

    # Step 2: Ensure precomputed embeddings are on the same device
    device = user_embedding.device
    precomputed_embeddings = torch.tensor(precomputed_embeddings).to(device)

    # Step 3: Compute cosine similarity
    cosine_similarities = util.cos_sim(user_embedding, precomputed_embeddings)

    # Step 4: Find the most similar question
    most_similar_index = cosine_similarities.argmax().item()
    similarity_score = cosine_similarities[0][most_similar_index].item()

    # Step 5: Return result
    return most_similar_index, similarity_score

# Function to get a question by index from the database
def get_question_by_index(index):
    query = """
    MATCH (q:Question)
    RETURN q.question AS Question
    SKIP $index LIMIT 1
    """
    with driver.session() as session:
        result = session.run(query, index=index)
        record = result.single()
        return record["Question"] if record else None

# Function to get details for a question
def get_question_details(question):
    query = """
    MATCH (q:Question {question: $question})-[:RELATED_TO]->(k:KPI_Name)
    MATCH (kg:KPI_Group)-[:CONTAINS_KPI]->(k)
    MATCH (d:Domain)-[:HAS_KPI]->(k)
    RETURN k.name AS KPI_Name,
           collect(DISTINCT kg.name) AS KPI_Groups,
           collect(DISTINCT d.name) AS Domains
    """
    with driver.session() as session:
        result = session.run(query, question=question)
        details = []
        for record in result:
            details.append({
                "KPI_Name": record["KPI_Name"],
                "KPI_Groups": record["KPI_Groups"],
                "Domains": record["Domains"]
            })
        return details

# Main logic
try:
    # Step 1: Get the user question
    user_question = input("Enter your question: ").strip()

    # Step 2: Load precomputed embeddings
    precomputed_embeddings = load_precomputed_embeddings()

    # Step 3: Find the most similar question
    most_similar_index, similarity_score = find_similar_question(user_question, precomputed_embeddings)

    if similarity_score > 0.5:  # Similarity threshold
        # Step 4: Retrieve matched question
        similar_question = get_question_by_index(most_similar_index)

        # Step 5: Fetch details
        details = get_question_details(similar_question)

        # Output
        print(f"\nMatched Question: {similar_question}\n")

        for detail in details:
            kpi_name = detail['KPI_Name']
            kpi_groups = ", ".join(detail['KPI_Groups']) if detail['KPI_Groups'] else "N/A"
            domains = ", ".join(detail['Domains']) if detail['Domains'] else "N/A"

            print(f"KPI: {kpi_name}")
            print(f"KPI Group: {kpi_groups}")
            print(f"Domain: {domains}")
            print()
    else:
        print("Sorry, no similar question was found.")

except Exception as e:
    print(f"Error: {e}")

# Close the Neo4j driver
driver.close()
