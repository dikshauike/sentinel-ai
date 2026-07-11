import json
import os
import streamlit as st
import plotly.express as px
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
    
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# Try to get the key from Streamlit secrets, if not, use the local one
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except:
    st.error("Groq API Key missing. Please add it to Streamlit Secrets.")
    
@st.cache_resource
def setup_rag_system():
    try:
        loader = PyPDFLoader(r"C:\Users\Diksha Uike\Downloads\ET AI\safety_rules.pdf") # Use your full path!
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(pages)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(chunks, embeddings)
        return vector_db.as_retriever()
    except Exception as e:
        st.error(f"Could not load safety_rules.pdf. Error: {e}")
        return None

retriever = setup_rag_system()

def get_shift_data():
    iot_data = {
        "shift_end_time": "16:00",
        "zone_a_gas_max_ppm": 120,  
        "zone_b_gas_max_ppm": 380,  
        "zone_b_valve_status": "Open 20%", 
        "active_permits": ["Hot Work - Zone B", "Confined Space - Zone C"], # Changed Zone A to Zone B!
        "historical_incidents": "2023-11-04: Minor gas leak in Zone B due to open valve. 2024-01-15: Fire near Zone B due to unreported gas." # <--- ADD THIS LINE
    }    
    supervisor_logbook = """
    Shift A Summary by Supervisor Raj:
    - Shift was mostly quiet. 
    - Zone A gas levels were normal. Hot work permit is active there.
    - Zone B was completely clear all shift. No issues to report.
    - All valves in Zone B are fully closed.
    - Nothing to warn Shift B about.
    """
    return iot_data, supervisor_logbook

def analyze_shift_handover(iot_data, logbook, retriever):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    search_query = "safety regulations for gas leaks, open valves, and hot work permits"
    retrieved_docs = retriever.invoke(search_query)
    rules_context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    system_prompt = """You are Sentinel AI, an expert industrial safety assistant and Emergency Response Orchestrator.
    Your job is to compare the RAW IoT sensor data against the Outgoing Supervisor's manual logbook to find discrepancies and compound risks.
    
    If a CRITICAL compound risk is detected (e.g., gas leak + hot work permit):
    1. Act as an Emergency Response Orchestrator.
    2. Generate a preliminary regulatory-compliant incident report.
    3. The report MUST include:
       - Immediate Evacuation Protocol (which zones to evacuate)
       - Alert Routing (who to notify, e.g., Fire Safety, Plant Manager)
       - Preserved Sensor Evidence (the exact IoT readings that triggered this)
       - Regulatory Violations (cited from the provided Safety Regulations Context)
    
    Also, provide a 3-bullet point mandatory safety briefing for the incoming shift.
    Finally, generate a "Corrective Action Workflow" with step-by-step instructions to fix the compliance deviations.
    Format your response clearly using Markdown.
    """

    
    human_prompt = f"""
    Here is the RAW IoT DATA:
    {json.dumps(iot_data, indent=2)}
    Here is the SUPERVISOR LOGBOOK:
    {logbook}
    Here is the Safety Regulations Context retrieved from the PDF:
    {rules_context}
    Analyze this, cite the regulations, and provide your output.
    """
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    response = llm.invoke(messages)
    return response.content

# --- WEB UI ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stButton>button { background-color: #ff4b4b; color: white; border-radius:5px; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Sentinel AI: Shift-Change Context Bridge")
st.markdown("##### Automating the 16:00 Shift Handover - Detecting human error before it becomes a fatality.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 RAW IoT Sensor Data (The Truth)")
    iot_data, logbook = get_shift_data()
    st.json(iot_data)

with col2:
    st.subheader("📝 Outgoing Supervisor Logbook")
    st.info(logbook)

st.divider()

# --- THE GEOSPATIAL HEATMAP ---
st.subheader("🗺️ Geospatial Plant Risk Map")

# Risk Zones
df_risk = pd.DataFrame({
    "Zone": ["Zone A (Hot Work)", "Zone B (Gas Leak)", "Zone C (Confined Space)"],
    "X_Coord": [10, 20, 30],
    "Y_Coord": [10, 20, 10],
    "Risk_Level": [50, 100, 20] # 100 is highest risk (Zone B)
})

# Worker Locations (Simulating worker tracking)
df_workers = pd.DataFrame({
    "Worker_ID": ["W-01", "W-02", "W-03"],
    "X_Coord": [12, 22, 30],
    "Y_Coord": [11, 19, 12]
})

fig = px.scatter(df_risk, x="X_Coord", y="Y_Coord", size="Risk_Level", color="Risk_Level",
                 color_continuous_scale=["green", "yellow", "red"], text="Zone",
                 title="Real-Time Plant Risk & Worker Location Map")
# Add worker markers
fig.add_scatter(x=df_workers["X_Coord"], y=df_workers["Y_Coord"], mode="markers+text", 
                text=df_workers["Worker_ID"], marker=dict(size=15, color="blue", symbol="x"),
                name="Workers")
fig.update_layout(plot_bgcolor='#0e1117', paper_bgcolor='#0e1117', font_color='white')
st.plotly_chart(fig, use_container_width=True)

st.divider()

if st.button("🚨 Analyze Shift Handover", use_container_width=True):
    if retriever is None:
        st.warning("Please add safety_rules.pdf to the folder.")
    
    with st.spinner("Sentinel AI is cross-referencing logs, sensors, and legal regulations..."):
        ai_result = analyze_shift_handover(iot_data, logbook, retriever)
    
    st.success("Analysis Complete! Discrepancies & Regulatory Violations Found.")
    st.subheader("🤖 Sentinel AI Briefing for Incoming Shift B:")
    st.markdown(ai_result)
    
    # --- THE KNOWLEDGE GRAPH INNOVATION ---
    st.divider()
    st.subheader("🧠 AI Knowledge Graph: Compound Risk Mapping")
    st.markdown("##### How Sentinel AI connected the dots:")
    
    import networkx as nx
    import plotly.graph_objects as go
    
    # Create the graph
    G = nx.Graph()
    
    # Add nodes (Entities)
    G.add_node("Zone B\nEquipment", color="lightblue", size=1500)
    G.add_node("Valve Open 20%\nStatus", color="red", size=2000)
    G.add_node("Gas Leak (380ppm)\nSensor", color="red", size=2500)
    G.add_node("Hot Work Permit\nActivity", color="orange", size=2000)
    G.add_node("EXPLOSION RISK\nCompound Threat", color="darkred", size=3000)
    
    # Add edges (Relationships)
    G.add_edge("Zone B\nEquipment", "Valve Open 20%\nStatus")
    G.add_edge("Valve Open 20%\nStatus", "Gas Leak (380ppm)\nSensor")
    G.add_edge("Gas Leak (380ppm)\nSensor", "Hot Work Permit\nActivity")
    G.add_edge("Hot Work Permit\nActivity", "EXPLOSION RISK\nCompound Threat")
    
    # Get positions for the nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Create Plotly traces for the graph
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=3, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(G.nodes[node]['color'])
        node_text.append(node)
        
        node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        textfont=dict(color="white", size=12), # <--- THIS MAKES THE LABELS VISIBLE!
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_colors,
            size=30,
            line_width=2))

       
    # Draw the graph
    # Draw the graph
    fig_graph = go.Figure(data=[edge_trace, node_trace])

    fig_graph.update_layout(
        title=dict(text='AI Reasoning: Equipment -> Sensor -> Permit -> Risk', font=dict(size=16)),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

    # Make sure this line is indented exactly like the others!
    st.plotly_chart(fig_graph, use_container_width=True)