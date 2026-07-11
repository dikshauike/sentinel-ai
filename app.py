import json
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. API Key Setup (Reads from Streamlit Secrets safely)
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except:
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

st.set_page_config(page_title="Sentinel AI - Shift Handover", layout="wide", page_icon="🛡️")

# Initialize session state for Forensic Snapshots
if 'snapshots' not in st.session_state:
    st.session_state.snapshots = {}

@st.cache_resource
def setup_rag_system():
    try:
        loader = PyPDFLoader("safety_rules.pdf")
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
    # Includes CCTV Mock Alert, Historical Incidents, and Zone B Hot Work Permit
    iot_data = {
        "shift_end_time": "16:00",
        "zone_a_gas_max_ppm": 120,  
        "zone_b_gas_max_ppm": 380,  
        "zone_b_valve_status": "Open 20%", 
        "active_permits": ["Hot Work - Zone B", "Confined Space - Zone C"],
        "cctv_analytics_alert": "Person detected entering Zone B during active gas leak",
        "historical_incidents": "2023-11-04: Minor gas leak in Zone B due to open valve. 2024-01-15: Fire near Zone B due to unreported gas."
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
    search_query = "safety regulations for gas leaks, open valves, hot work permits, and worker safety"
    retrieved_docs = retriever.invoke(search_query)
    rules_context = "\n".join([doc.page_content for doc in retrieved_docs])

    system_prompt = """You are Sentinel AI, an expert industrial safety assistant and Emergency Response Orchestrator.
    Your job is to compare the RAW IoT sensor data against the Outgoing Supervisor's manual logbook to find discrepancies and compound risks.
    
    Format your response EXACTLY like this using Markdown. DO NOT write long paragraphs. Use bullet points only.
    
    EXECUTIVE SUMMARY:
    [Exactly 3 bullet points. The most critical actions requiring human attention right now. Be blunt and urgent.]
    
    ### 🚨 DETAILED INCIDENT REPORT
    **Immediate Evacuation Protocol:**
    - [Bullet point]
    **Alert Routing:**
    - [Bullet point]
    **Preserved Sensor Evidence:**
    - [Bullet point]
    **Regulatory Violations:**
    - [Bullet point citing the exact rule/section]
    
    ### 📋 MANDATORY BRIEFING & CORRECTIVE ACTION WORKFLOW
    **Incoming Shift Briefing:**
    - [3 bullet points]
    **Corrective Action Workflow:**
    1. [Step 1]
    2. [Step 2]
    3. [Step 3]
    ...
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

# --- FORENSIC TIME-TRAVEL SIDEBAR ---
st.sidebar.title("📜 Incident Rewind")
st.sidebar.markdown("View past saved incident snapshots for audit & compliance.")
selected_snapshot = st.sidebar.selectbox("Select Snapshot", ["None"] + list(st.session_state.snapshots.keys()))

if selected_snapshot != "None":
    st.warning(f"⏪ VIEWING HISTORICAL SNAPSHOT: {selected_snapshot}")
    snapshot_data = st.session_state.snapshots[selected_snapshot]
    iot_data = snapshot_data['iot']
    logbook = snapshot_data['log']
    ai_result = snapshot_data['ai']
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📡 RAW IoT Sensor Data (Historical)")
        st.json(iot_data)
    with col2:
        st.subheader("📝 Outgoing Supervisor Logbook (Historical)")
        st.info(logbook)
        
    # Show Historical Map
    st.subheader("🗺️ Geospatial Plant Risk Map (Historical)")
    df_risk = pd.DataFrame({
        "Zone": ["Zone A (Hot Work)", "Zone B (Gas Leak)", "Zone C (Confined Space)"],
        "X_Coord": [10, 20, 30],
        "Y_Coord": [10, 20, 10],
        "Risk_Level": [50, 100, 20]
    })
    df_workers = pd.DataFrame({
        "Worker_ID": ["W-01", "W-02", "W-03"],
        "X_Coord": [12, 22, 30],
        "Y_Coord": [11, 19, 12]
    })
    fig = px.scatter(df_risk, x="X_Coord", y="Y_Coord", size="Risk_Level", color="Risk_Level",
                     color_continuous_scale=["green", "yellow", "red"], text="Zone",
                     title="Frozen Plant Risk Map")
    fig.add_scatter(x=df_workers["X_Coord"], y=df_workers["Y_Coord"], mode="markers+text", 
                    text=df_workers["Worker_ID"], marker=dict(size=15, color="blue", symbol="x"),
                    name="Workers")
    fig.update_layout(plot_bgcolor='#0e1117', paper_bgcolor='#0e1117', font=dict(color='white'))
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🤖 Sentinel AI Historical Briefing & Incident Report:")
    st.markdown(ai_result)
    st.stop()

# --- LIVE DASHBOARD ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 RAW IoT Sensor Data (The Truth)")
    iot_data, logbook = get_shift_data()
    st.json(iot_data)

with col2:
    st.subheader("📝 Outgoing Supervisor Logbook")
    st.info(logbook)

# --- MOCK CCTV ALERT ---
if iot_data.get("cctv_analytics_alert"):
    st.error(f"📹 CCTV ANALYTICS ALERT: {iot_data['cctv_analytics_alert']}")

# --- THE GEOSPATIAL HEATMAP ---
st.subheader("🗺️ Geospatial Plant Risk Map")
df_risk = pd.DataFrame({
    "Zone": ["Zone A (Hot Work)", "Zone B (Gas Leak)", "Zone C (Confined Space)"],
    "X_Coord": [10, 20, 30],
    "Y_Coord": [10, 20, 10],
    "Risk_Level": [50, 100, 20]
})
df_workers = pd.DataFrame({
    "Worker_ID": ["W-01", "W-02", "W-03"],
    "X_Coord": [12, 22, 30],
    "Y_Coord": [11, 19, 12]
})

fig = px.scatter(df_risk, x="X_Coord", y="Y_Coord", size="Risk_Level", color="Risk_Level",
                 color_continuous_scale=["green", "yellow", "red"], text="Zone",
                 title="Real-Time Plant Risk & Worker Location Map")
fig.add_scatter(x=df_workers["X_Coord"], y=df_workers["Y_Coord"], mode="markers+text", 
                text=df_workers["Worker_ID"], marker=dict(size=15, color="blue", symbol="x"),
                name="Workers")
fig.update_layout(plot_bgcolor='#0e1117', paper_bgcolor='#0e1117', font=dict(color='white'))
st.plotly_chart(fig, use_container_width=True)

st.divider()

if st.button("🚨 Analyze Shift Handover", use_container_width=True):
    if retriever is None:
        st.warning("Please add safety_rules.pdf to the folder.")
    
    with st.spinner("Sentinel AI is cross-referencing logs, sensors, and legal regulations..."):
        ai_result = analyze_shift_handover(iot_data, logbook, retriever)
    
    st.success("Analysis Complete! Discrepancies & Regulatory Violations Found.")
    
    # --- THE RED ZONE EXECUTIVE SUMMARY ---
    if "EXECUTIVE SUMMARY:" in ai_result:
        parts = ai_result.split("DETAILED INCIDENT REPORT:")
        summary_part = parts[0].replace("EXECUTIVE SUMMARY:", "").strip()
        detailed_part = "DETAILED INCIDENT REPORT:" + parts[1] if len(parts) > 1 else ""
        
        # The massive, impossible-to-miss Red Banner
        st.markdown(f"""
        <div style="background-color:#ffcccc; padding:20px; border-radius:10px; border-left:10px solid #ff0000;">
        <h3 style="color:#cc0000; margin-bottom:10px;">🚨 EXECUTIVE SUMMARY: ACT NOW</h3>
        <ul style="color:#990000; font-size:18px; font-weight:bold;">
        {''.join(f'<li>{line.strip().replace("- ", "")}</li>' for line in summary_part.split(chr(10)) if line.strip())}
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🤖 Sentinel AI Detailed Report for Incoming Shift B:")
        st.markdown(detailed_part)
    else:
        st.subheader("🤖 Sentinel AI Briefing for Incoming Shift B:")
        st.markdown(ai_result)
    
    # --- SAVE FORENSIC SNAPSHOT ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshot_name = f"Incident {timestamp}"
    st.session_state.snapshots[snapshot_name] = {
        'iot': iot_data,
        'log': logbook,
        'ai': ai_result
    }
    st.info(f"💾 Forensic Snapshot saved to sidebar as '{snapshot_name}' for audit purposes.")
    
    # --- THE KNOWLEDGE GRAPH ---
    st.divider()
    st.subheader("🧠 AI Knowledge Graph: Compound Risk Mapping")
    st.markdown("##### How Sentinel AI connected the dots:")
    
    G = nx.Graph()
    G.add_node("Zone B\nEquipment", color="lightblue", size=1500)
    G.add_node("Valve Open 20%\nStatus", color="red", size=2000)
    G.add_node("Gas Leak (380ppm)\nSensor", color="red", size=2500)
    G.add_node("Hot Work Permit\nActivity", color="orange", size=2000)
    G.add_node("CCTV: Worker in Zone B\nVideo Analytics", color="purple", size=2000)
    G.add_node("EXPLOSION RISK\nCompound Threat", color="darkred", size=3000)
    
    G.add_edge("Zone B\nEquipment", "Valve Open 20%\nStatus")
    G.add_edge("Valve Open 20%\nStatus", "Gas Leak (380ppm)\nSensor")
    G.add_edge("Gas Leak (380ppm)\nSensor", "Hot Work Permit\nActivity")
    G.add_edge("Gas Leak (380ppm)\nSensor", "CCTV: Worker in Zone B\nVideo Analytics")
    G.add_edge("Hot Work Permit\nActivity", "EXPLOSION RISK\nCompound Threat")
    G.add_edge("CCTV: Worker in Zone B\nVideo Analytics", "EXPLOSION RISK\nCompound Threat")
    
    pos = nx.spring_layout(G, seed=42)
    
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

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=3, color='#888'), hoverinfo='none', mode='lines')

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
        textfont=dict(color="white", size=12),
        hoverinfo='text',
        marker=dict(showscale=False, color=node_colors, size=30, line_width=2))
    
    fig_graph = go.Figure(data=[edge_trace, node_trace])
    fig_graph.update_layout(
        title=dict(text='AI Reasoning: Equipment -> Sensor -> Permit -> Risk', font=dict(size=16)),
        showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40),
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font=dict(color='white'),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    st.plotly_chart(fig_graph, use_container_width=True)
