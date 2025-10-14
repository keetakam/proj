# -*- coding: utf-8 -*-
"""
Unified Business Data Analyzer - COMPLETE VERSION
รวม: Complaint Chatbot + Single/Multi-File Data Analyzer + Chat Mode
"""

import streamlit as st
import pandas as pd
import json
import requests
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import time
from datetime import datetime
import hashlib
from io import BytesIO
import os
import logging

# Import PandasAI
try:
    from pandasai import SmartDataframe
    from pandasai.llm import LLM
    PANDASAI_AVAILABLE = True
except ImportError:
    PANDASAI_AVAILABLE = False

# ==================== Configuration ====================
st.set_page_config(
    page_title="🎯 Business Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATABASE = 'complaints.db'

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .mode-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        transition: all 0.3s;
    }
    .mode-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        animation: fadeIn 0.3s ease-in;
    }
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
    }
    .ai-message {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-left: 4px solid #9c27b0;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .file-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff9800;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# ==================== Database Functions ====================
def init_db():
    """Initialize complaints database with new structure"""
    with sqlite3.connect(DATABASE) as conn:
        # ตรวจสอบว่ามีตารางเก่าอยู่หรือไม่
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='complaints'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            # ตรวจสอบว่ามีคอลัมน์เก่าอยู่หรือไม่
            cursor.execute("PRAGMA table_info(complaints)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # ถ้ามีคอลัมน์เก่า ให้สร้างตารางใหม่และย้ายข้อมูล
            if 'complaint' in columns and 'subject' not in columns:
                # สร้างตารางใหม่
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS complaints_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        subject TEXT NOT NULL,
                        details TEXT NOT NULL,
                        name TEXT NOT NULL,
                        category TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # ย้ายข้อมูลจากตารางเก่าไปใหม่
                cursor.execute('''
                    INSERT INTO complaints_new (subject, details, name, category, timestamp)
                    SELECT 
                        SUBSTR(complaint, 1, 100) as subject,
                        complaint as details,
                        name,
                        NULL as category,
                        timestamp
                    FROM complaints
                ''')
                
                # ลบตารางเก่าและเปลี่ยนชื่อตารางใหม่
                conn.execute('DROP TABLE complaints')
                conn.execute('ALTER TABLE complaints_new RENAME TO complaints')
                conn.commit()
                
                st.success("✅ อัปเกรดฐานข้อมูลเรียบร้อยแล้ว")
            elif 'subject' not in columns:
                # ถ้าไม่มีตารางเลย ให้สร้างใหม่
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS complaints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        subject TEXT NOT NULL,
                        details TEXT NOT NULL,
                        name TEXT NOT NULL,
                        category TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
        else:
            # ถ้าไม่มีตารางเลย ให้สร้างใหม่
            conn.execute('''
                CREATE TABLE IF NOT EXISTS complaints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    details TEXT NOT NULL,
                    name TEXT NOT NULL,
                    category TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

def save_complaint(subject: str, details: str, name: str, category: str = None) -> int:
    """Save complaint to database with new structure"""
    # ตรวจสอบข้อมูลว่าง
    if not subject or not details or not name:
        raise ValueError("เรื่อง รายละเอียด และชื่อผู้ร้องเรียนต้องไม่ว่างเปล่า")
    
    # ทำความสะอาดข้อมูล
    subject = subject.strip()[:100]  # จำกัดความยาวหัวข้อ
    details = details.strip()[:2000]  # จำกัดความยาวรายละเอียด
    name = name.strip()[:100]  # จำกัดความยาวชื่อ
    
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO complaints (subject, details, name, category) VALUES (?, ?, ?, ?)",
            (subject, details, name, category)
        )
        conn.commit()
        return cursor.lastrowid

def get_complaints(limit: int = 100) -> List[Dict]:
    """Get complaints from database with new structure"""
    with sqlite3.connect(DATABASE) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            'SELECT id, subject, details, name, category, timestamp FROM complaints ORDER BY timestamp DESC LIMIT ?',
            (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

def get_recent_complaints_context(limit: int = 3) -> str:
    """Get recent complaints as context for RAG with new structure"""
    complaints = get_complaints(limit)
    return "\n".join([f"- {c['subject']}: {c['details']}" for c in complaints])

# ==================== Session State ====================
def init_session_state():
    """Initialize session state"""
    defaults = {
        'mode': 'home',
        'dataframes': {},
        'chat_history': [],
        'complaint_history': [],
        'new_mode_history': [],
        'llm': None,
        'api_stats': {'requests': 0, 'errors': 0, 'cache_hits': 0},
        'question_cache': {},
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==================== OpenRouter LLM ====================
def create_llm(api_key: str, model: str):
    """Create and return an OpenRouter LLM instance"""
    if not PANDASAI_AVAILABLE:
        raise RuntimeError("pandasai ไม่ได้ติดตั้ง แต่จำเป็นสำหรับโหมดนี้")
    
    # ตอนนี้เราแน่ใจว่า LLM มีอยู่
    class OpenRouterDirectLLM(LLM):
        """OpenRouter LLM using direct HTTP requests"""
        
        def __init__(
            self, 
            api_token: str, 
            model: str = "anthropic/claude-3-haiku",
            temperature: float = 0.2,
            max_tokens: int = 2000,
            **kwargs
        ):
            self.api_token = api_token
            self.model = model
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.kwargs = kwargs
            self.request_count = 0
            self.error_count = 0

        def call(self, instruction: str, context: str = "") -> str:
            """Call OpenRouter API"""
            # สร้าง system prompt สำหรับ pandasai
            system_prompt = {
                "role": "system",
                "content": (
                    "คุณเป็นผู้ช่วยวิเคราะห์ข้อมูลที่เชี่ยวชาญ "
                    "ตอบคำถามเป็นภาษาไทยที่ชัดเจน กระชับ และเป็นประโยชน์"
                )
            }
            
            user_content = f"{context}\n\n{instruction}" if context else instruction
            
            messages = [
                system_prompt,
                {
                    "role": "user",
                    "content": user_content
                }
            ]
            
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            try:
                self.request_count += 1
                st.session_state.api_stats['requests'] += 1
                
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                response.raise_for_status()
                data = response.json()
                
                if 'choices' in data and len(data['choices']) > 0:
                    return data['choices'][0]['message']['content']
                else:
                    return "❌ ไม่ได้รับคำตอบจาก AI"
                    
            except Exception as e:
                self.error_count += 1
                st.session_state.api_stats['errors'] += 1
                return f"❌ เกิดข้อผิดพลาด: {str(e)}"

        @property
        def type(self) -> str:
            return "openrouter"
        
        # เพิ่ม properties ที่จำเป็นสำหรับ pandasai LLM
        @property
        def model_name(self) -> str:
            return self.model
        
        def generate(self, prompt: str) -> str:
            """Generate response using OpenRouter API"""
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            }
            
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                response.raise_for_status()
                data = response.json()
                
                if 'choices' in data and len(data['choices']) > 0:
                    return data['choices'][0]['message']['content']
                else:
                    return "❌ ไม่ได้รับคำตอบจาก AI"
                    
            except Exception as e:
                return f"❌ เกิดข้อผิดพลาด: {str(e)}"

    return OpenRouterDirectLLM(api_token=api_key, model=model)

# ==================== File Processing Functions ====================
def load_json_file(file_content: bytes) -> tuple:
    """Load JSON file and return DataFrame"""
    try:
        raw_data = json.loads(file_content.decode('utf-8'))
        
        if isinstance(raw_data, list):
            if len(raw_data) == 0:
                return pd.DataFrame(), "ข้อมูลว่างเปล่า"
            
            if isinstance(raw_data[0], list):
                num_cols = len(raw_data[0])
                columns = [f"col_{i+1}" for i in range(num_cols)]
                df = pd.DataFrame(raw_data, columns=columns)
            else:
                df = pd.DataFrame(raw_data)
                
        elif isinstance(raw_data, dict):
            df = pd.DataFrame([raw_data])
        else:
            return pd.DataFrame(), "รูปแบบ JSON ไม่ถูกต้อง"
        
        return df, ""
        
    except Exception as e:
        return pd.DataFrame(), f"เกิดข้อผิดพลาด: {str(e)}"

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe"""
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Convert complex types to string
    for col in df_clean.columns:
        if df_clean[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df_clean[col] = df_clean[col].apply(
                lambda x: str(x) if isinstance(x, (list, dict)) else x
            )
    
    # Clean text columns
    text_cols = df_clean.select_dtypes(include=['object']).columns
    for col in text_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()
    
    return df_clean

def ask_question_about_data(question: str, dataframes: Dict[str, pd.DataFrame], llm: "LLM") -> str:
    """Ask question about data using SmartDataframe"""
    
    # Check cache
    q_hash = hashlib.md5(question.encode()).hexdigest()
    if q_hash in st.session_state.question_cache:
        st.session_state.api_stats['cache_hits'] += 1
        return st.session_state.question_cache[q_hash] + "\n\n💾 (จากแคช)"
    
    try:
        df_list = list(dataframes.values())
        
        if len(df_list) == 0:
            return "❌ ไม่มีข้อมูลให้วิเคราะห์"
        
        config = {
            "llm": llm,
            "verbose": False,
            "enable_cache": True,
            "conversational": False,
            "save_charts": False,
        }
        
        # Create SmartDataframe
        if len(df_list) == 1:
            sdf = SmartDataframe(df_list[0], config=config)
        else:
            sdf = SmartDataframe(df_list, config=config)
        
        # Ask question
        response = sdf.chat(question)
        
        # Format response
        if response is None:
            answer = "❌ ไม่ได้รับคำตอบจาก AI"
        elif isinstance(response, pd.DataFrame):
            answer = "📊 ผลลัพธ์:\n\n" + response.to_markdown(index=False)
        elif isinstance(response, (int, float)):
            answer = f"✅ ผลลัพธ์: {response:,.2f}"
        else:
            answer = str(response)
        
        # Cache successful response
        if not answer.startswith("❌"):
            st.session_state.question_cache[q_hash] = answer
        
        return answer
        
    except Exception as e:
        return f"❌ เกิดข้อผิดพลาด: {str(e)}\n\n💡 ลองถามคำถามแบบอื่น เช่น:\n- แสดงข้อมูล 10 แถวแรก\n- นับจำนวนแถวทั้งหมด"

# ==================== Chat Mode Handler ====================
def handle_new_mode_chat(api_key: str, model: str):
    """Handle new mode chatbot interface"""
    
    st.subheader("💬 โหมดแชท OpenRouter")
    
    # Initialize LLM if needed
    if not st.session_state.llm and api_key:
        try:
            st.session_state.llm = create_llm(api_key, model)
        except Exception as e:
            st.error(f"❌ สร้าง LLM ไม่ได้: {e}")
            return
    
    # Chat interface with instructions
    with st.expander("💡 วิธีใช้งาน", expanded=False):
        st.markdown(""" 
        1. พิมพ์ข้อความในช่องด้านล่าง
        2. กดปุ่ม "ส่ง" หรือกด Enter
        3. AI จะตอบกลับทันที
        """)
    
    # Use Form for Enter key support
    with st.form("new_mode_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_message = st.text_input(
                "ข้อความ",
                placeholder="พิมพ์คำถามหรือข้อความที่ต้องการถาม...",
                label_visibility="collapsed"
            )
        
        with col2:
            send_btn = st.form_submit_button("📤 ส่ง", type="primary", use_container_width=True)
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ ล้างการสนทนา", use_container_width=True):
            st.session_state.new_mode_history = []
            st.success("✅ ล้างการสนทนาแล้ว")
            st.rerun()
    
    with col2:
        if st.button("💾 ส่งออกประวัติ", use_container_width=True):
            if st.session_state.new_mode_history:
                df = pd.DataFrame(st.session_state.new_mode_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 ดาวน์โหลด CSV",
                    csv,
                    "chat_history.csv",
                    "text/csv",
                    use_container_width=True
                )
    
    # Process message
    if send_btn and user_message and st.session_state.llm:
        # Add user message
        st.session_state.new_mode_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Build conversation
        full_conversation = "\n\n".join([
            f"{m['role']}: {m['content']}" 
            for m in st.session_state.new_mode_history
        ])
        
        with st.spinner("🤔 AI กำลังตอบกลับ..."):
            bot_reply = st.session_state.llm.call(full_conversation)
        
        st.session_state.new_mode_history.append({
            "role": "assistant",
            "content": bot_reply,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        st.rerun()
    
    # Display chat history
    if st.session_state.new_mode_history:
        st.markdown("---")
        st.subheader("📜 ประวัติการสนทนา")
        
        for msg in st.session_state.new_mode_history[-10:]:
            if msg['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 คุณ [{msg.get('timestamp', '')}]:</strong><br/>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message ai-message">
                    <strong>🤖 AI [{msg.get('timestamp', '')}]:</strong><br/>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)

# ==================== Complaint Chat Handler ====================
def extract_complaint_info(conversation_history: List[Dict]) -> tuple:
    """Extract complaint info from conversation with new order: 1.แจ้งเรื่อง 2.แจ้งรายละเอียด 3.แจ้งชื่อ"""
    user_messages = [m['content'] for m in conversation_history if m['role'] == 'user']
    
    if len(user_messages) < 3:
        return None, None, None
    
    # ข้อความแรกคือเรื่องร้องเรียน (หัวข้อ)
    subject = user_messages[0].strip()
    
    # ข้อความที่สองคือรายละเอียด
    details = user_messages[1].strip()
    
    # ข้อความสุดท้ายคือชื่อผู้ร้องเรียน
    name = user_messages[-1].strip()
    
    # ตรวจสอบความยาวขั้นต่ำ
    if len(subject) < 2 or len(details) < 10 or len(name) < 2:
        return None, None, None
    
    return subject, details, name

def validate_complaint_data(subject: str, details: str, name: str) -> tuple:
    """Validate complaint data and return (is_valid, error_message)"""
    if not subject or not subject.strip():
        return False, "กรุณาระบุเรื่องร้องเรียน"
    
    if not details or not details.strip():
        return False, "กรุณาระบุรายละเอียดเรื่องร้องเรียน"
    
    if not name or not name.strip():
        return False, "กรุณาระบุชื่อผู้ร้องเรียน"
    
    if len(subject.strip()) < 2:
        return False, "เรื่องร้องเรียนต้องมีอย่างน้อย 2 ตัวอักษร"
    
    if len(details.strip()) < 10:
        return False, "รายละเอียดเรื่องร้องเรียนต้องมีอย่างน้อย 10 ตัวอักษร"
    
    if len(name.strip()) < 2:
        return False, "ชื่อผู้ร้องเรียนต้องมีอย่างน้อย 2 ตัวอักษร"
    
    return True, None

def handle_complaint_chat(api_key: str, model: str):
    """Handle complaint chatbot interface with new order: 1.แจ้งเรื่อง 2.แจ้งรายละเอียด 3.แจ้งชื่อ"""
    
    st.subheader("💬 ระบบรับเรื่องร้องเรียน")
    
    # Initialize LLM if needed
    if not st.session_state.llm and api_key:
        try:
            st.session_state.llm = create_llm(api_key, model)
        except Exception as e:
            st.error(f"❌ สร้าง LLM ไม่ได้: {e}")
            return
    
    # Chat interface
    with st.expander("💡 วิธีใช้งาน", expanded=False):
        st.markdown(""" 
        1. แจ้งเรื่องร้องเรียน (หัวข้อ)
        2. แจ้งรายละเอียดเรื่องร้องเรียน
        3. แจ้งชื่อผู้ร้องเรียน
        4. ระบบจะบันทึกอัตโนมัติเมื่อข้อมูลครบ
        """)
    
    # Use Form
    with st.form("complaint_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_message = st.text_input(
                "ข้อความ",
                placeholder="สวัสดีครับ ผมต้องการร้องเรียนเรื่อง...",
                label_visibility="collapsed"
            )
        
        with col2:
            send_btn = st.form_submit_button("📤 ส่ง", type="primary", use_container_width=True)
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ ล้างการสนทนา", use_container_width=True):
            st.session_state.complaint_history = []
            st.success("✅ ล้างแล้ว")
    
    with col2:
        if st.button("📋 ดูเรื่องร้องเรียนทั้งหมด", use_container_width=True):
            complaints = get_complaints(50)
            if complaints:
                st.dataframe(
                    pd.DataFrame(complaints),
                    use_container_width=True,
                    height=300
                )
            else:
                st.info("ยังไม่มีเรื่องร้องเรียน")
    
    # Process message
    if send_btn and user_message and st.session_state.llm:
        # Add user message
        st.session_state.complaint_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Get RAG context
        rag_context = get_recent_complaints_context()
        
        # นับจำนวนข้อความจากผู้ใช้
        user_messages_count = len([m for m in st.session_state.complaint_history if m['role'] == 'user'])
        
        # กำหนดคำถามตามขั้นตอน
        if user_messages_count == 1:
            # ขั้นที่ 1: รับเรื่องร้องเรียน
            prompt = f"""คุณคือผู้ช่วยที่เป็นมิตรในการรับเรื่องร้องเรียน

ข้อมูลเรื่องร้องเรียนล่าสุด:
{rag_context}

ผู้ใช้แจ้งเรื่องร้องเรียนว่า: "{user_message}"

ตอบกลับเพื่อขอรายละเอียดเพิ่มเติมเกี่ยวกับเรื่องร้องเรียนนี้
ตอบเป็นภาษาไทยที่เป็นกันเอง
"""
        elif user_messages_count == 2:
            # ขั้นที่ 2: รับรายละเอียดเรื่องร้องเรียน
            prompt = f"""คุณคือผู้ช่วยที่เป็นมิตรในการรับเรื่องร้องเรียน

ผู้ใช้แจ้งรายละเอียดเพิ่มเติมว่า: "{user_message}"

ตอบกลับเพื่อขอชื่อผู้ร้องเรียน
ตอบเป็นภาษาไทยที่เป็นกันเอง
"""
        else:
            # ขั้นที่ 3: รับชื่อผู้ร้องเรียน
            prompt = f"""คุณคือผู้ช่วยที่เป็นมิตรในการรับเรื่องร้องเรียน

ผู้ใช้แจ้งชื่อว่า: "{user_message}"

ตอบกลับเพื่อยืนยันการรับเรื่องร้องเรียนและขอบคุณ
ตอบเป็นภาษาไทยที่เป็นกันเอง
"""
        
        messages = [{"role": "system", "content": prompt}]
        
        full_conversation = "\n\n".join([
            f"{m['role']}: {m['content']}" for m in messages
        ])
        
        with st.spinner("🤔 AI กำลังตอบกลับ..."):
            bot_reply = st.session_state.llm.call(full_conversation)
        
        st.session_state.complaint_history.append({
            "role": "assistant",
            "content": bot_reply
        })
        
        # แสดงขั้นตอนปัจจุบัน
        if user_messages_count == 1:
            st.info("📝 ขั้นตอนที่ 1: รับเรื่องร้องเรียน (หัวข้อ) - สำเร็จ ✅")
        elif user_messages_count == 2:
            st.info("📝 ขั้นตอนที่ 2: รับรายละเอียดเรื่องร้องเรียน - สำเร็จ ✅")
        elif user_messages_count >= 3:
            st.info("📝 ขั้นตอนที่ 3: รับชื่อผู้ร้องเรียน - สำเร็จ ✅")
            
            # ดึงข้อมูลจากประวัติการสนทนา
            subject, details, name = extract_complaint_info(st.session_state.complaint_history)
            
            if subject and details and name:
                # ตรวจสอบความถูกต้องของข้อมูล
                is_valid, error_msg = validate_complaint_data(subject, details, name)
                
                if is_valid:
                    try:
                        complaint_id = save_complaint(subject, details, name)
                        st.success(f"✅ บันทึกเรื่องร้องเรียน ID: {complaint_id}")
                        st.balloons()
                        
                        # รีเซ็ตประวัติการสนทนาหลังจากบันทึกสำเร็จ
                        st.session_state.complaint_history = []
                    except Exception as e:
                        st.error(f"❌ เกิดข้อผิดพลาดในการบันทึก: {e}")
                else:
                    st.error(f"❌ ข้อมูลไม่ถูกต้อง: {error_msg}")
            else:
                st.error("❌ ไม่สามารถดึงข้อมูลจากการสนทนาได้ กรุณาลองใหม่")
        
        st.rerun()
    
    # Display chat history
    if st.session_state.complaint_history:
        st.markdown("---")
        for msg in st.session_state.complaint_history[-10:]:
            if msg['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 คุณ:</strong> {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message ai-message">
                    <strong>🤖 AI:</strong> {msg['content']}
                </div>
                """, unsafe_allow_html=True)

# ==================== Data Analysis Handler ====================
def handle_data_analysis(api_key: str, model: str, multiple: bool):
    """Handle data analysis interface"""
    
    mode_name = "หลายไฟล์" if multiple else "ไฟล์เดียว"
    st.subheader(f"📊 วิเคราะห์ข้อมูล ({mode_name})")
    
    # File upload
    uploaded_files = st.file_uploader(
        f"เลือกไฟล์ JSON" + (" (หลายไฟล์ได้)" if multiple else ""),
        type=['json'],
        accept_multiple_files=multiple,
        help="อัปโหลดไฟล์ JSON ที่ต้องการวิเคราะห์"
    )
    
    if uploaded_files:
        if not multiple:
            uploaded_files = [uploaded_files]
        
        if st.button("🚀 โหลดข้อมูล", type="primary"):
            st.session_state.dataframes = {}
            
            with st.spinner("📂 กำลังโหลดไฟล์..."):
                for file in uploaded_files:
                    file_content = file.read()
                    df, error = load_json_file(file_content)
                    
                    if error:
                        st.error(f"❌ {file.name}: {error}")
                    else:
                        df_clean = clean_dataframe(df)
                        if not df_clean.empty:
                            key = file.name.replace('.json', '')
                            st.session_state.dataframes[key] = df_clean
                            st.success(f"✅ {file.name}: {len(df_clean):,} แถว, {len(df_clean.columns)} คอลัมน์")
            
            # Setup LLM
            if st.session_state.dataframes:
                try:
                    st.session_state.llm = create_llm(api_key, model)
                    st.success("✅ เตรียม AI พร้อมแล้ว")
                except Exception as e:
                    st.error(f"❌ สร้าง LLM ไม่ได้: {e}")
    
    # Show data if loaded
    if st.session_state.dataframes:
        st.markdown("---")
        st.subheader(f"📊 ข้อมูลที่โหลด ({len(st.session_state.dataframes)} ไฟล์)")
        
        # Summary metrics
        total_rows = sum(len(df) for df in st.session_state.dataframes.values())
        total_cols = sum(len(df.columns) for df in st.session_state.dataframes.values())
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📁 ไฟล์", len(st.session_state.dataframes))
        col2.metric("📊 แถวรวม", f"{total_rows:,}")
        col3.metric("📋 คอลัมน์รวม", total_cols)
        col4.metric("🤖 AI", "พร้อม" if st.session_state.llm else "ยังไม่พร้อม")
        
        # Show dataframes
        for name, df in st.session_state.dataframes.items():
            with st.expander(f"📄 {name} ({len(df):,} แถว × {len(df.columns)} คอลัมน์)", expanded=False):
                
                # Show data info
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**คอลัมน์:**", ", ".join(df.columns.tolist()))
                with col2:
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if numeric_cols:
                        st.write("**คอลัมน์ตัวเลข:**", ", ".join(numeric_cols))
                
                # Show data preview
                st.dataframe(df.head(20), use_container_width=True)
                
                # Basic stats for numeric columns
                if numeric_cols:
                    st.write("**สถิติพื้นฐาน:**")
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Chat interface for asking questions
        if st.session_state.llm:
            st.markdown("---")
            st.subheader("💬 ถามคำถามเกี่ยวกับข้อมูล")
            
            # Example questions
            with st.expander("💡 ตัวอย่างคำถาม", expanded=False):
                st.markdown("""
                - แสดงข้อมูล 10 แถวแรก
                - มีข้อมูลทั้งหมดกี่แถว?
                - หาค่าเฉลี่ยของคอลัมน์ [ชื่อคอลัมน์]
                - แสดงกราฟแท่งของ [ชื่อคอลัมน์]
                - หาค่าสูงสุดและต่ำสุด
                - นับจำนวนแต่ละประเภทใน [ชื่อคอลัมน์]
                """)
            
            # Question form
            with st.form("question_form", clear_on_submit=True):
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    question = st.text_input(
                        "คำถาม",
                        placeholder="เช่น: มีข้อมูลกี่แถว? แสดง 5 แถวแรก",
                        label_visibility="collapsed"
                    )
                
                with col2:
                    ask_btn = st.form_submit_button("🚀 ถาม", type="primary", use_container_width=True)
            
            # Process question
            if ask_btn and question:
                with st.spinner("🤔 AI กำลังวิเคราะห์..."):
                    start = time.time()
                    answer = ask_question_about_data(
                        question,
                        st.session_state.dataframes,
                        st.session_state.llm
                    )
                    elapsed = time.time() - start
                
                st.session_state.chat_history.append({
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'question': question,
                    'answer': answer,
                    'elapsed': f"{elapsed:.2f}s"
                })
                
                st.rerun()
            
            # Show chat history
            if st.session_state.chat_history:
                st.markdown("---")
                st.subheader("📜 ประวัติคำถาม")
                
                for chat in reversed(st.session_state.chat_history[-5:]):
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 [{chat['time']}]:</strong> {chat['question']}
                    </div>
                    <div class="chat-message ai-message">
                        <strong>🤖 ({chat['elapsed']}):</strong><br/>{chat['answer']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Clear chat button
                if st.button("🗑️ ล้างประวัติการสนทนา", use_container_width=True):
                    st.session_state.chat_history = []
                    st.session_state.question_cache = {}
                    st.success("✅ ล้างแล้ว")

# ==================== Main Application ====================
def main():
    """Main application"""
    
    init_session_state()
    init_db()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Unified Business Platform</h1>
        <p>ระบบรับเรื่องร้องเรียน + วิเคราะห์ข้อมูลธุรกิจ</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ การตั้งค่า")
        
        # API Key
        api_key = st.text_input(
            "🔑 OpenRouter API Key",
            type="password",
            help="รับฟรีที่ https://openrouter.ai/keys"
        )
        
        # Model selection
        model = st.selectbox(
            "🤖 AI Model",
            [
                "anthropic/claude-3-haiku",
                "anthropic/claude-3-5-sonnet",
                "openai/gpt-4-turbo",
                "openai/gpt-3.5-turbo",
                "google/gemini-pro"
            ],
            help="เลือกโมเดล AI ที่ต้องการใช้"
        )
        
        # Mode selection
        st.markdown("---")
        st.subheader("📱 โหมดการใช้งาน")
        
        mode_options = {
            'home': '🏠 หน้าหลัก',
            'complaint': '💬 รับเรื่องร้องเรียน',
            'single_file': '📊 วิเคราะห์ไฟล์เดียว',
            'multi_file': '📁 วิเคราะห์หลายไฟล์',
            'new_mode': '📈 โหมดแชท'
        }
        
        for mode_key, mode_name in mode_options.items():
            if st.button(
                mode_name, 
                use_container_width=True, 
                type="primary" if st.session_state.mode == mode_key else "secondary"
            ):
                st.session_state.mode = mode_key
                st.rerun()
        
        # Stats
        st.markdown("---")
        st.subheader("📊 สถิติ")
        stats = st.session_state.api_stats
        col1, col2 = st.columns(2)
        col1.metric("📡 API Calls", stats['requests'])
        col2.metric("❌ Errors", stats['errors'])
        col1.metric("💾 Cache Hits", stats['cache_hits'])
        
        # System status
        st.markdown("---")
        st.subheader("🔧 สถานะระบบ")
        st.text(f"{'✅' if api_key else '⚠️'} API Key")
        st.text(f"{'✅' if PANDASAI_AVAILABLE else '⚠️'} PandasAI")
        st.text(f"{'✅' if st.session_state.llm else '⚠️'} AI Model")
    
    # Main content based on mode
    if st.session_state.mode == 'home':
        # Home page
        st.subheader("🏠 ยินดีต้อนรับสู่ Business Platform")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="mode-card">
                <h3>💬 ระบบรับเรื่องร้องเรียน</h3>
                <p>• แชทกับ AI เพื่อบันทึกเรื่องร้องเรียน</p>
                <p>• บันทึกอัตโนมัติลง Database</p>
                <p>• ดึงข้อมูลเดิมมาช่วย (RAG)</p>
                <p>• ดูประวัติเรื่องร้องเรียนทั้งหมด</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="mode-card">
                <h3>📊 วิเคราะห์ไฟล์เดียว</h3>
                <p>• อัปโหลด JSON ไฟล์เดียว</p>
                <p>• วิเคราะห์ข้อมูลด้วย AI</p>
                <p>• ถามคำถามเป็นภาษาธรรมชาติ</p>
                <p>• รองรับกราฟและสถิติ</p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="mode-card">
                <h3>📁 วิเคราะห์หลายไฟล์</h3>
                <p>• อัปโหลดหลาย JSON พร้อมกัน</p>
                <p>• วิเคราะห์ข้ามตาราง</p>
                <p>• เชื่อมโยงข้อมูลอัตโนมัติ</p>
                <p>• ประหยัดเวลาในการวิเคราะห์</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="mode-card">
                <h3>📈 โหมดแชท</h3>
                <p>• พูดคุยกับ AI แบบทั่วไป</p>
                <p>• ตอบคำถามได้หลากหลาย</p>
                <p>• บันทึกประวัติการสนทนา</p>
                <p>• ส่งออกเป็นไฟล์ CSV</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Features
        st.markdown("---")
        st.subheader("✨ คุณสมบัติเด่น")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🚀 ใช้งานง่าย
            - ไม่ต้องเขียนโค้ด
            - อินเทอร์เฟซเข้าใจง่าย
            - ถามคำถามเป็นภาษาไทย
            """)
        
        with col2:
            st.markdown("""
            ### 🧠 AI ฉลาด
            - วิเคราะห์ข้อมูลอัตโนมัติ
            - ตอบเป็นภาษาไทย
            - รองรับหลายโมเดล
            """)
        
        with col3:
            st.markdown("""
            ### ⚡ ประหยัด
            - แคชคำตอบซ้ำ
            - ประหยัดเครดิต API
            - รวดเร็วทันใจ
            """)
        
        # Getting started
        st.markdown("---")
        st.info("👈 **เริ่มต้นใช้งาน:** เลือกโหมดที่ต้องการจากแถบซ้าย และใส่ API Key")
        
        with st.expander("❓ วิธีรับ API Key ฟรี"):
            st.markdown("""
            1. ไปที่ https://openrouter.ai/keys
            2. สร้างบัญชี (ฟรี)
            3. คลิก "Create Key"
            4. Copy API Key มาใส่ในแถบซ้าย
            5. เลือกโมเดล AI ที่ต้องการ
            """)
    
    elif st.session_state.mode == 'complaint':
        if not api_key:
            st.warning("⚠️ กรุณาใส่ API Key ที่แถบซ้าย")
        else:
            handle_complaint_chat(api_key, model)
    
    elif st.session_state.mode == 'single_file':
        if not PANDASAI_AVAILABLE:
            st.error("⚠️ กรุณาติดตั้ง: `pip install pandasai requests`")
            return
        
        if not api_key:
            st.warning("⚠️ กรุณาใส่ API Key ที่แถบซ้าย")
            return
        
        handle_data_analysis(api_key, model, multiple=False)
    
    elif st.session_state.mode == 'multi_file':
        if not PANDASAI_AVAILABLE:
            st.error("⚠️ กรุณาติดตั้ง: `pip install pandasai requests`")
            return
        
        if not api_key:
            st.warning("⚠️ กรุณาใส่ API Key ที่แถบซ้าย")
            return
        
        handle_data_analysis(api_key, model, multiple=True)
    
    elif st.session_state.mode == 'new_mode':
        if not api_key:
            st.warning("⚠️ กรุณาใส่ API Key ที่แถบซ้าย")
        else:
            handle_new_mode_chat(api_key, model)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>🎯 Unified Business Platform</strong></p>
        <p style='font-size: 0.9em;'>
            รับเรื่องร้องเรียน • วิเคราะห์ข้อมูล • แชทกับ AI • ใช้งานง่าย
        </p>
        <p style='font-size: 0.8em; color: #999;'>
            Powered by PandasAI & OpenRouter | Version 2.0 | Made with ❤️ in Thailand
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"""
        ### ❌ เกิดข้อผิดพลาด
        
        **รายละเอียด:** {str(e)}
        
        **แนะนำการแก้ไข:**
        
        1. **ติดตั้ง Dependencies:**
        ```bash
        pip install streamlit pandas plotly pandasai requests
        ```
        
        2. **ตรวจสอบ API Key:**
        - ไปที่ https://openrouter.ai/keys
        - สร้าง API Key ใหม่
        - Copy มาใส่ในแอป
        
        3. **ตรวจสอบไฟล์ JSON:**
        - ต้องเป็น valid JSON format
        - ลองใช้ JSONLint.com ตรวจสอบ
        
        4. **ตรวจสอบ Python Version:**
        - ต้องเป็น Python 3.8 ขึ้นไป
        """)
        
        if st.button("🔄 รีเฟรชแอป", type="primary"):
            st.rerun()