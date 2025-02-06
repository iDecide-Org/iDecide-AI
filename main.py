import re
import uuid
import json
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import google.generativeai as genai
import os
import logging

app = FastAPI()

# IMPORTANT: Set allow_credentials to True so that cookies can be set!
origins = [
    "http://localhost:5173",  # React frontend URL
    # Add other origins if necessary
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # <-- Enable cookies!
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Securely Load API Key ---
genai.configure(api_key="AIzaSyAs7X_8HaTKP44Y5YCRN2Nbnx58Yb59zQk")
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# --- Data Models ---

class UserPersonal(BaseModel):
    full_name: Optional[str] = None
    age: Optional[int] = None
    location: Optional[str] = None
    interests: Optional[List[str]] = None

class UserAcademic(BaseModel):
    education_level: Optional[str] = None
    major: Optional[str] = None
    grade: Optional[float] = None
    skills: Optional[List[str]] = None

class UserCareer(BaseModel):
    career_goals: Optional[str] = None
    preferred_industry: Optional[str] = None
    desired_salary: Optional[str] = None

class UserFinancial(BaseModel):
    budget: Optional[str] = None

class UserData(BaseModel):
    personal: UserPersonal = UserPersonal()
    academic: UserAcademic = UserAcademic()
    career: UserCareer = UserCareer()
    financial: UserFinancial = UserFinancial()
    recommended_college_majors: Optional[List[str]] = None  # New field added

class ChatResponse(BaseModel):
    response: str
    recommended_college_majors: Optional[List[str]] = None  # New field added

# --- Request Model ---

class ChatRequest(BaseModel):
    message: str

# --- Helper Functions ---

def generate_gemini_response(prompt: str, history: Optional[List[str]] = None) -> str:
    """Generates a response using the Gemini Pro model."""
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(candidate_count=1),
            safety_settings=None,
            stream=False
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating Gemini response: {e}")
        return "عذرًا، حدث خطأ أثناء معالجة طلبك. يرجى المحاولة لاحقًا."

def is_user_question(message: str) -> bool:
    """Determines if the user's message is a question."""
    question_prompt = f"""
    باللغة العربية، حدد ما إذا كانت الرسالة التالية سؤالاً أم لا.
    الرسالة: "{message}"
    الجواب بنعم أو لا فقط.
    """
    response = generate_gemini_response(question_prompt)
    return 'نعم' in response.lower()

def validate_input_with_ai(user_input: str, context: str) -> (bool, Optional[str]):
    """
    Uses AI to validate user input and provides feedback.  Improved feedback extraction.
    """
    validation_prompt = f"""
    باللغة العربية، قم بتقييم ما إذا كان إدخال المستخدم التالي منطقيًا وذو صلة بالسياق الموضح أدناه.
    السياق: {context}
    إدخال المستخدم: {user_input}
    قيم الإدخال بـ "نعم" إذا كان مناسبًا، أو "لا" متبوعًا بتفسير قصير إذا كان غير مناسب.
    """
    response = generate_gemini_response(validation_prompt)
    if 'نعم' in response.lower():
        return True, None
    else:
        # Find the index of "لا"
        no_index = response.lower().find('لا')
        if no_index != -1:
            # Find the start of the explanation (after "لا")
            # Look for the first letter, number, or common punctuation mark
            match = re.search(r"[\u0600-\u06FF\d،؛؟!. ]", response[no_index + len('لا'):]) #Arabic letters, digits and punctuations
            if match:
                start_index = no_index + len('لا') + match.start()
                feedback = response[start_index:].strip()
                return False, feedback or "الإدخال غير صالح. يرجى المحاولة مرة أخرى."

        # Fallback if "لا" is not found or parsing fails
        return False, "الإدخال غير صالح. يرجى المحاولة مرة أخرى."

def generate_recommended_college_majors(user_data: UserData) -> List[str]:
    """Generates a list of recommended college majors based on collected user data."""
    prompt = f"""
    بناءً على المعلومات التالية للمستخدم، قم بإنشاء قائمة من التخصصات الجامعية الموصى بها باللغة العربية.
    البيانات الشخصية: {user_data.personal.json()}
    البيانات الأكاديمية: {user_data.academic.json()}
    البيانات المهنية: {user_data.career.json()}
    البيانات المالية: {user_data.financial.json()}

    قائمة التخصصات الموصى بها:
    1.
    """
    response = generate_gemini_response(prompt)
    
    # Attempt to parse the response into a list of majors
    majors = []
    for line in response.split('\n'):
        if line.strip().startswith(tuple(str(i) for i in range(1, 100))):
            # Remove the numbering (e.g., "1. Computer Science" -> "Computer Science")
            major = line.split('.', 1)[-1].strip()
            if major:
                majors.append(major)
    if not majors:
        majors = [line.strip() for line in response.split('\n') if line.strip()]
    return majors[:5]  # Limit to top 5 recommendations

# --- Chatbot Logic ---

class ChatState:
    def __init__(self):
        self.user_data = UserData()
        self.conversation_history = []
        self.faq_answers = {
            "ما هي الخدمات التي نقدمها؟": "نحن هنا لمساعدتك في جمع معلوماتك وتوفير توصيات مخصصة لك.",
            "كيف يمكنني التواصل معكم؟": "يمكنك التواصل معنا من خلال هذه المحادثة.",
            "هل معلوماتي آمنة معكم؟": "نحن نولي أهمية كبيرة لخصوصية معلوماتك.",
            # Add more FAQs if needed
        }
        self.current_question_type = None  # To track which data is being collected
        self.flow_steps = [
            ("personal_name", "personal.full_name", "أهلاً بك! ما هو اسمك الكامل؟"),
            ("personal_age", "personal.age", "ما هو عمرك؟"),
            ("personal_location", "personal.location", "أين تقيم؟"),
            ("personal_interests", "personal.interests", "ما هي بعض اهتماماتك؟"),
            ("academic_level", "academic.education_level", "ما هو أعلى مستوى تعليمي حصلت عليه؟ (مثال: الثانوية العامة, الشهادة الجامعية, تعليم فني)"),
            ("academic_major", "academic.major", "ما هو تخصصك الدراسي؟ (مثال: علمي علوم, علمي رياضة, أدبي)"),
            ("academic_grade", "academic.grade", "ما هو مجموعك الدراسي؟ (مثال: 50 - 100)"),
            ("academic_skills", "academic.skills", "ما هي بعض مهاراتك؟"),
            ("career_goals", "career.career_goals", "ما هي أهدافك المهنية؟"),
            ("career_industry", "career.preferred_industry", "ما هو القطاع الصناعي الذي تفضله؟"),
            ("career_salary", "career.desired_salary", "ما هو الراتب الذي تطمح إليه؟"),
            ("financial_budget", "financial.budget", "ما هي ميزانيتك التقريبية؟"),
        ]
        self.current_step = 0

    def handle_message(self, message: str) -> str:
        self.conversation_history.append(f"المستخدم: {message}")
        if self.current_step < len(self.flow_steps):
            step_key, field_path, prompt = self.flow_steps[self.current_step]
            context = f"السؤال عن {prompt}"
            final_field = field_path.split('.')[-1]

            # Validate numeric fields manually.
            if final_field in ("age", "grade"):
                try:
                    if final_field == "age":
                        int(message)
                    else:
                        float(message)
                    is_valid = True
                    feedback = None
                except ValueError:
                    is_valid = False
                    feedback = "يرجى إدخال رقم صالح."
            else:
                is_valid, feedback = validate_input_with_ai(message, context)

            if is_valid:
                self.set_user_data(field_path, message)
                self.current_step += 1
                if self.current_step < len(self.flow_steps):
                    response = self.flow_steps[self.current_step][2]
                else:
                    recommended_majors = generate_recommended_college_majors(self.user_data)
                    self.user_data.recommended_college_majors = recommended_majors
                    response = (
                        "شكرًا لك على تزويدنا بالمعلومات. إليك قائمة التخصصات الجامعية الموصى بها لك:\n\n" +
                        "\n".join([f"{i+1}. {major}" for i, major in enumerate(recommended_majors)])
                    )
            else:
                response = feedback or "لم أفهم ذلك. هل يمكنك إعادة المحاولة؟"

            self.conversation_history.append(f"الرد الآلي: {response}")
            return response

        if is_user_question(message):
            answer = generate_gemini_response(f"أجب على السؤال التالي باللغة العربية: {message}")
            reminder = "هل تحتاج إلى مساعدة إضافية؟ يرجى الاستمرار في تزويدنا بمعلوماتك."
            full_response = f"{answer}\n\n{reminder}"
            self.conversation_history.append(f"الرد الآلي: {full_response}")
            return full_response

        for question, answer in self.faq_answers.items():
            if question in message:
                self.conversation_history.append(f"الرد الآلي: {answer}")
                return answer

        response = "شكرًا لك على تزويدنا بالمعلومات."
        self.conversation_history.append(f"الرد الآلي: {response}")
        return response

    def set_user_data(self, field_path: str, value: str):
        fields = field_path.split('.')
        data = self.user_data
        for field in fields[:-1]:
            data = getattr(data, field)
        final_field = fields[-1]
        if isinstance(getattr(data, final_field), list):
            setattr(data, final_field, [item.strip() for item in value.split(',')])
        elif final_field in ("age", "grade"):
            try:
                setattr(data, final_field, int(value) if final_field == "age" else float(value))
            except ValueError:
                setattr(data, final_field, None)
        else:
            setattr(data, final_field, value)

# --- Global Chat Sessions ---

chat_sessions: Dict[str, ChatState] = {}

# --- FastAPI Endpoints ---

@app.post("/chat/")  # Keep the trailing slash
async def chat_endpoint(request: Request, response: Response, chat_request: ChatRequest):
    session_cookie = request.cookies.get("chat_session")
    session_id = None
    if session_cookie:
        try:
            session_data = json.loads(session_cookie)
            session_id = session_data.get("id")
        except json.JSONDecodeError:
            session_id = None

    if not session_id or session_id not in chat_sessions:
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = ChatState()

    chat_state = chat_sessions[session_id]
    res_message = chat_state.handle_message(chat_request.message)
    recommended_majors = chat_state.user_data.recommended_college_majors

    if chat_state.current_step < len(chat_state.flow_steps):
        current_question = chat_state.flow_steps[chat_state.current_step][2]
    else:
        current_question = None

    session_info = {
        "id": session_id,
        "current_step": chat_state.current_step,
        "last_user_message": chat_request.message,
        "last_ai_response": res_message,
        "current_question": current_question,
    }

    # Set the cookie with a defined path and max_age.
    response.set_cookie(
        "chat_session",
        json.dumps(session_info),
        httponly=True,
        max_age=3600,
        path="/"
    )
    return ChatResponse(response=res_message, recommended_college_majors=recommended_majors)

@app.get("/data/", response_model=UserData)
async def get_user_data(request: Request):
    session_cookie = request.cookies.get("chat_session")
    if session_cookie:
        try:
            session_data = json.loads(session_cookie)
            session_id = session_data.get("id")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="بيانات تعريف الجلسة غير صالحة")
        if session_id in chat_sessions:
            return chat_sessions[session_id].user_data
    raise HTTPException(status_code=404, detail="لم يتم العثور على جلسة مستخدم")
