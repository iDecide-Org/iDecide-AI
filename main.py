from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import google.generativeai as genai
import os
import logging

app = FastAPI()

origins = [
    "http://localhost:5173",  # React frontend URL
    # Add other origins if your frontend is hosted elsewhere
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your React app's URL
    allow_credentials=False,  # Changed to False since we're not using credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Securely Load API Key ---
genai.configure(api_key="AIzaSyAs7X_8HaTKP44Y5YCRN2Nbnx58Yb59zQk")
model = genai.GenerativeModel('gemini-pro')

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
    Uses AI to validate if the user input is relevant and makes sense in the context.
    Returns a tuple of validation status and an optional feedback message.
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
        # Extract the feedback message after 'لا'
        feedback = response.split('لا')[-1].strip()
        return False, feedback or "الإدخال غير صالح. يرجى المحاولة مرة أخرى."

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
            major = ''.join(filter(str.isalpha, line))
            major = line.split('.', 1)[-1].strip()
            if major:
                majors.append(major)
    # If parsing fails or no majors found, return an empty list
    if not majors:
        # Fallback: return the raw response split by newlines
        majors = [line.strip() for line in response.split('\n') if line.strip()]
    return majors[:5]  # Limit to top 5 recommendations

# --- Chatbot Logic ---

class ChatState:
    def __init__(self):
        self.user_data = UserData()
        self.conversation_history = []
        self.faq_answers = {
            "ما هي الخدمات التي تقدمونها؟": "نحن هنا لمساعدتك في جمع معلوماتك وتوفير توصيات مخصصة لك.",
            "كيف يمكنني التواصل معكم؟": "يمكنك التواصل معنا من خلال هذه المحادثة.",
            "هل معلوماتي آمنة معكم؟": "نحن نولي أهمية كبيرة لخصوصية معلوماتك.",
            # Add more FAQs here
        }
        self.current_question_type = None  # To track the current information being collected
        self.flow_steps = [
            ("personal_name", "personal.full_name", "أهلاً بك! ما هو اسمك الكامل؟"),
            ("personal_age", "personal.age", "ما هو عمرك؟"),
            ("personal_location", "personal.location", "أين تقيم؟"),
            ("personal_interests", "personal.interests", "ما هي بعض اهتماماتك؟"),
            ("academic_level", "academic.education_level", "ما هو أعلى مستوى تعليمي حصلت عليه؟ (مثال: الثانوية العامة, الشهادة الجامعية, تعليم فني)"),
            ("academic_major", "academic.major", "ما هو تخصصك الدراسي؟ (مثال: علمي علوم, علمي رياضة, ادبي)"),
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

        # Check if the user message is a question
        if is_user_question(message):
            answer = generate_gemini_response(f"أجب على السؤال التالي باللغة العربية: {message}")
            reminder = "هل تحتاج إلى مساعدة إضافية؟ يرجى الاستمرار في تزويدنا بمعلوماتك."
            full_response = f"{answer}\n\n{reminder}"
            self.conversation_history.append(f"الرد الآلي: {full_response}")
            return full_response

        # Check for FAQs first
        for question, answer in self.faq_answers.items():
            if question in message:
                self.conversation_history.append(f"الرد الآلي: {answer}")
                return answer

        # Proceed with data collection flow
        if self.current_step < len(self.flow_steps):
            step_key, field_path, prompt = self.flow_steps[self.current_step]
            context = f"السؤال عن {prompt}"
            is_valid, feedback = validate_input_with_ai(message, context)

            if is_valid:
                # Set the value in user_data based on field_path
                self.set_user_data(field_path, message)
                self.current_step += 1
                if self.current_step < len(self.flow_steps):
                    next_prompt = self.flow_steps[self.current_step][2]
                    response = next_prompt
                else:
                    # All data collected, generate recommended college majors
                    recommended_majors = generate_recommended_college_majors(self.user_data)
                    self.user_data.recommended_college_majors = recommended_majors
                    response = "شكرًا لك على تزويدنا بالمعلومات. إليك قائمة التخصصات الجامعية الموصى بها لك:\n\n" + "\n".join([f"{i+1}. {major}" for i, major in enumerate(recommended_majors)])
            else:
                response = feedback or "لم أفهم ذلك. هل يمكنك إعادة المحاولة؟"
        else:
            response = "شكرًا لك على تزويدنا بالمعلومات."

        self.conversation_history.append(f"الرد الآلي: {response}")
        return response

    def set_user_data(self, field_path: str, value: str):
        """Sets the user data based on the field path."""
        fields = field_path.split('.')
        data = self.user_data
        for field in fields[:-1]:
            data = getattr(data, field)
        final_field = fields[-1]
        if isinstance(getattr(data, final_field), list):
            setattr(data, final_field, [item.strip() for item in value.split(',')])
        elif final_field == "age" or final_field == "grade":
            try:
                setattr(data, final_field, int(value) if final_field == "age" else float(value))
            except ValueError:
                setattr(data, final_field, None)
        else:
            setattr(data, final_field, value)

# --- FastAPI Endpoints ---

chat_state = ChatState()

@app.post("/chat/")  # Keep the trailing slash
async def chat_endpoint(request: ChatRequest):
    try:
        response = chat_state.handle_message(request.message)
        recommended_college_majors = chat_state.user_data.recommended_college_majors
        return ChatResponse(response=response, recommended_college_majors=recommended_college_majors)
    except Exception as e:
        logger.error(f"Error in chat_endpoint: {e}")
        raise HTTPException(status_code=500, detail="حدث خطأ داخلي. يرجى المحاولة لاحقًا.")

@app.get("/data/", response_model=UserData)
async def get_user_data():
    return chat_state.user_data
