import re
import uuid
import json
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List, Any
import google.generativeai as genai
import os
import logging
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine # Added for similarity calculation
from sklearn.preprocessing import StandardScaler # Added import for StandardScaler

app = FastAPI()

# --- Global Variables for Model and Scaler ---
MODEL_PROFILES: Optional[pd.DataFrame] = None # Will hold the major profiles DataFrame
SCALER: Optional[StandardScaler] = None # type: ignore
MAJOR_NAMES: Optional[List[str]] = [] # List of major names

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Securely Load API Key & Configure Gemini ---
# IMPORTANT: Consider moving the API key to an environment variable for security
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyAs7X_8HaTKP44Y5YCRN2Nbnx58Yb59zQk") # Fallback to hardcoded if not in env
    if gemini_api_key == "AIzaSyAs7X_8HaTKP44Y5YCRN2Nbnx58Yb59zQk":
        logger.warning("Using a hardcoded API key for Gemini. Consider setting GEMINI_API_KEY environment variable.")
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Updated model name
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    gemini_model = None

# --- CORS Configuration ---
origins = [
    "http://localhost:5173",  # React frontend URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model and Scaler at Startup ---
@app.on_event("startup")
async def load_model_resources():
    global MODEL_PROFILES, SCALER, MAJOR_NAMES
    try:
        profiles_path = os.path.join("model", "major_profiles.joblib")
        scaler_path = os.path.join("model", "major_scaler.joblib")
        csv_path = os.path.join("model", "cleaned_riasec_big5_major.csv")

        if not os.path.exists(profiles_path):
            logger.error(f"Major profiles file not found at {profiles_path}")
            raise FileNotFoundError(f"Major profiles file not found at {profiles_path}")
        if not os.path.exists(scaler_path):
            logger.error(f"Scaler file not found at {scaler_path}")
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

        MODEL_PROFILES = joblib.load(profiles_path)
        SCALER = joblib.load(scaler_path)
        logger.info("Major profiles (DataFrame) and Scaler loaded successfully.")

        if not isinstance(MODEL_PROFILES, pd.DataFrame):
            logger.error(f"Loaded major profiles is not a pandas DataFrame. Type: {type(MODEL_PROFILES)}")
            MODEL_PROFILES = None # Invalidate if not a DataFrame
            raise TypeError("Loaded major_profiles.joblib is not a DataFrame.")
        
        logger.info(f"Type of loaded SCALER: {type(SCALER)}")
        if not hasattr(SCALER, 'transform') or not hasattr(SCALER, 'feature_names_in_'):
             logger.error(f"Loaded scaler does not appear to be a valid scikit-learn scaler. Type: {type(SCALER)}")
             SCALER = None # Invalidate
             raise TypeError("Loaded major_scaler.joblib is not a valid scaler.")


        # --- Determine MAJOR_NAMES ---
        MAJOR_NAMES = []
        major_names_populated = False

        if MODEL_PROFILES is not None and not MODEL_PROFILES.empty and MODEL_PROFILES.index.name is not None and len(MODEL_PROFILES.index) > 0:
            try:
                temp_list_from_index = [str(s).strip() for s in MODEL_PROFILES.index.tolist() if s is not None and str(s).strip()]
                if temp_list_from_index:
                    MAJOR_NAMES = sorted(list(set(temp_list_from_index)))
                    if MAJOR_NAMES:
                        logger.info(f"Loaded {len(MAJOR_NAMES)} unique major names from DataFrame index.")
                        major_names_populated = True
                    else:
                        logger.info("DataFrame index found but was empty or yielded no valid major names.")
                else:
                    logger.info("DataFrame index found but was empty or yielded no valid major names.")
            except Exception as e:
                logger.error(f"Error processing DataFrame index for major names: {e}. Will attempt CSV fallback.")
        
        if not major_names_populated:
            logger.warning("Could not populate major names from DataFrame index. Attempting CSV fallback.")
            if os.path.exists(csv_path):
                logger.info(f"Attempting to load major names from CSV: {csv_path}.")
                try:
                    df_csv = pd.read_csv(csv_path)
                    major_column_name = 'major' # As corrected previously
                    if major_column_name in df_csv.columns:
                        csv_majors = [str(s).strip() for s in df_csv[major_column_name].astype(str).unique().tolist() if s is not None and str(s).strip()]
                        if csv_majors:
                            MAJOR_NAMES = sorted(list(set(csv_majors)))
                            logger.info(f"Successfully loaded {len(MAJOR_NAMES)} unique major names from CSV column '{major_column_name}'.")
                            major_names_populated = True
                        else:
                            logger.warning(f"CSV column '{major_column_name}' found but contained no valid major names.")
                    else:
                        logger.error(f"Column '{major_column_name}' not found in {csv_path}. Available columns: {df_csv.columns.tolist()}")
                except Exception as e:
                    logger.error(f"Error loading major names from CSV {csv_path}: {e}")
            else:
                logger.info(f"CSV file for major names not found at {csv_path}.")

        if not MAJOR_NAMES:
             logger.critical("MAJOR_NAMES is empty after all attempts. Application will not be able to provide major recommendations.")
             # raise RuntimeError("Failed to load major names. Application cannot start.")
        else:
            logger.info(f"Final MAJOR_NAMES count: {len(MAJOR_NAMES)}")


    except FileNotFoundError as fnf_error:
        logger.critical(f"Essential model resource file not found: {fnf_error}")
        MODEL_PROFILES = None
        SCALER = None
        MAJOR_NAMES = []
    except TypeError as type_error:
        logger.critical(f"Type error with loaded model resources: {type_error}")
        MODEL_PROFILES = None
        SCALER = None
        MAJOR_NAMES = []
    except Exception as e:
        logger.critical(f"An unexpected error occurred during model resource loading: {e}")
        MODEL_PROFILES = None
        SCALER = None
        MAJOR_NAMES = []

# --- Data Models ---
class RIASECInput(BaseModel):
    realistic: Optional[float] = None
    investigative: Optional[float] = None
    artistic: Optional[float] = None
    social: Optional[float] = None
    enterprising: Optional[float] = None
    conventional: Optional[float] = None

class BIG5Input(BaseModel):
    openness: Optional[float] = None
    conscientiousness: Optional[float] = None
    extraversion: Optional[float] = None
    agreeableness: Optional[float] = None
    emotional_stability: Optional[float] = None # Changed from neuroticism

class UserData(BaseModel):
    riasec: RIASECInput = RIASECInput()
    big5: BIG5Input = BIG5Input()
    recommended_college_majors: Optional[List[str]] = None

class ChatResponse(BaseModel):
    response: str
    recommended_college_majors: Optional[List[str]] = None
    chatbot_completed: bool = False

class ChatRequest(BaseModel):
    message: str

# --- Helper Functions ---
def generate_gemini_response(prompt: str) -> str:
    """Generates a response using the Gemini model."""
    if not gemini_model:
        logger.error("Gemini model not available for generating response.")
        return "عذرًا، خدمة الذكاء الاصطناعي غير متاحة حاليًا."
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(candidate_count=1), # type: ignore
            # safety_settings=None, # Consider configuring safety settings
            stream=False
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating Gemini response: {e}")
        return "عذرًا، حدث خطأ أثناء معالجة طلبك. يرجى المحاولة لاحقًا."

def is_user_question(message: str) -> bool:
    """Determines if the user's message is a question using Gemini."""
    question_prompt = f"""
    باللغة العربية، حدد ما إذا كانت الرسالة التالية سؤالاً أم لا.
    الرسالة: "{message}"
    الجواب بنعم أو لا فقط.
    """
    response = generate_gemini_response(question_prompt)
    return 'نعم' in response.lower()

def predict_majors_from_model(user_data: UserData, top_n: int = 5) -> List[str]:
    """Generates a list of recommended college majors using cosine similarity with major profiles."""
    if MODEL_PROFILES is None or SCALER is None or not MAJOR_NAMES:
        logger.error("Model profiles, Scaler, or MAJOR_NAMES not loaded. Cannot predict majors.")
        return ["تعذر التنبؤ بالتخصصات بسبب عدم توفر النموذج أو بيانات التخصصات."]

    try:
        # Define the feature order based on train.py
        # ['realistic', 'investigative', 'artistic', 'social', 'enterprising', 'conventional',
        #  'extraversion', 'agreeableness', 'conscientiousness', 'emotional_stability', 'openness']
        
        user_scores_map = {
            'realistic': user_data.riasec.realistic,
            'investigative': user_data.riasec.investigative,
            'artistic': user_data.riasec.artistic,
            'social': user_data.riasec.social,
            'enterprising': user_data.riasec.enterprising,
            'conventional': user_data.riasec.conventional,
            'extraversion': user_data.big5.extraversion,
            'agreeableness': user_data.big5.agreeableness,
            'conscientiousness': user_data.big5.conscientiousness,
            'emotional_stability': user_data.big5.emotional_stability, # Matched to train.py
            'openness': user_data.big5.openness
        }

        # Ensure all data is present
        if any(score is None for score in user_scores_map.values()):
            logger.warning("Not all RIASEC/BIG5 scores are available for prediction.")
            return ["يرجى إكمال جميع مدخلات RIASEC و BIG5 أولاً."]

        # Get feature names from the scaler to ensure correct order
        # This makes it robust if SCALER.feature_names_in_ is available and matches
        try:
            model_feature_names = SCALER.feature_names_in_
        except AttributeError:
            logger.error("SCALER does not have 'feature_names_in_'. Falling back to predefined order. This might be risky.")
            # Fallback to the order explicitly defined, hoping it matches the scaler's training
            model_feature_names = [
                'realistic', 'investigative', 'artistic', 'social', 'enterprising', 'conventional',
                'extraversion', 'agreeableness', 'conscientiousness', 'emotional_stability', 'openness'
            ]


        # Construct the feature vector in the order defined by model_feature_names
        features_vector = [user_scores_map[name] for name in model_feature_names]
        
        # Check if all features required by the scaler are present in user_scores_map
        # This is an additional safety check
        for feature_name in model_feature_names:
            if feature_name not in user_scores_map or user_scores_map[feature_name] is None:
                logger.error(f"Missing score for feature '{feature_name}' required by the model/scaler.")
                return [f"يرجى تقديم قيمة لـ '{feature_name}'."]


        user_features_array = np.array([features_vector])
        
        # Scale the user's features
        user_scaled_features = SCALER.transform(user_features_array)

        recommendations = {}
        # MODEL_PROFILES is the DataFrame of major_category -> feature_means
        for major_category, profile_vector in MODEL_PROFILES.iterrows():
            # Ensure profile_vector is also in the same order as user_scaled_features
            # If MODEL_PROFILES columns are already ordered correctly (matching scaler.feature_names_in_)
            # then profile_vector.values should be fine.
            # For safety, reorder profile_vector if its columns might not match scaler.feature_names_in_
            # However, MODEL_PROFILES was created using 'features' list in train.py, which should match.
            
            # Check if profile_vector needs reordering based on scaler's feature names
            # This is an extra safety, assuming MODEL_PROFILES.columns are the feature names
            ordered_profile_values = profile_vector[model_feature_names].values
            
            similarity = 1 - cosine(user_scaled_features[0], ordered_profile_values)
            recommendations[str(major_category)] = similarity # Ensure major_category is string

        # Sort recommendations by similarity score in descending order
        sorted_recs = sorted(recommendations.items(), key=lambda item: item[1], reverse=True)
        
        recommended_majors = [major for major, score in sorted_recs[:top_n] if score > 0] # Optionally filter by min score

        return recommended_majors if recommended_majors else ["لم يتم العثور على تخصصات موصى بها بناءً على مدخلاتك."]

    except Exception as e:
        logger.error(f"Error during major prediction using cosine similarity: {e}", exc_info=True)
        return [f"حدث خطأ أثناء التنبؤ بالتخصصات: {e}"]

# --- Chatbot Logic ---
class ChatState:
    def __init__(self):
        self.user_data = UserData()
        self.conversation_history = []
        self.faq_answers = {
            "ما هي الخدمات التي نقدمها؟": "نحن هنا لمساعدتك في تحديد التخصصات الجامعية المناسبة بناءً على تقييمات RIASEC و BIG5.",
            "كيف يمكنني التواصل معكم؟": "يمكنك التواصل معنا من خلال هذه المحادثة.",
            "هل معلوماتي آمنة معكم؟": "نحن نولي أهمية كبيرة لخصوصية معلوماتك.",
        }
        self.flow_steps = [
            ("riasec_r", "riasec.realistic", "ما هي درجة واقعيتك (Realistic)؟ (أدخل قيمة رقمية، مثال: 3.5)", "numeric"),
            ("riasec_i", "riasec.investigative", "ما هي درجة بحثك (Investigative)؟ (أدخل قيمة رقمية)", "numeric"),
            ("riasec_a", "riasec.artistic", "ما هي درجة حسك الفني (Artistic)؟ (أدخل قيمة رقمية)", "numeric"),
            ("riasec_s", "riasec.social", "ما هي درجة اجتماعيتك (Social)؟ (أدخل قيمة رقمية)", "numeric"),
            ("riasec_e", "riasec.enterprising", "ما هي درجة إقدامك (Enterprising)؟ (أدخل قيمة رقمية)", "numeric"),
            ("riasec_c", "riasec.conventional", "ما هي درجة تقليديتك (Conventional)؟ (أدخل قيمة رقمية)", "numeric"),
            ("big5_o", "big5.openness", "ما هي درجة انفتاحك على التجارب (Openness)؟ (أدخل قيمة رقمية)", "numeric"),
            ("big5_c", "big5.conscientiousness", "ما هي درجة ضميرك الحي (Conscientiousness)؟ (أدخل قيمة رقمية)", "numeric"),
            ("big5_e", "big5.extraversion", "ما هي درجة انبساطك (Extraversion)؟ (أدخل قيمة رقمية)", "numeric"),
            ("big5_a", "big5.agreeableness", "ما هي درجة مقبوليتك (Agreeableness)؟ (أدخل قيمة رقمية)", "numeric"),
            ("big5_es", "big5.emotional_stability", "ما هي درجة استقرارك العاطفي (Emotional Stability)؟ (أدخل قيمة رقمية)", "numeric"), # Changed
        ]
        self.current_step = 0
        self.chatbot_completed = False

    def handle_message(self, message: str) -> str:
        self.conversation_history.append(f"المستخدم: {message}")

        if MODEL_PROFILES is None or SCALER is None or not MAJOR_NAMES:
            response = "عذرًا، نظام التوصية غير جاهز حاليًا بسبب مشكلة في تحميل الموارد الأساسية. يرجى المحاولة لاحقًا أو الاتصال بالدعم."
            self.conversation_history.append(f"الرد الآلي: {response}")
            self.chatbot_completed = True 
            self.user_data.recommended_college_majors = []
            return response

        if self.current_step < len(self.flow_steps):
            step_key, field_path, prompt_question, input_type = self.flow_steps[self.current_step]
            
            is_valid = False
            feedback = None
            processed_value = None

            if input_type == "numeric":
                try:
                    processed_value = float(message)
                    is_valid = True
                except ValueError:
                    is_valid = False
                    feedback = "يرجى إدخال قيمة رقمية صالحة."
            else: 
                is_valid = True 
                processed_value = message

            if is_valid:
                self.set_user_data(field_path, processed_value)
                self.current_step += 1
                if self.current_step < len(self.flow_steps):
                    response = self.flow_steps[self.current_step][2] 
                else:
                    recommended_majors = predict_majors_from_model(self.user_data)
                    self.user_data.recommended_college_majors = recommended_majors
                    self.chatbot_completed = True
                    if recommended_majors and "تعذر" not in recommended_majors[0] and "يرجى إكمال" not in recommended_majors[0]:
                        response = (
                            "شكرًا لك على تزويدنا بالمعلومات. بناءً على إجاباتك، إليك قائمة التخصصات الجامعية الموصى بها لك:\\n\\n" +
                            "\\n".join([f"- {major}" for major in recommended_majors]) +
                            "\\n\\nهل لديك أي أسئلة أخرى؟"
                        )
                    else:
                         response = (
                            "شكرًا لك على تزويدنا بالمعلومات.\\n" +
                            (recommended_majors[0] if recommended_majors else "لم نتمكن من تحديد توصيات في الوقت الحالي.") +
                            "\\n\\nهل لديك أي أسئلة أخرى؟"
                        )
            else:
                response = feedback or "لم أفهم ذلك. هل يمكنك إعادة المحاولة؟"
        
        else: 
            if is_user_question(message):
                answer = generate_gemini_response(f"أجب على السؤال التالي باللغة العربية مع الأخذ في الاعتبار أن المستخدم قد أكمل للتو استبيان RIASEC/BIG5 للتوصية بالتخصصات الجامعية: {message}")
                full_response = f"{answer}"
                self.conversation_history.append(f"الرد الآلي: {full_response}")
                return full_response

            for question, answer in self.faq_answers.items():
                if question in message or message in question: 
                    self.conversation_history.append(f"الرد الآلي: {answer}")
                    return answer
            
            response = generate_gemini_response(f"المستخدم قال: \"{message}\". قدم ردًا عامًا ومساعدًا باللغة العربية، مع العلم أن المستخدم قد أكمل استبيان التوصية بالتخصصات.")

        self.conversation_history.append(f"الرد الآلي: {response}")
        return response

    def set_user_data(self, field_path: str, value: Any):
        parts = field_path.split('.')
        obj = self.user_data
        for part in parts[:-1]:
            if getattr(obj, part) is None: 
                if part == "riasec": setattr(obj, part, RIASECInput())
                elif part == "big5": setattr(obj, part, BIG5Input())
            obj = getattr(obj, part)
        
        if hasattr(obj, parts[-1]):
            setattr(obj, parts[-1], value)
        else:
            logger.error(f"Attempted to set non-existent attribute {parts[-1]} on {type(obj)}")


# --- Global Chat Sessions ---
chat_sessions: Dict[str, ChatState] = {}

# --- FastAPI Endpoints ---
@app.post("/chat/")
async def chat_endpoint(request: Request, response: Response, chat_request: ChatRequest):
    session_cookie = request.cookies.get("chat_session")
    session_id = None
    if session_cookie:
        try:
            session_data = json.loads(session_cookie)
            session_id = session_data.get("id")
        except json.JSONDecodeError:
            logger.warning("Failed to decode session cookie JSON.")
            session_id = None

    chat_state: ChatState
    if session_id and session_id in chat_sessions:
        chat_state = chat_sessions[session_id]
        if chat_state.chatbot_completed and (MODEL_PROFILES is None or SCALER is None or not MAJOR_NAMES):
            logger.warning(f"Session {session_id} was completed, but model resources are now unavailable. Resetting state slightly.")
    else:
        session_id = str(uuid.uuid4())
        chat_state = ChatState()
        chat_sessions[session_id] = chat_state
        if MODEL_PROFILES is not None and SCALER is not None and MAJOR_NAMES and chat_state.current_step < len(chat_state.flow_steps) :
            initial_response = chat_state.flow_steps[0][2]
            chat_state.conversation_history.append(f"الرد الآلي: {initial_response}")
            
            session_info_new = {
                "id": session_id,
                "current_step": chat_state.current_step, 
                "last_user_message": "", 
                "last_ai_response": initial_response,
                "chatbot_completed": chat_state.chatbot_completed,
            }
            response.set_cookie(
                "chat_session", json.dumps(session_info_new),
                httponly=True, max_age=3600, path="/"
            )
            return ChatResponse(
                response=initial_response,
                recommended_college_majors=None,
                chatbot_completed=False
            )

    res_message = chat_state.handle_message(chat_request.message)
    
    recommended_majors = chat_state.user_data.recommended_college_majors
    chatbot_completed_status = chat_state.chatbot_completed

    session_info = {
        "id": session_id,
        "current_step": chat_state.current_step,
        "last_user_message": chat_request.message,
        "last_ai_response": res_message,
        "chatbot_completed": chatbot_completed_status,
    }

    response.set_cookie(
        "chat_session", json.dumps(session_info),
        httponly=True, max_age=3600, path="/" 
    )

    return ChatResponse(
        response=res_message,
        recommended_college_majors=recommended_majors,
        chatbot_completed=chatbot_completed_status
    )

@app.get("/data/", response_model=UserData)
async def get_user_data(request: Request):
    session_cookie = request.cookies.get("chat_session")
    if session_cookie:
        try:
            session_data = json.loads(session_cookie)
            session_id = session_data.get("id")
            if session_id in chat_sessions:
                return chat_sessions[session_id].user_data
        except json.JSONDecodeError:
            logger.warning("Failed to decode session cookie JSON for /data/ endpoint.")
            raise HTTPException(status_code=400, detail="بيانات تعريف الجلسة غير صالحة")
    raise HTTPException(status_code=404, detail="لم يتم العثور على جلسة مستخدم أو بيانات")

# --- Entry point for running with Uvicorn (optional, for local dev) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local development.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
