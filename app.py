# app.py (FastAPI backend with Supabase Client)
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Header, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import List, Optional
from datetime import datetime, timedelta
import os
import uuid
import PyPDF2
import io
import anthropic
import logging
import json
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel
from supabase import create_client, Client
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="PDF Q&A System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://qqdpepqbpdlodpadnvwr.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFxZHBlcHFicGRsb2RwYWRudndyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDIyMTkwMjgsImV4cCI6MjA1Nzc5NTAyOH0.vx97UL49iYkafRLyN3gsia3Bqfj2ISk-XV1C0qfK2PY")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFxZHBlcHFicGRsb2RwYWRudndyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDIyMTkwMjgsImV4cCI6MjA1Nzc5NTAyOH0.vx97UL49iYkafRLyN3gsia3Bqfj2ISk-XV1C0qfK2PY")

# Initialize Supabase client
print(SUPABASE_SERVICE_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Security config
SECRET_KEY = os.getenv("SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic Models for API
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    id: int
    is_active: bool

class PDFDocumentSchema(BaseModel):
    id: int
    filename: str
    upload_date: datetime

class QuestionCreate(BaseModel):
    question_text: str
    document_ids: List[int]

class QuestionResponse(BaseModel):
    id: int
    question_text: str
    answer_text: str
    timestamp: datetime

# Initialize Anthropic client
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY", "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")
)

# Security Functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str):
    # Query Supabase for the user
    try:
        response = supabase.table("users").select("*").eq("username", username).execute()
        users = response.data
        
        if not users or len(users) == 0:
            return False
        
        user = users[0]
        if not verify_password(password, user["hashed_password"]):
            return False
        
        return user
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    # Query Supabase for the user
    try:
        response = supabase.table("users").select("*").eq("username", token_data.username).execute()
        users = response.data
        
        if not users or len(users) == 0:
            raise credentials_exception
        
        user = users[0]
        return user
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        raise credentials_exception

def get_current_active_user(current_user = Depends(get_current_user)):
    if not current_user["is_active"]:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Utility function to extract text from PDF
def extract_text_from_pdf(file_content):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# API Endpoints
@app.post("/register", response_model=UserInDB)
async def register_user(user: UserCreate):
    try:
        # Check if username exists
        username_response = supabase.table("users").select("username").eq("username", user.username).execute()
        if username_response.data and len(username_response.data) > 0:
            raise HTTPException(status_code=400, detail="Username already registered")
        
        # Check if email exists
        email_response = supabase.table("users").select("email").eq("email", user.email).execute()
        if email_response.data and len(email_response.data) > 0:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create new user
        hashed_password = get_password_hash(user.password)
        new_user = {
            "username": user.username,
            "email": user.email,
            "hashed_password": hashed_password,
            "is_active": True
        }
        
        insert_response = supabase.table("users").insert(new_user).execute()
        
        if not insert_response.data or len(insert_response.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create user")
        
        created_user = insert_response.data[0]
        return UserInDB(
            id=created_user["id"],
            username=created_user["username"],
            email=created_user["email"],
            is_active=created_user["is_active"]
        )
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserInDB)
async def read_users_me(current_user = Depends(get_current_active_user)):
    return UserInDB(
        id=current_user["id"],
        username=current_user["username"],
        email=current_user["email"],
        is_active=current_user["is_active"]
    )

@app.post("/upload/", response_model=PDFDocumentSchema)
async def upload_pdf(
    file: UploadFile = File(...),
    current_user = Depends(get_current_active_user)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read file content
        content = await file.read()
        
        # Create a unique file path
        file_id = f"{current_user['id']}_{uuid.uuid4()}"
        file_path = f"{file_id}.pdf"
        
        # Create temporary file for Supabase upload
        temp_file_path = f"/tmp/{file_path}"
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        # Upload to Supabase Storage
        try:
            # Upload the temporary file
            storage_response = supabase.storage.from_("pdfs").upload(
                file_path,
                temp_file_path,
                file_options={"content-type": "application/pdf"}
            )
            
            # Remove temporary file after upload
            os.remove(temp_file_path)
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            logger.error(f"Storage upload error: {e}")
            raise HTTPException(status_code=500, detail=f"Error uploading to storage: {str(e)}")
        
        # Extract text from PDF
        text_content = extract_text_from_pdf(content)
        
        # Create database entry with owner_id
        new_document = {
            "filename": file.filename,
            "file_path": file_path,
            "content_text": text_content,
            "owner_id": current_user["id"]
        }
        
        document_response = supabase.table("documents").insert(new_document).execute()
        
        if not document_response.data or len(document_response.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create document record")
        
        created_doc = document_response.data[0]
        
        return PDFDocumentSchema(
            id=created_doc["id"],
            filename=created_doc["filename"],
            upload_date=created_doc["upload_date"]
        )
    except Exception as e:
        logger.error(f"Error during PDF upload: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/documents/", response_model=List[PDFDocumentSchema])
async def list_documents(current_user = Depends(get_current_active_user)):
    # Only return documents owned by the current user
    try:
        documents_response = supabase.table("documents") \
            .select("id,filename,upload_date") \
            .eq("owner_id", current_user["id"]) \
            .execute()
        
        documents = documents_response.data
        return documents
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching documents: {str(e)}")

@app.post("/ask-free/", response_model=QuestionResponse)
async def ask_free_question(
    question_data: QuestionCreate,
    current_user = Depends(get_current_active_user)
):
    # Check if user has reached the free question limit (3 questions)
    try:
        free_questions_response = supabase.table("questions") \
            .select("id") \
            .eq("user_id", current_user["id"]) \
            .eq("is_free_question", True) \
            .execute()
        
        free_questions = free_questions_response.data
        
        if free_questions and len(free_questions) >= 3:
            raise HTTPException(
                status_code=403, 
                detail="You have reached your free question limit. Please upload and select documents to continue."
            )
        
        # Generate answer using Claude
        try:
            message = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                temperature=0,
                system="You are a helpful assistant that answers general questions. You can help with a wide range of topics but will be honest when you don't know something.",
                messages=[
                    {
                        "role": "user",
                        "content": f"Question: {question_data.question_text}"
                    }
                ]
            )
            
            answer = message.content[0].text
            
            # Save question and answer to database as a free question
            new_question = {
                "document_id": None,  # No document associated
                "question_text": question_data.question_text,
                "answer_text": answer,
                "user_id": current_user["id"],
                "is_free_question": True
            }
            
            question_response = supabase.table("questions").insert(new_question).execute()
            
            if not question_response.data or len(question_response.data) == 0:
                raise HTTPException(status_code=500, detail="Failed to save question")
            
            created_question = question_response.data[0]
            
            return QuestionResponse(
                id=created_question["id"],
                question_text=created_question["question_text"],
                answer_text=created_question["answer_text"],
                timestamp=created_question["timestamp"]
            )
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in free question: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error processing free question: {str(e)}")

@app.post("/ask/", response_model=QuestionResponse)
async def ask_question(
    question_data: QuestionCreate,
    current_user = Depends(get_current_active_user)
):
    # If no document IDs are provided, redirect to free question endpoint
    if not question_data.document_ids:
        return await ask_free_question(question_data, current_user)
        
    try:
        # Retrieve the documents and verify ownership
        documents = []
        for doc_id in question_data.document_ids:
            doc_response = supabase.table("documents") \
                .select("*") \
                .eq("id", doc_id) \
                .eq("owner_id", current_user["id"]) \
                .execute()
            
            doc_data = doc_response.data
            
            if doc_data and len(doc_data) > 0:
                documents.append(doc_data[0])
            else:
                raise HTTPException(status_code=404, detail=f"Document with ID {doc_id} not found or access denied")

        if not documents:
            raise HTTPException(status_code=404, detail="No valid documents found")
        
        # Prepare context from all documents
        context = ""
        for i, doc in enumerate(documents):
            context += f"\nDocument {i+1} ({doc['filename']}):\n{doc['content_text']}\n"
        
        # Generate answer using Claude
        try:
            message = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                temperature=0,
                system="You are a helpful assistant that answers questions based on the provided PDF documents. Only use information from the provided documents to answer questions. If the information is not in the documents, say you don't know.",
                messages=[
                    {
                        "role": "user",
                        "content": f"Here are the contents of the documents:\n{context}\n\nQuestion: {question_data.question_text}"
                    }
                ]
            )
            
            answer = message.content[0].text
            
            # Save question and answer to database
            new_question = {
                "document_id": documents[0]["id"],  # Link to the first document for simplicity
                "question_text": question_data.question_text,
                "answer_text": answer,
                "user_id": current_user["id"],
                "is_free_question": False
            }
            
            question_response = supabase.table("questions").insert(new_question).execute()
            
            if not question_response.data or len(question_response.data) == 0:
                raise HTTPException(status_code=500, detail="Failed to save question")
            
            created_question = question_response.data[0]
            
            return QuestionResponse(
                id=created_question["id"],
                question_text=created_question["question_text"],
                answer_text=created_question["answer_text"],
                timestamp=created_question["timestamp"]
            )
        except Exception as e:
            logger.error(f"Error generating document answer: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing document question: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in document question: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error processing document question: {str(e)}")

@app.get("/questions/", response_model=List[QuestionResponse])
async def list_questions(
    document_id: Optional[int] = None,
    current_user = Depends(get_current_active_user)
):
    try:
        # Base query for user's questions
        query = supabase.table("questions") \
            .select("id,question_text,answer_text,timestamp") \
            .eq("user_id", current_user["id"])
        
        # Add document filter if provided
        if document_id:
            # Verify document belongs to user
            doc_response = supabase.table("documents") \
                .select("id") \
                .eq("id", document_id) \
                .eq("owner_id", current_user["id"]) \
                .execute()
            
            if not doc_response.data or len(doc_response.data) == 0:
                raise HTTPException(status_code=404, detail="Document not found or access denied")
                
            query = query.eq("document_id", document_id)
        
        # Execute the query
        response = query.execute()
        questions = response.data
        return questions
    except Exception as e:
        logger.error(f"Error listing questions: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error fetching questions: {str(e)}")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)