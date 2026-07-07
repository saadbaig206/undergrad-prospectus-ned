from typing import List, Dict, Optional
from pydantic import BaseModel

class AuthRequest(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    query: str
    history: Optional[List[Dict[str, str]]] = []

class CreateAdminRequest(BaseModel):
    username: str
    password: str
