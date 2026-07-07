from pydantic import BaseModel

class AuthRequest(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    query: str

class CreateAdminRequest(BaseModel):
    username: str
    password: str
