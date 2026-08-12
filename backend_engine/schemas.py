from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
import datetime

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    pin: str = Field(..., min_length=6, max_length=6)
    api_key: str
    client_id: str
    password: str
    totp_secret: str

class UserLogin(UserBase):
    pin: str = Field(..., min_length=6, max_length=6)

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None
