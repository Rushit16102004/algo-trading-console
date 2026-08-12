import os
from cryptography.fernet import Fernet
from backend_engine.config import ENCRYPTION_KEY

class CredentialsEncryptor:
    def __init__(self):
        # Fernet requires a 32-byte url-safe base64-encoded key
        key = ENCRYPTION_KEY.encode()
        self.cipher = Fernet(key)
        
    def encrypt(self, plain_text: str) -> str:
        if not plain_text:
            return ""
        return self.cipher.encrypt(plain_text.encode()).decode()
        
    def decrypt(self, cipher_text: str) -> str:
        if not cipher_text:
            return ""
        return self.cipher.decrypt(cipher_text.encode()).decode()

encryptor = CredentialsEncryptor()
