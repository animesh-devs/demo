from jose import jwt, JWTError
from typing import Dict, Any, Optional
from fastapi import HTTPException, status
import logging

from app.config import settings

logger = logging.getLogger(__name__)

def verify_token(token: str) -> Dict[str, Any]:
    """Verify a JWT token and return the payload"""
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=["HS256"]
        )
        return payload
    except JWTError as e:
        logger.error(f"JWT verification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Error verifying token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
