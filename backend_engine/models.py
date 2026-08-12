import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from backend_engine.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="USER") # ADMIN, USER
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    broker_accounts = relationship("BrokerAccount", back_populates="user", cascade="all, delete-orphan")
    paper_trades = relationship("PaperTrade", back_populates="user", cascade="all, delete-orphan")
    strategy_runs = relationship("StrategyRun", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")

class BrokerAccount(Base):
    __tablename__ = "broker_accounts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    broker = Column(String, default="ANGEL_ONE")
    client_id = Column(String, nullable=False)
    encrypted_api_key = Column(String, nullable=False)
    encrypted_password = Column(String, nullable=False)
    encrypted_totp_secret = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="broker_accounts")

class PaperTrade(Base):
    __tablename__ = "paper_trades"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    strategy = Column(String, nullable=False, index=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False) # BUY, SELL / LONG, SHORT
    quantity = Column(Integer, default=1)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=True)
    pnl = Column(Float, default=0.0)
    status = Column(String, default="OPEN") # OPEN, CLOSED
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="paper_trades")

class StrategyRun(Base):
    __tablename__ = "strategy_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    strategy = Column(String, nullable=False, index=True)
    mode = Column(String, nullable=False) # DEMO, PAPER, LIVE
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    stopped_at = Column(DateTime, nullable=True)
    status = Column(String, default="RUNNING") # RUNNING, STOPPED
    
    user = relationship("User", back_populates="strategy_runs")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    event = Column(String, nullable=False, index=True)
    details = Column(String, nullable=True)
    request_id = Column(String, nullable=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="audit_logs")

# Add database indexes where appropriate
Index("ix_users_email", User.email)
Index("ix_audit_logs_event_timestamp", AuditLog.event, AuditLog.timestamp)
Index("ix_paper_trades_user_strategy", PaperTrade.user_id, PaperTrade.strategy)
