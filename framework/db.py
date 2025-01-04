from sqlalchemy import create_engine, Column, DateTime, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import os
import uuid
from datetime import datetime

# "postgresql://user:pass@localhost/dbname" or "sqlite:///./test.db"
DATABASE_URL = os.getenv("DATABASE_URL") 


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class JobModel(Base):
    __tablename__ = "jobs"

    job_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    status = Column(Text, index=True, default="running")
    stdout = Column(Text, nullable=True)
    stderr = Column(Text, nullable=True)
    exit_code = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

Base.metadata.create_all(bind=engine)