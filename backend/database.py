import os
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, inspect, text  # pyre-ignore
from sqlalchemy.orm import declarative_base, sessionmaker  # pyre-ignore
from dotenv import load_dotenv  # pyre-ignore
import uuid
import datetime

load_dotenv()

# Use SQLite for local development by default
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sortiq.db")

# For SQLite, we need connect_args to allow multiple threads
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class WasteScan(Base):
    __tablename__ = "waste_scans"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    predicted_class = Column(String, index=True)
    confidence = Column(Float)
    image_thumbnail_url = Column(String, nullable=True)
    interaction_type = Column(String, default="waste", index=True)
    robot_message = Column(String, nullable=True)

def create_tables():
    from logger import logger  # pyre-ignore
    logger.info("Creating database tables")
    # Create any missing tables based on models
    Base.metadata.create_all(bind=engine)

    # Development-friendly migration: ensure new columns exist on legacy DBs
    try:
        inspector = inspect(engine)
        if "waste_scans" in inspector.get_table_names():
            existing_cols = [c["name"] for c in inspector.get_columns("waste_scans")]
            with engine.connect() as conn:
                if "interaction_type" not in existing_cols:
                    logger.info("Adding missing column 'interaction_type' to waste_scans")
                    conn.execute(text("ALTER TABLE waste_scans ADD COLUMN interaction_type TEXT DEFAULT 'waste'"))
                if "robot_message" not in existing_cols:
                    logger.info("Adding missing column 'robot_message' to waste_scans")
                    conn.execute(text("ALTER TABLE waste_scans ADD COLUMN robot_message TEXT"))
                try:
                    conn.commit()
                except Exception:
                    # Some SQLAlchemy/DBAPI setups auto-commit; ignore commit failures
                    pass
    except Exception as e:
        logger.warning(f"Could not run lightweight migration: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
