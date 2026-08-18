import os
import mysql.connector
from dotenv import load_dotenv

# Load .env
load_dotenv()


def get_connection():
    print("✅ USING SECURE DB CONFIGURATION")

    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME", "railsphere"),
        use_pure=True
    )