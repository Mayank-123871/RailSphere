import os
import logging
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -------------------------------------------------
# Logging Configuration
# -------------------------------------------------

logging.basicConfig(
    filename="railsphere.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Database Connection
# -------------------------------------------------

def get_connection():

    try:

        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            use_pure=True
        )

        if connection.is_connected():

            logger.info(
                "MySQL database connection established successfully."
            )

            print("✅ Database connected successfully")

            return connection

        raise ConnectionError(
            "Unable to establish MySQL database connection."
        )

    except Error as e:

        logger.error(
            f"MySQL connection error: {e}"
        )

        print(
            f"❌ Database connection failed: {e}"
        )

        raise


# -------------------------------------------------
# Close Database Connection
# -------------------------------------------------

def close_connection(connection):

    try:

        if connection and connection.is_connected():

            connection.close()

            logger.info(
                "MySQL database connection closed."
            )

    except Error as e:

        logger.error(
            f"Error while closing database connection: {e}"
        )