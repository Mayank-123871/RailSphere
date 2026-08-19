import os
import logging
import mysql.connector

from mysql.connector import Error
from dotenv import load_dotenv


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    filename="railsphere.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# DATABASE CONFIGURATION
# =========================================================

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")


# =========================================================
# CONFIGURATION VALIDATION
# =========================================================

def validate_database_config():

    missing_variables = []

    if not DB_USER:
        missing_variables.append("DB_USER")

    if not DB_PASSWORD:
        missing_variables.append("DB_PASSWORD")

    if not DB_NAME:
        missing_variables.append("DB_NAME")

    if missing_variables:

        message = (
            "Missing database environment variables: "
            + ", ".join(missing_variables)
        )

        logger.error(message)

        raise EnvironmentError(message)


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():

    validate_database_config()

    connection = None

    try:

        connection = mysql.connector.connect(

            host=DB_HOST,

            port=DB_PORT,

            user=DB_USER,

            password=DB_PASSWORD,

            database=DB_NAME,

            connection_timeout=10,

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
            "MySQL connection error: %s",
            e,
            exc_info=True
        )

        print(
            f"❌ Database connection failed: {e}"
        )

        if connection and connection.is_connected():

            connection.close()

        raise


# =========================================================
# CLOSE DATABASE CONNECTION
# =========================================================

def close_connection(connection):

    if connection is None:
        return

    try:

        if connection.is_connected():

            connection.close()

            logger.info(
                "MySQL database connection closed successfully."
            )

    except Error as e:

        logger.error(
            "Error while closing database connection: %s",
            e,
            exc_info=True
        )


# =========================================================
# DATABASE HEALTH CHECK
# =========================================================

def check_database_health():

    connection = None

    try:

        connection = get_connection()

        cursor = connection.cursor()

        cursor.execute("SELECT 1")

        result = cursor.fetchone()

        cursor.close()

        if result and result[0] == 1:

            logger.info(
                "Database health check passed."
            )

            return True

        return False

    except Exception as e:

        logger.error(
            "Database health check failed: %s",
            e,
            exc_info=True
        )

        return False

    finally:

        close_connection(connection)