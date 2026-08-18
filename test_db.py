import sys
from database.db import get_connection

# Show which Python interpreter is being used
print("Python Executable:", sys.executable)

try:
    conn = get_connection()

    if conn.is_connected():
        print("✅ Database Connected Successfully!")
    else:
        print("❌ Connection Failed")

    conn.close()

except Exception as e:
    print("❌ Error:", e)