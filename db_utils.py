import psycopg2

def get_connection():
    return psycopg2.connect(
        dbname="faces_db",
        user="postgres",
        password="nitya123",
        host="localhost",
        port="5432"
    )
