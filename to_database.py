import psycopg2 as pg
import numpy as np
import struct

conn = pg.connect(database="digger", user="postgres", password="root", host="localhost", port="5432")

db_cur = conn.cursor()

def insert_data(pic_id,url,is_pro):
    try:
        db_cur.execute("INSERT INTO URLS(ID,URL,IS_PRO) VALUES(%s,%s,%s);",(pic_id,url,is_pro))
        conn.commit()
        print('url inserted')
    except Exception as e:
        print(e)
        conn.rollback()

def select_data():
    try:
        db_cur.execute("SELECT * FROM DATABASE;")
        data = db_cur.fetchall()
        return data
    except Exception as e:
        print(e)

def close_conn():
    conn.close()

def main():
    db_cur.execute("SELECT * FROM DATABASE;")
    data = db_cur.fetchall()
    print(data[0])
    conn.close()

if __name__ == '__main__':
    main()
