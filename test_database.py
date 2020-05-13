from psycopg2 import connect
from .database import PGDataBase
import json
import pytest

with open('nogit_credentials.json', 'r') as f:
    creds = json.load(f)

class Test_database:

    def teardown_method(self):
        ''' cleanse the database '''
        conn = connect(**creds)
        with conn:
            with conn.cursor() as curs:
                curs.execute(
                    """
                    SELECT relname
                    FROM pg_class
                    WHERE relkind='r'
                    AND relname !~ '^(pg_|sql_)'
                    """
                )
                tables = curs.fetchall()
        for (table,) in tables:
            print(table, '', end='')
            with conn:
                with conn.cursor() as curs:
                    curs.execute(f'drop table {table} cascade')
                    print('dropped')

    def test_connection(self):
        connect(**creds).close()

    def test_basic(self):
        db = PGDataBase(creds)
        db.create_table('test_table', 'i INT, j CHAR(5)')
        assert len(db.tables) == 1
        assert 'test_table' in db.tables

    def test_table_primarykey(self):
        db = PGDataBase(creds)
        db.create_table('my_table',
                        'column1 int, column2 char(50), column3 int, primary key (column1, column2)',
        )
        pks = db.tables['my_table'].primary_key
        assert len(pks) == 2
        assert 'column1' in pks
        assert 'column2' in pks

    def test_add_rows(self):
        db = PGDataBase(creds)
        db.create_table('my_table',
                        'pk int, col2 VARCHAR(20)')
        db.tables['my_table'].insert_rows(
            ((i, f'{("a"*i)[:20]:< 20}') for i in range(50))
        )

    def test_key_present(self):
        db = PGDataBase(creds)
        raise NotImplementedError
