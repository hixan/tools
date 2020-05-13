import sys
print(sys.path)
from psycopg2 import connect
from psycopg2.errors import DatabaseError, ProgrammingError
from psycopg2.extensions import Column, connection as Connection  # Capitalise for readability as object
from logging import Logger
from typing import Union, Mapping


Value = Union[bool, int, float, str]  # value of a returned database object


class PGDataBase:

    def __init__(self, credentials):
        self._credentials = credentials
        for table in self._get_tables():
            pass

    def _get_tables(self, connection):
        ''' retrieve table names from the database. Does not commit. '''
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT relname
                FROM pg_class
                WHERE relkind='r'
                AND relname !~ '^(pg_|sql_)'
                """)
            rv = cursor.fetchall()
        return [t[0] for t in rv]

    def _connect(self):
        return connect(self._credentials)

    class Table:
        def __init__(self, table, connection):
            self._tablename = table
            with connection:
                cols = self._fetch_columns(connection)
            self.columns: Mapping[str, Column] = {c.name: c for c in cols}

        def _fetch_columns(self, connection: Connection):
            with connection.cursor() as cursor:
                cursor.execute(f'SELECT * FROM {self._tablename} LIMIT 0')
                return cursor.describe

        def _fetch_pks(self, connection: Connection):
            with connection.cursor() as cursor:
                cursor.execute(f''' SELECT a.attname FROM   pg_index i JOIN   pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) WHERE  i.indrelid = '{self._tablename}'::regclass AND    i.indisprimary;''')
                res = cursor.fetchall()
            print(res)

        def __contains__(self, primary_key):
            if len(primary_key) != 0:
                pass
            return
