from psycopg2 import connect
from psycopg2.extensions import Column, connection as Connection  # Capitalise for readability as object
from typing import Union, Mapping, Sequence, Iterable, TypeVar, Tuple
from .tools import batch


Value = Union[bool, int, float, str]  # value of a returned database object
T = TypeVar('T')

class PGDataBase:

    def __init__(self, credentials):
        self._credentials = credentials
        self.tables = {}
        self._init_tables()

    def _init_tables(self):
        ''' syncronisis tables with the database '''
        # fetch table information
        tableconn = self._connect()
        present = set()
        for table in self._get_tables():
            self.tables[table] = self.Table(table, tableconn)
            present.add(table)
        tableconn.close()

        # remove missing tables
        for t in set(self.tables.keys()) - present:
            del self.tables[t]

    def _get_tables(self):
        '''
        retrieve table names from the database. Does not commit. Is not a
        generator.
        '''
        with self._connect().cursor() as cursor:
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
        return connect(**self._credentials)

    def create_table(
            self,
            name: str,
            columnsdef: str,
            connection: Connection = None,
    ):
        sql = f'''CREATE TABLE {name} (
        {columnsdef}
        )
        '''
        if connection is None:
            commit = True
            connection = self._connect()
        with connection.cursor() as cur:
            cur.execute(sql)

        if commit:
            connection.commit()
            connection.close()
        self._init_tables()

    class Table:
        ''' table always manages its transactions '''

        def __init__(self, table: str, connection: Connection):
            self._tablename = table
            cols = self._fetch_columns(connection)
            self.columns: Mapping[str, Column] = {c.name: c for c in cols}
            self.primary_key = self._fetch_pks(connection)
            print(self.primary_key)
            connection.commit()

        def _fetch_columns(self, connection: Connection):
            with connection.cursor() as cursor:
                cursor.execute(f'SELECT * FROM {self._tablename} LIMIT 0')
                return cursor.description

        def _fetch_pks(self, connection: Connection):
            with connection.cursor() as cursor:
                cursor.execute(f'''
                SELECT a.attname
                FROM pg_index i
                     JOIN
                     pg_attribute a
                     ON a.attrelid = i.indrelid
                     AND a.attnum = ANY(i.indkey)
                WHERE  i.indrelid = '{self._tablename}'::regclass
                AND i.indisprimary;''')
                res = cursor.fetchall()
            return [r[0] for r in res]  # unpack the keys

        def __contains__(self, primary_key: Sequence[Value]):
            if len(primary_key) != 0:
                pass
            return

        def insert_rows(self, rows, connection: Connection, batch_size=20):
            pass


        def __len__(self):
            raise NotImplementedError
