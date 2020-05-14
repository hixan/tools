from psycopg2 import connect
from psycopg2.extensions import Column, connection as Connection  # Capitalise for readability as object
import psycopg2.errors as pge
from typing import Union, Mapping, Sequence, Iterable, TypeVar, Tuple, Deque, Dict
from itertools import chain
from .tools import batch
from collections import deque


Value = Union[bool, int, float, str]  # value of a returned database object
T = TypeVar('T')


class PGDataBase:

    def __init__(self, credentials):
        self._credentials = credentials
        self.tables: Dict[str, PGDataBase.Table] = {}
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

    def _get_tables(self) -> Sequence[str]:
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

    def _connect(self) -> Connection:
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

        def insert_rows(
                self,
                columns: Sequence[str],
                rows: Iterable[Tuple[str, ...]],
                connection: Connection,
                batch_size: int = 20
        ):
            ''' inserts rows in batches. (order is not guaranteed) '''
            sql = f'insert into {self._tablename}({",".join(columns)}) values '
            row_template = '(' + ','.join(['%s'] * len(columns)) + ')'
            normal_template = sql + ','.join([row_template] * batch_size)
            handler_queue: Deque[Tuple[str, ...]] = deque()

            curs = connection.cursor()

            # insert in batches
            for b in batch(rows, batch_size):
                # attempt to insert batch
                curs.execute('savepoint start_batch')
                try:
                    if len(b) == batch_size:
                        curs.execute(normal_template, tuple(chain.from_iterable(b)))
                    else:
                        curs.execute(sql + ','.join([row_template] * len(b)),
                                     tuple(chain.from_iterable(b)))

                    curs.execute('release savepoint start_batch')
                except (pge.DataError, pge.IntegrityError):
                    # add batch to the queue to be handled later
                    handler_queue.extend(b)
                    # roll back to clean state
                    curs.execute('rollback to savepoint start_batch')

            retval = []
            # handle the queue
            for row in handler_queue:
                curs.execute('savepoint start_insert')
                try:
                    curs.execute(sql+row_template, row)
                except (pge.DataError, pge.IntegrityError):
                    retval.append(row)  # this row could not be inserted
                    curs.execute('rollback to savepoint start_insert')
                curs.execute('release savepoint start_insert')

            curs.close()
            connection.commit()
            return retval

        def __len__(self):
            raise NotImplementedError
