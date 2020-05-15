from psycopg2 import connect  # type: ignore
# Capitalise for readability as object
from psycopg2.extensions import Column, connection as Connection  # type: ignore
import psycopg2.errors as pge  # type: ignore
from typing import (Union, Mapping, Sequence, Iterable, TypeVar, Tuple, Deque,
                    Dict, Optional, Collection, List)
from itertools import chain
from .tools import batch
from collections import deque
from datetime import datetime
import atexit


Value = Union[bool, float, int, str, datetime, None]  # value of a returned database object
T = TypeVar('T')


class PGDataBase:

    def __init__(self, credentials):
        ''' Create a database object to communicate with the database
        at <credentials>. Credentials follows the psycopg2.connect
        standard, and so needs:
            port (if not default), database, host, user and password.
        See psycopg2.connect for more details
        '''
        self._credentials = credentials
        self._tables: Dict[str, PGDataBase.Table] = {}
        self._connection_pool: List[Connection] = []
        self._opened_count = 0
        with self.connection() as conn:
            self._init_tables(conn)

        # disconnect when python exits
        atexit.register(self.disconnect_all)

    def _init_tables(self, connection):
        ''' syncronisis tables map with the database '''
        # fetch table information
        present = set()
        for table in self._get_tables(connection):
            self._tables[table] = self.Table(table, connection)
            present.add(table)

        # remove missing tables
        for t in set(self._tables.keys()) - present:
            del self._tables[t]

    def _get_tables(
            self, connection: Connection
    ) -> Sequence[str]:
        '''
        retrieve table names from the database. Does not commit. Is not a
        generator.
        '''
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

    def __getitem__(self, key: str):
        '''
        case insensitive indexing of the tables contained by the database
        '''
        return self._tables[key.lower()]

    def __contains__(self, table: str):
        ''' case insensitive containment test '''
        return table.lower() in self._tables

    def __iter__(self):
        ''' item retrieval (with case fixing) '''
        for item in self._tables:
            yield item.title()

    def create_table(
            self,
            name: str,
            columnsdef: str,
            connection: Connection = None,
    ):
        ''' Create a table in the database. THIS METHOD COMMITS CHANGES '''
        sql = f'''CREATE TABLE {name} (
        {columnsdef}
        )
        '''
        if connection is None:
            cmgr = self.connection()
            connection = cmgr.__enter__()
        else:
            cmgr = None

        with connection.cursor() as cur:
            cur.execute(sql)
        connection.commit()

        self._init_tables(connection)

        if cmgr:
            cmgr.__exit__(None, None, None)

    def disconnect_all(self):
        ''' close all connections to the database. This is automatically called
        on exit. '''
        for conn in self._connection_pool:
            conn.close()
        self._connection_pool = []

    def connection(self):
        ''' context manager for accessing connections to the database. Treat
        like psycopg2.connection
        >>> with db.connection() as connection:
        ...     # do stuff with conneciton, which is open
        ... # connections latest transaction is now committed.

        The connection should not be accessed after the with block as it is now
        owned by the database and not the current thread, and it may get lent
        again to another context while you are accessing it.
        '''
        return PGDataBase._ConnectionHandler(self)

    class _ConnectionHandler:

        def __init__(self, db):
            ''' Should not access this class directly. Use factory function
            'PGDataBase.connection()'.
            handle the connection safely closing with exceptions and
            efficiently allocate to tasks (if multiprocessing / nesting) '''
            self._db = db

        def __enter__(self):
            ''' enters this context and the connections context '''
            if len(self._db._connection_pool) == 0:
                self._db._connection_pool.append(connect(**self._db._credentials))
                self._db._opened_count += 1
            # get an open connection and discard closed connections
            self.connection = self._db._connection_pool.pop()
            while self.connection.closed:
                self.connection = self._db._connection_pool.pop()
            return self.connection.__enter__()

        def __exit__(self, exc_type, exc_value, traceback):
            ''' exits this context and the connections context '''
            self.connection.__exit__(exc_type, exc_value, traceback)
            self._db._connection_pool.append(self.connection)

    class Table:

        def __init__(self, table: str, connection: Connection):
            ''' should not access this class directly, instead use the
            create_table method or the refresh method. '''
            self._tablename = table
            cols = self._fetch_columns(connection)
            self.columns: Mapping[str, Column] = {c.name: c for c in cols}
            self.primary_key = self._fetch_pks(connection)
            self._pk_cache: Optional[Collection[Tuple[Value, ...]]] = None

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

        def has_primary_key(self, primary_key: Tuple[Value, ...], connection: Connection):
            ''' returns true if the table contains a row with that primary key
            '''
            # fetch cache if necessary
            if self._pk_cache is None:
                self._pk_cache = set(self.select(self.primary_key, connection))
            return primary_key in self._pk_cache

        def insert_rows(
                self,
                columns: Sequence[str],
                rows: Iterable[Tuple[Value, ...]],
                connection: Connection,
                batch_size: int = 20
        ):
            ''' inserts rows in batches. (order is not guaranteed) Returns rows
            that it could not insert. DOES NOT COMMIT '''
            sql = f'insert into {self._tablename}({",".join(columns)}) values '
            row_template = '(' + ','.join(['%s'] * len(columns)) + ')'
            normal_template = sql + ','.join([row_template] * batch_size)
            handler_queue: Deque[Tuple[str, ...]] = deque()

            inserted_count = 0

            with connection.cursor() as curs:

                # reset pk cache
                self._pk_cache = None

                # insert in batches
                for b in batch(rows, batch_size):
                    # attempt to insert batch
                    curs.execute('savepoint start_batch')
                    try:
                        if len(b) == batch_size:
                            curs.execute(normal_template, tuple(chain.from_iterable(b)))
                            inserted_count += batch_size
                        else:
                            curs.execute(sql + ','.join([row_template] * len(b)),
                                         tuple(chain.from_iterable(b)))
                            inserted_count += len(b)

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
                        inserted_count += 1
                    except (pge.DataError, pge.IntegrityError) as e:
                        retval.append((row, e))  # this row failed
                        curs.execute('rollback to savepoint start_insert')
                    curs.execute('release savepoint start_insert')

            return inserted_count, retval

        def select(
                self, columns: Sequence[str], connection: Connection,
                where: Sequence[str] = None, limit: int = None
        ) -> Iterable[Tuple[Value, ...]]:
            ''' simple implementation of SQL select, IS NOT SQL INJECTION SAFE

            :param where: sequence of strings that correspond to conditions
            that must be true.
            :param limit: number of lines to return. None returns all.
            :return: generator of resultant rows.

            Since this returns a generator, the connection must be dedicated to
            this process while it still going to return new values.

            TODO allow argument passing to protect against SQL injection '''
            sql = f'''SELECT {",".join(columns)} FROM {self._tablename}
            {{where}}
            {{limit}}'''
            sql = sql.format(where=' AND '.join(where) if where is not None else '',
                             limit=f' LIMIT {limit}' if limit is not None else '')
            with connection.cursor() as curs:
                curs.execute(sql)
                while True:
                    res = curs.fetchone()
                    if res is None:
                        break
                    yield res
