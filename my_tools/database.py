from psycopg2.extensions import connection as Connection  # Capitalise for readability as object
from psycopg2.errors import DatabaseError, ProgrammingError
from logging import Logger
from typing import Optional, List, Sequence, Generator, Union, Iterable, Iterator, Callable
from itertools import chain
import numpy as np
import abc


LOGGER: Optional[Logger] = None  # global, can be a logging.logger object

# Custom Types
Value = Union[bool, int, float, str]

Row = Iterable[Value]


def dbexec(
        connection: Connection,
        sql: str,
        args: List[str] = None,
        msg: str = '',
        commit: bool = True,
):
    '''
    runs  a single <sql> command with <connection>. Does raise sql errors but
    logs them. Values returned from the db are not returned here.
    '''

    if LOGGER:
        LOGGER.debug(msg, sql.replace('\n', ' '), args)
    with connection.cursor() as cur:
        try:
            cur.execute(sql, args)
        except (DatabaseError, ProgrammingError):
            if LOGGER:
                LOGGER.warning(
                    f'Database error encountered. (sql command, args)',
                    ' '.join(sql.split()), args)
            raise
    if commit:
        connection.commit()


def dbquery(connection: Connection,
            sql: str,
            args: Sequence[str] = None,
            msg: str = '') -> Generator[Sequence[Value], None, None]:
    if LOGGER:
        LOGGER.debug(sql.replace('\n', ' '))
    with connection:
        with connection.cursor() as cur:
            cur.execute(sql, args)
            for record in cur:
                yield record


def create_table(
    connection: Connection,
    tablename: str,
    tablecols: Iterable[str],  # including type and constraints
    tableconst: Iterable[str] = (),
    delete_existing: bool = True,
):
    ''' deletes (if exists), then creates table in database using connection '''

    if delete_existing:
        dbexec(connection,
               f'DROP TABLE IF EXISTS {tablename} CASCADE',
               msg=f"Delete table '{tablename}'")

    sqlcols = ','.join(tablecols)
    sqlconsts = ''.join(map(lambda x: f', CONSTRAINT {x}', tableconst))
    sql = f'''CREATE TABLE IF NOT EXISTS {tablename} (
    {sqlcols}
    {sqlconsts}
    )
    '''
    dbexec(connection, sql, msg=f"Create table '{tablename}'")


def add_rows(connection: Connection, tablename: str, values: Iterable[List[str]]):
    '''
    Populates table <tablename> with rows <values>. (They must already be clean)

    :param connection: Connection object to database
    :param tablename: name of table to insert data into
    :param values: Values to insert
    :param msg: message for debugging / logging
    '''

    # extract the first value
    it: Iterator[Row] = iter(values)
    firstval = next(it)
    it = chain([firstval], it)

    if isinstance(firstval, abc.Mapping):
        valuenames = firstval.keys()
        valuefmts = ','.join(map(lambda x: f'%({x})s', valuenames))
        valuenamess = ','.join(valuenames)
        sqlcmd = f'''
        INSERT INTO {tablename} (
        {valuenamess}
        )
        VALUES (
        {valuefmts}
        )
        '''
    elif isinstance(firstval, abc.Sequence) or isinstance(
            firstval, np.ndarray):
        valuefmts = ('%s,' * len(firstval))[:-1]
        sqlcmd = f'''
        INSERT INTO {tablename}
        VALUES (
        {valuefmts}
        )
        '''
    else:
        print(type(firstval))
        raise Exception  # TODO raise a sutible exception (should only raise
    # when transform_row is not compatible.

    for row in it:
        try:
            dbexec(connection,
                   sqlcmd,
                   args=row,
                   msg=f'Insert row to {tablename}')
        except KeyboardInterrupt:
            raise
        except:
            LOGGER.error('An unknown error occured while inserting a row.' +
                         ' (table, row, types in row)',
                         tablename,
                         row,
                         list(map(type, row)),
                         exc_info=True)


def clean_rows(
    dirty_rows: Iterable[Sequence[str]],
    transform_row: Callable[[Sequence[str]], Iterator[Row]]
) -> Iterator[Row]:
    for drow in dirty_rows:
        for crow in transform_row(drow):
            yield crow
