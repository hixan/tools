from psycopg2 import connect
from .database import PGDataBase
import json
import pytest

# in order for this testing module to work, a credentials file must be created
# pointing to an empty (and unused) postgres SQL database. THE TESTS WILL
# CLEAR ALL TABLES IN THE DATABASE! So dont use a database with actual data
# stored.


with open('nogit_credentials.json', 'r') as f:
    creds = json.load(f)


class Test_database:

    def _get_tables(self):
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
                return [t[0] for t in curs.fetchall()]

    def setup_method(self):
        print('setup')
        assert len(self._get_tables()) == 0

    def teardown_method(self):
        ''' cleanse the database '''
        print('teardown')
        conn = connect(**creds)
        tables = self._get_tables()
        for table in tables:
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
        db.create_table(
            'my_table',
            '''column1 int, column2 char(50), column3 int,
            primary key (column1, column2)'''
        )
        pks = db.tables['my_table'].primary_key
        assert len(pks) == 2
        assert 'column1' in pks
        assert 'column2' in pks

    @pytest.mark.parametrize('batch_size', (4, 2, 1, 10000))
    @pytest.mark.parametrize(
        "table_defn,columns_inserted,rows_inserted,rows_actual,rows_missing",
        [
            (  # test 1
                'x int primary key, y char(1)',  # table definition
                ('x', 'y'),  # columns inserted
                (  # rows inserted
                    (1, 'c'), (2, 'b'), (3, 'c'), (4, 'a')
                ),
                (  # rows actual
                    (1, 'c'), (2, 'b'), (3, 'c'), (4, 'a')
                ),
                (),  # rows missing
            ),
            (  # test 2
                'x int, y int, z char(1), primary key (x, z)',
                ('x', 'z'),
                (
                    (1, 'c'), (2, 'b'), (3, 'c'), (4, 'a')
                ),
                (
                    (1, None, 'c'), (2, None, 'b'), (3, None, 'c'),
                    (4, None, 'a')
                ),
                ()
            ),
            (
                'x int primary key, y char(1)',
                ('x', 'y'),
                (
                    (1, 'c'), (2, 'b'), (2, 'c'),  # duplicate shouldnt work
                    (3, 'c'), (4, 'a')
                ),
                (
                    (1, 'c'), (2, 'b'), (3, 'c'), (4, 'a')
                ),
                ((2, 'c'),)
            ),
        ]
    )
    def test_insert(self, table_defn, columns_inserted, rows_inserted,
                    rows_actual, rows_missing, batch_size):
        db = PGDataBase(creds)
        db.create_table('my_table', table_defn)
        table = db.tables['my_table']
        c = db._connect()
        missing_rows = table.insert_rows(columns_inserted, rows_inserted, c,
                                         batch_size)
        curs = c.cursor()
        curs.execute('select * from my_table')
        res = curs.fetchall()
        curs.close()
        c.close()

        print('inserted rows:')
        print(*rows_inserted, sep='\n')
        print('resultant rows:')
        print(*res, sep='\n')
        print('expected rows:')
        print(*rows_actual, sep='\n')
        print('rows missing:')
        print(*missing_rows, sep='\n')
        print('expected rows missing:')
        print(*rows_missing, sep='\n')

        assert self.sequences_equal(res, rows_actual)
        assert self.sequences_equal(rows_missing, missing_rows)

    @staticmethod
    def sequences_equal(a, b):
        for i in a:
            if i not in b:
                return False
        for i in b:
            if i not in a:
                return False
        if len(a) != len(b):
            return False
        return True

    def test_key_present(self):
        db = PGDataBase(creds)
        raise NotImplementedError
