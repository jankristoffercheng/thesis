from connection.Connection import Connection

class UsersDAO:
    def addUser(self, username, gender, birthyear, birthmonth, birthday, source):
        conn = Connection().getConnection()
        cursor = conn.cursor()
        sql = 'INSERT INTO users(username, gender, birthdate, source) VALUES (' + \
            '\'' + username + '\', ' + \
            '\'' + gender + '\', ' + \
            '\'' + birthyear + '-' + birthmonth + '-' + birthday + '\', ' + \
            '\'' + source + '\')'
        print(sql)
        cursor.execute(sql)