from requests import Session

class User(object):
    """
    class for user credentials and sending and posting requests
    """

    def __init__(self, username, password):
        super(User, self).__init__()
        self.username = username
        self.password = password
        self.session = Session()
        self.logged_in = False

    def __del__(self):
        self.session.close()

    def login(self):
        self.logged_in = True

    def logout(self):
        self.logged_in = False

    def check_login(self):
        if self.logged_in is False:
            raise AUTHError('%s is not logged in.' % self.username)

    def post(self, url, **kwargs):
        response = self.session.post(url, **kwargs)
        return response

    def get(self, url, **kwargs):
        response = self.session.get(url, **kwargs)
        return response

    @staticmethod
    def check_response(response):
        if response.ok:
            return response.json()
        else:
            response.raise_for_status()

class AUTHError(Exception):
    pass